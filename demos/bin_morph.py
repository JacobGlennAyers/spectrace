"""
Binary morphology operations for whale vocalization masks.

This module provides utilities for applying morphological operations
to binary masks stored in HDF5 files.
"""

import numpy as np
import h5py
from scipy import ndimage
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MorphologyResult:
    """Result of a morphology operation."""
    original_mask: np.ndarray
    processed_mask: np.ndarray
    operation: str
    kernel_size: int
    pixel_count_before: int
    pixel_count_after: int
    pixel_change: int
    
    @property
    def pixel_change_percent(self) -> float:
        """Percentage change in pixel count."""
        if self.pixel_count_before == 0:
            return 0.0
        return 100 * self.pixel_change / self.pixel_count_before


# =============================================================================
# Morphology Operations
# =============================================================================

class MaskMorphology:
    """
    Binary morphology operations for masks.
    
    Usage:
        morph = MaskMorphology()
        result = morph.apply(mask, operation="dilation", kernel_size=3)
        
        # Or use specific methods
        dilated = morph.dilate(mask, kernel_size=3)
        eroded = morph.erode(mask, kernel_size=3)
    """
    
    OPERATIONS = ["erosion", "dilation", "opening", "closing"]
    
    @staticmethod
    def _create_kernel(kernel_size: int, shape: str = "square") -> np.ndarray:
        """
        Create a structuring element (kernel).
        
        Args:
            kernel_size: Size of the kernel (must be odd)
            shape: "square" or "cross"
        
        Returns:
            Binary structuring element
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        if shape == "square":
            struct = np.ones((kernel_size, kernel_size), dtype=bool)
        elif shape == "cross":
            struct = ndimage.generate_binary_structure(2, 1)
            # Iterate to get desired size
            iterations = (kernel_size - 1) // 2
            struct = ndimage.iterate_structure(struct, iterations)
        else:
            raise ValueError(f"Unknown shape: {shape}")
        
        return struct
    
    def apply(
        self,
        mask: np.ndarray,
        operation: Literal["erosion", "dilation", "opening", "closing"],
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> MorphologyResult:
        """
        Apply a morphology operation to a mask.
        
        Args:
            mask: Binary mask (H, W)
            operation: Type of operation
            kernel_size: Size of structuring element (must be odd)
            kernel_shape: "square" or "cross"
        
        Returns:
            MorphologyResult with original and processed masks
        """
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}. Choose from {self.OPERATIONS}")
        
        # Ensure mask is binary
        mask = mask.astype(bool)
        
        # Create structuring element
        struct = self._create_kernel(kernel_size, kernel_shape)
        
        # Apply operation
        if operation == "erosion":
            processed = ndimage.binary_erosion(mask, structure=struct)
        elif operation == "dilation":
            processed = ndimage.binary_dilation(mask, structure=struct)
        elif operation == "opening":
            processed = ndimage.binary_opening(mask, structure=struct)
        elif operation == "closing":
            processed = ndimage.binary_closing(mask, structure=struct)
        
        # Convert back to uint8
        original_uint8 = mask.astype(np.uint8)
        processed_uint8 = processed.astype(np.uint8)
        
        # Count pixels
        pixel_count_before = int(mask.sum())
        pixel_count_after = int(processed.sum())
        pixel_change = pixel_count_after - pixel_count_before
        
        return MorphologyResult(
            original_mask=original_uint8,
            processed_mask=processed_uint8,
            operation=operation,
            kernel_size=kernel_size,
            pixel_count_before=pixel_count_before,
            pixel_count_after=pixel_count_after,
            pixel_change=pixel_change
        )
    
    def dilate(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> np.ndarray:
        """
        Apply dilation to a mask.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            kernel_shape: "square" or "cross"
        
        Returns:
            Dilated mask
        """
        result = self.apply(mask, "dilation", kernel_size, kernel_shape)
        return result.processed_mask
    
    def erode(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> np.ndarray:
        """
        Apply erosion to a mask.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            kernel_shape: "square" or "cross"
        
        Returns:
            Eroded mask
        """
        result = self.apply(mask, "erosion", kernel_size, kernel_shape)
        return result.processed_mask
    
    def open(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> np.ndarray:
        """
        Apply opening (erosion then dilation) to a mask.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            kernel_shape: "square" or "cross"
        
        Returns:
            Opened mask
        """
        result = self.apply(mask, "opening", kernel_size, kernel_shape)
        return result.processed_mask
    
    def close(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> np.ndarray:
        """
        Apply closing (dilation then erosion) to a mask.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            kernel_shape: "square" or "cross"
        
        Returns:
            Closed mask
        """
        result = self.apply(mask, "closing", kernel_size, kernel_shape)
        return result.processed_mask


# =============================================================================
# HDF5 Interface
# =============================================================================

class HDF5MorphologyProcessor:
    """
    Apply morphology operations to masks in HDF5 files.
    
    Usage:
        processor = HDF5MorphologyProcessor("path/to/file.hdf5")
        result = processor.apply_to_class(
            class_name="call_type_1",
            operation="dilation",
            kernel_size=3
        )
        
        # Save result
        processor.save_processed_mask(
            class_name="call_type_1",
            processed_mask=result.processed_mask,
            output_path="path/to/output.hdf5"
        )
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize processor.
        
        Args:
            hdf5_path: Path to HDF5 file
        """
        self.hdf5_path = Path(hdf5_path)
        self.morph = MaskMorphology()
        
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    def get_class_names(self) -> List[str]:
        """Get list of class names from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            return json.loads(f.attrs['class_names'])
    
    def get_non_empty_classes(self) -> List[str]:
        """Get list of classes with non-zero masks."""
        with h5py.File(self.hdf5_path, 'r') as f:
            class_names = json.loads(f.attrs['class_names'])
            masks = f['masks'][:]
            
            non_empty = []
            for idx, class_name in enumerate(class_names):
                if masks[idx].sum() > 0:
                    non_empty.append(class_name)
        
        return non_empty
    
    def load_class_mask(self, class_name: str) -> np.ndarray:
        """
        Load mask for a specific class.
        
        Args:
            class_name: Name of the class
        
        Returns:
            Binary mask (H, W)
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            class_names = json.loads(f.attrs['class_names'])
            
            if class_name not in class_names:
                raise ValueError(f"Class '{class_name}' not found. Available: {class_names}")
            
            idx = class_names.index(class_name)
            return f['masks'][idx, :, :]
    
    def load_spectrogram(self) -> np.ndarray:
        """Load spectrogram from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            return f['spectrogram'][:]
    
    def apply_to_class(
        self,
        class_name: str,
        operation: str,
        kernel_size: int = 3,
        kernel_shape: str = "square"
    ) -> MorphologyResult:
        """
        Apply morphology operation to a specific class mask.
        
        Args:
            class_name: Name of the class
            operation: Type of morphology operation
            kernel_size: Size of structuring element
            kernel_shape: "square" or "cross"
        
        Returns:
            MorphologyResult with original and processed masks
        """
        mask = self.load_class_mask(class_name)
        
        return self.morph.apply(
            mask=mask,
            operation=operation,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape
        )
    
    def save_processed_mask(
        self,
        class_name: str,
        processed_mask: np.ndarray,
        output_path: str,
        overwrite: bool = False
    ):
        """
        Save a processed mask to a new HDF5 file.
        
        This creates a copy of the original HDF5 file with one mask replaced.
        
        Args:
            class_name: Name of the class to replace
            processed_mask: Processed mask to save
            output_path: Path for output HDF5 file
            overwrite: Whether to overwrite existing file
        """
        output_path = Path(output_path)
        
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}. Set overwrite=True to replace.")
        
        # Load all data from original file
        with h5py.File(self.hdf5_path, 'r') as f_in:
            class_names = json.loads(f_in.attrs['class_names'])
            
            if class_name not in class_names:
                raise ValueError(f"Class '{class_name}' not found")
            
            # Create new file
            with h5py.File(output_path, 'w') as f_out:
                # Copy spectrogram
                f_out.create_dataset(
                    'spectrogram',
                    data=f_in['spectrogram'][:],
                    compression='gzip'
                )
                
                # Copy masks, replacing the specified class
                masks = f_in['masks'][:]
                idx = class_names.index(class_name)
                masks[idx] = processed_mask
                
                f_out.create_dataset('masks', data=masks, compression='gzip')
                
                # Copy metadata
                meta_group = f_out.create_group('metadata')
                for key, value in f_in['metadata'].attrs.items():
                    meta_group.attrs[key] = value
                
                # Copy class names
                f_out.attrs['class_names'] = f_in.attrs['class_names']
                f_out.attrs['num_classes'] = f_in.attrs['num_classes']
        
        print(f"✅ Saved processed mask to: {output_path}")


# =============================================================================
# Visualization
# =============================================================================

def visualize_morphology_result(
    result: MorphologyResult,
    spectrogram: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Visualize the result of a morphology operation.
    
    Args:
        result: MorphologyResult from morphology operation
        spectrogram: Optional spectrogram to display as background
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Display spectrogram as background if provided
    if spectrogram is not None:
        for ax in axes:
            ax.imshow(spectrogram, cmap='gray', aspect='auto', alpha=0.5)
    
    # Original mask
    axes[0].imshow(result.original_mask, cmap='Reds', aspect='auto', alpha=0.7)
    axes[0].set_title(f'Original\n({result.pixel_count_before} pixels)')
    axes[0].axis('off')
    
    # Processed mask
    axes[1].imshow(result.processed_mask, cmap='Blues', aspect='auto', alpha=0.7)
    axes[1].set_title(
        f'{result.operation.capitalize()} (k={result.kernel_size})\n'
        f'({result.pixel_count_after} pixels, {result.pixel_change:+d})'
    )
    axes[1].axis('off')
    
    # Difference
    difference = result.processed_mask.astype(int) - result.original_mask.astype(int)
    im = axes[2].imshow(difference, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title(
        f'Difference\n'
        f'(Red=removed, Blue=added)'
    )
    axes[2].axis('off')
    
    # Add colorbar for difference plot
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Removed', 'Unchanged', 'Added'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_multiple_operations(
    mask: np.ndarray,
    spectrogram: Optional[np.ndarray] = None,
    operations: List[str] = None,
    kernel_sizes: List[int] = None,
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None
):
    """
    Compare multiple morphology operations side by side.
    
    Args:
        mask: Binary mask to process
        spectrogram: Optional spectrogram background
        operations: List of operations to compare
        kernel_sizes: List of kernel sizes to compare
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if operations is None:
        operations = ["erosion", "dilation", "opening", "closing"]
    
    if kernel_sizes is None:
        kernel_sizes = [3]
    
    morph = MaskMorphology()
    
    n_ops = len(operations)
    n_sizes = len(kernel_sizes)
    
    fig, axes = plt.subplots(n_sizes, n_ops + 1, figsize=figsize)
    
    # Ensure axes is 2D
    if n_sizes == 1:
        axes = axes.reshape(1, -1)
    
    for size_idx, kernel_size in enumerate(kernel_sizes):
        # First column: original
        ax = axes[size_idx, 0]
        if spectrogram is not None:
            ax.imshow(spectrogram, cmap='gray', aspect='auto', alpha=0.5)
        ax.imshow(mask, cmap='Reds', aspect='auto', alpha=0.7)
        ax.set_title(f'Original\n({mask.sum()} pixels)')
        ax.axis('off')
        
        # Subsequent columns: operations
        for op_idx, operation in enumerate(operations):
            ax = axes[size_idx, op_idx + 1]
            
            result = morph.apply(mask, operation, kernel_size)
            
            if spectrogram is not None:
                ax.imshow(spectrogram, cmap='gray', aspect='auto', alpha=0.5)
            ax.imshow(result.processed_mask, cmap='Blues', aspect='auto', alpha=0.7)
            
            title = f'{operation.capitalize()}\nk={kernel_size}\n'
            title += f'{result.pixel_count_after} px ({result.pixel_change:+d})'
            ax.set_title(title, fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Demo Functions
# =============================================================================

def demo_basic_operations():
    """Demonstrate basic morphology operations on a simple example."""
    print("=" * 60)
    print("Binary Morphology Demo: Basic Operations")
    print("=" * 60)
    
    # Create a simple test mask
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[20:30, 40:60] = 1  # Rectangle
    mask[15:20, 45:55] = 1  # Small extension
    
    morph = MaskMorphology()
    
    # Apply different operations
    operations = ["erosion", "dilation", "opening", "closing"]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f'Original ({mask.sum()} pixels)')
    axes[0].axis('off')
    
    # Operations
    for idx, operation in enumerate(operations, start=1):
        result = morph.apply(mask, operation, kernel_size=3)
        axes[idx].imshow(result.processed_mask, cmap='gray')
        axes[idx].set_title(
            f'{operation.capitalize()}\n'
            f'({result.pixel_count_after} pixels, {result.pixel_change:+d})'
        )
        axes[idx].axis('off')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Demo complete!")


def demo_with_hdf5(hdf5_path: str, class_name: Optional[str] = None):
    """
    Demonstrate morphology operations on a real HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        class_name: Optional specific class to process
    """
    print("=" * 60)
    print("Binary Morphology Demo: HDF5 File")
    print("=" * 60)
    
    processor = HDF5MorphologyProcessor(hdf5_path)
    
    # Get non-empty classes
    non_empty = processor.get_non_empty_classes()
    
    if not non_empty:
        print("❌ No non-empty classes found in HDF5 file")
        return
    
    print(f"\n📊 Non-empty classes: {non_empty}")
    
    # Select class to process
    if class_name is None:
        class_name = non_empty[0]
        print(f"   Using first class: {class_name}")
    elif class_name not in non_empty:
        print(f"⚠️  Class '{class_name}' is empty, using {non_empty[0]} instead")
        class_name = non_empty[0]
    
    # Load data
    spectrogram = processor.load_spectrogram()
    mask = processor.load_class_mask(class_name)
    
    print(f"\n📊 Data shapes:")
    print(f"   Spectrogram: {spectrogram.shape}")
    print(f"   Mask: {mask.shape}, {mask.sum()} pixels")
    
    # Apply dilation
    print(f"\n🔧 Applying dilation with kernel_size=3...")
    result = processor.apply_to_class(class_name, "dilation", kernel_size=3)
    
    print(f"\n📊 Results:")
    print(f"   Before: {result.pixel_count_before} pixels")
    print(f"   After: {result.pixel_count_after} pixels")
    print(f"   Change: {result.pixel_change:+d} pixels ({result.pixel_change_percent:+.1f}%)")
    
    # Visualize
    visualize_morphology_result(result, spectrogram)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    print("Binary Morphology Module")
    print("=" * 60)
    print("\nThis module provides morphology operations for HDF5 mask data.")
    print("\nUsage examples:")
    print("  1. demo_basic_operations() - Simple example")
    print("  2. demo_with_hdf5('path/to/file.hdf5') - Real data")
    print("\nOr import and use:")
    print("  from bin_morph import MaskMorphology, HDF5MorphologyProcessor")