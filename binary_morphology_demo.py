#!/usr/bin/env python3
"""
Demo: Binary Morphology Operations on HDF5 Masks

This script demonstrates how the HDF5 format enables ML-ready image processing
operations on the extracted masks. We apply:
1. Opening - removes small noise/artifacts
2. Closing - fills small gaps in contours

This is a proof-of-concept showing the masks are ready for:
- Data augmentation
- Preprocessing for segmentation models
- Post-processing model predictions
- Feature extraction
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def apply_morphology_operation(
    mask: np.ndarray,
    operation: str = "closing",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply binary morphology operations to a mask.
    
    Args:
        mask: Binary mask (H, W)
        operation: "opening", "closing", "erosion", "dilation"
        kernel_size: Size of structuring element (larger = stronger effect)
    
    Returns:
        Processed mask
    """
    # Create structuring element (disk-like)
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, kernel_size // 2)
    
    if operation == "opening":
        # Opening = erosion followed by dilation
        # Effect: removes small bright spots (noise)
        result = ndimage.binary_opening(mask, structure=struct)
    elif operation == "closing":
        # Closing = dilation followed by erosion
        # Effect: fills small gaps/holes
        result = ndimage.binary_closing(mask, structure=struct)
    elif operation == "erosion":
        # Erosion: shrinks bright regions
        result = ndimage.binary_erosion(mask, structure=struct)
    elif operation == "dilation":
        # Dilation: expands bright regions
        result = ndimage.binary_dilation(mask, structure=struct)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result.astype(np.uint8)


def load_sample_from_hdf5(hdf5_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load spectrogram, masks, and metadata from HDF5 file.
    
    Returns:
        (spectrogram, masks, metadata) tuple
    """
    with h5py.File(hdf5_path, 'r') as f:
        spectrogram = f['spectrogram'][:]
        masks = f['masks'][:]  # Shape: (num_classes, H, W)
        
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        class_names = json.loads(f.attrs['class_names'])
    
    metadata['class_names'] = class_names
    return spectrogram, masks, metadata


def visualize_morphology_comparison(
    spectrogram: np.ndarray,
    original_mask: np.ndarray,
    opened_mask: np.ndarray,
    closed_mask: np.ndarray,
    class_name: str,
    output_path: str = None
):
    """
    Create a comparison visualization showing original and processed masks.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Original mask variations
    axes[0, 0].imshow(spectrogram, cmap='gray', aspect='auto')
    axes[0, 0].set_title('Spectrogram', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_mask, cmap='hot', aspect='auto')
    axes[0, 1].set_title(f'Original Mask\n({original_mask.sum()} pixels)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay original on spectrogram
    axes[0, 2].imshow(spectrogram, cmap='gray', aspect='auto')
    mask_overlay = np.ma.masked_where(original_mask == 0, original_mask)
    axes[0, 2].imshow(mask_overlay, cmap='Reds', alpha=0.6, aspect='auto')
    axes[0, 2].set_title('Original Overlay', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Morphology results
    axes[1, 0].imshow(opened_mask, cmap='hot', aspect='auto')
    pixel_diff_open = int(original_mask.sum() - opened_mask.sum())
    axes[1, 0].set_title(f'After Opening\n({opened_mask.sum()} px, Δ={pixel_diff_open})', 
                         fontsize=12, color='blue')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(closed_mask, cmap='hot', aspect='auto')
    pixel_diff_close = int(closed_mask.sum() - original_mask.sum())
    axes[1, 1].set_title(f'After Closing\n({closed_mask.sum()} px, Δ=+{pixel_diff_close})', 
                         fontsize=12, color='green')
    axes[1, 1].axis('off')
    
    # Show difference map
    diff_map = np.zeros_like(original_mask, dtype=float)
    diff_map[opened_mask > original_mask] = -1  # Pixels removed by opening
    diff_map[closed_mask > original_mask] = 1   # Pixels added by closing
    
    axes[1, 2].imshow(diff_map, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[1, 2].set_title('Changes\n(Blue=removed, Red=added)', fontsize=12)
    axes[1, 2].axis('off')
    
    fig.suptitle(f'Binary Morphology Demo: {class_name}', 
                 fontsize=14, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def demo_morphology_on_sample(
    hdf5_path: Path,
    output_folder: Path,
    kernel_size: int = 5
):
    """
    Run morphology demo on a single HDF5 file.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {hdf5_path.name}")
    print(f"{'='*60}")
    
    # Load data
    spectrogram, masks, metadata = load_sample_from_hdf5(hdf5_path)
    class_names = metadata['class_names']
    
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Process each class
    results = []
    
    for class_idx, class_name in enumerate(class_names):
        original_mask = masks[class_idx]
        
        # Skip empty masks
        if original_mask.sum() == 0:
            print(f"\n⏭️  Skipping '{class_name}' (empty mask)")
            continue
        
        print(f"\n🔬 Processing class: {class_name}")
        print(f"   Original pixels: {original_mask.sum()}")
        
        # Apply opening (removes noise)
        opened_mask = apply_morphology_operation(
            original_mask,
            operation="opening",
            kernel_size=kernel_size
        )
        print(f"   After opening: {opened_mask.sum()} pixels (removed {original_mask.sum() - opened_mask.sum()})")
        
        # Apply closing (fills gaps)
        closed_mask = apply_morphology_operation(
            original_mask,
            operation="closing",
            kernel_size=kernel_size
        )
        print(f"   After closing: {closed_mask.sum()} pixels (added {closed_mask.sum() - original_mask.sum()})")
        
        # Create visualization
        output_path = output_folder / f"{hdf5_path.stem}_{class_name.replace('/', '_')}_morphology.png"
        
        visualize_morphology_comparison(
            spectrogram=spectrogram,
            original_mask=original_mask,
            opened_mask=opened_mask,
            closed_mask=closed_mask,
            class_name=class_name,
            output_path=str(output_path)
        )
        
        results.append({
            'class': class_name,
            'original_pixels': int(original_mask.sum()),
            'opened_pixels': int(opened_mask.sum()),
            'closed_pixels': int(closed_mask.sum()),
            'opening_removed': int(original_mask.sum() - opened_mask.sum()),
            'closing_added': int(closed_mask.sum() - original_mask.sum())
        })
    
    return results


def demo_additional_operations(hdf5_path: Path, output_folder: Path):
    """
    Demonstrate additional morphology operations (erosion, dilation).
    """
    print(f"\n{'='*60}")
    print(f"Additional Operations Demo: {hdf5_path.name}")
    print(f"{'='*60}")
    
    # Load data
    spectrogram, masks, metadata = load_sample_from_hdf5(hdf5_path)
    class_names = metadata['class_names']
    
    # Find first non-empty mask
    for class_idx, class_name in enumerate(class_names):
        mask = masks[class_idx]
        if mask.sum() > 0:
            break
    else:
        print("⚠️  No non-empty masks found")
        return
    
    print(f"Using class: {class_name}")
    
    # Apply various operations
    eroded = apply_morphology_operation(mask, "erosion", kernel_size=3)
    dilated = apply_morphology_operation(mask, "dilation", kernel_size=3)
    opened = apply_morphology_operation(mask, "opening", kernel_size=5)
    closed = apply_morphology_operation(mask, "closing", kernel_size=5)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    operations = [
        (mask, "Original", f"{mask.sum()} px"),
        (eroded, "Erosion", f"{eroded.sum()} px"),
        (dilated, "Dilation", f"{dilated.sum()} px"),
        (opened, "Opening", f"{opened.sum()} px"),
        (closed, "Closing", f"{closed.sum()} px"),
        (spectrogram, "Spectrogram", "")
    ]
    
    for idx, (data, title, info) in enumerate(operations):
        ax = axes[idx // 3, idx % 3]
        
        if idx == 5:  # Spectrogram
            ax.imshow(data, cmap='gray', aspect='auto')
        else:
            ax.imshow(data, cmap='hot', aspect='auto')
        
        ax.set_title(f'{title}\n{info}', fontsize=11, weight='bold')
        ax.axis('off')
    
    fig.suptitle(f'All Morphology Operations: {class_name}', 
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    output_path = output_folder / f"{hdf5_path.stem}_all_operations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    plt.close()


def main():
    """Main execution function."""
    # Configuration
    ml_data_folder = "./hdf5_files"
    output_folder = "./morphology_demo"
    kernel_size = 5  # Size of structuring element (larger = stronger effect)
    
    print("="*60)
    print("🔬 Binary Morphology Demo for ML-Ready Masks")
    print("="*60)
    print(f"Input folder: {ml_data_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Kernel size: {kernel_size}")
    
    ml_data_path = Path(ml_data_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find HDF5 files
    hdf5_files = list(ml_data_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"\n❌ No HDF5 files found in {ml_data_folder}")
        print("   Run the conversion script first: python convert_to_hdf5.py")
        return
    
    print(f"\n📁 Found {len(hdf5_files)} HDF5 files")
    
    # Process first file in detail
    print(f"\n{'='*60}")
    print("DEMO 1: Opening & Closing Operations")
    print(f"{'='*60}")
    
    all_results = []
    
    for hdf5_file in hdf5_files[:3]:  # Process up to 3 files for demo
        try:
            results = demo_morphology_on_sample(
                hdf5_file,
                output_path,
                kernel_size=kernel_size
            )
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error processing {hdf5_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Additional operations demo on first file
    if hdf5_files:
        print(f"\n{'='*60}")
        print("DEMO 2: All Morphology Operations")
        print(f"{'='*60}")
        
        try:
            demo_additional_operations(hdf5_files[0], output_path)
        except Exception as e:
            print(f"❌ Error in additional demo: {e}")
    
    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("📊 Summary Statistics")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"\n{result['class']}:")
            print(f"  Original: {result['original_pixels']} pixels")
            print(f"  Opening removed: {result['opening_removed']} pixels ({100*result['opening_removed']/result['original_pixels']:.1f}%)")
            print(f"  Closing added: {result['closing_added']} pixels ({100*result['closing_added']/result['original_pixels']:.1f}%)")
    
    print(f"\n{'='*60}")
    print("✅ Demo Complete!")
    print(f"{'='*60}")
    print(f"\nVisualization outputs saved to: {output_folder}")
    print("\n💡 Key Takeaways:")
    print("   • Opening removes small artifacts/noise")
    print("   • Closing fills small gaps in contours")
    print("   • These operations are essential for:")
    print("     - Cleaning noisy annotations")
    print("     - Post-processing model predictions")
    print("     - Data augmentation in training")
    print("     - Feature extraction for downstream tasks")
    print("\n🚀 Next Steps:")
    print("   • Use these masks for training segmentation models")
    print("   • Extract time-frequency features for classification")
    print("   • Apply data augmentation for robust training")


if __name__ == "__main__":
    main()