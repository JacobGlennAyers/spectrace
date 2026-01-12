"""
Proposed architecture for ML-ready whale vocalization dataset.

This demonstrates best practices for:
1. Converting XCF annotations to standardized format
2. Loading data efficiently for ML training
3. Handling multi-class binary masks
4. Downstream processing (morphology, time-frequency extraction)
"""

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from scipy import ndimage
from dataclasses import dataclass, asdict


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class SpectrogramMetadata:
    """Metadata for a single spectrogram."""
    sample_rate: int
    nfft: int
    noverlap: int
    duration_sec: float
    max_freq_hz: float
    time_per_pixel: float
    freq_per_pixel: float
    audio_path: str
    clip_basename: str
    project_index: int


@dataclass
class MaskAnnotation:
    """Single mask annotation with metadata."""
    layer_name: str
    mask: np.ndarray  # Binary mask (H, W)
    visible: bool
    pixel_count: int
    color_hex: str


# =============================================================================
# Phase 1: Convert XCF to Standardized Format
# =============================================================================

class XCFToStandardConverter:
    """
    Converts raw XCF project folders into standardized HDF5 or NPZ format.
    
    This is a one-time preprocessing step that:
    1. Scans all project folders
    2. Extracts XCF layers as binary masks
    3. Saves in efficient ML-ready format
    4. Creates index/manifest file
    
    Usage:
        converter = XCFToStandardConverter(
            project_folder="path/to/projects",
            output_folder="path/to/ml_data",
            format="hdf5"  # or "npz"
        )
        converter.convert_all()
    """
    
    def __init__(
        self,
        project_folder: str,
        output_folder: str,
        layer_group_name: str = "OrcinusOrca_FrequencyContours",
        format: str = "hdf5",  # "hdf5" or "npz"
        color_mapping: Optional[Dict[str, str]] = None
    ):
        self.project_folder = Path(project_folder)
        self.output_folder = Path(output_folder)
        self.layer_group_name = layer_group_name
        self.format = format
        self.color_mapping = color_mapping
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def convert_all(self) -> pd.DataFrame:
        """
        Convert all projects to standardized format.
        
        Returns:
            DataFrame with index of all converted samples
        """
        # Import your existing functions
        from utils import (
            extract_layers_from_xcf,
            load_metadata,
            get_or_create_color_mapping
        )
        
        # Load or create color mapping
        if self.color_mapping is None:
            self.color_mapping = get_or_create_color_mapping(
                str(self.project_folder),
                self.layer_group_name
            )
        
        index_data = []
        
        # Scan all project folders
        for project_dir in self.project_folder.iterdir():
            if not project_dir.is_dir():
                continue
            
            try:
                sample_data = self._convert_project(project_dir)
                index_data.append(sample_data)
                print(f"✅ Converted: {project_dir.name}")
            except Exception as e:
                print(f"❌ Error converting {project_dir.name}: {e}")
        
        # Create and save index
        index_df = pd.DataFrame(index_data)
        index_path = self.output_folder / "dataset_index.csv"
        index_df.to_csv(index_path, index=False)
        print(f"\n📊 Created index with {len(index_df)} samples: {index_path}")
        
        return index_df
    
    def _convert_project(self, project_dir: Path) -> Dict:
        """Convert a single project folder."""
        from utils import extract_layers_from_xcf, load_metadata
        
        # Find files - look for *_spectrogram.png and *_spectrogram.xcf
        spectrogram_files = list(project_dir.glob("*_spectrogram.png"))
        xcf_files = list(project_dir.glob("*_spectrogram.xcf"))
        
        if not spectrogram_files:
            raise FileNotFoundError(f"No spectrogram PNG found in {project_dir}")
        if not xcf_files:
            raise FileNotFoundError(f"No XCF file found in {project_dir}")
            
        spectrogram_path = spectrogram_files[0]
        xcf_path = xcf_files[0]
        
        # Load metadata
        metadata = load_metadata(str(project_dir))
        
        # Load spectrogram
        from PIL import Image
        spectrogram_img = Image.open(spectrogram_path)
        
        # Convert to grayscale if needed and normalize to 0-255 range
        if spectrogram_img.mode == 'L':
            # Already grayscale
            spectrogram = np.array(spectrogram_img)
        elif spectrogram_img.mode == 'RGB':
            # Convert RGB to grayscale (take first channel since it's grayscale saved as RGB)
            spectrogram = np.array(spectrogram_img)[:, :, 0]
        elif spectrogram_img.mode == 'RGBA':
            # Convert RGBA to grayscale
            spectrogram = np.array(spectrogram_img.convert('L'))
        else:
            # Fallback: force to grayscale
            spectrogram = np.array(spectrogram_img.convert('L'))
        
        print(f"    Loaded spectrogram: shape={spectrogram.shape}, dtype={spectrogram.dtype}, range=[{spectrogram.min()}, {spectrogram.max()}]")
        
        # Extract masks from XCF
        layer_data = extract_layers_from_xcf(
            str(xcf_path),
            self.layer_group_name,
            verbose=False
        )
        
        # Parse project name to get clip_basename and index
        # Expected format: "clipname_N" where N is the project index
        project_name = project_dir.name
        parts = project_name.rsplit('_', 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            clip_basename = parts[0]
            project_index = int(parts[1])
        else:
            # Fallback: treat entire name as basename, index as 0
            clip_basename = project_name
            project_index = 0
            print(f"  ⚠️  Could not parse index from '{project_name}', using index=0")
        
        # Prepare standardized data structure
        spec_metadata = SpectrogramMetadata(
            sample_rate=metadata.get('sample_rate', 44100),
            nfft=metadata.get('nfft', 2048),
            noverlap=metadata.get('noverlap', 1024),
            duration_sec=spectrogram.shape[1] * metadata.get('time_per_pixel', 0.01),
            max_freq_hz=metadata.get('sample_rate', 44100) / 2,
            time_per_pixel=metadata.get('time_per_pixel', 0.01),
            freq_per_pixel=metadata.get('frequency_spacing', 10.0),
            audio_path=metadata.get('copied_audio_path', ''),
            clip_basename=clip_basename,
            project_index=project_index
        )
        
        # Organize masks by class
        class_names = sorted(self.color_mapping.keys())
        masks_dict = {}
        
        print(f"    Processing {len(class_names)} classes: {class_names}")
        
        for class_name in class_names:
            if class_name in layer_data['layers']:
                layer_info = layer_data['layers'][class_name]
                mask = layer_info['mask']
                
                print(f"      {class_name}: original mask shape={mask.shape}, pixels={mask.sum()}")
                
                # Resize if needed
                if mask.shape != spectrogram.shape[:2]:
                    from scipy.ndimage import zoom
                    scale_y = spectrogram.shape[0] / mask.shape[0]
                    scale_x = spectrogram.shape[1] / mask.shape[1]
                    mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
                    print(f"      {class_name}: resized to {mask.shape}, pixels={mask.sum()}")
                
                masks_dict[class_name] = mask.astype(np.uint8)
            else:
                # Class not present in this sample
                print(f"      {class_name}: not present, creating empty mask")
                masks_dict[class_name] = np.zeros(
                    spectrogram.shape[:2],
                    dtype=np.uint8
                )
        
        # Save in chosen format
        output_path = self.output_folder / f"{project_dir.name}.{self.format}"
        
        if self.format == "hdf5":
            self._save_hdf5(
                output_path,
                spectrogram,
                masks_dict,
                spec_metadata,
                class_names
            )
        elif self.format == "npz":
            self._save_npz(
                output_path,
                spectrogram,
                masks_dict,
                spec_metadata
            )
        
        # Return index entry
        return {
            'sample_id': project_dir.name,
            'clip_basename': clip_basename,
            'project_index': project_index,
            'file_path': str(output_path.relative_to(self.output_folder)),
            'num_classes': len(class_names),
            'spectrogram_shape': spectrogram.shape,
            'has_annotations': any(m.sum() > 0 for m in masks_dict.values()),
            'duration_sec': spec_metadata.duration_sec
        }
    
    def _save_hdf5(
        self,
        path: Path,
        spectrogram: np.ndarray,
        masks_dict: Dict[str, np.ndarray],
        metadata: SpectrogramMetadata,
        class_names: List[str]
    ):
        """Save data in HDF5 format."""
        with h5py.File(path, 'w') as f:
            # Store spectrogram
            f.create_dataset('spectrogram', data=spectrogram, compression='gzip')
            
            # Store masks as multi-channel array
            masks_array = np.stack([masks_dict[name] for name in class_names], axis=0)
            f.create_dataset('masks', data=masks_array, compression='gzip')
            
            # Store metadata
            meta_group = f.create_group('metadata')
            for key, value in asdict(metadata).items():
                meta_group.attrs[key] = value
            
            # Store class mapping
            f.attrs['class_names'] = json.dumps(class_names)
            f.attrs['num_classes'] = len(class_names)
    
    def _save_npz(
        self,
        path: Path,
        spectrogram: np.ndarray,
        masks_dict: Dict[str, np.ndarray],
        metadata: SpectrogramMetadata
    ):
        """Save data in NPZ format."""
        save_dict = {
            'spectrogram': spectrogram,
            'metadata': json.dumps(asdict(metadata)),
            **{f'mask_{name}': mask for name, mask in masks_dict.items()}
        }
        np.savez_compressed(path, **save_dict)


# =============================================================================
# Phase 2: PyTorch Dataset for ML Training
# =============================================================================

class WhaleVocalizationDataset(Dataset):
    """
    PyTorch Dataset for whale vocalization spectrograms with multi-class masks.
    
    Supports:
    - Efficient loading from HDF5 or NPZ
    - On-the-fly augmentations
    - Multi-class binary masks
    - Flexible output formats (for different model architectures)
    
    Usage:
        dataset = WhaleVocalizationDataset(
            data_folder="path/to/ml_data",
            format="hdf5",
            transform=my_transforms,
            output_format="multi_binary"  # or "dense_class"
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4
        )
    """
    
    def __init__(
        self,
        data_folder: str,
        format: str = "hdf5",
        transform: Optional[callable] = None,
        output_format: str = "multi_binary",  # "multi_binary" or "dense_class"
        class_names: Optional[List[str]] = None,
        filter_empty: bool = False
    ):
        """
        Args:
            data_folder: Folder containing converted HDF5/NPZ files
            format: "hdf5" or "npz"
            transform: Optional transform to apply (albumentations recommended)
            output_format: 
                - "multi_binary": (C, H, W) with C binary masks
                - "dense_class": (H, W) with class indices (assumes non-overlapping)
            class_names: List of class names (loaded from index if None)
            filter_empty: Whether to exclude samples with no annotations
        """
        self.data_folder = Path(data_folder)
        self.format = format
        self.transform = transform
        self.output_format = output_format
        self.filter_empty = filter_empty
        
        # Load index
        index_path = self.data_folder / "dataset_index.csv"
        self.index = pd.read_csv(index_path)
        
        # Filter empty samples if requested
        if filter_empty:
            self.index = self.index[self.index['has_annotations']].reset_index(drop=True)
        
        # Load class names
        if class_names is None:
            # Load from first sample
            sample_path = self.data_folder / self.index.iloc[0]['file_path']
            self.class_names = self._load_class_names(sample_path)
        else:
            self.class_names = class_names
        
        print(f"📚 Loaded dataset: {len(self)} samples, {len(self.class_names)} classes")
    
    def _load_class_names(self, sample_path: Path) -> List[str]:
        """Load class names from a sample file."""
        if self.format == "hdf5":
            with h5py.File(sample_path, 'r') as f:
                return json.loads(f.attrs['class_names'])
        elif self.format == "npz":
            data = np.load(sample_path)
            return [k.replace('mask_', '') for k in data.keys() if k.startswith('mask_')]
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - 'spectrogram': (H, W) or (H, W, C) tensor
                - 'masks': (C, H, W) or (H, W) depending on output_format
                - 'metadata': dict with sample information
        """
        row = self.index.iloc[idx]
        sample_path = self.data_folder / row['file_path']
        
        # Load data
        if self.format == "hdf5":
            spectrogram, masks, metadata = self._load_hdf5(sample_path)
        elif self.format == "npz":
            spectrogram, masks, metadata = self._load_npz(sample_path)
        
        # Apply transforms if provided (use albumentations for image + mask transforms)
        if self.transform is not None:
            # Albumentations example:
            # transformed = self.transform(image=spectrogram, masks=[masks[i] for i in range(len(masks))])
            # spectrogram = transformed['image']
            # masks = np.array(transformed['masks'])
            pass
        
        # Ensure spectrogram is 2D (H, W) for grayscale
        if len(spectrogram.shape) == 3 and spectrogram.shape[2] == 1:
            spectrogram = spectrogram[:, :, 0]
        elif len(spectrogram.shape) == 3 and spectrogram.shape[2] == 3:
            # Convert RGB to grayscale if needed (take first channel as they should be identical for grayscale images)
            spectrogram = spectrogram[:, :, 0]
        
        # Convert to tensors
        spectrogram = torch.from_numpy(spectrogram).float()
        
        # Ensure masks are the right shape (C, H, W)
        if len(masks.shape) == 2:
            # Single mask case - expand to (1, H, W)
            masks = masks[np.newaxis, ...]
        
        # Handle mask output format
        if self.output_format == "multi_binary":
            masks = torch.from_numpy(masks).float()  # (C, H, W)
        elif self.output_format == "dense_class":
            # Convert to single-channel class indices
            masks_binary = masks.astype(bool)
            class_mask = np.zeros(masks.shape[1:], dtype=np.int64)
            for i, mask in enumerate(masks_binary):
                class_mask[mask] = i + 1  # 0 is background
            masks = torch.from_numpy(class_mask).long()  # (H, W)
        
        return {
            'spectrogram': spectrogram,
            'masks': masks,
            'metadata': {
                'sample_id': row['sample_id'],
                'clip_basename': row['clip_basename'],
                'class_names': self.class_names,
                **metadata
            }
        }
    
    def _load_hdf5(self, path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load data from HDF5 file."""
        with h5py.File(path, 'r') as f:
            spectrogram = f['spectrogram'][:]
            masks = f['masks'][:]
            metadata = dict(f['metadata'].attrs)
        return spectrogram, masks, metadata
    
    def _load_npz(self, path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load data from NPZ file."""
        data = np.load(path)
        spectrogram = data['spectrogram']
        
        # Stack all masks
        mask_keys = sorted([k for k in data.keys() if k.startswith('mask_')])
        masks = np.stack([data[k] for k in mask_keys], axis=0)
        
        metadata = json.loads(str(data['metadata']))
        return spectrogram, masks, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        Useful for weighted loss functions.
        
        Returns:
            Tensor of shape (num_classes,) with weights
        """
        # Count pixels per class across entire dataset
        class_pixel_counts = np.zeros(len(self.class_names))
        
        for idx in range(len(self)):
            sample_path = self.data_folder / self.index.iloc[idx]['file_path']
            if self.format == "hdf5":
                with h5py.File(sample_path, 'r') as f:
                    masks = f['masks'][:]
            elif self.format == "npz":
                data = np.load(sample_path)
                mask_keys = sorted([k for k in data.keys() if k.startswith('mask_')])
                masks = np.stack([data[k] for k in mask_keys], axis=0)
            
            class_pixel_counts += masks.sum(axis=(1, 2))
        
        # Inverse frequency weighting
        total_pixels = class_pixel_counts.sum()
        weights = total_pixels / (len(self.class_names) * class_pixel_counts + 1e-6)
        
        return torch.from_numpy(weights).float()


# =============================================================================
# Phase 3: Downstream Processing Utilities
# =============================================================================

class MaskProcessor:
    """
    Utilities for processing masks after loading.
    
    Includes:
    - Binary morphology operations
    - Time-frequency contour extraction
    - Statistics computation
    """
    
    @staticmethod
    def apply_morphology(
        mask: np.ndarray,
        operation: str = "closing",
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply binary morphology operations.
        
        Args:
            mask: Binary mask (H, W)
            operation: "erosion", "dilation", "opening", "closing"
            kernel_size: Size of structuring element
        
        Returns:
            Processed mask
        """
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, kernel_size // 2)
        
        if operation == "erosion":
            return ndimage.binary_erosion(mask, structure=struct)
        elif operation == "dilation":
            return ndimage.binary_dilation(mask, structure=struct)
        elif operation == "opening":
            return ndimage.binary_opening(mask, structure=struct)
        elif operation == "closing":
            return ndimage.binary_closing(mask, structure=struct)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def extract_time_frequency_contours(
        mask: np.ndarray,
        metadata: SpectrogramMetadata,
        method: str = "centroid"
    ) -> pd.DataFrame:
        """
        Extract time-frequency contour from a mask.
        
        Args:
            mask: Binary mask (H, W) where H=freq, W=time
            metadata: Spectrogram metadata with time/freq scaling
            method: 
                - "centroid": frequency centroid per time bin
                - "min_max": min and max frequency per time bin
                - "all_points": all active pixels
        
        Returns:
            DataFrame with columns: time_sec, freq_hz, (intensity_optional)
        """
        height, width = mask.shape
        contour_data = []
        
        if method == "centroid":
            for t in range(width):
                col = mask[:, t]
                if col.sum() > 0:
                    # Frequency is inverted (0 = max freq)
                    active_freqs = np.where(col)[0]
                    centroid_idx = active_freqs.mean()
                    
                    time_sec = t * metadata.time_per_pixel
                    freq_hz = metadata.max_freq_hz - (centroid_idx * metadata.freq_per_pixel)
                    
                    contour_data.append({
                        'time_sec': time_sec,
                        'freq_hz': freq_hz,
                        'pixel_count': len(active_freqs)
                    })
        
        elif method == "min_max":
            for t in range(width):
                col = mask[:, t]
                if col.sum() > 0:
                    active_freqs = np.where(col)[0]
                    
                    time_sec = t * metadata.time_per_pixel
                    freq_min = metadata.max_freq_hz - (active_freqs.max() * metadata.freq_per_pixel)
                    freq_max = metadata.max_freq_hz - (active_freqs.min() * metadata.freq_per_pixel)
                    
                    contour_data.append({
                        'time_sec': time_sec,
                        'freq_min_hz': freq_min,
                        'freq_max_hz': freq_max,
                        'bandwidth_hz': freq_max - freq_min
                    })
        
        elif method == "all_points":
            active_pixels = np.argwhere(mask)
            for freq_idx, time_idx in active_pixels:
                time_sec = time_idx * metadata.time_per_pixel
                freq_hz = metadata.max_freq_hz - (freq_idx * metadata.freq_per_pixel)
                
                contour_data.append({
                    'time_sec': time_sec,
                    'freq_hz': freq_hz
                })
        
        return pd.DataFrame(contour_data)
    
    @staticmethod
    def compute_mask_statistics(
        mask: np.ndarray,
        metadata: SpectrogramMetadata
    ) -> Dict:
        """
        Compute statistics for a mask annotation.
        
        Returns:
            Dict with statistics like duration, bandwidth, pixel count, etc.
        """
        if mask.sum() == 0:
            return {
                'pixel_count': 0,
                'duration_sec': 0,
                'bandwidth_hz': 0,
                'start_time_sec': None,
                'end_time_sec': None,
                'min_freq_hz': None,
                'max_freq_hz': None
            }
        
        # Find bounding box
        active_pixels = np.argwhere(mask)
        freq_indices = active_pixels[:, 0]
        time_indices = active_pixels[:, 1]
        
        # Time statistics
        time_min_idx = time_indices.min()
        time_max_idx = time_indices.max()
        start_time_sec = time_min_idx * metadata.time_per_pixel
        end_time_sec = time_max_idx * metadata.time_per_pixel
        duration_sec = end_time_sec - start_time_sec
        
        # Frequency statistics (remember: inverted axis)
        freq_min_idx = freq_indices.min()
        freq_max_idx = freq_indices.max()
        max_freq_hz = metadata.max_freq_hz - (freq_min_idx * metadata.freq_per_pixel)
        min_freq_hz = metadata.max_freq_hz - (freq_max_idx * metadata.freq_per_pixel)
        bandwidth_hz = max_freq_hz - min_freq_hz
        
        return {
            'pixel_count': int(mask.sum()),
            'duration_sec': float(duration_sec),
            'bandwidth_hz': float(bandwidth_hz),
            'start_time_sec': float(start_time_sec),
            'end_time_sec': float(end_time_sec),
            'min_freq_hz': float(min_freq_hz),
            'max_freq_hz': float(max_freq_hz),
            'time_coverage': float((time_max_idx - time_min_idx + 1) / mask.shape[1]),
            'freq_coverage': float((freq_max_idx - freq_min_idx + 1) / mask.shape[0])
        }


# =============================================================================
# Example Usage
# =============================================================================

def example_conversion():
    """Example: Convert XCF projects to ML-ready format."""
    converter = XCFToStandardConverter(
        project_folder="path/to/projects",
        output_folder="path/to/ml_data",
        format="hdf5"
    )
    
    index_df = converter.convert_all()
    print(index_df.head())


def example_training():
    """Example: Use dataset for training."""
    # Create dataset
    dataset = WhaleVocalizationDataset(
        data_folder="path/to/ml_data",
        format="hdf5",
        output_format="multi_binary",
        filter_empty=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get class weights for balanced training
    class_weights = dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Training loop example
    for batch in dataloader:
        spectrograms = batch['spectrogram']  # (B, H, W)
        masks = batch['masks']  # (B, C, H, W)
        
        # Your model training here
        # outputs = model(spectrograms)
        # loss = criterion(outputs, masks)
        break


def example_morphology():
    """Example: Apply morphology to masks."""
    dataset = WhaleVocalizationDataset(
        data_folder="path/to/ml_data",
        format="hdf5"
    )
    
    sample = dataset[0]
    mask = sample['masks'][0].numpy()  # First class mask
    
    # Apply closing to fill small gaps
    closed_mask = MaskProcessor.apply_morphology(
        mask,
        operation="closing",
        kernel_size=5
    )
    
    # Visualize difference
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(closed_mask, cmap='gray')
    axes[1].set_title('After Closing')
    plt.show()


def example_contour_extraction():
    """Example: Extract time-frequency contours and save to CSV."""
    dataset = WhaleVocalizationDataset(
        data_folder="path/to/ml_data",
        format="hdf5"
    )
    
    sample = dataset[0]
    metadata_dict = sample['metadata']
    
    # Reconstruct metadata object
    metadata = SpectrogramMetadata(**{
        k: metadata_dict[k] for k in SpectrogramMetadata.__dataclass_fields__
        if k in metadata_dict
    })
    
    # Extract contours for each class
    all_contours = []
    
    for class_idx, class_name in enumerate(dataset.class_names):
        mask = sample['masks'][class_idx].numpy()
        
        if mask.sum() > 0:
            contours = MaskProcessor.extract_time_frequency_contours(
                mask,
                metadata,
                method="centroid"
            )
            contours['class'] = class_name
            contours['sample_id'] = metadata_dict['sample_id']
            
            all_contours.append(contours)
    
    # Combine and save
    if all_contours:
        combined_df = pd.concat(all_contours, ignore_index=True)
        output_path = f"contours_{metadata_dict['sample_id']}.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"✅ Saved contours to {output_path}")
        print(combined_df.head())


def example_statistics():
    """Example: Compute statistics for all annotations."""
    dataset = WhaleVocalizationDataset(
        data_folder="path/to/ml_data",
        format="hdf5"
    )
    
    all_stats = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata_dict = sample['metadata']
        
        metadata = SpectrogramMetadata(**{
            k: metadata_dict[k] for k in SpectrogramMetadata.__dataclass_fields__
            if k in metadata_dict
        })
        
        for class_idx, class_name in enumerate(dataset.class_names):
            mask = sample['masks'][class_idx].numpy()
            
            stats = MaskProcessor.compute_mask_statistics(mask, metadata)
            stats['class'] = class_name
            stats['sample_id'] = metadata_dict['sample_id']
            stats['clip_basename'] = metadata_dict['clip_basename']
            
            all_stats.append(stats)
    
    # Create summary dataframe
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df[stats_df['pixel_count'] > 0]  # Filter empty masks
    
    # Save
    stats_df.to_csv("annotation_statistics.csv", index=False)
    
    # Print summary
    print("\n📊 Annotation Statistics Summary:")
    print(stats_df.groupby('class').agg({
        'pixel_count': ['count', 'mean', 'std'],
        'duration_sec': ['mean', 'std'],
        'bandwidth_hz': ['mean', 'std']
    }).round(2))


if __name__ == "__main__":
    print("This is a proposal/template file.")
    print("Uncomment and modify the examples below to use.")
    
    # example_conversion()
    # example_training()
    # example_morphology()
    # example_contour_extraction()
    # example_statistics()