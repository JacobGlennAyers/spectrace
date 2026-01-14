"""
ML-ready whale vocalization dataset preparation.

This module handles:
1. Converting XCF annotations to standardized HDF5 format
2. Loading data efficiently for ML training
3. Handling multi-class binary masks
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
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
# HDF5 Data Loader
# =============================================================================

class HDF5SpectrogramLoader:
    """
    Efficient loader for HDF5 spectrogram data.
    
    Usage:
        loader = HDF5SpectrogramLoader("path/to/file.hdf5")
        spec, masks, metadata = loader.load()
        
        # Get specific class mask
        mask = loader.get_class_mask("call_type_1")
        
        # Get metadata
        meta = loader.get_metadata()
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize loader.
        
        Args:
            hdf5_path: Path to HDF5 file
        """
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        self._file = None
        self._class_names = None
        self._metadata = None
    
    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()
    
    def _ensure_open(self):
        """Ensure file is open."""
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
    
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        if self._class_names is None:
            self._ensure_open()
            self._class_names = json.loads(self._file.attrs['class_names'])
        return self._class_names
    
    def get_metadata(self) -> SpectrogramMetadata:
        """Get spectrogram metadata."""
        if self._metadata is None:
            self._ensure_open()
            meta_group = self._file['metadata']
            self._metadata = SpectrogramMetadata(
                sample_rate=int(meta_group.attrs['sample_rate']),
                nfft=int(meta_group.attrs['nfft']),
                noverlap=int(meta_group.attrs['noverlap']),
                duration_sec=float(meta_group.attrs['duration_sec']),
                max_freq_hz=float(meta_group.attrs['max_freq_hz']),
                time_per_pixel=float(meta_group.attrs['time_per_pixel']),
                freq_per_pixel=float(meta_group.attrs['freq_per_pixel']),
                audio_path=str(meta_group.attrs['audio_path']),
                clip_basename=str(meta_group.attrs['clip_basename']),
                project_index=int(meta_group.attrs['project_index'])
            )
        return self._metadata
    
    def load_spectrogram(self) -> np.ndarray:
        """
        Load spectrogram.
        
        Returns:
            Spectrogram array (H, W)
        """
        self._ensure_open()
        return self._file['spectrogram'][:]
    
    def load_masks(self) -> np.ndarray:
        """
        Load all masks.
        
        Returns:
            Masks array (C, H, W) where C is number of classes
        """
        self._ensure_open()
        return self._file['masks'][:]
    
    def get_class_mask(self, class_name: str) -> np.ndarray:
        """
        Get mask for a specific class.
        
        Args:
            class_name: Name of the class
        
        Returns:
            Binary mask (H, W)
        """
        class_names = self.get_class_names()
        if class_name not in class_names:
            raise ValueError(f"Class '{class_name}' not found. Available: {class_names}")
        
        idx = class_names.index(class_name)
        self._ensure_open()
        return self._file['masks'][idx, :, :]
    
    def get_non_empty_classes(self) -> List[str]:
        """
        Get list of classes that have non-zero masks.
        
        Returns:
            List of class names with annotations
        """
        class_names = self.get_class_names()
        masks = self.load_masks()
        
        non_empty = []
        for idx, class_name in enumerate(class_names):
            if masks[idx].sum() > 0:
                non_empty.append(class_name)
        
        return non_empty
    
    def load(self) -> tuple:
        """
        Load all data at once.
        
        Returns:
            Tuple of (spectrogram, masks, metadata)
        """
        return (
            self.load_spectrogram(),
            self.load_masks(),
            self.get_metadata()
        )
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        self._ensure_open()
        metadata = self.get_metadata()
        class_names = self.get_class_names()
        masks = self.load_masks()
        
        summary = {
            'file_path': str(self.path),
            'clip_basename': metadata.clip_basename,
            'project_index': metadata.project_index,
            'spectrogram_shape': self._file['spectrogram'].shape,
            'num_classes': len(class_names),
            'class_names': class_names,
            'non_empty_classes': self.get_non_empty_classes(),
            'duration_sec': metadata.duration_sec,
            'sample_rate': metadata.sample_rate,
            'max_freq_hz': metadata.max_freq_hz
        }
        
        # Add per-class pixel counts
        class_pixel_counts = {}
        for idx, class_name in enumerate(class_names):
            class_pixel_counts[class_name] = int(masks[idx].sum())
        summary['class_pixel_counts'] = class_pixel_counts
        
        return summary


# =============================================================================
# Phase 1: Convert XCF to Standardized Format
# =============================================================================

class XCFToStandardConverter:
    """
    Converts raw XCF project folders into standardized HDF5 format.
    
    This is a one-time preprocessing step that:
    1. Scans all project folders
    2. Extracts XCF layers as binary masks
    3. Saves in efficient ML-ready format
    4. Creates index/manifest file
    
    Usage:
        converter = XCFToStandardConverter(
            project_folder="path/to/projects",
            output_folder="path/to/ml_data",
            format="hdf5"
        )
        converter.convert_all()
    """
    
    def __init__(
        self,
        project_folder: str,
        output_folder: str,
        layer_group_name: str = "OrcinusOrca_FrequencyContours",
        format: str = "hdf5",
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
        
        # Find files - look for *_spectrogram.png and *.xcf
        spectrogram_files = list(project_dir.glob("*_spectrogram.png"))
        xcf_files = list(project_dir.glob("*.xcf"))
        
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
            spectrogram = np.array(spectrogram_img)
        elif spectrogram_img.mode == 'RGB':
            spectrogram = np.array(spectrogram_img)[:, :, 0]
        elif spectrogram_img.mode == 'RGBA':
            spectrogram = np.array(spectrogram_img.convert('L'))
        else:
            spectrogram = np.array(spectrogram_img.convert('L'))
        
        print(f"    Loaded spectrogram: shape={spectrogram.shape}, dtype={spectrogram.dtype}, range=[{spectrogram.min()}, {spectrogram.max()}]")
        
        # Extract masks from XCF
        layer_data = extract_layers_from_xcf(
            str(xcf_path),
            self.layer_group_name,
            verbose=False
        )
        
        # Parse project name to get clip_basename and index
        project_name = project_dir.name
        parts = project_name.rsplit('_', 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            clip_basename = parts[0]
            project_index = int(parts[1])
        else:
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
                print(f"      {class_name}: not present, creating empty mask")
                masks_dict[class_name] = np.zeros(
                    spectrogram.shape[:2],
                    dtype=np.uint8
                )
        
        # Save in HDF5 format
        output_path = self.output_folder / f"{project_dir.name}.{self.format}"
        
        self._save_hdf5(
            output_path,
            spectrogram,
            masks_dict,
            spec_metadata,
            class_names
        )
        
        # Return index entry
        return {
            'sample_id': project_dir.name,
            'clip_basename': clip_basename,
            'project_index': project_index,
            'file_path': str(output_path.relative_to(self.output_folder)),
            'num_classes': len(class_names),
            'spectrogram_shape': str(spectrogram.shape),
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


# =============================================================================
# Utility Functions
# =============================================================================

def list_hdf5_files(directory: str) -> List[Path]:
    """
    List all HDF5 files in a directory.
    
    Args:
        directory: Path to directory
    
    Returns:
        List of Path objects for HDF5 files
    """
    directory = Path(directory)
    return sorted(directory.glob("*.hdf5"))


def load_dataset_index(ml_data_folder: str) -> pd.DataFrame:
    """
    Load the dataset index CSV.
    
    Args:
        ml_data_folder: Path to ML data folder
    
    Returns:
        DataFrame with dataset index
    """
    index_path = Path(ml_data_folder) / "dataset_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    return pd.read_csv(index_path)