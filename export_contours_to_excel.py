#!/usr/bin/env python3
"""
Export time-frequency contours from HDF5 files to Excel spreadsheet.

This script:
1. Loads all HDF5 files from the ml_data folder
2. Extracts time-frequency information for each layer/class
3. Outputs a comprehensive Excel file with multiple sheets:
   - Summary: Overview of all samples
   - Contours: Detailed time-frequency points for each annotation
   - Statistics: Per-annotation statistics (duration, bandwidth, etc.)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_hdf5_metadata(hdf5_path: Path) -> Dict:
    """Load metadata from HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        metadata = dict(f['metadata'].attrs)
        class_names = json.loads(f.attrs['class_names'])
        num_classes = f.attrs['num_classes']
    
    metadata['class_names'] = class_names
    metadata['num_classes'] = num_classes
    return metadata


def extract_contours_from_mask(
    mask: np.ndarray,
    sample_rate: int,
    nfft: int,
    noverlap: int,
    method: str = "centroid"
) -> pd.DataFrame:
    """
    Extract time-frequency contours from a binary mask.
    
    Args:
        mask: Binary mask (H, W) where H=freq bins, W=time frames
        sample_rate: Audio sample rate (Hz)
        nfft: FFT window length
        noverlap: Overlap between windows
        method: "centroid", "min_max", or "all_points"
    
    Returns:
        DataFrame with time and frequency information
    """
    height, width = mask.shape
    
    # Calculate physical parameters
    hop_length = nfft - noverlap
    time_per_frame = hop_length / sample_rate
    max_freq = sample_rate / 2
    freq_per_bin = max_freq / height
    
    contour_data = []
    
    if method == "centroid":
        # One point per time frame: frequency centroid
        for t_idx in range(width):
            col = mask[:, t_idx]
            if col.sum() > 0:
                # Get active frequency bins
                active_freqs = np.where(col)[0]
                
                # Calculate centroid (weighted average)
                centroid_idx = active_freqs.mean()
                
                # Convert to physical units
                time_sec = t_idx * time_per_frame
                # Note: frequency axis is inverted (0 = top = max freq)
                freq_hz = max_freq - (centroid_idx * freq_per_bin)
                
                contour_data.append({
                    'time_sec': time_sec,
                    'freq_hz': freq_hz,
                    'pixel_count': len(active_freqs)
                })
    
    elif method == "min_max":
        # Min and max frequency per time frame
        for t_idx in range(width):
            col = mask[:, t_idx]
            if col.sum() > 0:
                active_freqs = np.where(col)[0]
                
                time_sec = t_idx * time_per_frame
                freq_min = max_freq - (active_freqs.max() * freq_per_bin)
                freq_max = max_freq - (active_freqs.min() * freq_per_bin)
                
                contour_data.append({
                    'time_sec': time_sec,
                    'freq_min_hz': freq_min,
                    'freq_max_hz': freq_max,
                    'bandwidth_hz': freq_max - freq_min,
                    'pixel_count': len(active_freqs)
                })
    
    elif method == "all_points":
        # All active pixels
        active_pixels = np.argwhere(mask)
        for freq_idx, time_idx in active_pixels:
            time_sec = time_idx * time_per_frame
            freq_hz = max_freq - (freq_idx * freq_per_bin)
            
            contour_data.append({
                'time_sec': time_sec,
                'freq_hz': freq_hz
            })
    
    return pd.DataFrame(contour_data)


def compute_annotation_statistics(
    mask: np.ndarray,
    sample_rate: int,
    nfft: int,
    noverlap: int
) -> Dict:
    """
    Compute comprehensive statistics for an annotation mask.
    
    Returns:
        Dictionary with temporal and spectral statistics
    """
    if mask.sum() == 0:
        return {
            'pixel_count': 0,
            'duration_sec': 0,
            'bandwidth_hz': 0,
            'start_time_sec': None,
            'end_time_sec': None,
            'min_freq_hz': None,
            'max_freq_hz': None,
            'center_freq_hz': None,
            'time_coverage_pct': 0,
            'freq_coverage_pct': 0
        }
    
    height, width = mask.shape
    hop_length = nfft - noverlap
    time_per_frame = hop_length / sample_rate
    max_freq = sample_rate / 2
    freq_per_bin = max_freq / height
    
    # Find bounding box
    active_pixels = np.argwhere(mask)
    freq_indices = active_pixels[:, 0]
    time_indices = active_pixels[:, 1]
    
    # Temporal statistics
    time_min_idx = time_indices.min()
    time_max_idx = time_indices.max()
    start_time_sec = time_min_idx * time_per_frame
    end_time_sec = time_max_idx * time_per_frame
    duration_sec = end_time_sec - start_time_sec
    
    # Spectral statistics (frequency axis is inverted)
    freq_min_idx = freq_indices.min()
    freq_max_idx = freq_indices.max()
    max_freq_hz = max_freq - (freq_min_idx * freq_per_bin)
    min_freq_hz = max_freq - (freq_max_idx * freq_per_bin)
    bandwidth_hz = max_freq_hz - min_freq_hz
    center_freq_hz = (max_freq_hz + min_freq_hz) / 2
    
    return {
        'pixel_count': int(mask.sum()),
        'duration_sec': float(duration_sec),
        'bandwidth_hz': float(bandwidth_hz),
        'start_time_sec': float(start_time_sec),
        'end_time_sec': float(end_time_sec),
        'min_freq_hz': float(min_freq_hz),
        'max_freq_hz': float(max_freq_hz),
        'center_freq_hz': float(center_freq_hz),
        'time_coverage_pct': float(100 * (time_max_idx - time_min_idx + 1) / width),
        'freq_coverage_pct': float(100 * (freq_max_idx - freq_min_idx + 1) / height)
    }


def process_single_hdf5(hdf5_path: Path, contour_method: str = "centroid") -> Tuple[List, List]:
    """
    Process a single HDF5 file and extract contours + statistics.
    
    Returns:
        (contours_list, stats_list) where each is a list of dictionaries
    """
    print(f"  Processing: {hdf5_path.name}")
    
    # Load metadata
    metadata = load_hdf5_metadata(hdf5_path)
    
    sample_rate = metadata['sample_rate']
    nfft = metadata['nfft']
    noverlap = metadata['noverlap']
    class_names = metadata['class_names']
    clip_basename = metadata['clip_basename']
    project_index = metadata['project_index']
    
    sample_id = hdf5_path.stem
    
    # Load masks
    with h5py.File(hdf5_path, 'r') as f:
        masks = f['masks'][:]  # Shape: (num_classes, H, W)
    
    contours_list = []
    stats_list = []
    
    # Process each class
    for class_idx, class_name in enumerate(class_names):
        mask = masks[class_idx]
        
        # Skip empty masks
        if mask.sum() == 0:
            continue
        
        # Extract contours
        contours = extract_contours_from_mask(
            mask, sample_rate, nfft, noverlap, method=contour_method
        )
        
        if len(contours) > 0:
            contours['sample_id'] = sample_id
            contours['clip_basename'] = clip_basename
            contours['project_index'] = project_index
            contours['class'] = class_name
            contours['sample_rate'] = sample_rate
            contours['nfft'] = nfft
            contours['noverlap'] = noverlap
            
            contours_list.append(contours)
        
        # Compute statistics
        stats = compute_annotation_statistics(mask, sample_rate, nfft, noverlap)
        stats['sample_id'] = sample_id
        stats['clip_basename'] = clip_basename
        stats['project_index'] = project_index
        stats['class'] = class_name
        stats['sample_rate'] = sample_rate
        stats['nfft'] = nfft
        stats['noverlap'] = noverlap
        
        stats_list.append(stats)
    
    return contours_list, stats_list


def export_to_excel(
    ml_data_folder: str,
    output_excel: str,
    contour_method: str = "centroid"
):
    """
    Export all HDF5 files to a comprehensive Excel spreadsheet.
    
    Args:
        ml_data_folder: Path to folder containing HDF5 files
        output_excel: Output Excel filename
        contour_method: "centroid", "min_max", or "all_points"
    """
    ml_data_path = Path(ml_data_folder)
    
    # Find all HDF5 files
    hdf5_files = list(ml_data_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"❌ No HDF5 files found in {ml_data_folder}")
        return
    
    print(f"\n📊 Found {len(hdf5_files)} HDF5 files")
    print(f"Extraction method: {contour_method}")
    
    # Process all files
    all_contours = []
    all_stats = []
    
    for hdf5_path in sorted(hdf5_files):
        try:
            contours_list, stats_list = process_single_hdf5(hdf5_path, contour_method)
            all_contours.extend(contours_list)
            all_stats.extend(stats_list)
        except Exception as e:
            print(f"  ⚠️  Error processing {hdf5_path.name}: {e}")
            continue
    
    if not all_contours and not all_stats:
        print("❌ No data extracted from any files!")
        return
    
    print(f"\n✅ Extracted data from {len(hdf5_files)} files")
    
    # Combine dataframes
    if all_contours:
        contours_df = pd.concat(all_contours, ignore_index=True)
        print(f"   Total contour points: {len(contours_df)}")
    else:
        contours_df = pd.DataFrame()
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        # Filter out empty annotations
        stats_df = stats_df[stats_df['pixel_count'] > 0]
        print(f"   Total annotations: {len(stats_df)}")
    else:
        stats_df = pd.DataFrame()
    
    # Load dataset index for summary
    index_path = ml_data_path / "dataset_index.csv"
    if index_path.exists():
        summary_df = pd.read_csv(index_path)
    else:
        summary_df = pd.DataFrame()
    
    # Create Excel writer with multiple sheets
    print(f"\n💾 Writing to Excel: {output_excel}")
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Sheet 1: Summary
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✓ Sheet 'Summary': {len(summary_df)} samples")
        
        # Sheet 2: Contours (detailed time-frequency points)
        if not contours_df.empty:
            contours_df.to_excel(writer, sheet_name='Contours', index=False)
            print(f"   ✓ Sheet 'Contours': {len(contours_df)} data points")
        
        # Sheet 3: Statistics (per-annotation summary)
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            print(f"   ✓ Sheet 'Statistics': {len(stats_df)} annotations")
        
        # Sheet 4: Class Summary
        if not stats_df.empty:
            class_summary = stats_df.groupby('class').agg({
                'pixel_count': ['count', 'mean', 'std', 'min', 'max'],
                'duration_sec': ['mean', 'std', 'min', 'max'],
                'bandwidth_hz': ['mean', 'std', 'min', 'max'],
                'center_freq_hz': ['mean', 'std']
            }).round(2)
            
            # Flatten multi-index columns
            class_summary.columns = ['_'.join(col).strip() for col in class_summary.columns.values]
            class_summary = class_summary.reset_index()
            
            class_summary.to_excel(writer, sheet_name='Class_Summary', index=False)
            print(f"   ✓ Sheet 'Class_Summary': {len(class_summary)} classes")
    
    print(f"\n✅ Excel export complete: {output_excel}")
    
    # Print summary statistics
    if not stats_df.empty:
        print("\n📈 Dataset Statistics:")
        print(f"   Total samples: {stats_df['sample_id'].nunique()}")
        print(f"   Total annotations: {len(stats_df)}")
        print(f"   Classes: {sorted(stats_df['class'].unique())}")
        print(f"\n   Per-class annotation counts:")
        class_counts = stats_df['class'].value_counts().sort_index()
        for class_name, count in class_counts.items():
            print(f"      {class_name}: {count}")


def main():
    """Main execution function."""
    # Configuration
    ml_data_folder = "./hdf5_files"
    output_excel = "whale_contours_export.xlsx"
    
    # Contour extraction method:
    # - "centroid": One freq value per time frame (smoothest, recommended)
    # - "min_max": Min/max freq per time frame (captures bandwidth)
    # - "all_points": Every pixel (most detailed, largest file)
    contour_method = "centroid"
    
    print("="*60)
    print("🐋 HDF5 to Excel Time-Frequency Exporter")
    print("="*60)
    print(f"Input folder: {ml_data_folder}")
    print(f"Output file: {output_excel}")
    print(f"Contour method: {contour_method}")
    
    # Run export
    export_to_excel(
        ml_data_folder=ml_data_folder,
        output_excel=output_excel,
        contour_method=contour_method
    )
    
    print("\n" + "="*60)
    print("✅ Export complete!")
    print("="*60)
    print(f"\nOpen '{output_excel}' to view:")
    print("  - Summary: Dataset overview")
    print("  - Contours: Time-frequency points for each annotation")
    print("  - Statistics: Per-annotation metrics")
    print("  - Class_Summary: Aggregate statistics per class")


if __name__ == "__main__":
    main()