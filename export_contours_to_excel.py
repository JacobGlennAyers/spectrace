#!/usr/bin/env python3
"""
Export time-frequency contours from HDF5 files (schema v2) to Excel.

HDF5 schema v2 layout (one file per audio clip):
    clip_basename.hdf5
    ├── audio_wav
    ├── spectrogram            (uint8, HxW)
    ├── metadata/              (shared audio/spectrogram params)
    │   ├── @sample_rate
    │   ├── @nfft
    │   ├── @noverlap
    │   ├── @duration_sec
    │   ├── @max_freq_hz
    │   ├── @time_per_pixel
    │   └── @freq_per_pixel
    ├── @class_names           (JSON string)
    ├── @num_classes
    ├── @num_annotations
    └── annotations/
        ├── 0/
        │   ├── masks          (uint8, CxHxW)
        │   ├── @notes
        │   └── @timing_drift
        ├── 1/
        │   └── ...
        └── ...

Output Excel sheets:
  Summary       – one row per annotation set (from dataset_index.csv)
  Contours      – time-frequency points, one row per time frame per annotation
  Statistics    – per-annotation bounding-box metrics
  Class_Summary – aggregate stats per class across all annotations
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours_from_mask(
    mask: np.ndarray,
    sample_rate: int,
    nfft: int,
    noverlap: int,
    method: str = "centroid",
) -> pd.DataFrame:
    """
    Extract time-frequency contours from a binary mask.

    Args:
        mask:        Binary mask (H, W) — H=freq bins, W=time frames
        sample_rate: Audio sample rate in Hz
        nfft:        FFT window length
        noverlap:    Overlap between windows
        method:      "centroid" | "min_max" | "all_points"

    Returns:
        DataFrame with time/frequency columns.
    """
    height, width  = mask.shape
    hop_length     = nfft - noverlap
    time_per_frame = hop_length / sample_rate
    max_freq       = sample_rate / 2
    freq_per_bin   = max_freq / height

    rows = []

    if method == "centroid":
        for t in range(width):
            col = mask[:, t]
            if col.sum():
                active = np.where(col)[0]
                rows.append({
                    'time_sec':    t * time_per_frame,
                    'freq_hz':     max_freq - (active.mean() * freq_per_bin),
                    'pixel_count': len(active),
                })

    elif method == "min_max":
        for t in range(width):
            col = mask[:, t]
            if col.sum():
                active = np.where(col)[0]
                f_min = max_freq - (active.max() * freq_per_bin)
                f_max = max_freq - (active.min() * freq_per_bin)
                rows.append({
                    'time_sec':     t * time_per_frame,
                    'freq_min_hz':  f_min,
                    'freq_max_hz':  f_max,
                    'bandwidth_hz': f_max - f_min,
                    'pixel_count':  len(active),
                })

    elif method == "all_points":
        for f_idx, t_idx in np.argwhere(mask):
            rows.append({
                'time_sec': t_idx * time_per_frame,
                'freq_hz':  max_freq - (f_idx * freq_per_bin),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_annotation_statistics(
    mask: np.ndarray,
    sample_rate: int,
    nfft: int,
    noverlap: int,
) -> Dict:
    """Compute temporal and spectral bounding-box statistics for one mask."""

    if mask.sum() == 0:
        return dict(
            pixel_count=0, duration_sec=0, bandwidth_hz=0,
            start_time_sec=None, end_time_sec=None,
            min_freq_hz=None, max_freq_hz=None, center_freq_hz=None,
            time_coverage_pct=0, freq_coverage_pct=0,
        )

    height, width  = mask.shape
    hop_length     = nfft - noverlap
    time_per_frame = hop_length / sample_rate
    max_freq       = sample_rate / 2
    freq_per_bin   = max_freq / height

    active    = np.argwhere(mask)
    f_indices = active[:, 0]
    t_indices = active[:, 1]

    t_min, t_max = int(t_indices.min()), int(t_indices.max())
    f_min, f_max = int(f_indices.min()), int(f_indices.max())

    # Low pixel index (f_min) → high frequency; high pixel index → low freq
    freq_high = max_freq - (f_min * freq_per_bin)
    freq_low  = max_freq - (f_max * freq_per_bin)

    return dict(
        pixel_count       = int(mask.sum()),
        start_time_sec    = float(t_min * time_per_frame),
        end_time_sec      = float(t_max * time_per_frame),
        duration_sec      = float((t_max - t_min) * time_per_frame),
        min_freq_hz       = float(freq_low),
        max_freq_hz       = float(freq_high),
        bandwidth_hz      = float(freq_high - freq_low),
        center_freq_hz    = float((freq_high + freq_low) / 2),
        time_coverage_pct = float(100 * (t_max - t_min + 1) / width),
        freq_coverage_pct = float(100 * (f_max - f_min + 1) / height),
    )


# ---------------------------------------------------------------------------
# Per-file processor  (schema v2 — annotations/0/, annotations/1/, …)
# ---------------------------------------------------------------------------

def process_single_hdf5(
    hdf5_path: Path,
    contour_method: str = "centroid",
) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """
    Process one HDF5 file (schema v2).

    Reads shared metadata once from `metadata/`, then iterates every
    annotation set under `annotations/<index>/masks`.

    Returns:
        (contours_list, stats_list)
    """
    print(f"  {hdf5_path.name}")

    contours_list: List[pd.DataFrame] = []
    stats_list:    List[Dict]         = []

    with h5py.File(hdf5_path, 'r') as f:

        # ── shared metadata (stored once per clip) ───────────────────────────
        meta         = f['metadata']
        sample_rate  = int(meta.attrs['sample_rate'])
        nfft         = int(meta.attrs['nfft'])
        noverlap     = int(meta.attrs['noverlap'])
        duration_sec = float(meta.attrs['duration_sec'])

        clip_basename = hdf5_path.stem
        class_names   = json.loads(f.attrs['class_names'])

        # ── iterate every annotation set ────────────────────────────────────
        ann_grp  = f['annotations']
        ann_keys = sorted(ann_grp.keys(), key=int)

        for ann_key in ann_keys:
            ann_index    = int(ann_key)
            ann          = ann_grp[ann_key]
            masks        = ann['masks'][:]        # (C, H, W)
            notes        = str(ann.attrs.get('notes', ''))
            timing_drift = bool(ann.attrs.get('timing_drift', False))

            for class_idx, class_name in enumerate(class_names):
                mask = masks[class_idx]

                if mask.sum() == 0:
                    continue

                shared = {
                    'clip_basename':    clip_basename,
                    'annotation_index': ann_index,
                    'class':            class_name,
                    'notes':            notes,
                    'timing_drift':     timing_drift,
                    'sample_rate':      sample_rate,
                    'nfft':             nfft,
                    'noverlap':         noverlap,
                    'duration_sec':     duration_sec,
                }

                # contours
                df = extract_contours_from_mask(
                    mask, sample_rate, nfft, noverlap, contour_method
                )
                if len(df):
                    for col, val in shared.items():
                        df[col] = val
                    contours_list.append(df)

                # statistics
                stats = compute_annotation_statistics(
                    mask, sample_rate, nfft, noverlap
                )
                stats.update(shared)
                stats_list.append(stats)

    return contours_list, stats_list


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_to_excel(
    ml_data_folder: str,
    output_excel: str,
    contour_method: str = "centroid",
):
    """
    Walk ml_data_folder for .hdf5 files (schema v2) and write a
    multi-sheet Excel file.

    Args:
        ml_data_folder: Folder containing HDF5 files + dataset_index.csv
        output_excel:   Output .xlsx path
        contour_method: "centroid" | "min_max" | "all_points"
    """
    ml_data_path = Path(ml_data_folder)
    hdf5_files   = sorted(ml_data_path.glob("*.hdf5"))

    if not hdf5_files:
        print(f"❌ No HDF5 files found in {ml_data_folder}")
        return

    print(f"\n📊 {len(hdf5_files)} HDF5 file(s) found  (method: {contour_method})\n")

    all_contours: List[pd.DataFrame] = []
    all_stats:    List[Dict]         = []
    errors = 0

    for hdf5_path in hdf5_files:
        try:
            c, s = process_single_hdf5(hdf5_path, contour_method)
            all_contours.extend(c)
            all_stats.extend(s)
        except Exception as e:
            print(f"  ⚠️  {hdf5_path.name}: {e}")
            errors += 1

    if not all_contours and not all_stats:
        print("❌ No data extracted.")
        return

    ok = len(hdf5_files) - errors
    print(f"\n✅ {ok}/{len(hdf5_files)} files processed"
          + (f"  ({errors} error(s))" if errors else ""))

    # ── build DataFrames ─────────────────────────────────────────────────────
    contours_df = (
        pd.concat(all_contours, ignore_index=True)
        if all_contours else pd.DataFrame()
    )
    stats_df = pd.DataFrame(all_stats)
    if not stats_df.empty:
        stats_df = stats_df[stats_df['pixel_count'] > 0].reset_index(drop=True)

    if not contours_df.empty:
        print(f"   Contour points   : {len(contours_df):,}")
    if not stats_df.empty:
        print(f"   Annotations      : {len(stats_df):,}")

    # Optional summary from dataset_index.csv
    index_csv  = ml_data_path / "dataset_index.csv"
    summary_df = pd.read_csv(index_csv) if index_csv.exists() else pd.DataFrame()

    # ── column ordering ───────────────────────────────────────────────────────
    ID_COLS = ['clip_basename', 'annotation_index', 'class',
               'notes', 'timing_drift']

    def front_cols(df: pd.DataFrame) -> pd.DataFrame:
        front = [c for c in ID_COLS if c in df.columns]
        rest  = [c for c in df.columns if c not in front]
        return df[front + rest]

    # ── write Excel ──────────────────────────────────────────────────────────
    print(f"\n💾 Writing → {output_excel}")

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:

        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✓ Summary        : {len(summary_df)} rows")

        if not contours_df.empty:
            front_cols(contours_df).to_excel(
                writer, sheet_name='Contours', index=False
            )
            print(f"   ✓ Contours       : {len(contours_df):,} points")

        if not stats_df.empty:
            front_cols(stats_df).to_excel(
                writer, sheet_name='Statistics', index=False
            )
            print(f"   ✓ Statistics     : {len(stats_df):,} annotations")

            class_summary = (
                stats_df.groupby('class')
                .agg(
                    annotation_count    = ('pixel_count',    'count'),
                    duration_mean_sec   = ('duration_sec',   'mean'),
                    duration_std_sec    = ('duration_sec',   'std'),
                    duration_min_sec    = ('duration_sec',   'min'),
                    duration_max_sec    = ('duration_sec',   'max'),
                    bandwidth_mean_hz   = ('bandwidth_hz',   'mean'),
                    bandwidth_std_hz    = ('bandwidth_hz',   'std'),
                    bandwidth_min_hz    = ('bandwidth_hz',   'min'),
                    bandwidth_max_hz    = ('bandwidth_hz',   'max'),
                    center_freq_mean_hz = ('center_freq_hz', 'mean'),
                    center_freq_std_hz  = ('center_freq_hz', 'std'),
                    pixel_count_mean    = ('pixel_count',    'mean'),
                    pixel_count_total   = ('pixel_count',    'sum'),
                )
                .round(3)
                .reset_index()
            )
            class_summary.to_excel(
                writer, sheet_name='Class_Summary', index=False
            )
            print(f"   ✓ Class_Summary  : {len(class_summary)} classes")

    print(f"\n✅ Done → {output_excel}")

    # ── console summary ───────────────────────────────────────────────────────
    if not stats_df.empty:
        n_ann_sets = stats_df.groupby(['clip_basename', 'annotation_index']).ngroups
        print(f"\n📈 Dataset overview:")
        print(f"   Clips            : {stats_df['clip_basename'].nunique()}")
        print(f"   Annotation sets  : {n_ann_sets}")
        print(f"   Total annotations: {len(stats_df)}")
        print(f"   Classes          : {sorted(stats_df['class'].unique())}")
        print(f"\n   Annotations per class:")
        for cls, cnt in stats_df['class'].value_counts().sort_index().items():
            print(f"      {cls}: {cnt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ml_data_folder = "/media/songbird/SSD3/whale_freq_contour_trace_data/hdf5_data"
    output_excel   = "/media/songbird/SSD3/whale_freq_contour_trace_data/whale_contours_export.xlsx"

    # "centroid"   – one freq value per time frame (recommended)
    # "min_max"    – min + max freq per time frame (captures bandwidth)
    # "all_points" – every active pixel (largest output)
    contour_method = "all_points"

    print("=" * 60)
    print("🐋 HDF5 → Excel Exporter  (schema v2 — consolidated format)")
    print("=" * 60)
    print(f"Input  : {ml_data_folder}")
    print(f"Output : {output_excel}")
    print(f"Method : {contour_method}")

    export_to_excel(ml_data_folder, output_excel, contour_method)

    print("\n" + "=" * 60)
    print("Sheets written:")
    print("  Summary       – annotation sets (from dataset_index.csv)")
    print("  Contours      – time-frequency points per annotation")
    print("  Statistics    – per-annotation bounding-box metrics")
    print("  Class_Summary – aggregate stats per class")
    print("=" * 60)


if __name__ == "__main__":
    main()