#!/usr/bin/env python3
"""
Heterodyne validation: predict heterodyne frequencies from annotated HFC/LFC
fundamentals and compare against labelled heterodyne contours.

Key formula: heterodyne_freq = (n+1) * f_HFC +/- k * f_LFC
where n is the heterodyne order (0 = affiliated with HFC fundamental,
1 = affiliated with 1st HFC harmonic, etc.)

Supports two comparison modes:
  - Node-level (pixel IoU): strict pixel-by-pixel mask comparison with dilation
  - Contour-level: frequency curve comparison with MAE, RMSE, tolerance accuracy

Tolerance configuration
-----------------------
All accuracy tolerances are defined ONCE in ``TOLERANCES_HZ`` below, and the
single "headline" tolerance reported in the summary table is
``PRIMARY_TOLERANCE_HZ``. Column names, empty-case fills, the results table,
and the negative control all derive their tolerance columns from these two
constants via ``acc_key()`` — there are no hardcoded ``ba_acc_<n>hz`` strings
anywhere else in the file. To change which tolerance the table highlights,
edit ``PRIMARY_TOLERANCE_HZ`` (and make sure it appears in ``TOLERANCES_HZ``).

Usage (run from demos/ or from the repo root):
    python demos/heterodyne_validation.py --hdf5 ml_data/clip.hdf5
    python demos/heterodyne_validation.py --hdf5-dir ml_data/ --kernel-size 7 --max-k 3
    python demos/heterodyne_validation.py --hdf5 clip.hdf5 --no-plots
    python demos/heterodyne_validation.py --hdf5-dir ml_data/ --estimate-max-k
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Path setup — this file now lives in demos/.
# hdf5_utils and class_registry live one level up (repo root).
# bin_morph lives alongside this file in demos/.
# ---------------------------------------------------------------------------
_DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_DEMOS_DIR, "..")

sys.path.insert(0, _DEMOS_DIR)
sys.path.insert(0, _REPO_ROOT)

from hdf5_utils import HDF5SpectrogramLoader  # noqa: E402 (path setup must come first)
from bin_morph import MaskMorphology           # noqa: E402

HETERODYNE_ORDERS = list(range(13))  # 0 through 12

_HETERODYNE_CLASS_RE = re.compile(r"^Heterodynes/\d+$")

# ===========================================================================
# SINGLE SOURCE OF TRUTH for tolerance configuration
# ===========================================================================
# Every band-aware accuracy column is generated from this list. The summary
# table and the negative control report the PRIMARY tolerance. Nothing else in
# the file may hardcode a tolerance value or a "ba_acc_<n>hz" column name; use
# acc_key() / acc_keys() instead.
TOLERANCES_HZ: List[int] = [200, 250, 500, 1000, 2000]
PRIMARY_TOLERANCE_HZ: int = 250

# Variant suffixes for the three sub-band-selection strategies plus the raw fan.
BA_VARIANTS = ("full", "pruned", "fitted")

# Scalar (non-accuracy) band-aware metric keys, shared by every variant.
BA_SCALAR_KEYS = ("ba_mae_hz", "ba_median_hz", "ba_n_samples")


def acc_key(tol_hz: int) -> str:
    """Canonical band-aware accuracy column name for a tolerance in Hz.

    This is the ONLY place the ``ba_acc_<n>hz`` naming convention is defined.
    """
    return f"ba_acc_{int(tol_hz)}hz"


def acc_keys() -> List[str]:
    """All accuracy column names implied by TOLERANCES_HZ (base, no variant)."""
    return [acc_key(t) for t in TOLERANCES_HZ]


def ba_keys() -> List[str]:
    """All band-aware metric keys (scalars + accuracies), base names."""
    return list(BA_SCALAR_KEYS) + acc_keys()


def ba_empty_fill(result: dict, variant: str) -> None:
    """Fill every band-aware key for one variant with NaN.

    Derives the full key set from ba_keys() so a new tolerance added to
    TOLERANCES_HZ is automatically NaN-filled in the empty case — no manual
    tuple to keep in sync. This is what previously dropped ba_acc_250hz and
    produced a ragged DataFrame.
    """
    for k in ba_keys():
        result[f"{k}_{variant}"] = np.nan


def primary_acc_col(variant: str) -> str:
    """Column name for the PRIMARY tolerance under a given variant.

    e.g. primary_acc_col("fitted") -> "ba_acc_250hz_fitted".
    Used by the table and negative control so the headline metric is wired to
    PRIMARY_TOLERANCE_HZ with no fallback substitution.
    """
    return f"{acc_key(PRIMARY_TOLERANCE_HZ)}_{variant}"


# Sanity: the primary tolerance must be one of the configured tolerances,
# otherwise the table would request a column that is never computed.
assert PRIMARY_TOLERANCE_HZ in TOLERANCES_HZ, (
    f"PRIMARY_TOLERANCE_HZ={PRIMARY_TOLERANCE_HZ} is not in "
    f"TOLERANCES_HZ={TOLERANCES_HZ}"
)


# ---------------------------------------------------------------------------
# Pre-scan helpers
# ---------------------------------------------------------------------------

def get_valid_annotation_indices(hdf5_path: str) -> Tuple[List[int], int]:
    """Return (valid_indices, total_heterodyne_count) for a clip.

    valid_indices: annotation indices that have drawn Heterodynes/N masks
        AND both f0_HFC and f0_LFC in the same annotation set.
    total_heterodyne_count: total number of non-empty Heterodynes/N masks
        found across all valid annotation sets (one per order per set).

    Heterodynes/unsure is excluded by the regex: it has no defined order N
    and cannot be matched to any biphonic prediction.
    """
    valid_indices = []
    total_het_count = 0
    try:
        with HDF5SpectrogramLoader(hdf5_path) as loader:
            class_names = loader.get_class_names()

            het_class_names = [
                name for name in class_names
                if _HETERODYNE_CLASS_RE.match(name)
            ]
            if not het_class_names:
                return [], 0

            num_annotations = loader.get_num_annotations()
            for ann_idx in range(num_annotations):
                drawn_hets = [
                    name for name in het_class_names
                    if loader.get_class_mask(name, ann_idx) is not None
                    and loader.get_class_mask(name, ann_idx).sum() > 0
                ]
                if not drawn_hets:
                    continue

                hfc_mask = loader.get_class_mask("f0_HFC", ann_idx)
                lfc_mask = loader.get_class_mask("f0_LFC", ann_idx)

                missing = []
                if "f0_HFC" not in class_names or hfc_mask is None or hfc_mask.sum() == 0:
                    missing.append("f0_HFC")
                if "f0_LFC" not in class_names or lfc_mask is None or lfc_mask.sum() == 0:
                    missing.append("f0_LFC")

                if missing:
                    print(f"  WARNING: {Path(hdf5_path).name} annotation {ann_idx} "
                          f"has drawn heterodynes but missing fundamentals "
                          f"({', '.join(missing)}) — skipping this index.")
                    continue

                valid_indices.append(ann_idx)
                total_het_count += len(drawn_hets)

    except Exception as exc:
        print(f"  WARNING: could not read {hdf5_path}: {exc}")
    return valid_indices, total_het_count


def estimate_max_k_from_annotations(hdf5_path: str) -> dict:
    """Estimate the maximum k drawn in any Heterodynes/N mask in a clip.

    Each Heterodynes/N layer groups all +k/-k sidebands around one HFC
    harmonic into a single mask. In any given time column, the number of
    disconnected frequency bands present equals the number of distinct k
    values drawn (both the +k and -k sideband for the same k typically
    appear as separate blobs separated by the HFC harmonic frequency).

    We count the maximum number of disconnected bands in any single column
    across all Heterodynes/N masks and annotation sets. That count is the
    empirical max_k present in the data.

    Returns a dict with:
        max_k_observed   : int  — the maximum band count seen in any column
        k_distribution   : dict — {k_count: n_columns} across all masks
        orders_with_data : list — which Heterodynes/N orders had drawn masks
    """
    k_counts: dict = {}
    orders_with_data: List[int] = []

    try:
        with HDF5SpectrogramLoader(hdf5_path) as loader:
            class_names = loader.get_class_names()
            het_class_names = [n for n in class_names if _HETERODYNE_CLASS_RE.match(n)]
            if not het_class_names:
                return {"max_k_observed": 0, "k_distribution": {}, "orders_with_data": []}

            for ann_idx in range(loader.get_num_annotations()):
                for het_name in het_class_names:
                    mask = loader.get_class_mask(het_name, ann_idx)
                    if mask is None or mask.sum() == 0:
                        continue

                    order = int(het_name.split("/")[1])
                    if order not in orders_with_data:
                        orders_with_data.append(order)

                    H, W = mask.shape
                    for t in range(W):
                        col = mask[:, t]
                        if not col.any():
                            continue
                        # Count disconnected runs of active pixels in this column.
                        # Each run is one sideband (one k value, one sign).
                        # Gap threshold of >3 rows matches _labelled_bands_per_column.
                        rows = np.where(col)[0]
                        gaps = np.diff(rows)
                        n_bands = int(np.sum(gaps > 3)) + 1
                        k_counts[n_bands] = k_counts.get(n_bands, 0) + 1

    except Exception as exc:
        print(f"  WARNING: estimate_max_k could not read {hdf5_path}: {exc}")
        return {"max_k_observed": 0, "k_distribution": {}, "orders_with_data": []}

    max_k = max(k_counts.keys()) if k_counts else 0
    return {
        "max_k_observed": max_k,
        "k_distribution": dict(sorted(k_counts.items())),
        "orders_with_data": sorted(orders_with_data),
    }


def print_max_k_estimate(hdf5_dir: str) -> None:
    """Scan all HDF5 files and report the empirical max_k across the dataset."""
    hdf5_files = sorted(Path(hdf5_dir).glob("*.hdf5"))
    if not hdf5_files:
        print("No .hdf5 files found.")
        return

    print(f"\nEstimating max_k from annotations in {len(hdf5_files)} clip(s)…")
    global_k_counts: dict = {}
    global_max_k = 0
    all_orders: set = set()

    for f in hdf5_files:
        result = estimate_max_k_from_annotations(str(f))
        if result["max_k_observed"] == 0:
            continue
        clip_max = result["max_k_observed"]
        global_max_k = max(global_max_k, clip_max)
        all_orders.update(result["orders_with_data"])
        for k, n in result["k_distribution"].items():
            global_k_counts[k] = global_k_counts.get(k, 0) + n
        print(f"  {Path(f).name:<50}  max_k={clip_max}  "
              f"orders={result['orders_with_data']}")

    if not global_k_counts:
        print("  No heterodyne masks found.")
        return

    total_cols = sum(global_k_counts.values())
    print(f"\n  Band count distribution across all clips ({total_cols} columns):")
    for k in sorted(global_k_counts):
        pct = global_k_counts[k] / total_cols * 100
        bar = "█" * int(pct / 2)
        print(f"    {k} band(s): {global_k_counts[k]:>7} columns  ({pct:5.1f}%)  {bar}")
    print(f"\n  Heterodyne orders annotated: {sorted(all_orders)}")
    print(f"  → Recommended max_k = {global_max_k}  "
          f"(use this or lower based on the distribution above)")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_f0_contour(
    mask: np.ndarray,
    max_freq: float,
) -> np.ndarray:
    """Extract f0 frequency values as a dense 1D array indexed by time column.

    For each column, compute the frequency centroid of active pixels.
    Returns NaN for frames with no active pixels.

    ``max_freq`` is passed in (not recomputed from sample_rate) so that this
    function uses exactly the same frequency ceiling as the rest of the
    pipeline (meta.max_freq_hz). Recomputing sample_rate/2 locally was a
    latent inconsistency when max_freq_hz != Nyquist.

        freq_per_bin = max_freq / height
        freq_hz = max_freq - (mean_active_row * freq_per_bin)
    """
    height, width = mask.shape
    freq_per_bin = max_freq / height

    contour = np.full(width, np.nan)
    for t in range(width):
        col = mask[:, t]
        if col.any():
            active = np.where(col)[0]
            contour[t] = max_freq - (active.mean() * freq_per_bin)
    return contour


def smooth_f0_contour(
    contour: np.ndarray,
    window: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing of an f0 centroid contour, within gaps.

    Smoothing is applied only within contiguous non-NaN runs; NaN positions
    are preserved and runs shorter than ``window`` are returned unchanged.
    """
    out = contour.copy()
    if out.size == 0:
        return out

    valid = ~np.isnan(out)
    if not valid.any():
        return out

    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]  # exclusive

    for s, e in zip(run_starts, run_ends):
        if e - s >= window:
            out[s:e] = savgol_filter(
                contour[s:e], window_length=window, polyorder=polyorder,
            )
    return out


def compute_predicted_heterodyne_freqs(
    f0_hfc: np.ndarray,
    f0_lfc: np.ndarray,
    hfc_multiplier: int,
    max_k: int = 1,
    max_freq: float = np.inf,
) -> List[np.ndarray]:
    """Compute predicted heterodyne frequencies for one heterodyne order.

    ``hfc_multiplier`` is the effective HFC multiple, i.e. (heterodyne_order
    + 1): order 0 is affiliated with the HFC fundamental (multiplier 1),
    order 1 with the 1st harmonic (multiplier 2), etc. The caller is
    responsible for passing order+1; this parameter is named explicitly to
    avoid the previous ``order_n`` ambiguity where the docstring's ``n`` and
    the caller's ``n`` disagreed.

    For each time frame where both f0s are annotated:
        freq = hfc_multiplier * f_HFC +/- k * f_LFC   (k = 1..max_k)

    Returns a list of 1D arrays, one per (k, sign) sub-band. Values outside
    [0, max_freq] are set to NaN.
    """
    W = len(f0_hfc)
    both_valid = ~np.isnan(f0_hfc) & ~np.isnan(f0_lfc)
    results = []

    for k in range(1, max_k + 1):
        for sign in [1, -1]:
            freqs = np.full(W, np.nan)
            freqs[both_valid] = (
                hfc_multiplier * f0_hfc[both_valid]
                + sign * k * f0_lfc[both_valid]
            )
            freqs[(freqs < 0) | (freqs > max_freq)] = np.nan
            results.append(freqs)

    return results


def render_frequency_to_mask(
    freq_arrays: List[np.ndarray],
    height: int,
    width: int,
    max_freq: float,
    line_thickness: int = 1,
) -> np.ndarray:
    """Render frequency contour arrays into a binary mask.

    Inverse of the row-to-frequency mapping:
        row = round((max_freq - freq_hz) / freq_per_bin)

    Consecutive valid columns whose rows differ by more than one are bridged
    by linear interpolation so the rendered line is 8-connected with no holes.
    """
    freq_per_bin = max_freq / height
    mask = np.zeros((height, width), dtype=np.uint8)
    half = line_thickness // 2

    def _stamp(r: int, t: int) -> None:
        if 0 <= t < width:
            for rr in range(r - half, r + half + 1):
                if 0 <= rr < height:
                    mask[rr, t] = 1

    for freqs in freq_arrays:
        prev_row = None
        prev_t = None
        for t in range(width):
            f = freqs[t]
            if np.isnan(f):
                prev_row = None
                prev_t = None
                continue
            row = int(round((max_freq - f) / freq_per_bin))
            _stamp(row, t)

            if (
                prev_t is not None
                and prev_t == t - 1
                and abs(row - prev_row) > 1
            ):
                n_steps = abs(row - prev_row) + 1
                cs = np.linspace(prev_t, t, n_steps)
                rs = np.linspace(prev_row, row, n_steps)
                for c, r in zip(cs, rs):
                    _stamp(int(round(r)), int(round(c)))

            prev_row = row
            prev_t = t
    return mask


# ---------------------------------------------------------------------------
# Sub-band assignment helpers
# ---------------------------------------------------------------------------

def _labelled_bands_per_column(
    labelled_mask: np.ndarray,
    max_freq: float,
) -> dict:
    """Per-column labelled frequency bands (split at gaps > 3 rows).

    Returns {col -> list of centroid frequencies}.
    """
    H, W = labelled_mask.shape
    freq_per_bin = max_freq / H
    bands_per_col = {}
    for t in range(W):
        col = labelled_mask[:, t]
        if not col.any():
            continue
        rows = np.where(col)[0]
        gaps = np.diff(rows)
        gap_positions = np.where(gaps > 3)[0]
        bands = []
        start = 0
        for gi in gap_positions:
            bands.append(rows[start:gi + 1])
            start = gi + 1
        bands.append(rows[start:])
        bands_per_col[t] = [max_freq - b.mean() * freq_per_bin for b in bands]
    return bands_per_col


def assign_drawn_subbands_tolerance(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    tolerance_hz: float = 500.0,
) -> List[int]:
    """Return sub-band indices whose predictions come within tolerance_hz of
    at least one labelled frame."""
    lab_bands = _labelled_bands_per_column(labelled_mask, max_freq)
    drawn = set()
    for t, bands in lab_bands.items():
        pred_at_t = [
            (i, fa[t]) for i, fa in enumerate(pred_freqs)
            if not np.isnan(fa[t])
        ]
        if not pred_at_t:
            continue
        for lab_freq in bands:
            best_i = min(pred_at_t, key=lambda x: abs(lab_freq - x[1]))[0]
            if abs(lab_freq - pred_freqs[best_i][t]) <= tolerance_hz:
                drawn.add(best_i)
    return sorted(drawn)


def _segment_contour(
    segment_mask: np.ndarray,
    max_freq: float,
) -> np.ndarray:
    """Centroid frequency per column for a single connected-component mask."""
    H, W = segment_mask.shape
    freq_per_bin = max_freq / H
    contour = np.full(W, np.nan)
    for t in range(W):
        col = segment_mask[:, t]
        if col.any():
            active = np.where(col)[0]
            contour[t] = max_freq - (active.mean() * freq_per_bin)
    return contour


def fit_subband_per_segment(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    min_segment_size: int = 3,
) -> tuple:
    """Fit one sub-band per connected component of the label.

    Returns (fitted_freqs, fit_info).
    """
    H, W = labelled_mask.shape
    labeled_array, n_components = ndimage.label(
        labelled_mask > 0, structure=np.ones((3, 3), dtype=int)
    )

    fitted_freqs = []
    fit_info = []

    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled_array == comp_id).astype(np.uint8)
        n_pixels = int(comp_mask.sum())
        if n_pixels < min_segment_size:
            continue

        seg_contour = _segment_contour(comp_mask, max_freq)
        seg_cols = np.where(~np.isnan(seg_contour))[0]
        if len(seg_cols) == 0:
            continue
        t_min, t_max = int(seg_cols.min()), int(seg_cols.max())

        best_idx = None
        best_residual = np.inf
        best_n = 0
        for i, pred in enumerate(pred_freqs):
            both = ~np.isnan(seg_contour) & ~np.isnan(pred)
            if both.sum() < 2:
                continue
            residual = float(np.abs(seg_contour[both] - pred[both]).mean())
            if residual < best_residual:
                best_residual = residual
                best_idx = i
                best_n = int(both.sum())

        if best_idx is None:
            continue

        restricted = np.full(W, np.nan)
        restricted[t_min:t_max + 1] = pred_freqs[best_idx][t_min:t_max + 1]
        fitted_freqs.append(restricted)
        fit_info.append({
            "segment_id": comp_id,
            "sub_band_index": best_idx,
            "residual_hz": best_residual,
            "n_frames": best_n,
            "n_pixels": n_pixels,
            "t_min": t_min,
            "t_max": t_max,
        })

    return fitted_freqs, fit_info


# ---------------------------------------------------------------------------
# Band-aware metrics — accuracy columns derived from TOLERANCES_HZ
# ---------------------------------------------------------------------------

def compute_band_aware_metrics(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    tolerances_hz: Optional[List[int]] = None,
) -> dict:
    """Per-band MAE and tolerance-accuracy that handles multi-curve labels.

    Each (column, labelled-band) pair contributes exactly one error sample.
    Accuracy columns are produced for every tolerance in ``tolerances_hz``,
    which defaults to the module-level TOLERANCES_HZ. The returned keys use
    acc_key(), so they are guaranteed consistent with the rest of the file.

    Empty-input case returns the SAME key set (all NaN / zero), so the output
    schema does not depend on whether any samples were found — this is what
    keeps the assembled DataFrame rectangular.
    """
    if tolerances_hz is None:
        tolerances_hz = TOLERANCES_HZ

    lab_bands = _labelled_bands_per_column(labelled_mask, max_freq)
    errors = []
    for t, bands in lab_bands.items():
        pred_at_t = [fa[t] for fa in pred_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        for lab_freq in bands:
            best_err = min(abs(lab_freq - p) for p in pred_at_t)
            errors.append(best_err)

    if not errors:
        out = {"ba_mae_hz": np.nan, "ba_median_hz": np.nan, "ba_n_samples": 0}
        for tol in tolerances_hz:
            out[acc_key(tol)] = np.nan
        return out

    errors = np.array(errors)
    out = {
        "ba_mae_hz": float(errors.mean()),
        "ba_median_hz": float(np.median(errors)),
        "ba_n_samples": int(len(errors)),
    }
    for tol in tolerances_hz:
        out[acc_key(tol)] = float((errors <= tol).mean())
    return out


def compute_iou(
    predicted: np.ndarray,
    labelled: np.ndarray,
    kernel_size: int = 5,
) -> dict:
    """IoU between predicted and labelled binary masks (both dilated)."""
    morph = MaskMorphology()

    pred_px = int(predicted.sum())
    lab_px = int(labelled.sum())

    if pred_px == 0 and lab_px == 0:
        return {"iou": np.nan, "intersection": 0, "union": 0,
                "predicted_px": 0, "labelled_px": 0, "both_empty": True}

    pred_d = morph.dilate(predicted, kernel_size) if pred_px > 0 else predicted
    lab_d = morph.dilate(labelled, kernel_size) if lab_px > 0 else labelled

    intersection = int(np.logical_and(pred_d, lab_d).sum())
    union = int(np.logical_or(pred_d, lab_d).sum())
    iou = intersection / union if union > 0 else 0.0

    return {"iou": iou, "intersection": intersection, "union": union,
            "predicted_px": pred_px, "labelled_px": lab_px, "both_empty": False}


def compute_node_level_metrics(
    predicted: np.ndarray,
    labelled: np.ndarray,
    kernel_size: int = 5,
) -> dict:
    """Node-level per-bin TP/FP/FN with precision, recall, F1 (both dilated)."""
    morph = MaskMorphology()

    pred_px = int(predicted.sum())
    lab_px = int(labelled.sum())

    if pred_px == 0 and lab_px == 0:
        return {"node_precision": np.nan, "node_recall": np.nan,
                "node_f1": np.nan, "node_tp": 0, "node_fp": 0, "node_fn": 0}

    pred_d = morph.dilate(predicted, kernel_size) if pred_px > 0 else predicted
    lab_d = morph.dilate(labelled, kernel_size) if lab_px > 0 else labelled

    tp = int(np.logical_and(pred_d, lab_d).sum())
    fp = int(np.logical_and(pred_d, ~lab_d.astype(bool)).sum())
    fn = int(np.logical_and(~pred_d.astype(bool), lab_d).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"node_precision": float(precision), "node_recall": float(recall),
            "node_f1": float(f1), "node_tp": tp, "node_fp": fp, "node_fn": fn}


def compute_contour_level_metrics(
    predicted_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    freq_tolerance_hz: float = 500.0,
) -> dict:
    """Contour-level evaluation (coverage, fragmentation, deviation, P/R).

    The previously-unused ``overlap_threshold`` parameter has been removed.
    """
    height, width = labelled_mask.shape

    lab_bands_per_col = _labelled_bands_per_column(labelled_mask, max_freq)

    if not lab_bands_per_col:
        return {"contour_coverage": np.nan, "contour_fragmentation": np.nan,
                "contour_freq_deviation_hz": np.nan, "contour_precision": np.nan,
                "contour_precision_matched": np.nan, "contour_recall": np.nan,
                "contour_matched_sub_bands": 0}

    lab_frames = sorted(lab_bands_per_col.keys())
    matched_errors = []
    all_lab_errors = []

    for t in lab_frames:
        pred_at_t = [fa[t] for fa in predicted_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            all_lab_errors.append(np.inf)
            continue
        best_err = np.inf
        for lab_freq in lab_bands_per_col[t]:
            for p in pred_at_t:
                err = abs(lab_freq - p)
                if err < best_err:
                    best_err = err
        all_lab_errors.append(best_err)
        if best_err <= freq_tolerance_hz:
            matched_errors.append((t, best_err))

    pred_errors = []
    for fa in predicted_freqs:
        for t in range(width):
            if np.isnan(fa[t]):
                continue
            if t in lab_bands_per_col:
                best_err = min(abs(lf - fa[t]) for lf in lab_bands_per_col[t])
                pred_errors.append(best_err)
            else:
                pred_errors.append(np.inf)

    all_lab_errors = np.array(all_lab_errors)
    pred_errors = np.array(pred_errors) if pred_errors else np.array([np.inf])

    n_lab = len(lab_frames)
    n_matched = len(matched_errors)
    coverage = n_matched / n_lab if n_lab > 0 else 0.0

    if n_matched > 1:
        matched_cols = sorted([t for t, _ in matched_errors])
        runs = []
        run_len = 1
        for i in range(1, len(matched_cols)):
            if matched_cols[i] - matched_cols[i - 1] <= 2:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        fragmentation = 1.0 - max(runs) / n_matched
    elif n_matched == 1:
        fragmentation = 0.0
    else:
        fragmentation = np.nan

    freq_deviation = float(np.mean([e for _, e in matched_errors])) if matched_errors else np.nan
    contour_recall = float((all_lab_errors <= freq_tolerance_hz).mean()) if n_lab > 0 else 0.0
    contour_precision = float((pred_errors <= freq_tolerance_hz).mean()) if len(pred_errors) > 0 else 0.0

    matched_sub_bands = set()
    for t in lab_frames:
        pred_at_t = [(i, fa[t]) for i, fa in enumerate(predicted_freqs) if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        for lab_freq in lab_bands_per_col[t]:
            best_i = min(pred_at_t, key=lambda x: abs(lab_freq - x[1]))[0]
            if abs(lab_freq - predicted_freqs[best_i][t]) <= freq_tolerance_hz:
                matched_sub_bands.add(best_i)

    if matched_sub_bands:
        matched_pred_errors = []
        for i in matched_sub_bands:
            fa = predicted_freqs[i]
            for t in range(width):
                if np.isnan(fa[t]):
                    continue
                if t in lab_bands_per_col:
                    best_err = min(abs(lf - fa[t]) for lf in lab_bands_per_col[t])
                    matched_pred_errors.append(best_err)
                else:
                    matched_pred_errors.append(np.inf)
        matched_pred_errors = np.array(matched_pred_errors)
        precision_matched = float((matched_pred_errors <= freq_tolerance_hz).mean()) if len(matched_pred_errors) > 0 else 0.0
    else:
        precision_matched = 0.0

    return {
        "contour_coverage": float(coverage),
        "contour_fragmentation": float(fragmentation) if not np.isnan(fragmentation) else np.nan,
        "contour_freq_deviation_hz": freq_deviation,
        "contour_precision": float(contour_precision),
        "contour_precision_matched": float(precision_matched),
        "contour_recall": float(contour_recall),
        "contour_matched_sub_bands": len(matched_sub_bands),
    }


def compute_contour_metrics(
    predicted_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    tolerance_hz: Optional[List[int]] = None,
) -> dict:
    """Basic contour comparison: MAE, RMSE, correlation, tolerance accuracy.

    ``max_freq`` is passed in (was previously recomputed from sample_rate);
    the unused nfft/noverlap parameters have been removed. The contour_acc_*
    columns use the same TOLERANCES_HZ list as everything else.
    """
    if tolerance_hz is None:
        tolerance_hz = TOLERANCES_HZ

    height, width = labelled_mask.shape
    freq_per_bin = max_freq / height

    lab_contour = np.full(width, np.nan)
    for t in range(width):
        col = labelled_mask[:, t]
        if col.any():
            active = np.where(col)[0]
            lab_contour[t] = max_freq - (active.mean() * freq_per_bin)

    lab_valid = ~np.isnan(lab_contour)
    empty = {"contour_mae_hz": np.nan, "contour_rmse_hz": np.nan,
             "contour_corr": np.nan, "contour_n_points": 0,
             **{f"contour_acc_{int(t)}hz": np.nan for t in tolerance_hz}}
    if lab_valid.sum() == 0:
        return empty

    errors = []
    for t in range(width):
        if not lab_valid[t]:
            continue
        pred_at_t = [fa[t] for fa in predicted_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        errors.append(min(abs(lab_contour[t] - p) for p in pred_at_t))

    if not errors:
        return empty

    errors = np.array(errors)
    result = {
        "contour_mae_hz": float(errors.mean()),
        "contour_rmse_hz": float(np.sqrt((errors ** 2).mean())),
        "contour_n_points": len(errors),
    }
    for tol in tolerance_hz:
        result[f"contour_acc_{int(tol)}hz"] = float((errors <= tol).mean())

    pred_nearest = np.full(width, np.nan)
    for t in range(width):
        if not lab_valid[t]:
            continue
        pred_at_t = [(fa[t], abs(lab_contour[t] - fa[t]))
                     for fa in predicted_freqs if not np.isnan(fa[t])]
        if pred_at_t:
            pred_nearest[t] = min(pred_at_t, key=lambda x: x[1])[0]

    both_valid = ~np.isnan(lab_contour) & ~np.isnan(pred_nearest)
    if both_valid.sum() >= 3:
        result["contour_corr"] = float(np.corrcoef(
            lab_contour[both_valid], pred_nearest[both_valid])[0, 1])
    else:
        result["contour_corr"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def validate_single_clip(
    hdf5_path: str,
    annotation_index: int = 0,
    kernel_size: int = 5,
    max_k: int = 1,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run full heterodyne validation on a single HDF5 file."""
    clip_name = Path(hdf5_path).stem
    print(f"\n{'='*60}")
    print(f"Validating: {clip_name}")
    print(f"{'='*60}")

    with HDF5SpectrogramLoader(hdf5_path) as loader:
        meta = loader.get_metadata()
        class_names = loader.get_class_names()
        spectrogram = loader.load_spectrogram()

        for required in ["f0_HFC", "f0_LFC"]:
            if required not in class_names:
                print(f"  WARNING: '{required}' not in class registry, skipping clip")
                return pd.DataFrame()

        f0_hfc_mask = loader.get_class_mask("f0_HFC", annotation_index)
        f0_lfc_mask = loader.get_class_mask("f0_LFC", annotation_index)

        max_freq = meta.max_freq_hz

        f0_hfc = smooth_f0_contour(extract_f0_contour(f0_hfc_mask, max_freq))
        f0_lfc = smooth_f0_contour(extract_f0_contour(f0_lfc_mask, max_freq))

        H, W = f0_hfc_mask.shape

        both_coverage = np.sum(~np.isnan(f0_hfc) & ~np.isnan(f0_lfc)) / W * 100
        print(f"  f0_HFC coverage: {np.sum(~np.isnan(f0_hfc)) / W * 100:.1f}%")
        print(f"  f0_LFC coverage: {np.sum(~np.isnan(f0_lfc)) / W * 100:.1f}%")
        print(f"  Both annotated:  {both_coverage:.1f}%")
        if both_coverage < 1:
            print("  WARNING: Less than 1% of frames have both fundamentals annotated")

        rows = []
        predicted_masks = {}
        predicted_freq_arrays = {}
        labelled_masks = {}

        for n in HETERODYNE_ORDERS:
            het_name = f"Heterodynes/{n}"
            if het_name not in class_names:
                continue

            labelled = loader.get_class_mask(het_name, annotation_index)
            labelled_masks[n] = labelled

            # Heterodynes/N is affiliated with the (N+1)th HFC multiple.
            pred_freqs = compute_predicted_heterodyne_freqs(
                f0_hfc, f0_lfc, hfc_multiplier=n + 1, max_k=max_k, max_freq=max_freq
            )
            predicted = render_frequency_to_mask(pred_freqs, H, W, max_freq)
            predicted_masks[n] = predicted
            predicted_freq_arrays[n] = pred_freqs

            result = compute_iou(predicted, labelled, kernel_size)
            result["clip"] = clip_name
            result["order"] = n
            result["f0_coverage_pct"] = both_coverage
            result.update(compute_node_level_metrics(predicted, labelled, kernel_size))
            result.update(compute_contour_metrics(pred_freqs, labelled, max_freq))
            result.update(compute_contour_level_metrics(pred_freqs, labelled, max_freq))

            # Band-aware: full fan
            ba_full = compute_band_aware_metrics(pred_freqs, labelled, max_freq)
            for k, v in ba_full.items():
                result[f"{k}_full"] = v

            # Band-aware: tolerance-pruned
            drawn_idx = assign_drawn_subbands_tolerance(
                pred_freqs, labelled, max_freq, tolerance_hz=500.0)
            if drawn_idx:
                pruned_freqs = [pred_freqs[i] for i in drawn_idx]
                pruned_mask = render_frequency_to_mask(pruned_freqs, H, W, max_freq)
                result["iou_pruned"] = compute_iou(pruned_mask, labelled, kernel_size)["iou"]
                result["n_drawn_subbands"] = len(drawn_idx)
                result["drawn_sub_band_indices"] = ",".join(str(i) for i in drawn_idx)
                for k, v in compute_band_aware_metrics(pruned_freqs, labelled, max_freq).items():
                    result[f"{k}_pruned"] = v
            else:
                result["iou_pruned"] = np.nan
                result["n_drawn_subbands"] = 0
                result["drawn_sub_band_indices"] = ""
                ba_empty_fill(result, "pruned")

            # Band-aware: per-segment fit
            fitted_freqs, fit_info = fit_subband_per_segment(pred_freqs, labelled, max_freq)
            if fitted_freqs:
                fitted_mask = render_frequency_to_mask(fitted_freqs, H, W, max_freq)
                result["iou_fitted"] = compute_iou(fitted_mask, labelled, kernel_size)["iou"]
                result["n_segments"] = len(fit_info)
                result["fitted_sub_band_indices"] = ",".join(
                    str(f["sub_band_index"]) for f in fit_info)
                total_frames = sum(f["n_frames"] for f in fit_info) or 1
                result["fit_residual_hz_mean"] = float(
                    sum(f["residual_hz"] * f["n_frames"] for f in fit_info) / total_frames)
                for k, v in compute_band_aware_metrics(fitted_freqs, labelled, max_freq).items():
                    result[f"{k}_fitted"] = v
            else:
                result["iou_fitted"] = np.nan
                result["n_segments"] = 0
                result["fitted_sub_band_indices"] = ""
                result["fit_residual_hz_mean"] = np.nan
                ba_empty_fill(result, "fitted")

            rows.append(result)

            iou_str = f"{result['iou']:.3f}" if not np.isnan(result["iou"]) else "N/A"
            if result["both_empty"]:
                print(f"  Heterodynes/{n:>2d}: (both empty)")
            elif result["labelled_px"] == 0:
                print(f"  Heterodynes/{n:>2d}: (no labels, pred={result['predicted_px']}px)")
            else:
                p = result.get("iou_pruned", np.nan)
                f = result.get("iou_fitted", np.nan)
                mae_fit = result.get("ba_mae_hz_fitted", np.nan)
                acc_fit = result.get(primary_acc_col("fitted"), np.nan)
                p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
                f_str = f"{f:.3f}" if not np.isnan(f) else "N/A"
                mae_fit_s = f"{mae_fit:.0f}" if not np.isnan(mae_fit) else "N/A"
                acc_fit_s = f"{acc_fit:.0%}" if not np.isnan(acc_fit) else "N/A"
                print(f"  Heterodynes/{n:>2d}:  IoU[full={iou_str} prune={p_str} fit={f_str}]  "
                      f"MAE_fit={mae_fit_s}Hz  Acc@{PRIMARY_TOLERANCE_HZ}_fit={acc_fit_s}  "
                      f"segs={result.get('n_segments', 0)}")

    df = pd.DataFrame(rows)

    if output_dir is not None and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        generate_visualizations(
            spectrogram, df, predicted_masks, labelled_masks,
            clip_name, max_freq, meta.duration_sec, output_dir,
            predicted_freq_arrays=predicted_freq_arrays, f0_hfc=f0_hfc, f0_lfc=f0_lfc)
        csv_path = os.path.join(output_dir, f"{clip_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Results saved to: {csv_path}")

    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_visualizations(
    spectrogram: np.ndarray,
    results_df: pd.DataFrame,
    predicted_masks: Dict[int, np.ndarray],
    labelled_masks: Dict[int, np.ndarray],
    clip_name: str,
    max_freq: float,
    duration_sec: float,
    output_dir: str,
    predicted_freq_arrays: Optional[Dict[int, List[np.ndarray]]] = None,
    f0_hfc: Optional[np.ndarray] = None,
    f0_lfc: Optional[np.ndarray] = None,
):
    """Generate metrics charts and per-order contour overlay plots."""
    short_name = clip_name.split("--")[-1] if "--" in clip_name else clip_name

    labelled_orders = sorted(n for n, m in labelled_masks.items() if m.sum() > 0)
    if not labelled_orders:
        return

    labelled_df = results_df[results_df["order"].isin(labelled_orders)].copy()
    H, W = spectrogram.shape
    col_axis = np.arange(W)

    if not labelled_df.empty:
        has_contour = "contour_mae_hz" in labelled_df.columns
        fig, axes = plt.subplots(1, 2 if has_contour else 1,
                                 figsize=(14 if has_contour else 8, 5))
        axes = list(axes) if has_contour else [axes]

        orders = labelled_df["order"].values
        ious = labelled_df["iou"].values
        x = np.arange(len(orders))
        colors = ["#2ecc71" if v >= 0.5 else "#f39c12" if v >= 0.2 else "#e74c3c"
                  for v in np.nan_to_num(ious)]
        axes[0].bar(x, np.nan_to_num(ious), color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("Heterodyne Order", fontsize=12)
        axes[0].set_ylabel("IoU (Node-Level)", fontsize=12)
        axes[0].set_title(f"{short_name} - Node-Level IoU", fontsize=13)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(orders)
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.5, color="green", linestyle="--", alpha=0.3)

        if has_contour:
            # Bar series derive from the three smallest configured tolerances,
            # so adding/removing a tolerance reshapes the chart automatically.
            chart_tols = sorted(TOLERANCES_HZ)[:3]
            palette = ["#e74c3c", "#f39c12", "#2ecc71"]
            w = 0.8 / max(len(chart_tols), 1)
            for i, tol in enumerate(chart_tols):
                col = f"contour_acc_{int(tol)}hz"
                if col in labelled_df.columns:
                    vals = labelled_df[col].fillna(0).values
                    axes[1].bar(x + (i - (len(chart_tols) - 1) / 2) * w, vals, width=w,
                                color=palette[i % len(palette)], edgecolor="black",
                                linewidth=0.5, label=f"<{tol} Hz")
            axes[1].set_xlabel("Heterodyne Order", fontsize=12)
            axes[1].set_ylabel("Accuracy", fontsize=12)
            axes[1].set_title(f"{short_name} - Contour-Level Tolerance", fontsize=13)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(orders)
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{clip_name}_metrics_chart.png"), dpi=150)
        plt.close(fig)

    if predicted_freq_arrays is None:
        return

    primary_contour_acc = f"contour_acc_{PRIMARY_TOLERANCE_HZ}hz"
    for n in labelled_orders:
        pred_freqs = predicted_freq_arrays.get(n)
        lab_mask = labelled_masks.get(n)
        if pred_freqs is None or lab_mask is None:
            continue

        freq_per_bin = max_freq / H
        row = results_df[results_df["order"] == n]
        mae = row["contour_mae_hz"].values[0] if len(row) else np.nan
        acc = row[primary_contour_acc].values[0] if (len(row) and primary_contour_acc in row) else np.nan
        iou = row["iou"].values[0] if len(row) else np.nan

        fig, ax = plt.subplots(figsize=(max(10, W / 40), max(6, H / 40)))
        ax.imshow(spectrogram, aspect="equal", cmap="gray", origin="upper",
                  extent=[0, W, H, 0], interpolation="nearest")

        lab_overlay = np.zeros((H, W, 4))
        lab_overlay[lab_mask > 0] = [1, 0, 1, 1]
        ax.imshow(lab_overlay, aspect="equal", origin="upper",
                  extent=[0, W, H, 0], interpolation="nearest", zorder=3)

        colors_pred = ['cyan', 'lime', 'yellow', 'orange', 'red', 'white']
        for i, fa in enumerate(pred_freqs):
            valid = ~np.isnan(fa)
            if valid.sum() == 0:
                continue
            c = colors_pred[i % len(colors_pred)]
            sign = "+" if i % 2 == 0 else "-"
            k = i // 2 + 1
            pred_rows = (max_freq - fa[valid]) / freq_per_bin
            ax.plot(col_axis[valid], pred_rows, '-', color=c, linewidth=1.5,
                    alpha=0.8, label=f'Pred k={k} ({sign})', zorder=4)

        if f0_hfc is not None:
            hfc_harmonic = (n + 1) * f0_hfc
            hfc_valid = ~np.isnan(hfc_harmonic)
            if hfc_valid.sum() > 0:
                hfc_rows = (max_freq - hfc_harmonic[hfc_valid]) / freq_per_bin
                ax.plot(col_axis[hfc_valid], hfc_rows, '--', color='white',
                        linewidth=1, alpha=0.5, label=f'{n+1}x HFC', zorder=2)

        yticks_rows = np.linspace(0, H, 8)
        ax.set_yticks(yticks_rows)
        ax.set_yticklabels([f"{(max_freq - r * freq_per_bin)/1000:.0f}" for r in yticks_rows])

        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor='magenta', label='Labelled'))
        labels.insert(0, 'Labelled')

        iou_str = f"IoU={iou:.3f}" if not np.isnan(iou) else ""
        mae_str = f"MAE={mae:.0f}Hz" if not np.isnan(mae) else ""
        acc_str = f"Acc@{PRIMARY_TOLERANCE_HZ}Hz={acc:.0%}" if not np.isnan(acc) else ""
        metrics = "  ".join(s for s in [iou_str, mae_str, acc_str] if s)

        ax.set_ylim(H, 0)
        ax.set_xlim(0, W)
        ax.set_title(f"{short_name} - Heterodynes/{n}    {metrics}", fontsize=13)
        ax.set_xlabel("Column (pixel)", fontsize=12)
        ax.set_ylabel("Frequency (kHz)", fontsize=12)
        ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{clip_name}_contour_order_{n}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_validation_table(df: pd.DataFrame) -> None:
    """Per-clip breakdown and aggregate summary (Table 2 in the paper).

    Reports the FITTED variant at the PRIMARY tolerance. The accuracy column
    is resolved once via primary_acc_col("fitted"); if it is genuinely absent
    (e.g. a malformed older CSV) we raise rather than silently substituting a
    different tolerance under the same header.
    """
    labelled = df[df["labelled_px"] > 0].copy()
    if labelled.empty:
        print("\nNo labelled heterodyne rows to summarise.")
        return

    acc_col = primary_acc_col("fitted")
    if acc_col not in labelled.columns:
        raise KeyError(
            f"Expected column '{acc_col}' (PRIMARY_TOLERANCE_HZ="
            f"{PRIMARY_TOLERANCE_HZ}) not found. Columns present: "
            f"{[c for c in labelled.columns if c.startswith('ba_acc_')]}. "
            f"Re-run validation with the current TOLERANCES_HZ.")

    acc_hdr = f"Acc@{PRIMARY_TOLERANCE_HZ}Hz"
    col_w = 76
    header = (f"{'Clip':<36}  {'Order':>5}  {'IoU':>6}  "
              f"{'MAE(Hz)':>7}  {acc_hdr:>9}  {'Heterodynes':>11}")
    divider = "-" * col_w
    thick = "=" * col_w

    print(f"\n{thick}")
    print("HETERODYNE VALIDATION RESULTS")
    print(thick)
    print(header)
    print(divider)

    clip_summaries = []
    for clip_name, clip_df in labelled.groupby("clip", sort=False):
        parts = clip_name.split("--")
        short = "--".join(parts[-2:]) if len(parts) >= 2 else clip_name
        if len(short) > 36:
            short = short[:33] + "..."

        for _, row in clip_df.sort_values("order").iterrows():
            iou = row.get("iou_fitted", np.nan)
            mae = row.get("ba_mae_hz_fitted", np.nan)
            acc = row.get(acc_col, np.nan)
            segs = row.get("n_segments", 0)
            iou_s = f"{iou:.3f}" if not np.isnan(iou) else "  N/A"
            mae_s = f"{mae:.0f}" if not np.isnan(mae) else "  N/A"
            acc_s = f"{acc:.0%}" if not np.isnan(acc) else "  N/A"
            print(f"{short:<36}  {int(row['order']):>5}  {iou_s:>6}  "
                  f"{mae_s:>7}  {acc_s:>9}  {int(segs):>11}")

        iou_mean = clip_df["iou_fitted"].dropna().mean()
        mae_mean = clip_df["ba_mae_hz_fitted"].dropna().mean()
        acc_mean = clip_df[acc_col].dropna().mean()
        segs_sum = clip_df["n_segments"].sum()
        iou_s = f"{iou_mean:.3f}" if not np.isnan(iou_mean) else "  N/A"
        mae_s = f"{mae_mean:.0f}" if not np.isnan(mae_mean) else "  N/A"
        acc_s = f"{acc_mean:.0%}" if not np.isnan(acc_mean) else "  N/A"

        print(divider)
        print(f"{'  >> summary':<36}  {'':>5}  {iou_s:>6}  "
              f"{mae_s:>7}  {acc_s:>9}  {int(segs_sum):>11}")
        print(divider)

        clip_summaries.append({"iou_mean": iou_mean, "mae_mean": mae_mean,
                               "acc_mean": acc_mean, "segs_sum": segs_sum})

    agg_iou = float(np.nanmean([s["iou_mean"] for s in clip_summaries]))
    agg_mae = float(np.nanmean([s["mae_mean"] for s in clip_summaries]))
    agg_acc = float(np.nanmean([s["acc_mean"] for s in clip_summaries]))
    agg_segs = int(sum(s["segs_sum"] for s in clip_summaries))

    print(thick)
    print(f"{'  ALL CLIPS (mean)':<36}  {'':>5}  {agg_iou:.3f}  "
          f"{agg_mae:>7.0f}  {agg_acc:>9.0%}  {agg_segs:>11}")
    print(thick)


def aggregate_positive_metrics(df: pd.DataFrame, variant: str = "fitted") -> dict:
    """Mean-of-per-clip-means for MAE and the PRIMARY-tolerance accuracy.

    Returns the same numbers the ALL CLIPS row shows, so the negative control
    can compare against an identically-derived positive baseline. ``variant``
    selects which sub-band strategy to summarise; the negative control passes
    the variant it itself uses so the comparison is apples-to-apples.
    """
    labelled = df[df["labelled_px"] > 0] if "labelled_px" in df.columns else pd.DataFrame()
    mae_col = f"ba_mae_hz_{variant}"
    acc_col = primary_acc_col(variant)
    if labelled.empty or mae_col not in labelled.columns or acc_col not in labelled.columns:
        return {"mae": float("nan"), "acc": float("nan"), "variant": variant}
    return {
        "mae": float(labelled.groupby("clip")[mae_col].mean().mean()),
        "acc": float(labelled.groupby("clip")[acc_col].mean().mean()),
        "variant": variant,
    }


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------

def run_negative_control(
    skipped_files: List[Path],
    max_k: int,
    kernel_size: int,
    output_dir: Optional[str],
    pos_metrics: Optional[dict] = None,
) -> None:
    """Run the biphonic formula on clips with fundamentals but NO heterodynes,
    measuring how close predictions land to OTHER drawn contours.

    The negative-side metric is the PRIMARY-tolerance band-aware accuracy on
    the FULL fan, and the positive baseline passed in via ``pos_metrics`` must
    be derived from the same variant (see aggregate_positive_metrics) so the
    two columns are the same metric. The accuracy column is resolved through
    acc_key(PRIMARY_TOLERANCE_HZ) — no hardcoded 250.
    """
    _REFERENCE_LAYER_PREFIXES = (
        "harmonics_HFC", "harmonics_LFC", "subharmonics_HFC", "subharmonics_LFC",
        "heterodyne_or_subharmonic_or_other", "Cetacean_AdditionalContours",
        "unsure_HFC", "unsure_LFC", "Heterodynes/unsure",
    )
    primary_acc = acc_key(PRIMARY_TOLERANCE_HZ)
    rows = []

    for hdf5_path in skipped_files:
        try:
            with HDF5SpectrogramLoader(str(hdf5_path)) as loader:
                class_names = loader.get_class_names()
                if "f0_HFC" not in class_names or "f0_LFC" not in class_names:
                    continue

                meta = loader.get_metadata()
                max_freq = meta.max_freq_hz

                for ann_idx in range(loader.get_num_annotations()):
                    hfc_mask = loader.get_class_mask("f0_HFC", ann_idx)
                    lfc_mask = loader.get_class_mask("f0_LFC", ann_idx)
                    if hfc_mask is None or hfc_mask.sum() == 0:
                        continue
                    if lfc_mask is None or lfc_mask.sum() == 0:
                        continue

                    f0_hfc = smooth_f0_contour(extract_f0_contour(hfc_mask, max_freq))
                    f0_lfc = smooth_f0_contour(extract_f0_contour(lfc_mask, max_freq))

                    ref_masks = {}
                    for name in class_names:
                        if not any(name.startswith(p) for p in _REFERENCE_LAYER_PREFIXES):
                            continue
                        mask = loader.get_class_mask(name, ann_idx)
                        if mask is not None and mask.sum() > 0:
                            ref_masks[name] = mask
                    if not ref_masks:
                        continue

                    for order in range(7):  # orders 0-6, matching positive
                        pred_freqs = compute_predicted_heterodyne_freqs(
                            f0_hfc, f0_lfc, hfc_multiplier=order + 1,
                            max_k=max_k, max_freq=max_freq)
                        for ref_name, ref_mask in ref_masks.items():
                            ba = compute_band_aware_metrics(pred_freqs, ref_mask, max_freq)
                            rows.append({
                                "clip": hdf5_path.stem, "ann_idx": ann_idx,
                                "order": order, "reference_layer": ref_name,
                                "ba_mae_hz": ba["ba_mae_hz"],
                                primary_acc: ba[primary_acc],
                                "ba_n_samples": ba["ba_n_samples"],
                            })
        except Exception as exc:
            print(f"  WARNING: negative control — could not read {hdf5_path.name}: {exc}")

    if not rows:
        print("\nNegative control: no clips with fundamentals and other drawn content found.")
        return

    nc_df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nc_path = os.path.join(output_dir, "negative_control_results.csv")
        nc_df.to_csv(nc_path, index=False)
        print(f"  Negative control results saved to: {nc_path}")

    valid_nc = nc_df[nc_df["ba_n_samples"] > 0].copy()
    if valid_nc.empty:
        print("\nNegative control: no valid reference mask comparisons found.")
        return

    col_w = 72
    thick = "=" * col_w
    divider = "-" * col_w
    acc_hdr = f"Acc@{PRIMARY_TOLERANCE_HZ}Hz"

    print(f"\n{thick}")
    print("NEGATIVE CONTROL: predictions vs non-heterodyne contours")
    print(thick)
    print(f"  Clips used:          {nc_df['clip'].nunique()}")
    print(f"  Annotation sets:     {nc_df.groupby(['clip','ann_idx']).ngroups}")
    print(f"  Reference layers:    {nc_df['reference_layer'].nunique()} unique classes")
    print(f"  Comparisons made:    {len(valid_nc)}")
    print(divider)
    print(f"  {'Order':>5}  {'MAE(Hz)':>9}  {acc_hdr:>9}  {'N samples':>9}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}")
    for order, grp in valid_nc.groupby("order"):
        weights = grp["ba_n_samples"].values.astype(float)
        w_mae = float(np.average(grp["ba_mae_hz"].values, weights=weights))
        w_acc = float(np.average(grp[primary_acc].fillna(0).values, weights=weights))
        print(f"  {int(order):>5}  {w_mae:>9.0f}  {w_acc:>9.0%}  {int(weights.sum()):>9}")

    print(divider)
    weights = valid_nc["ba_n_samples"].values.astype(float)
    overall_mae = float(np.average(valid_nc["ba_mae_hz"].values, weights=weights))
    overall_acc = float(np.average(valid_nc[primary_acc].fillna(0).values, weights=weights))
    print(f"  {'all':>5}  {overall_mae:>9.0f}  {overall_acc:>9.0%}  {int(weights.sum()):>9}")
    print(thick)

    # Positive baseline — must be the SAME variant/tolerance as the negative side.
    print("  Compare against positive validation (all clips mean):")
    if pos_metrics is None:
        pos_metrics = {"mae": float("nan"), "acc": float("nan"), "variant": "?"}
    pos_mae = pos_metrics.get("mae", float("nan"))
    pos_acc = pos_metrics.get("acc", float("nan"))
    pos_variant = pos_metrics.get("variant", "?")
    pos_mae_s = f"{pos_mae:.0f} Hz" if not np.isnan(pos_mae) else "N/A"
    pos_acc_s = f"{pos_acc:.0%}" if not np.isnan(pos_acc) else "N/A"
    print(f"    MAE {pos_mae_s}  |  {acc_hdr} {pos_acc_s}   "
          f"(positive variant: {pos_variant}; negative variant: full)")
    if pos_variant != "full":
        print(f"    NOTE: positive baseline uses the '{pos_variant}' sub-band variant "
              f"while the negative control uses the full fan.")
        print(f"    These are different sub-band selections; the MAE gap is the "
              f"robust signal. See aggregate_positive_metrics(variant='full') "
              f"for a strict like-for-like comparison.")
    print("  A substantially higher negative-control MAE and lower accuracy")
    print("  confirms the formula is specific to labelled heterodynes.")
    print(thick)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heterodyne validation: predict heterodynes from f0 "
                    "annotations and compare against labelled contours.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5", type=str, help="Single HDF5 file to validate")
    group.add_argument("--hdf5-dir", type=str, help="Directory of HDF5 files (batch)")

    parser.add_argument("--annotation-index", type=int, default=0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=1)
    parser.add_argument("--output-dir", type=str,
                        default="visualizations/heterodyne_validation")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--negative-control", action="store_true",
                        help="Run the biphonic formula against non-heterodyne "
                             "contours in skipped clips. Only valid with --hdf5-dir.")
    parser.add_argument("--neg-control-variant", choices=BA_VARIANTS, default="full",
                        help="Positive-baseline variant to compare the negative "
                             "control against (default: full = strict like-for-like, "
                             "since the negative side uses the full fan).")
    parser.add_argument("--estimate-max-k", action="store_true",
                        help="Scan all HDF5 files and report the empirical max_k "
                             "present in the annotations, then exit. "
                             "Only valid with --hdf5-dir.")
    args = parser.parse_args()

    out_dir = None if args.no_plots else args.output_dir

    if args.hdf5:
        if not os.path.isfile(args.hdf5):
            print(f"ERROR: HDF5 file not found: {args.hdf5}")
            sys.exit(1)
        valid_indices, _ = get_valid_annotation_indices(args.hdf5)
        if not valid_indices:
            print(f"ERROR: {args.hdf5} has no annotation set with drawn "
                  f"f0_HFC, f0_LFC, and at least one Heterodynes/N mask.")
            sys.exit(1)
        if args.annotation_index in valid_indices:
            ann_idx = args.annotation_index
        else:
            print(f"  WARNING: annotation index {args.annotation_index} not valid "
                  f"(valid: {valid_indices}). Using {valid_indices[0]}.")
            ann_idx = valid_indices[0]
        df = validate_single_clip(args.hdf5, ann_idx, args.kernel_size, args.max_k, out_dir)
        if not df.empty:
            print_validation_table(df)
    else:
        hdf5_dir = Path(args.hdf5_dir)
        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"ERROR: No .hdf5 files found in {hdf5_dir}")
            sys.exit(1)

        if args.estimate_max_k:
            print_max_k_estimate(args.hdf5_dir)
            sys.exit(0)

        print(f"Scanning {len(hdf5_files)} HDF5 file(s) for heterodyne annotations…")
        heterodyne_files = []
        skipped_paths: List[Path] = []
        skipped_names = []
        total_het_count = 0
        for f in hdf5_files:
            valid_indices, het_count = get_valid_annotation_indices(str(f))
            if valid_indices:
                heterodyne_files.append((f, valid_indices))
                total_het_count += het_count
            else:
                skipped_paths.append(f)
                skipped_names.append(f.name)

        if skipped_names:
            print(f"  Skipped {len(skipped_names)} clip(s) with no valid heterodyne "
                  f"annotation sets: {', '.join(skipped_names)}")
        if not heterodyne_files:
            print("No clips with valid heterodyne annotations found. Nothing to validate.")
            sys.exit(0)

        n_indices = sum(len(idxs) for _, idxs in heterodyne_files)
        print(f"  Found {len(heterodyne_files)} clip(s) with {n_indices} valid "
              f"annotation set(s) and {total_het_count} annotated heterodyne "
              f"contour(s) — proceeding with validation.\n")

        all_dfs = []
        for f, valid_indices in heterodyne_files:
            for ann_idx in valid_indices:
                clip_df = validate_single_clip(str(f), ann_idx, args.kernel_size,
                                               args.max_k, out_dir)
                if not clip_df.empty:
                    all_dfs.append(clip_df)

        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            if out_dir:
                agg_path = os.path.join(out_dir, "aggregate_results.csv")
                df.to_csv(agg_path, index=False)
                print(f"\nAggregate results: {agg_path}")
            print_validation_table(df)
        else:
            print("No clips had valid results.")
            df = pd.DataFrame()

        if args.negative_control:
            if not skipped_paths:
                print("\nNegative control: all clips had valid heterodyne "
                      "annotations — no negative pool available.")
            else:
                print(f"\nRunning negative control on {len(skipped_paths)} skipped clip(s)…")
                pos_metrics = aggregate_positive_metrics(df, variant=args.neg_control_variant)
                run_negative_control(skipped_paths, args.max_k, args.kernel_size,
                                     out_dir, pos_metrics=pos_metrics)


if __name__ == "__main__":
    main()