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

Usage:
    python heterodyne_validation.py --hdf5 ml_data/clip.hdf5
    python heterodyne_validation.py --hdf5-dir ml_data/ --kernel-size 7 --max-k 3
    python heterodyne_validation.py --hdf5 clip.hdf5 --no-plots
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import savgol_filter

from hdf5_utils import HDF5SpectrogramLoader

# MaskMorphology lives in demos/ (no __init__.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos"))
from bin_morph import MaskMorphology

HETERODYNE_ORDERS = list(range(13))  # 0 through 12


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_f0_contour(
    mask: np.ndarray,
    sample_rate: int,
    nfft: int,
    noverlap: int,
) -> np.ndarray:
    """Extract f0 frequency values as a dense 1D array indexed by time column.

    For each column, compute the frequency centroid of active pixels.
    Returns NaN for frames with no active pixels.

    Math matches export_contours_to_excel.py:80-89 exactly:
        max_freq = sample_rate / 2
        freq_per_bin = max_freq / height
        freq_hz = max_freq - (mean_active_row * freq_per_bin)
    """
    height, width = mask.shape
    max_freq = sample_rate / 2
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

    The annotator draws HFC and LFC with a 1-pixel pen, so the raw
    centroid contour snaps to integer rows and carries ±0.5-row
    quantization noise. The biphonic model `(n+1)*f_HFC ± k*f_LFC`
    then multiplies that noise by `(n+1)`, producing a visibly
    staircased prediction at higher heterodyne orders. Smoothing the
    inputs (before they enter the biphonic math) removes the
    quantization noise at its source, so both the metrics and the
    rendered figure become clean.

    Smoothing is applied **only within contiguous non-NaN runs** of
    the contour — we must not interpolate across gaps because they
    typically correspond to silent frames or different calls. Runs
    shorter than `window` samples are returned unchanged.

    Parameters
    ----------
    contour : 1D array of frequencies, NaN where no pen stroke exists.
    window  : Savitzky-Golay window length (must be odd, > polyorder).
    polyorder : polynomial order for the local fit.

    Returns
    -------
    A new array of the same length; NaN positions are preserved.
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
    order_n: int,
    max_k: int = 1,
    max_freq: float = np.inf,
) -> List[np.ndarray]:
    """Compute predicted heterodyne frequencies for a given order.

    For each time frame where both f0s are annotated:
        freq_plus  = n * f_HFC + k * f_LFC
        freq_minus = n * f_HFC - k * f_LFC

    Returns list of 1D arrays (one per k/sign combination).
    Values outside [0, max_freq] are set to NaN.
    """
    W = len(f0_hfc)
    both_valid = ~np.isnan(f0_hfc) & ~np.isnan(f0_lfc)
    results = []

    for k in range(1, max_k + 1):
        for sign in [1, -1]:
            freqs = np.full(W, np.nan)
            valid = both_valid.copy()
            freqs[valid] = order_n * f0_hfc[valid] + sign * k * f0_lfc[valid]
            # Discard out-of-range
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

    Consecutive valid columns are connected by linearly interpolating
    the line between their predicted rows: when the contour jumps by
    more than one row between column ``t-1`` and ``t``, every integer
    pixel on the segment from (t-1, row_{t-1}) to (t, row_t) is
    stamped. Without this, sharp frequency jumps would leave vertical
    gaps in the rendered line because each column only marks a single
    endpoint row.
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

            # Bridge any vertical gap to the immediately preceding
            # valid column. Linear interpolation over |dr|+1 steps
            # splits the fill half-and-half across the two columns,
            # producing an 8-connected line with no visual holes.
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
# Exact-integer sub-band rasterisation from the HFC/LFC masks
# ---------------------------------------------------------------------------
#
# `render_frequency_to_mask` above takes a 1D centroid-per-column contour
# array and rounds it to the nearest pixel row. When the annotator's HFC or
# LFC pen was occasionally 2 pixels wide, the centroid becomes a half-integer
# row and the rounding introduces a staircase artefact in the rendered
# heterodyne prediction.
#
# The function below bypasses the centroid intermediate entirely. At each
# column it enumerates every (r_h, r_l) pair of annotated HFC and LFC pixel
# rows, and computes the predicted heterodyne row directly in integer
# arithmetic:
#
#   row_pred = H * (1 - order_n - s*k) + order_n * r_h + s*k * r_l
#
# which follows from f_pred = order_n * f_HFC + s*k * f_LFC with
# f = max_freq * (H - r) / H. All terms are integers, so no rounding is
# needed, and the prediction's pixel thickness automatically mirrors the
# annotation's pen width.


def render_subband_masks_from_f0(
    f0_hfc_mask: np.ndarray,
    f0_lfc_mask: np.ndarray,
    order_n: int,
    sub_band_specs: list,
    H: int,
    W: int,
) -> np.ndarray:
    """Union-rasterise one or more sub-bands of the biphonic model.

    Parameters
    ----------
    f0_hfc_mask, f0_lfc_mask : HxW uint8/bool annotation masks.
    order_n : the effective HFC multiplier, i.e. (heterodyne_order + 1).
    sub_band_specs : list of dicts. Each dict has keys
        - ``sub_band_index`` (int, 0..2*max_k-1) — encodes (k, sign)
          via k = idx//2 + 1, sign = +1 if idx%2==0 else -1
        - ``t_min`` (optional int, default 0)
        - ``t_max`` (optional int, default W-1)
      The rasterisation is restricted to columns in [t_min, t_max].
    H, W : spectrogram shape.

    Returns
    -------
    uint8 HxW mask, the union across all specs.
    """
    out = np.zeros((H, W), dtype=np.uint8)
    for spec in sub_band_specs:
        sb = spec["sub_band_index"]
        t_min = int(spec.get("t_min", 0))
        t_max = int(spec.get("t_max", W - 1))
        k = sb // 2 + 1
        sign = 1 if sb % 2 == 0 else -1
        offset = H * (1 - order_n - sign * k)

        # Loop over columns in the requested range. Numpy broadcasting
        # handles the cartesian product (r_h, r_l) for each column.
        t_lo = max(0, t_min)
        t_hi = min(W - 1, t_max)
        for t in range(t_lo, t_hi + 1):
            hfc_rows = np.flatnonzero(f0_hfc_mask[:, t])
            if len(hfc_rows) == 0:
                continue
            lfc_rows = np.flatnonzero(f0_lfc_mask[:, t])
            if len(lfc_rows) == 0:
                continue
            # (n_hfc, n_lfc) grid of integer predicted rows
            rp = (
                offset
                + order_n * hfc_rows[:, None]
                + sign * k * lfc_rows[None, :]
            ).ravel()
            rp = rp[(rp >= 0) & (rp < H)]
            if rp.size:
                out[rp, t] = 1
    return out


# ---------------------------------------------------------------------------
# Sub-band assignment — address the 6-fan ambiguity problem
# ---------------------------------------------------------------------------
#
# The raw predicted mask for a given heterodyne order is the union of up to
# 2*max_k sub-bands: (n+1)*f_HFC ± k*f_LFC for k in 1..max_k. The annotator's
# `Heterodynes/N` layer usually contains only one or two of those sub-bands.
# Computing IoU on the full union therefore penalises the pipeline for
# generating physically valid sub-bands that the annotator simply didn't draw.
#
# We provide two principled ways to narrow the prediction down to what was
# actually annotated:
#
#   (a) Tolerance pruning — keep every sub-band whose prediction comes within
#       `tolerance_hz` of at least one labelled frame. Inclusive; allows
#       multiple sub-bands through when the label is genuinely broadband.
#
#   (b) Per-segment fit — split the labelled mask into connected components
#       and, for each component, pick the single sub-band with the smallest
#       mean residual. Committal; exactly one predicted curve per labelled
#       curve component. This is the "recover the hidden parameter" approach.
#
# Both strategies produce a sub-set of the predicted frequency arrays that
# can be fed unchanged into render_frequency_to_mask / compute_iou /
# compute_contour_metrics to get "fair" numbers.


def _labelled_bands_per_column(
    labelled_mask: np.ndarray,
    max_freq: float,
) -> dict:
    """Extract per-column labelled frequency bands (split at gaps > 3 rows).

    Shared helper — used by both tolerance pruning and the existing
    compute_contour_level_metrics. Returns a dict {col -> list of freqs}.
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
    """Return sorted sub-band indices whose predictions come within
    tolerance_hz of at least one labelled frame.

    For each labelled band at each column, pick the sub-band with the
    smallest distance; if that distance is within tolerance, record the
    sub-band as "drawn" by the annotator.
    """
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
    """Centroid frequency per column for a single connected-component mask.

    Returns a length-W array with NaN for columns not in the component.
    """
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

    For each connected component of `labelled_mask`:
      1. Extract its frequency contour (centroid per column).
      2. For every candidate sub-band index, compute mean |label - pred|
         over frames where both are defined.
      3. Pick the sub-band with the smallest residual.
      4. Restrict that sub-band's prediction to this segment's time range.

    Returns (fitted_freqs, fit_info):
      fitted_freqs: list of length-W arrays, one per segment, with NaN
                    outside the segment's time range.
      fit_info:     list of dicts with per-segment diagnostics
                    (sub_band_index, residual_hz, n_frames, n_pixels).
    """
    H, W = labelled_mask.shape
    # 8-connectivity so diagonally adjacent pixels count as one component
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

        # Find best-fitting sub-band for this segment
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

        # Restrict the fitted prediction to this segment's time range
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


def compute_band_aware_metrics(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    tolerance_hz: List[float] = None,
) -> dict:
    """Per-band MAE and tolerance-accuracy that handles multi-curve labels.

    The centroid-based `compute_contour_metrics` silently averages across
    multiple vertically stacked curves at the same column, giving a
    meaningless "middle" frequency. This function instead extracts each
    labelled band separately and evaluates it against the closest
    prediction. Each (column, labelled-band) pair contributes exactly one
    error sample.

    Returns mean/median/Acc@{200,500,1000,2000}Hz over these per-band errors.
    """
    if tolerance_hz is None:
        tolerance_hz = [200, 500, 1000, 2000]

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
        out = {
            "ba_mae_hz": np.nan,
            "ba_median_hz": np.nan,
            "ba_n_samples": 0,
        }
        for tol in tolerance_hz:
            out[f"ba_acc_{int(tol)}hz"] = np.nan
        return out

    errors = np.array(errors)
    out = {
        "ba_mae_hz": float(errors.mean()),
        "ba_median_hz": float(np.median(errors)),
        "ba_n_samples": int(len(errors)),
    }
    for tol in tolerance_hz:
        out[f"ba_acc_{int(tol)}hz"] = float((errors <= tol).mean())
    return out


def compute_iou(
    predicted: np.ndarray,
    labelled: np.ndarray,
    kernel_size: int = 5,
) -> dict:
    """Compute IoU between predicted and labelled binary masks.

    Both masks are dilated before comparison to account for annotation
    imprecision.
    """
    morph = MaskMorphology()

    pred_px = int(predicted.sum())
    lab_px = int(labelled.sum())

    if pred_px == 0 and lab_px == 0:
        return {
            "iou": np.nan,
            "intersection": 0,
            "union": 0,
            "predicted_px": 0,
            "labelled_px": 0,
            "both_empty": True,
        }

    if pred_px > 0:
        pred_d = morph.dilate(predicted, kernel_size)
    else:
        pred_d = predicted

    if lab_px > 0:
        lab_d = morph.dilate(labelled, kernel_size)
    else:
        lab_d = labelled

    intersection = int(np.logical_and(pred_d, lab_d).sum())
    union = int(np.logical_or(pred_d, lab_d).sum())
    iou = intersection / union if union > 0 else 0.0

    return {
        "iou": iou,
        "intersection": intersection,
        "union": union,
        "predicted_px": pred_px,
        "labelled_px": lab_px,
        "both_empty": False,
    }


def compute_node_level_metrics(
    predicted: np.ndarray,
    labelled: np.ndarray,
    kernel_size: int = 5,
) -> dict:
    """Node-level evaluation: per-bin TP/FP/FN with precision, recall, F1.

    Following the SAM-whistle paper convention: each time-frequency bin is
    classified independently. Both masks are dilated to allow tolerance.
    """
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

    return {
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
        "node_tp": tp,
        "node_fp": fp,
        "node_fn": fn,
    }


def compute_contour_level_metrics(
    predicted_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    freq_tolerance_hz: float = 500.0,
    overlap_threshold: float = 0.3,
) -> dict:
    """Contour-level evaluation following the SAM-whistle paper convention.

    Extracts separate contour segments from the labelled mask, matches each
    to the nearest predicted sub-band, then computes 5 metrics:
      - coverage: fraction of labelled time frames that have a matched prediction
      - fragmentation: 1 - (longest_matched_run / total_matched_frames)
      - freq_deviation_hz: mean absolute frequency error on matched frames
      - contour_precision: fraction of predicted frames within tolerance of a label
      - contour_recall: fraction of labelled frames within tolerance of a prediction
    """
    height, width = labelled_mask.shape
    freq_per_bin = max_freq / height

    # Extract labelled contour bands per column (split at gaps > 3 rows)
    lab_bands_per_col = {}  # col -> list of centroid frequencies
    for t in range(width):
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

        lab_bands_per_col[t] = [max_freq - b.mean() * freq_per_bin for b in bands]

    if not lab_bands_per_col:
        return {"contour_coverage": np.nan, "contour_fragmentation": np.nan,
                "contour_freq_deviation_hz": np.nan, "contour_precision": np.nan,
                "contour_recall": np.nan}

    # For each labelled point, find closest predicted frequency and its error
    lab_frames = sorted(lab_bands_per_col.keys())
    matched_errors = []  # (col, error_hz) for frames within tolerance
    all_lab_errors = []  # error for every labelled frame (for recall calc)

    for t in lab_frames:
        pred_at_t = [fa[t] for fa in predicted_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            all_lab_errors.append(np.inf)
            continue

        # Best match across all labelled bands at this column
        best_err = np.inf
        for lab_freq in lab_bands_per_col[t]:
            for p in pred_at_t:
                err = abs(lab_freq - p)
                if err < best_err:
                    best_err = err

        all_lab_errors.append(best_err)
        if best_err <= freq_tolerance_hz:
            matched_errors.append((t, best_err))

    # For each predicted frame, find closest labelled frequency (for precision)
    pred_frames = set()
    pred_errors = []
    for fa in predicted_freqs:
        for t in range(width):
            if np.isnan(fa[t]):
                continue
            pred_frames.add(t)
            if t in lab_bands_per_col:
                best_err = min(abs(lf - fa[t]) for lf in lab_bands_per_col[t])
                pred_errors.append(best_err)
            else:
                pred_errors.append(np.inf)

    all_lab_errors = np.array(all_lab_errors)
    pred_errors = np.array(pred_errors) if pred_errors else np.array([np.inf])

    # Coverage: fraction of labelled frames that matched a prediction
    n_lab = len(lab_frames)
    n_matched = len(matched_errors)
    coverage = n_matched / n_lab if n_lab > 0 else 0.0

    # Fragmentation: 1 - (longest contiguous matched run / total matched)
    if n_matched > 1:
        matched_cols = sorted([t for t, _ in matched_errors])
        runs = []
        run_len = 1
        for i in range(1, len(matched_cols)):
            if matched_cols[i] - matched_cols[i - 1] <= 2:  # allow 1-col gap
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

    # Frequency deviation: mean error on matched frames
    if matched_errors:
        freq_deviation = float(np.mean([e for _, e in matched_errors]))
    else:
        freq_deviation = np.nan

    # Contour recall: fraction of labelled frames within tolerance of any prediction
    contour_recall = float((all_lab_errors <= freq_tolerance_hz).mean()) if n_lab > 0 else 0.0

    # Contour precision (all): fraction of predicted frames within tolerance of any label
    contour_precision = float((pred_errors <= freq_tolerance_hz).mean()) if len(pred_errors) > 0 else 0.0

    # Contour precision (matched only): only count the sub-bands the annotator drew.
    # For each labelled band at each column, identify which predicted sub-band index
    # is the best match. Then compute precision using only those sub-band predictions.
    matched_sub_bands = set()  # set of (col, sub_band_index)
    for t in lab_frames:
        pred_at_t = [(i, fa[t]) for i, fa in enumerate(predicted_freqs) if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        for lab_freq in lab_bands_per_col[t]:
            best_i = min(pred_at_t, key=lambda x: abs(lab_freq - x[1]))[0]
            if abs(lab_freq - predicted_freqs[best_i][t]) <= freq_tolerance_hz:
                matched_sub_bands.add(best_i)

    if matched_sub_bands:
        # Recompute precision using only the matched sub-band predictions
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
    sample_rate: int,
    nfft: int,
    noverlap: int,
    tolerance_hz: List[float] = None,
) -> dict:
    """Contour-level comparison: extract frequency curves and compare.

    For the labelled mask, extract the centroid frequency per time frame.
    For the predicted frequencies (already computed as 1D arrays), find the
    closest predicted sub-band to each labelled point and measure the error.

    Returns dict with MAE, RMSE, correlation, and tolerance accuracy.
    """
    if tolerance_hz is None:
        tolerance_hz = [200, 500, 1000]

    height, width = labelled_mask.shape
    max_freq = sample_rate / 2
    freq_per_bin = max_freq / height

    # Extract labelled contour (centroid per column)
    lab_contour = np.full(width, np.nan)
    for t in range(width):
        col = labelled_mask[:, t]
        if col.any():
            active = np.where(col)[0]
            lab_contour[t] = max_freq - (active.mean() * freq_per_bin)

    lab_valid = ~np.isnan(lab_contour)
    if lab_valid.sum() == 0:
        return {"contour_mae_hz": np.nan, "contour_rmse_hz": np.nan,
                "contour_corr": np.nan, "contour_n_points": 0,
                **{f"contour_acc_{int(t)}hz": np.nan for t in tolerance_hz}}

    # For each labelled time frame, find the closest predicted frequency
    errors = []
    for t in range(width):
        if not lab_valid[t]:
            continue
        # Collect all predicted sub-band frequencies at this time
        pred_at_t = [fa[t] for fa in predicted_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        # Minimum distance to any predicted sub-band
        min_err = min(abs(lab_contour[t] - p) for p in pred_at_t)
        errors.append(min_err)

    if not errors:
        return {"contour_mae_hz": np.nan, "contour_rmse_hz": np.nan,
                "contour_corr": np.nan, "contour_n_points": 0,
                **{f"contour_acc_{int(t)}hz": np.nan for t in tolerance_hz}}

    errors = np.array(errors)
    result = {
        "contour_mae_hz": float(errors.mean()),
        "contour_rmse_hz": float(np.sqrt((errors ** 2).mean())),
        "contour_n_points": len(errors),
    }

    # Tolerance accuracy
    for tol in tolerance_hz:
        result[f"contour_acc_{int(tol)}hz"] = float((errors <= tol).mean())

    # Correlation: match labelled centroid to nearest predicted, compare curves
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
            lab_contour[both_valid], pred_nearest[both_valid]
        )[0, 1])
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
    """Run full heterodyne validation on a single HDF5 file.

    Returns DataFrame with one row per heterodyne order.
    """
    clip_name = Path(hdf5_path).stem
    print(f"\n{'='*60}")
    print(f"Validating: {clip_name}")
    print(f"{'='*60}")

    with HDF5SpectrogramLoader(hdf5_path) as loader:
        meta = loader.get_metadata()
        class_names = loader.get_class_names()
        spectrogram = loader.load_spectrogram()

        # Check required classes exist
        for required in ["f0_HFC", "f0_LFC"]:
            if required not in class_names:
                print(f"  WARNING: '{required}' not in class registry, skipping clip")
                return pd.DataFrame()

        # Load fundamental masks
        f0_hfc_mask = loader.get_class_mask("f0_HFC", annotation_index)
        f0_lfc_mask = loader.get_class_mask("f0_LFC", annotation_index)

        # Extract f0 contours — these are the per-column centroid of
        # the annotator's 1-pixel pen strokes, so they snap to integer
        # rows and carry ±0.5-row quantization noise.
        f0_hfc = extract_f0_contour(f0_hfc_mask, meta.sample_rate, meta.nfft, meta.noverlap)
        f0_lfc = extract_f0_contour(f0_lfc_mask, meta.sample_rate, meta.nfft, meta.noverlap)

        # Savitzky-Golay denoising within contiguous runs (see
        # smooth_f0_contour docstring for rationale). This is the
        # single place pen quantization is addressed: every downstream
        # prediction, mask, and metric inherits the smoothed inputs,
        # so the figure rendering and the Table 2 metrics stay
        # coherent by construction.
        f0_hfc = smooth_f0_contour(f0_hfc)
        f0_lfc = smooth_f0_contour(f0_lfc)

        H, W = f0_hfc_mask.shape
        max_freq = meta.max_freq_hz

        # Coverage statistics
        hfc_coverage = np.sum(~np.isnan(f0_hfc)) / W * 100
        lfc_coverage = np.sum(~np.isnan(f0_lfc)) / W * 100
        both_coverage = np.sum(~np.isnan(f0_hfc) & ~np.isnan(f0_lfc)) / W * 100
        print(f"  f0_HFC coverage: {hfc_coverage:.1f}%")
        print(f"  f0_LFC coverage: {lfc_coverage:.1f}%")
        print(f"  Both annotated:  {both_coverage:.1f}%")

        if both_coverage < 1:
            print("  WARNING: Less than 1% of frames have both fundamentals annotated")

        # Validate each heterodyne order
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

            # Compute predicted — Heterodynes/N is affiliated with the
            # (N+1)th harmonic of HFC (N=0 → fundamental, N=1 → 1st harmonic, etc.)
            # so the HFC multiplier is (n + 1), not n.
            pred_freqs = compute_predicted_heterodyne_freqs(
                f0_hfc, f0_lfc, order_n=n + 1, max_k=max_k, max_freq=max_freq
            )
            predicted = render_frequency_to_mask(pred_freqs, H, W, max_freq)
            predicted_masks[n] = predicted
            predicted_freq_arrays[n] = pred_freqs

            # Node-level: IoU + precision/recall/F1 on binary masks
            result = compute_iou(predicted, labelled, kernel_size)
            result["clip"] = clip_name
            result["order"] = n
            result["f0_coverage_pct"] = both_coverage

            node_metrics = compute_node_level_metrics(predicted, labelled, kernel_size)
            result.update(node_metrics)

            # Contour-level: frequency curve comparison (basic)
            contour_result = compute_contour_metrics(
                pred_freqs, labelled, meta.sample_rate, meta.nfft, meta.noverlap
            )
            result.update(contour_result)

            # Contour-level: 5-metric evaluation (SAM-whistle style)
            contour5 = compute_contour_level_metrics(
                pred_freqs, labelled, max_freq
            )
            result.update(contour5)

            # Band-aware metrics on the full 6-fan (handles multi-curve
            # labels correctly — each labelled band contributes one error)
            ba_full = compute_band_aware_metrics(pred_freqs, labelled, max_freq)
            for k, v in ba_full.items():
                result[f"{k}_full"] = v

            # -------------------------------------------------------------
            # Sub-band-aware variants: prune the 6-fan to "what the
            # annotator drew" before recomputing IoU / MAE / Acc@1kHz.
            # See the docstring on assign_drawn_subbands_tolerance /
            # fit_subband_per_segment for the reasoning.
            # -------------------------------------------------------------

            # (a) Tolerance pruning — keep sub-bands whose predictions land
            #     within 500 Hz of at least one labelled frame
            drawn_idx = assign_drawn_subbands_tolerance(
                pred_freqs, labelled, max_freq, tolerance_hz=500.0
            )
            if drawn_idx:
                pruned_freqs = [pred_freqs[i] for i in drawn_idx]
                # Rasterise the smoothed predictions directly: fractional
                # rows → nearest-integer row. This matches the figure
                # overlay exactly and stays consistent with the smoothing
                # applied upstream.
                pruned_mask = render_frequency_to_mask(
                    pruned_freqs, H, W, max_freq, line_thickness=1,
                )
                pruned_iou = compute_iou(pruned_mask, labelled, kernel_size)
                # Band-aware MAE / Acc stay in frequency space (centroid-based
                # predictions are fine here — they are the right domain for a
                # frequency-distance metric).
                ba_pruned = compute_band_aware_metrics(
                    pruned_freqs, labelled, max_freq
                )
                result["iou_pruned"] = pruned_iou["iou"]
                result["n_drawn_subbands"] = len(drawn_idx)
                result["drawn_sub_band_indices"] = ",".join(
                    str(i) for i in drawn_idx
                )
                for k, v in ba_pruned.items():
                    result[f"{k}_pruned"] = v
            else:
                result["iou_pruned"] = np.nan
                result["n_drawn_subbands"] = 0
                result["drawn_sub_band_indices"] = ""
                for k in ("ba_mae_hz", "ba_median_hz", "ba_n_samples",
                          "ba_acc_200hz", "ba_acc_500hz",
                          "ba_acc_1000hz", "ba_acc_2000hz"):
                    result[f"{k}_pruned"] = np.nan

            # (b) Per-segment fit — connected components of the label,
            #     one best-fit sub-band per component
            fitted_freqs, fit_info = fit_subband_per_segment(
                pred_freqs, labelled, max_freq
            )
            if fitted_freqs:
                # Rasterise the smoothed per-segment predictions at
                # fractional row resolution, then round to the nearest
                # integer row. Same function used by the figure, so
                # what the paper shows is exactly what is measured.
                fitted_mask = render_frequency_to_mask(
                    fitted_freqs, H, W, max_freq, line_thickness=1,
                )
                fitted_iou = compute_iou(fitted_mask, labelled, kernel_size)
                ba_fitted = compute_band_aware_metrics(
                    fitted_freqs, labelled, max_freq
                )
                result["iou_fitted"] = fitted_iou["iou"]
                result["n_segments"] = len(fit_info)
                result["fitted_sub_band_indices"] = ",".join(
                    str(f["sub_band_index"]) for f in fit_info
                )
                # Weighted mean residual: weight each segment's residual
                # by how many frames contributed to it
                total_frames = sum(f["n_frames"] for f in fit_info) or 1
                result["fit_residual_hz_mean"] = float(
                    sum(f["residual_hz"] * f["n_frames"] for f in fit_info)
                    / total_frames
                )
                for k, v in ba_fitted.items():
                    result[f"{k}_fitted"] = v
            else:
                result["iou_fitted"] = np.nan
                result["n_segments"] = 0
                result["fitted_sub_band_indices"] = ""
                result["fit_residual_hz_mean"] = np.nan
                for k in ("ba_mae_hz", "ba_median_hz", "ba_n_samples",
                          "ba_acc_200hz", "ba_acc_500hz",
                          "ba_acc_1000hz", "ba_acc_2000hz"):
                    result[f"{k}_fitted"] = np.nan

            rows.append(result)

            iou_str = f"{result['iou']:.3f}" if not np.isnan(result["iou"]) else "N/A"
            f1_str = f"{result['node_f1']:.3f}" if not np.isnan(result.get("node_f1", np.nan)) else "N/A"
            dev_str = f"{result['contour_freq_deviation_hz']:.0f}" if not np.isnan(result.get("contour_freq_deviation_hz", np.nan)) else "N/A"
            cov_str = f"{result['contour_coverage']:.0%}" if not np.isnan(result.get("contour_coverage", np.nan)) else "N/A"
            cr_str = f"{result['contour_recall']:.0%}" if not np.isnan(result.get("contour_recall", np.nan)) else "N/A"
            cp_str = f"{result['contour_precision']:.0%}" if not np.isnan(result.get("contour_precision", np.nan)) else "N/A"

            pm_str = f"{result.get('contour_precision_matched', 0):.0%}" if not np.isnan(result.get("contour_precision_matched", np.nan)) else "N/A"
            n_sb = result.get("contour_matched_sub_bands", 0)

            if result["both_empty"]:
                print(f"  Heterodynes/{n:>2d}: (both empty)")
            elif result["labelled_px"] == 0:
                print(f"  Heterodynes/{n:>2d}: (no labels, pred={result['predicted_px']}px)")
            else:
                iou_p = result.get("iou_pruned", np.nan)
                iou_f = result.get("iou_fitted", np.nan)
                mae_full = result.get("ba_mae_hz_full", np.nan)
                mae_prune = result.get("ba_mae_hz_pruned", np.nan)
                mae_fit = result.get("ba_mae_hz_fitted", np.nan)
                acc_fit = result.get("ba_acc_1000hz_fitted", np.nan)
                n_seg = result.get("n_segments", 0)
                p_str = f"{iou_p:.3f}" if not np.isnan(iou_p) else "N/A"
                f_str = f"{iou_f:.3f}" if not np.isnan(iou_f) else "N/A"
                mae_full_s = f"{mae_full:.0f}" if not np.isnan(mae_full) else "N/A"
                mae_prune_s = f"{mae_prune:.0f}" if not np.isnan(mae_prune) else "N/A"
                mae_fit_s = f"{mae_fit:.0f}" if not np.isnan(mae_fit) else "N/A"
                acc_fit_s = f"{acc_fit:.0%}" if not np.isnan(acc_fit) else "N/A"
                print(f"  Heterodynes/{n:>2d}:  "
                      f"IoU[full={iou_str} prune={p_str} fit={f_str}]  "
                      f"MAE[full={mae_full_s} prune={mae_prune_s} fit={mae_fit_s}]Hz  "
                      f"Acc@1k_fit={acc_fit_s}  segs={n_seg}")

    df = pd.DataFrame(rows)

    # Visualizations
    if output_dir is not None and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        generate_visualizations(
            spectrogram, df, predicted_masks, labelled_masks,
            clip_name, max_freq, meta.duration_sec, output_dir,
            predicted_freq_arrays=predicted_freq_arrays,
            f0_hfc=f0_hfc, f0_lfc=f0_lfc,
        )
        # Save CSV
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
    # Short name for titles
    short_name = clip_name.split("--")[-1] if "--" in clip_name else clip_name

    # Only show orders that have labelled data
    labelled_orders = sorted(
        n for n, m in labelled_masks.items() if m.sum() > 0
    )
    if not labelled_orders:
        return

    labelled_df = results_df[results_df["order"].isin(labelled_orders)].copy()

    H, W = spectrogram.shape
    col_axis = np.arange(W)  # pixel column indices — no time conversion

    # --- 1. Combined metrics bar chart (only labelled orders) ---
    if not labelled_df.empty:
        has_contour = "contour_mae_hz" in labelled_df.columns
        fig, axes = plt.subplots(1, 2 if has_contour else 1,
                                 figsize=(14 if has_contour else 8, 5))
        if not has_contour:
            axes = [axes]
        else:
            axes = list(axes)

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
            w = 0.25
            for i, (tol, color, label) in enumerate([
                (200, "#e74c3c", "<200 Hz"),
                (500, "#f39c12", "<500 Hz"),
                (1000, "#2ecc71", "<1000 Hz"),
            ]):
                col = f"contour_acc_{tol}hz"
                if col in labelled_df.columns:
                    vals = labelled_df[col].fillna(0).values
                    axes[1].bar(x + (i - 1) * w, vals, width=w, color=color,
                               edgecolor="black", linewidth=0.5, label=label)
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

    # --- 2. Per-order contour plots (zoomed, line-based like Jacob's script) ---
    if predicted_freq_arrays is None:
        return

    for n in labelled_orders:
        pred_freqs = predicted_freq_arrays.get(n)
        lab_mask = labelled_masks.get(n)
        if pred_freqs is None or lab_mask is None:
            continue

        freq_per_bin = max_freq / H

        # Get metrics for title
        row = results_df[results_df["order"] == n]
        mae = row["contour_mae_hz"].values[0] if len(row) else np.nan
        acc = row.get("contour_acc_1000hz", pd.Series([np.nan])).values[0] if len(row) else np.nan
        iou = row["iou"].values[0] if len(row) else np.nan

        # Full spectrogram, no zoom
        fig, ax = plt.subplots(figsize=(max(10, W / 40), max(6, H / 40)))

        ax.imshow(spectrogram, aspect="equal", cmap="gray",
                  origin="upper", extent=[0, W, H, 0],
                  interpolation="nearest")

        # Labelled mask overlay (full size)
        lab_overlay = np.zeros((H, W, 4))
        lab_overlay[lab_mask > 0] = [1, 0, 1, 1]  # solid magenta
        ax.imshow(lab_overlay, aspect="equal", origin="upper",
                  extent=[0, W, H, 0],
                  interpolation="nearest", zorder=3)

        # Predicted sub-band contours (in row coordinates)
        colors_pred = ['cyan', 'lime', 'yellow', 'orange', 'red', 'white']
        for i, fa in enumerate(pred_freqs):
            valid = ~np.isnan(fa)
            if valid.sum() == 0:
                continue
            c = colors_pred[i % len(colors_pred)]
            sign = "+" if i % 2 == 0 else "-"
            k = i // 2 + 1
            pred_rows = (max_freq - fa[valid]) / freq_per_bin
            ax.plot(col_axis[valid], pred_rows,
                    '-', color=c, linewidth=1.5, alpha=0.8,
                    label=f'Pred k={k} ({sign})', zorder=4)

        # HFC harmonic center line (dashed)
        if f0_hfc is not None:
            hfc_harmonic = (n + 1) * f0_hfc
            hfc_valid = ~np.isnan(hfc_harmonic)
            if hfc_valid.sum() > 0:
                hfc_rows = (max_freq - hfc_harmonic[hfc_valid]) / freq_per_bin
                ax.plot(col_axis[hfc_valid], hfc_rows,
                        '--', color='white', linewidth=1, alpha=0.5,
                        label=f'{n+1}x HFC', zorder=2)

        # Y-axis: show frequency labels
        yticks_rows = np.linspace(0, H, 8)
        ytick_labels = [f"{(max_freq - r * freq_per_bin)/1000:.0f}" for r in yticks_rows]
        ax.set_yticks(yticks_rows)
        ax.set_yticklabels(ytick_labels)

        # Legend
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor='magenta', label='Labelled'))
        labels.insert(0, 'Labelled')

        mae_str = f"MAE={mae:.0f}Hz" if not np.isnan(mae) else ""
        acc_str = f"Acc@1kHz={acc:.0%}" if not np.isnan(acc) else ""
        iou_str = f"IoU={iou:.3f}" if not np.isnan(iou) else ""
        metrics = "  ".join(s for s in [iou_str, mae_str, acc_str] if s)

        ax.set_ylim(H, 0)
        ax.set_xlim(0, W)
        ax.set_title(f"{short_name} - Heterodynes/{n}    {metrics}", fontsize=13)
        ax.set_xlabel("Column (pixel)", fontsize=12)
        ax.set_ylabel("Frequency (kHz)", fontsize=12)
        ax.legend(handles=handles, labels=labels,
                  loc="upper right", fontsize=9, framealpha=0.8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{clip_name}_contour_order_{n}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heterodyne validation: predict heterodynes from f0 "
                    "annotations and compare against labelled contours using IoU."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5", type=str, help="Single HDF5 file to validate")
    group.add_argument("--hdf5-dir", type=str, help="Directory of HDF5 files (batch)")

    parser.add_argument("--annotation-index", type=int, default=0,
                        help="Which annotation set to use (default: 0)")
    parser.add_argument("--kernel-size", type=int, default=5,
                        help="Dilation kernel size for IoU tolerance (default: 5, must be odd)")
    parser.add_argument("--max-k", type=int, default=1,
                        help="Max LFC harmonic mixing order (default: 1)")
    parser.add_argument("--output-dir", type=str,
                        default="visualizations/heterodyne_validation",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip visualization, just compute IoU")
    args = parser.parse_args()

    out_dir = None if args.no_plots else args.output_dir

    if args.hdf5:
        if not os.path.isfile(args.hdf5):
            print(f"ERROR: HDF5 file not found: {args.hdf5}")
            print(f"\nIf you only have XCF projects, convert first:")
            print(f"  python xcf_to_hdf5.py")
            sys.exit(1)
        df = validate_single_clip(
            args.hdf5, args.annotation_index, args.kernel_size, args.max_k, out_dir
        )
    else:
        hdf5_dir = Path(args.hdf5_dir)
        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"ERROR: No .hdf5 files found in {hdf5_dir}")
            print(f"\nConvert XCF projects first:")
            print(f"  python xcf_to_hdf5.py")
            sys.exit(1)

        all_dfs = []
        for f in hdf5_files:
            clip_df = validate_single_clip(
                str(f), args.annotation_index, args.kernel_size, args.max_k, out_dir
            )
            if not clip_df.empty:
                all_dfs.append(clip_df)

        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            if out_dir:
                agg_path = os.path.join(out_dir, "aggregate_results.csv")
                df.to_csv(agg_path, index=False)
                print(f"\nAggregate results: {agg_path}")

            # Print summary
            print(f"\n{'='*60}")
            print("AGGREGATE RESULTS")
            print(f"{'='*60}")
            valid = df[~df["both_empty"]]
            if not valid.empty:
                agg_cols = {"iou": ["mean", "std"]}
                if "contour_mae_hz" in valid.columns:
                    agg_cols["contour_mae_hz"] = "mean"
                    agg_cols["contour_acc_1000hz"] = "mean"
                    agg_cols["contour_corr"] = "mean"
                agg_dict = {
                    "node_f1": ("node_f1", "mean"),
                    "iou": ("iou", "mean"),
                    "count": ("iou", "count"),
                }
                if "contour_freq_deviation_hz" in valid.columns:
                    agg_dict.update({
                        "dev_hz": ("contour_freq_deviation_hz", "mean"),
                        "coverage": ("contour_coverage", "mean"),
                        "c_recall": ("contour_recall", "mean"),
                        "c_prec": ("contour_precision", "mean"),
                    })
                summary = valid.groupby("order").agg(**agg_dict)
                print(summary.to_string(float_format="%.3f"))
        else:
            print("No clips had valid results.")


if __name__ == "__main__":
    main()
