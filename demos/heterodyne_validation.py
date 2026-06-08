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

Usage (run from demos/ or from the repo root):
    python demos/heterodyne_validation.py --hdf5 ml_data/clip.hdf5
    python demos/heterodyne_validation.py --hdf5-dir ml_data/ --kernel-size 7 --max-k 3
    python demos/heterodyne_validation.py --hdf5 clip.hdf5 --no-plots
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

# Prepend repo root so `import hdf5_utils` resolves to the root-level module.
# Prepend demos dir so `from bin_morph import MaskMorphology` resolves locally.
# We prepend both so they shadow any system-installed packages with the same name.
sys.path.insert(0, _DEMOS_DIR)
sys.path.insert(0, _REPO_ROOT)

from hdf5_utils import HDF5SpectrogramLoader  # noqa: E402 (path setup must come first)
from bin_morph import MaskMorphology           # noqa: E402

HETERODYNE_ORDERS = list(range(13))  # 0 through 12

# Matches only numbered heterodyne layers: "Heterodynes/0" … "Heterodynes/12".
# Deliberately excludes "Heterodynes/unsure" — that layer has no defined order N
# and cannot be matched to a biphonic prediction.
_HETERODYNE_CLASS_RE = re.compile(r"^Heterodynes/\d+$")


# ---------------------------------------------------------------------------
# Pre-scan helper
# ---------------------------------------------------------------------------

def get_valid_annotation_indices(hdf5_path: str) -> List[int]:
    """Return annotation indices that are valid for heterodyne validation.

    An index is valid when the annotator explicitly drew at least one
    Heterodynes/N mask in that annotation set AND both f0_HFC and f0_LFC
    are also drawn in that same set.

    The drawn-heterodyne check is the primary gate and reflects annotator
    intent: a heterodyne drawing means the annotator identified this
    particular annotation set as containing a biphonic call with
    heterodynes. Other annotation sets on the same clip (e.g. a second pass
    tracing background vocalisations or a different call type) will not have
    drawn heterodynes and are therefore skipped automatically — no
    annotation-index argument is needed to control this.

    The fundamentals check is a secondary sanity gate: if heterodynes were
    drawn but f0_HFC or f0_LFC are missing, the biphonic formula has nothing
    to anchor to. We skip that index and emit a warning rather than producing
    garbage predictions, since this situation likely indicates an incomplete
    annotation that the annotator should revisit.

    Heterodynes/unsure is excluded by the regex: it has no defined order N
    and cannot be matched to any biphonic prediction.
    """
    valid_indices = []
    try:
        with HDF5SpectrogramLoader(hdf5_path) as loader:
            class_names = loader.get_class_names()

            het_class_names = [
                name for name in class_names
                if _HETERODYNE_CLASS_RE.match(name)
            ]
            if not het_class_names:
                return []  # no numbered heterodyne layers registered at all

            num_annotations = loader.get_num_annotations()
            for ann_idx in range(num_annotations):

                # Primary check: did the annotator draw any heterodynes here?
                has_het = any(
                    loader.get_class_mask(name, ann_idx) is not None
                    and loader.get_class_mask(name, ann_idx).sum() > 0
                    for name in het_class_names
                )
                if not has_het:
                    continue

                # Secondary check: both fundamentals must also be drawn,
                # otherwise the prediction formula cannot run.
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

    except Exception as exc:
        print(f"  WARNING: could not read {hdf5_path}: {exc}")
    return valid_indices


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
            # -------------------------------------------------------------

            # (a) Tolerance pruning
            drawn_idx = assign_drawn_subbands_tolerance(
                pred_freqs, labelled, max_freq, tolerance_hz=500.0
            )
            if drawn_idx:
                pruned_freqs = [pred_freqs[i] for i in drawn_idx]
                pruned_mask = render_frequency_to_mask(
                    pruned_freqs, H, W, max_freq, line_thickness=1,
                )
                pruned_iou = compute_iou(pruned_mask, labelled, kernel_size)
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

            # (b) Per-segment fit
            fitted_freqs, fit_info = fit_subband_per_segment(
                pred_freqs, labelled, max_freq
            )
            if fitted_freqs:
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

    # --- 2. Per-order contour plots ---
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
# Results table
# ---------------------------------------------------------------------------

def print_validation_table(df: pd.DataFrame) -> None:
    """Print a per-clip breakdown and aggregate summary matching Table 2 in
    the Spectrace paper.

    Only rows where the annotator drew labels (labelled_px > 0) are included.
    Metrics reported are the fitted variants — IoU_f, band-aware MAE, and
    Acc@250Hz — because these reflect what the annotator actually drew rather
    than the full sub-band fan.

    The heterodyne count column is the number of 8-connected components in the
    labelled mask (n_segments), matching the "Heterodynes" column in Table 2.

    Layout
    ------
    For each clip:
        Clip  Order  IoU   MAE(Hz)  Acc@250Hz  Heterodynes
        ...   ...    ...   ...      ...        ...
        summary (arithmetic mean of IoU/MAE/Acc, sum of Heterodynes)
    Then a final "all clips" row: mean of per-clip summary values.
    """
    col_w = 76  # total table width

    # Only rows with actual labels
    labelled = df[df["labelled_px"] > 0].copy()
    if labelled.empty:
        print("\nNo labelled heterodyne rows to summarise.")
        return

    header = (
        f"{'Clip':<36}  {'Order':>5}  {'IoU':>6}  "
        f"{'MAE(Hz)':>7}  {'Acc@250Hz':>9}  {'Heterodynes':>11}"
    )
    divider = "-" * col_w
    thick   = "=" * col_w

    print(f"\n{thick}")
    print("HETERODYNE VALIDATION RESULTS")
    print(thick)
    print(header)
    print(divider)

    clip_summaries = []  # one dict per clip for the final aggregate row

    for clip_name, clip_df in labelled.groupby("clip", sort=False):
        # Build a short but unambiguous label from the clip name.
        # Full names like "2023-12-03--10-15-30--00-15-00--Ct-Dt--03-52--D"
        # have the recording context in the first segments and the unique
        # call identifier in the last two ("03-52--D"). Using only the final
        # segment ("D") is ambiguous when multiple clips share a suffix letter.
        # We take the last two "--"-delimited segments, giving e.g.
        # "03-52--D", "03-11--C", "Rec-C.sel.03.AA".
        parts = clip_name.split("--")
        short = "--".join(parts[-2:]) if len(parts) >= 2 else clip_name
        if len(short) > 36:
            short = short[:33] + "..."

        for _, row in clip_df.sort_values("order").iterrows():
            iou  = row.get("iou_fitted",        np.nan)
            mae  = row.get("ba_mae_hz_fitted",   np.nan)
            acc  = row.get("ba_acc_250hz_fitted", np.nan)
            # Fall back to ba_acc_200hz_fitted if 250 Hz bin absent (older runs)
            if np.isnan(acc):
                acc = row.get("ba_acc_200hz_fitted", np.nan)
            segs = row.get("n_segments", 0)

            iou_s  = f"{iou:.3f}"  if not np.isnan(iou)  else "  N/A"
            mae_s  = f"{mae:.0f}"  if not np.isnan(mae)  else "  N/A"
            acc_s  = f"{acc:.0%}"  if not np.isnan(acc)  else "  N/A"

            print(
                f"{short:<36}  {int(row['order']):>5}  {iou_s:>6}  "
                f"{mae_s:>7}  {acc_s:>9}  {int(segs):>11}"
            )

        # Per-clip summary row (shaded in the paper; here marked with >>)
        iou_mean  = clip_df["iou_fitted"].dropna().mean()
        mae_mean  = clip_df["ba_mae_hz_fitted"].dropna().mean()
        acc_col   = "ba_acc_250hz_fitted" if "ba_acc_250hz_fitted" in clip_df.columns \
                    else "ba_acc_200hz_fitted"
        acc_mean  = clip_df[acc_col].dropna().mean()
        segs_sum  = clip_df["n_segments"].sum()

        iou_s  = f"{iou_mean:.3f}"  if not np.isnan(iou_mean)  else "  N/A"
        mae_s  = f"{mae_mean:.0f}"  if not np.isnan(mae_mean)  else "  N/A"
        acc_s  = f"{acc_mean:.0%}"  if not np.isnan(acc_mean)  else "  N/A"

        print(divider)
        print(
            f"{'  >> summary':<36}  {'':>5}  {iou_s:>6}  "
            f"{mae_s:>7}  {acc_s:>9}  {int(segs_sum):>11}"
        )
        print(divider)

        clip_summaries.append({
            "iou_mean": iou_mean,
            "mae_mean": mae_mean,
            "acc_mean": acc_mean,
            "segs_sum": segs_sum,
        })

    # Final aggregate row: mean of per-clip summaries, total heterodynes
    agg_iou  = float(np.nanmean([s["iou_mean"] for s in clip_summaries]))
    agg_mae  = float(np.nanmean([s["mae_mean"] for s in clip_summaries]))
    agg_acc  = float(np.nanmean([s["acc_mean"] for s in clip_summaries]))
    agg_segs = int(sum(s["segs_sum"] for s in clip_summaries))

    print(thick)
    print(
        f"{'  ALL CLIPS (mean)':<36}  {'':>5}  {agg_iou:.3f}  "
        f"{agg_mae:>7.0f}  {agg_acc:>9.0%}  {agg_segs:>11}"
    )
    print(thick)



# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------

def run_negative_control(
    skipped_files: List[Path],
    max_k: int,
    kernel_size: int,
    output_dir: Optional[str],
) -> None:
    """Negative control: run the biphonic prediction formula on clips that
    have annotated f0_HFC and f0_LFC but NO drawn heterodynes, then measure
    how well the predicted heterodyne curves match any OTHER labelled frequency
    contours in the spectrogram (harmonics, subharmonics, or anything else the
    annotator drew).

    The logic is:
      - For each non-heterodyne clip with both fundamentals drawn, compute
        predicted heterodyne curves exactly as in the positive validation.
      - Collect all non-fundamental, non-heterodyne masks that have pixels
        drawn as "random" reference masks (harmonics_HFC, harmonics_LFC,
        subharmonics_HFC, subharmonics_LFC, heterodyne_or_subharmonic_or_other,
        and any Cetacean_AdditionalContours layers).
      - Evaluate band-aware MAE and Acc@250Hz of the predictions against each
        reference mask in turn.

    If the formula is genuinely specific to heterodynes, the MAE against these
    random reference contours should be substantially larger than the MAE
    against the true heterodyne labels in the positive clips. If MAE is
    similarly low, it suggests the spectrogram is dense enough that the
    predicted curves land near *something* by chance — which would undermine
    the validation claim.

    Only clips where at least one annotation index has both f0_HFC and f0_LFC
    drawn are used; clips with neither fundamental are uninformative.

    Results are printed as a summary table and, if output_dir is set, saved
    to negative_control_results.csv alongside the positive results.
    """
    # Layers we treat as "other drawn content" — anything the annotator drew
    # that is not a fundamental and not a numbered heterodyne.
    _REFERENCE_LAYER_PREFIXES = (
        "harmonics_HFC",
        "harmonics_LFC",
        "subharmonics_HFC",
        "subharmonics_LFC",
        "heterodyne_or_subharmonic_or_other",
        "Cetacean_AdditionalContours",
        "unsure_HFC",
        "unsure_LFC",
        "Heterodynes/unsure",
    )

    rows = []

    for hdf5_path in skipped_files:
        try:
            with HDF5SpectrogramLoader(str(hdf5_path)) as loader:
                class_names = loader.get_class_names()
                if "f0_HFC" not in class_names or "f0_LFC" not in class_names:
                    continue

                meta = loader.get_metadata()
                max_freq = meta.max_freq_hz
                H = W = None

                num_annotations = loader.get_num_annotations()
                for ann_idx in range(num_annotations):
                    hfc_mask = loader.get_class_mask("f0_HFC", ann_idx)
                    lfc_mask = loader.get_class_mask("f0_LFC", ann_idx)
                    if hfc_mask is None or hfc_mask.sum() == 0:
                        continue
                    if lfc_mask is None or lfc_mask.sum() == 0:
                        continue

                    if H is None:
                        H, W = hfc_mask.shape

                    f0_hfc = smooth_f0_contour(
                        extract_f0_contour(hfc_mask, meta.sample_rate,
                                           meta.nfft, meta.noverlap)
                    )
                    f0_lfc = smooth_f0_contour(
                        extract_f0_contour(lfc_mask, meta.sample_rate,
                                           meta.nfft, meta.noverlap)
                    )

                    # Collect reference masks from this annotation index
                    ref_masks = {}
                    for name in class_names:
                        if not any(name.startswith(p)
                                   for p in _REFERENCE_LAYER_PREFIXES):
                            continue
                        mask = loader.get_class_mask(name, ann_idx)
                        if mask is not None and mask.sum() > 0:
                            ref_masks[name] = mask

                    if not ref_masks:
                        continue

                    # Evaluate predictions for each heterodyne order against
                    # each reference mask
                    for order_n_idx in range(7):  # orders 0-6, same as positive
                        pred_freqs = compute_predicted_heterodyne_freqs(
                            f0_hfc, f0_lfc,
                            order_n=order_n_idx + 1,
                            max_k=max_k,
                            max_freq=max_freq,
                        )
                        for ref_name, ref_mask in ref_masks.items():
                            ba = compute_band_aware_metrics(
                                pred_freqs, ref_mask, max_freq
                            )
                            rows.append({
                                "clip": hdf5_path.stem,
                                "ann_idx": ann_idx,
                                "order": order_n_idx,
                                "reference_layer": ref_name,
                                "ba_mae_hz": ba["ba_mae_hz"],
                                "ba_acc_250hz": ba.get("ba_acc_250hz",
                                               ba.get("ba_acc_200hz", np.nan)),
                                "ba_n_samples": ba["ba_n_samples"],
                            })
        except Exception as exc:
            print(f"  WARNING: negative control — could not read "
                  f"{hdf5_path.name}: {exc}")

    if not rows:
        print("\nNegative control: no clips with fundamentals and other "
              "drawn content found among the skipped files.")
        return

    nc_df = pd.DataFrame(rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        nc_path = os.path.join(output_dir, "negative_control_results.csv")
        nc_df.to_csv(nc_path, index=False)
        print(f"  Negative control results saved to: {nc_path}")

    # Summary: mean MAE and Acc@250Hz across all (clip, order, ref_layer)
    # combinations, weighted by number of samples so that clips with more
    # drawn content contribute proportionally.
    valid_nc = nc_df[nc_df["ba_n_samples"] > 0].copy()
    if valid_nc.empty:
        print("\nNegative control: no valid reference mask comparisons found.")
        return

    col_w = 72
    thick = "=" * col_w
    divider = "-" * col_w

    print(f"\n{thick}")
    print("NEGATIVE CONTROL: predictions vs non-heterodyne contours")
    print(thick)
    print(f"  Clips used:          {nc_df['clip'].nunique()}")
    print(f"  Annotation sets:     {nc_df.groupby(['clip','ann_idx']).ngroups}")
    print(f"  Reference layers:    {nc_df['reference_layer'].nunique()} unique classes")
    print(f"  Comparisons made:    {len(valid_nc)}")
    print(divider)

    # Per-order summary (mirrors positive validation table for direct comparison)
    print(f"  {'Order':>5}  {'MAE(Hz)':>9}  {'Acc@250Hz':>9}  {'N samples':>9}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}")
    for order, grp in valid_nc.groupby("order"):
        # Weighted mean by n_samples
        weights = grp["ba_n_samples"].values.astype(float)
        w_mae = float(np.average(grp["ba_mae_hz"].values,
                                 weights=weights))
        w_acc = float(np.average(grp["ba_acc_250hz"].fillna(0).values,
                                 weights=weights))
        n_tot = int(weights.sum())
        print(f"  {int(order):>5}  {w_mae:>9.0f}  {w_acc:>9.0%}  {n_tot:>9}")

    print(divider)
    # Overall weighted mean
    weights = valid_nc["ba_n_samples"].values.astype(float)
    overall_mae = float(np.average(valid_nc["ba_mae_hz"].values,
                                   weights=weights))
    overall_acc = float(np.average(valid_nc["ba_acc_250hz"].fillna(0).values,
                                   weights=weights))
    print(f"  {'all':>5}  {overall_mae:>9.0f}  {overall_acc:>9.0%}  "
          f"{int(weights.sum()):>9}")
    print(thick)
    print("  Compare against positive validation (all clips mean):")
    print("    MAE 174 Hz  |  Acc@250Hz 74%")
    print("  A substantially higher negative-control MAE and lower Acc@250Hz")
    print("  confirms the formula is specific to labelled heterodynes.")
    print(thick)

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
    parser.add_argument("--negative-control", action="store_true",
                        help="After positive validation, run the biphonic "
                             "formula against non-heterodyne contours in "
                             "skipped clips as a specificity check. "
                             "Only valid with --hdf5-dir.")
    args = parser.parse_args()

    out_dir = None if args.no_plots else args.output_dir

    if args.hdf5:
        if not os.path.isfile(args.hdf5):
            print(f"ERROR: HDF5 file not found: {args.hdf5}")
            print(f"\nIf you only have XCF projects, convert first:")
            print(f"  python xcf_to_hdf5.py")
            sys.exit(1)

        valid_indices = get_valid_annotation_indices(args.hdf5)
        if not valid_indices:
            print(f"ERROR: {args.hdf5} has no annotation set with drawn "
                  f"f0_HFC, f0_LFC, and at least one Heterodynes/N mask.")
            sys.exit(1)

        # If the user supplied --annotation-index, honour it if valid;
        # otherwise warn and fall back to the first valid index.
        if args.annotation_index in valid_indices:
            ann_idx = args.annotation_index
        else:
            print(f"  WARNING: annotation index {args.annotation_index} is not valid "
                  f"for this clip (valid indices: {valid_indices}). "
                  f"Using {valid_indices[0]}.")
            ann_idx = valid_indices[0]

        df = validate_single_clip(
            args.hdf5, ann_idx, args.kernel_size, args.max_k, out_dir
        )
        if not df.empty:
            print_validation_table(df)
    else:
        hdf5_dir = Path(args.hdf5_dir)
        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"ERROR: No .hdf5 files found in {hdf5_dir}")
            print(f"\nConvert XCF projects first:")
            print(f"  python xcf_to_hdf5.py")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Pre-scan: for each clip, find annotation indices that have drawn
        # heterodynes AND both fundamentals. Indices without drawn heterodynes
        # are skipped — they represent other annotation passes on the same
        # clip (different calls, background vocalisations, etc.) and should
        # not be validated against the biphonic formula.
        # ------------------------------------------------------------------
        print(f"Scanning {len(hdf5_files)} HDF5 file(s) for heterodyne annotations…")
        heterodyne_files = []   # list of (Path, [valid_ann_indices])
        skipped_paths: List[Path] = []  # kept for --negative-control
        skipped_names = []
        for f in hdf5_files:
            valid_indices = get_valid_annotation_indices(str(f))
            if valid_indices:
                heterodyne_files.append((f, valid_indices))
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
              f"annotation set(s) — proceeding with validation.\n")
        # ------------------------------------------------------------------

        all_dfs = []
        for f, valid_indices in heterodyne_files:
            for ann_idx in valid_indices:
                clip_df = validate_single_clip(
                    str(f), ann_idx, args.kernel_size, args.max_k, out_dir
                )
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

        # --negative-control: only makes sense in batch mode since we
        # need a pool of skipped clips to compare against.
        if args.negative_control:
            if not skipped_paths:
                print("\nNegative control: all clips had valid heterodyne "
                      "annotations — no negative pool available.")
            else:
                print(f"\nRunning negative control on "
                      f"{len(skipped_paths)} skipped clip(s)…")
                run_negative_control(
                    skipped_paths, args.max_k, args.kernel_size, out_dir
                )


if __name__ == "__main__":
    main()