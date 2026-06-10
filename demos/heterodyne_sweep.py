#!/usr/bin/env python3
"""
Hyperparameter sweep + literature-comparison harness for heterodyne validation.

This is an orchestration layer on top of heterodyne_validation.py. It does NOT
re-implement any metric: it imports the validated functions and sweeps the
parameters that actually change predictions/scoring, then renders Pareto/robustness
figures with the *reported* operating point marked with a star.

Two axes are treated differently, on purpose:

  - ``max_k`` and ``kernel_size`` change what is predicted / how masks are
    dilated, so they require a fresh evaluation per grid cell.
  - frequency *tolerance* is free: a single run yields a per-band error array,
    and accuracy at any tolerance is just that array thresholded. So tolerance
    is swept analytically from one pass, never by re-running the pipeline.

Literature reference (Roch et al. 2011, JASA 130(4):2212-2223,
"Automated extraction of odontocete whistle contours"):
  * Per-bin bandwidth Df = 125 Hz (8 ms Hamming window).
  * Snap-to-ridge energy search: +/- 500 Hz (4 bins) around the spline GT.
  * Match/reject threshold: mean per-point deviation <= 350 Hz (~3 bins).
  * Reported quality on the 5-species corpus:
        graph search:   deviation  70 Hz, coverage 86.0%, fragmentation 1.2
        particle filter:deviation 161 Hz, coverage 79.7%, fragmentation 1.2
  These are the values we plot as the reference operating point / star.

IMPORTANT (anti-p-hacking): the starred point is fixed a priori (the tolerance
and max_k we report in the paper, specified via --reported-k and
--reported-tolerance), NOT chosen as the sweep argmax. The sweep exists to show
the conclusion is *robust around* that point, and the negative control is run at
every cell so specificity is demonstrated across the whole grid rather than at
one flattering setting.

Usage:
    python heterodyne_sweep.py --hdf5-dir ml_data/
    python heterodyne_sweep.py --hdf5-dir ml_data/ --max-k-grid 1 2 3 4 \
        --kernel-grid 3 5 7 --reported-k 5 --reported-tolerance 250 \
        --output-dir visualizations/sweep
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import heterodyne_validation as hv  # noqa: E402

# ---------------------------------------------------------------------------
# Literature reference values (Roch et al. 2011). Edit here only.
# ---------------------------------------------------------------------------
ROCH_BIN_HZ = 125.0
ROCH_SNAP_SEARCH_HZ = 500.0
ROCH_MATCH_THRESHOLD_HZ = 350.0
ROCH_REPORTED = {
    "graph_search":   {"deviation_hz": 70.0,  "coverage_pct": 86.0,
                       "fragmentation": 1.2, "recall_pct": 80.0, "precision_pct": 76.9},
    "particle_filter": {"deviation_hz": 161.0, "coverage_pct": 79.7,
                        "fragmentation": 1.2, "recall_pct": 71.5, "precision_pct": 60.8},
}

# Default reported operating point — overridden by --reported-k /
# --reported-tolerance at the CLI. These are used only as fallbacks when
# those flags are not supplied; change them here if your paper always uses
# the same values and you want to skip the flags.
DEFAULT_REPORTED_K: int = int(hv.PRIMARY_TOLERANCE_HZ // 250)  # conservative default
DEFAULT_REPORTED_TOLERANCE_HZ: float = float(hv.PRIMARY_TOLERANCE_HZ)  # 250 Hz


# ---------------------------------------------------------------------------
# Roch-parity metrics that the base script does not currently compute
# ---------------------------------------------------------------------------

def roch_style_fragmentation_and_match(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
    match_threshold_hz: float = ROCH_MATCH_THRESHOLD_HZ,
) -> dict:
    """Roch et al. 2011 detections-per-tonal fragmentation + match recall/prec.

    Roch's fragmentation is the average number of *distinct detected segments*
    that overlap a ground-truth tonal (their reported value is 1.2 — most
    tonals recovered as a single contour). Our base ``contour_fragmentation``
    is a different quantity (a 0-1 discontinuity score), so we add this so the
    comparison to the literature is like-for-like.

    Method
    ------
    1. Split the labelled mask into 8-connected components = ground-truth
       "tonals".
    2. Render the predicted fan to a mask and split *that* into components =
       "detections".
    3. For each GT tonal, count how many detection components overlap it within
       ``match_threshold_hz`` (mean per-overlapping-column deviation), exactly
       mirroring Roch's mean-deviation accept rule.
    4. Fragmentation = mean detections-per-matched-tonal (matched tonals only,
       to mirror "detections per ground truth tonal" on retrieved calls).
       Recall = fraction of GT tonals with >=1 matched detection.
       Precision = fraction of detection components that matched some GT tonal.
    """
    H, W = labelled_mask.shape
    freq_per_bin = max_freq / H

    gt_lab, n_gt = ndimage.label(labelled_mask > 0, structure=np.ones((3, 3), int))
    pred_mask = hv.render_frequency_to_mask(pred_freqs, H, W, max_freq)
    pred_lab, n_pred = ndimage.label(pred_mask > 0, structure=np.ones((3, 3), int))

    if n_gt == 0:
        return {"roch_fragmentation": np.nan, "roch_recall": np.nan,
                "roch_precision": np.nan, "roch_n_gt": 0, "roch_n_pred": n_pred}

    def comp_contour(lab_arr, cid):
        m = lab_arr == cid
        c = np.full(W, np.nan)
        for t in range(W):
            col = m[:, t]
            if col.any():
                c[t] = max_freq - np.where(col)[0].mean() * freq_per_bin
        return c

    gt_contours = {g: comp_contour(gt_lab, g) for g in range(1, n_gt + 1)}
    pred_contours = {p: comp_contour(pred_lab, p) for p in range(1, n_pred + 1)}

    matched_pred = set()
    detections_per_tonal = []
    matched_tonals = 0
    for g, gc in gt_contours.items():
        hits = 0
        for p, pc in pred_contours.items():
            both = ~np.isnan(gc) & ~np.isnan(pc)
            if both.sum() == 0:
                continue
            if float(np.abs(gc[both] - pc[both]).mean()) <= match_threshold_hz:
                hits += 1
                matched_pred.add(p)
        if hits > 0:
            matched_tonals += 1
            detections_per_tonal.append(hits)

    frag = float(np.mean(detections_per_tonal)) if detections_per_tonal else np.nan
    recall = matched_tonals / n_gt if n_gt else np.nan
    precision = len(matched_pred) / n_pred if n_pred else np.nan
    return {"roch_fragmentation": frag, "roch_recall": recall,
            "roch_precision": precision, "roch_n_gt": n_gt, "roch_n_pred": n_pred}


def per_band_error_array(
    pred_freqs: List[np.ndarray],
    labelled_mask: np.ndarray,
    max_freq: float,
) -> np.ndarray:
    """The raw per-(column,band) min-error array underlying band-aware accuracy.

    Returned once so tolerance can be swept analytically: Acc@tol = mean(err<=tol)
    and deviation = mean(err). This is the same matching logic as
    hv.compute_band_aware_metrics, exposed as the array instead of pre-thresholded.
    """
    lab_bands = hv._labelled_bands_per_column(labelled_mask, max_freq)
    errors = []
    for t, bands in lab_bands.items():
        pred_at_t = [fa[t] for fa in pred_freqs if not np.isnan(fa[t])]
        if not pred_at_t:
            continue
        for lab_freq in bands:
            errors.append(min(abs(lab_freq - p) for p in pred_at_t))
    return np.array(errors) if errors else np.array([])


# ---------------------------------------------------------------------------
# Sweep core
# ---------------------------------------------------------------------------

def _iter_clip_orders(hdf5_files_with_indices, max_k, kernel_size):
    """Yield (clip, order, pred_freqs, labelled, max_freq, H, W) for labelled orders."""
    for f, valid_indices in hdf5_files_with_indices:
        for ann_idx in valid_indices:
            try:
                with hv.HDF5SpectrogramLoader(str(f)) as loader:
                    meta = loader.get_metadata()
                    class_names = loader.get_class_names()
                    if "f0_HFC" not in class_names or "f0_LFC" not in class_names:
                        continue
                    max_freq = meta.max_freq_hz
                    hfc = hv.smooth_f0_contour(hv.extract_f0_contour(
                        loader.get_class_mask("f0_HFC", ann_idx), max_freq))
                    lfc = hv.smooth_f0_contour(hv.extract_f0_contour(
                        loader.get_class_mask("f0_LFC", ann_idx), max_freq))
                    H, W = loader.get_class_mask("f0_HFC", ann_idx).shape
                    for n in hv.HETERODYNE_ORDERS:
                        name = f"Heterodynes/{n}"
                        if name not in class_names:
                            continue
                        lab = loader.get_class_mask(name, ann_idx)
                        if lab is None or lab.sum() == 0:
                            continue
                        pred = hv.compute_predicted_heterodyne_freqs(
                            hfc, lfc, hfc_multiplier=n + 1, max_k=max_k, max_freq=max_freq)
                        yield (Path(f).stem, n, pred, lab, max_freq, H, W)
            except Exception as exc:
                print(f"  WARNING: sweep could not read {Path(f).name}: {exc}")


def run_sweep(
    hdf5_dir: str,
    max_k_grid: List[int],
    kernel_grid: List[int],
    tolerance_grid_hz: List[float],
    output_dir: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep (max_k x kernel) for fitted IoU/Roch metrics, and tolerance analytically.

    Returns (grid_df, tol_df):
      grid_df: one row per (max_k, kernel) — aggregate fitted IoU, Roch-style
               deviation/coverage/fragmentation/recall/precision, and the
               positive-minus-negative MAE gap (robustness signal).
      tol_df:  one row per (max_k, tolerance) — accuracy swept analytically from
               the cached per-band error arrays at kernel-independent cost.
    """
    hdf5_files = sorted(Path(hdf5_dir).glob("*.hdf5"))
    files_idx = []
    skipped = []
    for f in hdf5_files:
        idx = hv.get_valid_annotation_indices(str(f))
        if idx:
            files_idx.append((f, idx))
        else:
            skipped.append(f)
    print(f"Sweep over {len(files_idx)} clip(s) with heterodynes; "
          f"{len(skipped)} skipped (negative pool).")

    grid_rows = []
    tol_rows = []

    for max_k in max_k_grid:
        # Cache per-(clip,order) error arrays once per max_k (kernel-independent),
        # so the tolerance sweep is free and the kernel loop reuses predictions.
        cached = list(_iter_clip_orders(files_idx, max_k, kernel_size=kernel_grid[0]))

        # --- tolerance sweep (analytic, kernel-independent) ---
        err_by_clip = {}
        for clip, n, pred, lab, mf, H, W in cached:
            err = per_band_error_array(pred, lab, mf)
            if err.size:
                err_by_clip.setdefault(clip, []).append(err)
        for tol in tolerance_grid_hz:
            clip_devs, clip_accs = [], []
            for clip, arrs in err_by_clip.items():
                allerr = np.concatenate(arrs)
                clip_devs.append(allerr.mean())
                clip_accs.append((allerr <= tol).mean())
            if clip_devs:
                tol_rows.append({
                    "max_k": max_k, "tolerance_hz": tol,
                    "deviation_hz": float(np.mean(clip_devs)),
                    "accuracy": float(np.mean(clip_accs)),
                })

        # --- (max_k x kernel) grid for mask-based + Roch metrics ---
        for kernel in kernel_grid:
            pos_ious, pos_maes = [], []
            roch_devs, roch_covs, roch_frags, roch_recs, roch_precs = [], [], [], [], []
            for clip, n, pred, lab, mf, H, W in cached:
                fitted, _ = hv.fit_subband_per_segment(pred, lab, mf)
                if fitted:
                    fmask = hv.render_frequency_to_mask(fitted, H, W, mf)
                    pos_ious.append(hv.compute_iou(fmask, lab, kernel)["iou"])
                    ba = hv.compute_band_aware_metrics(fitted, lab, mf)
                    pos_maes.append(ba["ba_mae_hz"])
                cl = hv.compute_contour_level_metrics(pred, lab, mf)
                roch_devs.append(cl["contour_freq_deviation_hz"])
                roch_covs.append(cl["contour_coverage"])
                rm = roch_style_fragmentation_and_match(pred, lab, mf)
                roch_frags.append(rm["roch_fragmentation"])
                roch_recs.append(rm["roch_recall"])
                roch_precs.append(rm["roch_precision"])

            neg_mae = _negative_control_mae(skipped, max_k) if skipped else np.nan
            pos_mae = float(np.nanmean(pos_maes)) if pos_maes else np.nan
            grid_rows.append({
                "max_k": max_k, "kernel_size": kernel,
                "iou_fitted": float(np.nanmean(pos_ious)) if pos_ious else np.nan,
                "deviation_hz": float(np.nanmean(roch_devs)),
                "coverage_pct": float(np.nanmean(roch_covs)) * 100,
                "fragmentation": float(np.nanmean(roch_frags)),
                "roch_recall_pct": float(np.nanmean(roch_recs)) * 100,
                "roch_precision_pct": float(np.nanmean(roch_precs)) * 100,
                "pos_mae_hz": pos_mae,
                "neg_mae_hz": neg_mae,
                "mae_gap_hz": (neg_mae - pos_mae) if not (np.isnan(neg_mae) or np.isnan(pos_mae)) else np.nan,
            })

    grid_df = pd.DataFrame(grid_rows)
    tol_df = pd.DataFrame(tol_rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        grid_df.to_csv(os.path.join(output_dir, "sweep_grid.csv"), index=False)
        tol_df.to_csv(os.path.join(output_dir, "sweep_tolerance.csv"), index=False)
        print(f"  Sweep CSVs written to {output_dir}")

    return grid_df, tol_df


def _negative_control_mae(skipped_files, max_k) -> float:
    """Weighted mean band-aware MAE of predictions vs non-heterodyne contours."""
    _REF = ("harmonics_HFC", "harmonics_LFC", "subharmonics_HFC", "subharmonics_LFC",
            "heterodyne_or_subharmonic_or_other", "Cetacean_AdditionalContours",
            "unsure_HFC", "unsure_LFC", "Heterodynes/unsure")
    maes, weights = [], []
    for f in skipped_files:
        try:
            with hv.HDF5SpectrogramLoader(str(f)) as loader:
                cn = loader.get_class_names()
                if "f0_HFC" not in cn or "f0_LFC" not in cn:
                    continue
                meta = loader.get_metadata()
                mf = meta.max_freq_hz
                for ann_idx in range(loader.get_num_annotations()):
                    hm = loader.get_class_mask("f0_HFC", ann_idx)
                    lm = loader.get_class_mask("f0_LFC", ann_idx)
                    if hm is None or hm.sum() == 0 or lm is None or lm.sum() == 0:
                        continue
                    hfc = hv.smooth_f0_contour(hv.extract_f0_contour(hm, mf))
                    lfc = hv.smooth_f0_contour(hv.extract_f0_contour(lm, mf))
                    refs = {nm: loader.get_class_mask(nm, ann_idx) for nm in cn
                            if any(nm.startswith(p) for p in _REF)}
                    refs = {k: v for k, v in refs.items() if v is not None and v.sum() > 0}
                    for order in range(7):
                        pred = hv.compute_predicted_heterodyne_freqs(
                            hfc, lfc, hfc_multiplier=order + 1, max_k=max_k, max_freq=mf)
                        for v in refs.values():
                            ba = hv.compute_band_aware_metrics(pred, v, mf)
                            if ba["ba_n_samples"] > 0:
                                maes.append(ba["ba_mae_hz"])
                                weights.append(ba["ba_n_samples"])
        except Exception:
            continue
    if not maes:
        return np.nan
    return float(np.average(maes, weights=weights))


# ---------------------------------------------------------------------------
# Figures — star marks the REPORTED operating point, set via CLI flags
# ---------------------------------------------------------------------------

def _resolve_star(
    grid_df: pd.DataFrame,
    tol_df: pd.DataFrame,
    reported_k: Optional[int],
    reported_tolerance_hz: Optional[float],
) -> Tuple[int, float, int]:
    """Resolve the starred operating point, falling back to sane defaults.

    Returns (star_k, star_tol, star_kernel).
    """
    available_ks = sorted(grid_df["max_k"].unique())
    available_kernels = sorted(grid_df["kernel_size"].unique())

    star_k = reported_k if reported_k is not None else DEFAULT_REPORTED_K
    if star_k not in available_ks:
        fallback = available_ks[0]
        print(f"  WARNING: --reported-k {star_k} not in sweep grid {available_ks}; "
              f"falling back to {fallback}.")
        star_k = fallback

    star_tol = reported_tolerance_hz if reported_tolerance_hz is not None else DEFAULT_REPORTED_TOLERANCE_HZ

    star_kernel = 5 if 5 in available_kernels else available_kernels[0]

    return star_k, star_tol, star_kernel


def make_figures(
    grid_df: pd.DataFrame,
    tol_df: pd.DataFrame,
    output_dir: str,
    reported_k: Optional[int] = None,
    reported_tolerance_hz: Optional[float] = None,
):
    """Render the three sweep figures, starring the caller-specified operating point.

    Parameters
    ----------
    reported_k : int, optional
        The max_k value to mark with the star. Falls back to DEFAULT_REPORTED_K
        if not supplied (which itself falls back to the smallest k in the grid).
    reported_tolerance_hz : float, optional
        The frequency tolerance (Hz) to mark with the star on the tolerance
        curve. Falls back to DEFAULT_REPORTED_TOLERANCE_HZ if not supplied.
    """
    os.makedirs(output_dir, exist_ok=True)
    star_k, star_tol, star_kernel = _resolve_star(
        grid_df, tol_df, reported_k, reported_tolerance_hz)

    # --- Fig 1: robustness heatmap of positive-minus-negative MAE gap ---
    pivot = grid_df.pivot(index="max_k", columns="kernel_size", values="mae_gap_hz")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("kernel_size")
    ax.set_ylabel("max_k")
    ax.set_title("Robustness: negative-minus-positive MAE gap (Hz)\n"
                 "(large & positive everywhere = formula specific to heterodynes)")
    for i, mk in enumerate(pivot.index):
        for j, ks in enumerate(pivot.columns):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white", fontsize=9)
    if star_k in list(pivot.index) and star_kernel in list(pivot.columns):
        si = list(pivot.index).index(star_k)
        sj = list(pivot.columns).index(star_kernel)
        ax.scatter([sj], [si], marker="*", s=420, c="red", edgecolor="black",
                   linewidth=1.2, zorder=5, label="Reported operating point")
        ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="MAE gap (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_robustness_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- Fig 2: tolerance-as-CDF (accuracy vs tolerance), star @ reported point ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for max_k, grp in tol_df.groupby("max_k"):
        grp = grp.sort_values("tolerance_hz")
        ax.plot(grp["tolerance_hz"], grp["accuracy"], marker="o", label=f"max_k={max_k}")
    ax.axvline(star_tol, color="red", linestyle="--", alpha=0.7)
    # Star at reported tolerance on the reported-k curve
    star_curve = tol_df[tol_df["max_k"] == star_k].sort_values("tolerance_hz")
    if not star_curve.empty:
        acc_at_star = float(np.interp(
            star_tol, star_curve["tolerance_hz"], star_curve["accuracy"]))
        ax.scatter([star_tol], [acc_at_star], marker="*", s=420, c="red",
                   edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"Reported (k={star_k}, {star_tol:.0f} Hz)")
    ax.set_xlabel("Frequency tolerance (Hz)")
    ax.set_ylabel("Band-aware accuracy")
    ax.set_title("Accuracy vs tolerance (full error distribution)\n"
                 "reported point starred")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_tolerance_curve.png"), dpi=150)
    plt.close(fig)

    # --- Fig 3: Pareto — recall vs precision as max_k varies, Roch points overlaid ---
    fig, ax = plt.subplots(figsize=(7, 6))
    g = grid_df[grid_df["kernel_size"] == star_kernel].sort_values("max_k")
    ax.plot(g["roch_recall_pct"], g["roch_precision_pct"], "-o", color="C0",
            label="Spectrace (sweep over max_k)")
    for _, r in g.iterrows():
        ax.annotate(f"k={int(r['max_k'])}", (r["roch_recall_pct"], r["roch_precision_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    sr = g[g["max_k"] == star_k]
    if not sr.empty:
        ax.scatter(sr["roch_recall_pct"], sr["roch_precision_pct"], marker="*",
                   s=420, c="red", edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"Reported (max_k={star_k})")
    for name, vals in ROCH_REPORTED.items():
        ax.scatter([vals["recall_pct"]], [vals["precision_pct"]], marker="D", s=80,
                   edgecolor="black", label=f"Roch 2011 {name.replace('_', ' ')}")
    ax.set_xlabel("Recall (%)  — Roch-style match @ 350 Hz")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Recall–precision Pareto vs Roch et al. 2011")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_pareto_recall_precision.png"), dpi=150)
    plt.close(fig)

    print(f"  3 figures written to {output_dir}")


def print_literature_comparison(
    grid_df: pd.DataFrame,
    reported_k: Optional[int] = None,
    reported_tolerance_hz: Optional[float] = None,
):
    """Print our reported-point metrics beside Roch et al. 2011."""
    star_k, star_tol, star_kernel = _resolve_star(
        grid_df, pd.DataFrame(), reported_k, reported_tolerance_hz)
    ours = grid_df[(grid_df["max_k"] == star_k) & (grid_df["kernel_size"] == star_kernel)]
    w = 78
    print("\n" + "=" * w)
    print("LITERATURE COMPARISON — Roch et al. 2011 (JASA 130:2212) vs Spectrace")
    print("=" * w)
    print(f"  Roch tolerances: bin={ROCH_BIN_HZ:.0f}Hz  snap=±{ROCH_SNAP_SEARCH_HZ:.0f}Hz  "
          f"match≤{ROCH_MATCH_THRESHOLD_HZ:.0f}Hz")
    print(f"  Spectrace reported tolerance: {star_tol:.0f}Hz  max_k={star_k} "
          f"(set via --reported-k / --reported-tolerance)")
    print("-" * w)
    hdr = f"  {'Method':<28}{'Dev(Hz)':>9}{'Cover%':>9}{'Frag':>7}{'Rec%':>7}{'Prec%':>7}"
    print(hdr)
    print("-" * w)
    for name, v in ROCH_REPORTED.items():
        print(f"  {'Roch ' + name.replace('_', ' '):<28}"
              f"{v['deviation_hz']:>9.0f}{v['coverage_pct']:>9.1f}"
              f"{v['fragmentation']:>7.1f}{v['recall_pct']:>7.1f}{v['precision_pct']:>7.1f}")
    if not ours.empty:
        r = ours.iloc[0]
        print(f"  {'Spectrace (reported pt)':<28}"
              f"{r['deviation_hz']:>9.0f}{r['coverage_pct']:>9.1f}"
              f"{r['fragmentation']:>7.1f}{r['roch_recall_pct']:>7.1f}"
              f"{r['roch_precision_pct']:>7.1f}")
    print("=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Heterodyne hyperparameter sweep + Roch comparison")
    p.add_argument("--hdf5-dir", required=True)
    p.add_argument("--max-k-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--kernel-grid", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--tolerance-grid", type=float, nargs="+",
                   default=[100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000])
    p.add_argument("--output-dir", default="visualizations/sweep")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--reported-k", type=int, default=None,
        help="max_k value to mark with the star in all figures. "
             "Must be present in --max-k-grid. "
             f"Default: smallest value in --max-k-grid.")
    p.add_argument(
        "--reported-tolerance", type=float, default=None,
        help="Frequency tolerance (Hz) to mark with the star on the tolerance "
             "curve figure. Does not need to be in --tolerance-grid (interpolated). "
             f"Default: {DEFAULT_REPORTED_TOLERANCE_HZ:.0f} Hz "
             f"(PRIMARY_TOLERANCE_HZ from heterodyne_validation).")
    args = p.parse_args()

    out = None if args.no_plots else args.output_dir
    grid_df, tol_df = run_sweep(
        args.hdf5_dir, args.max_k_grid, args.kernel_grid,
        args.tolerance_grid, out)

    if grid_df.empty:
        print("No data swept.")
        sys.exit(0)

    print_literature_comparison(
        grid_df,
        reported_k=args.reported_k,
        reported_tolerance_hz=args.reported_tolerance,
    )

    if out:
        make_figures(
            grid_df, tol_df, out,
            reported_k=args.reported_k,
            reported_tolerance_hz=args.reported_tolerance,
        )


if __name__ == "__main__":
    main()