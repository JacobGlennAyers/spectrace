#!/usr/bin/env python3
"""
make_metric_pipeline_figure.py
==============================

A publication "magnification" figure that walks a SINGLE
heterodyne order through every metric in heterodyne_validation.py, and a
companion figure that runs the identical pipeline on a biphonic NULL CONTROL
clip (fundamentals present, no heterodynes) to show that the errors blow up.

It does NOT re-implement any metric. It imports the validated functions from
heterodyne_validation.py (just like heterodyne_sweep.py does), so the figure is
guaranteed to match Table 2. Put this file alongside heterodyne_validation.py
(i.e. in demos/).

Layout
------
FIGURE 1 (positive clip, one order):

    A  full spectrogram (TRUE pixel aspect) + f0_HFC / f0_LFC inputs
       + dashed zoom box on the order
    --------------------------------------------------------------------------
    B label + (n+1)*HFC reference   C sub-band fan (one color per k)   D fit
    --------------------------------------------------------------------------
    E raw 1-px overlap (IoU_raw)    F 5x5 dilation TP/FP/FN     G band-aware
                                      -> IoU only                 error stems
                                                                  (in/out of
                                                                  +/-tol) +
                                                                  inline error
                                                                  histogram
                                                                  -> MAE,
                                                                  Acc@tol

FIGURE 2 (negative control, same machinery):

    H full null spectrogram + fundamentals + nearest non-heterodyne reference
      + prediction fan + zoom box
    I zoom: prediction vs reference, in/out-of-tol stems + ribbon -> MAE, Acc@tol
    J bar: Acc@tol and MAE, positive vs negative

Outputs (into --output-dir):
    metric_pipeline.{pdf,svg,png}
    negative_control.{pdf,svg,png}   (only if --neg-hdf5 is supplied)

Design notes (greyscale-aware)
------------------------------
The spectrogram background is greyscale, so every overlay color is chosen to
sit clearly on top of mid/dark grey. Panel A is drawn at the spectrogram's true
pixel aspect (aspect="equal") so frequency/time are not distorted; the figure
is rendered at high DPI to keep it sharp. The +/-tolerance ribbon in G/I uses a
bright amber with a defined edge so it reads against the grey.

Usage
-----
    python make_metric_pipeline_figure.py \
        --hdf5 ml_data/Ct-Dt--03-52--D.hdf5 --order 3 --max-k 5 \
        --neg-hdf5 ml_data/<a-clip-with-no-heterodynes>.hdf5 \
        --output-dir figures

All knobs have defaults; --order, --tolerance, --kernel-size, --max-k and the
crop box are the ones you will most likely touch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / scripted
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))       # now = base_repo/demos/
_BASE_DIR = os.path.join(_HERE, "..")                    # = base_repo/
sys.path.insert(0, _HERE)                                # for anything else in demos/
sys.path.insert(0, _BASE_DIR)                            # for utils, bin_morph, etc.
import heterodyne_validation as hv          
from bin_morph import MaskMorphology          


# =============================================================================
# Style
# =============================================================================
LABEL_COLOR = "#e6007e"   # magenta  -> annotator's hand label
FITTED_COLOR = "#13b6c9"  # cyan     -> committed (fitted) prediction
HFC_REF_COLOR = "#ffffff"  # dashed   -> (n+1)*HFC harmonic reference
F0_HFC_COLOR = "#ff8c1a"   # orange   -> f0_HFC input (reads on grey)
F0_LFC_COLOR = "#22d3ee"   # bright cyan -> f0_LFC input (reads on grey)

# --- Panel C: ONE COLOR PER k (solid = +k sideband, dashed = -k sideband) ----
# A perceptually-spaced, greyscale-safe palette. Index 0 -> k=1, etc. If max_k
# exceeds the palette length we fall back to sampling a colormap (see k_color).
K_COLORS = [
    "#ff8c1a",  # orange
    "#22d3ee",  # cyan
    "#a78bfa",  # violet
    "#34d399",  # green
    "#f472b6",  # pink
    "#fbbf24",  # amber
    "#60a5fa",  # blue
    "#f87171",  # red
]


def k_color(k: int) -> str:
    """Color for sub-band order k (k = 1..max_k). One color per k."""
    if 1 <= k <= len(K_COLORS):
        return K_COLORS[k - 1]
    # graceful fallback for unusually large max_k
    cmap = plt.get_cmap("turbo")
    return cmap((k - 1) / max(k, 1))


# --- Tolerance window: binary in/out of the SINGLE reported tolerance --------
# We only report Acc@PRIMARY_TOLERANCE, so the error stems are colored simply
# by whether they fall inside that window or not. No multi-bin scheme.
TOL_IN_COLOR = "#2ecc71"    # within +/- tol  (matches "accurate")
TOL_OUT_COLOR = "#e74c3c"   # outside +/- tol
RIBBON_COLOR = "#ffb300"    # bright amber: the +/- tol window shading
RIBBON_EDGE = "#ff8f00"     # slightly darker amber edge for definition


def stem_color(err_hz: float, tol_hz: float) -> str:
    """Binary stem color: inside vs outside the reported tolerance."""
    return TOL_IN_COLOR if err_hz <= tol_hz else TOL_OUT_COLOR


# Panels H/I color the error by whether it is below (green) or above (red) the
# MAE, visualizing the spread of errors around their own mean.
MAE_BELOW_COLOR = "#2ecc71"   # <= MAE : better than average
MAE_ABOVE_COLOR = "#e74c3c"   # >  MAE : worse than average


# Non-heterodyne reference layers eligible for the negative control. Mirrors the
# _REFERENCE_LAYER_PREFIXES list inside heterodyne_validation.run_negative_control.
REF_PREFIXES = (
    "harmonics_HFC", "harmonics_LFC", "subharmonics_HFC", "subharmonics_LFC",
    "heterodyne_or_subharmonic_or_other", "Cetacean_AdditionalContours",
    "unsure_HFC", "unsure_LFC", "Heterodynes/unsure",
)


# =============================================================================
# Data assembly (uses only hv.* functions for anything that touches a metric)
# =============================================================================

class ClipBundle:
    """Everything needed to draw the panels for one (clip, order, label) case."""

    def __init__(self, spec, max_freq, tpp, f0_hfc, f0_lfc, label_mask,
                 pred_freqs, pred_mask, fitted_freqs, fitted_mask, order):
        self.spec = spec
        self.H, self.W = spec.shape
        self.fpb = max_freq / self.H           # Hz per pixel row
        self.max_freq = max_freq
        self.tpp = tpp                          # seconds per pixel column
        self.f0_hfc = f0_hfc
        self.f0_lfc = f0_lfc
        self.label_mask = label_mask
        self.pred_freqs = pred_freqs            # full fan (list of arrays)
        self.pred_mask = pred_mask
        self.fitted_freqs = fitted_freqs        # committed (one per segment)
        self.fitted_mask = fitted_mask
        self.order = order

    # the prediction set we score against: fitted if available, else the fan
    @property
    def pred_set(self) -> List[np.ndarray]:
        return self.fitted_freqs if self.fitted_freqs else self.pred_freqs

    @property
    def scored_mask(self) -> np.ndarray:
        return self.fitted_mask if self.fitted_freqs else self.pred_mask


def build_bundle(hdf5_path: str, ann_idx: int, order: int, max_k: int,
                 label_layer: str) -> ClipBundle:
    """Load a clip and run the prediction/fit pipeline for one order.

    ``label_layer`` is what we treat as the ground-truth mask:
      - "Heterodynes/<order>" for the positive case
      - a non-heterodyne reference layer for the negative control
    """
    with hv.HDF5SpectrogramLoader(hdf5_path) as loader:
        meta = loader.get_metadata()
        spec = loader.load_spectrogram()
        max_freq = float(meta.max_freq_hz)
        dur = float(meta.duration_sec)
        hfc_mask = loader.get_class_mask("f0_HFC", ann_idx)
        lfc_mask = loader.get_class_mask("f0_LFC", ann_idx)
        label_mask = loader.get_class_mask(label_layer, ann_idx)

    if label_mask is None:
        label_mask = np.zeros_like(spec, dtype=np.uint8)

    H, W = spec.shape
    tpp = dur / W

    f0_hfc = hv.smooth_f0_contour(hv.extract_f0_contour(hfc_mask, max_freq))
    f0_lfc = hv.smooth_f0_contour(hv.extract_f0_contour(lfc_mask, max_freq))

    pred_freqs = hv.compute_predicted_heterodyne_freqs(
        f0_hfc, f0_lfc, hfc_multiplier=order + 1, max_k=max_k, max_freq=max_freq)
    pred_mask = hv.render_frequency_to_mask(pred_freqs, H, W, max_freq)

    fitted_freqs, _ = hv.fit_subband_per_segment(pred_freqs, label_mask, max_freq)
    fitted_mask = (hv.render_frequency_to_mask(fitted_freqs, H, W, max_freq)
                   if fitted_freqs else np.zeros((H, W), dtype=np.uint8))

    return ClipBundle(spec, max_freq, tpp, f0_hfc, f0_lfc, label_mask,
                      pred_freqs, pred_mask, fitted_freqs, fitted_mask, order)


def resolve_positive_index(hdf5_path: str, order: int) -> int:
    valid, _ = hv.get_valid_annotation_indices(hdf5_path)
    if not valid:
        raise SystemExit(f"{hdf5_path}: no annotation set with f0_HFC, f0_LFC "
                         f"and a drawn Heterodynes/N mask.")
    # prefer an index whose chosen order is actually drawn
    for idx in valid:
        with hv.HDF5SpectrogramLoader(hdf5_path) as loader:
            m = loader.get_class_mask(f"Heterodynes/{order}", idx)
        if m is not None and m.sum() > 0:
            return idx
    print(f"  WARNING: order {order} not drawn in any valid index; using {valid[0]}.")
    return valid[0]


def resolve_negative(hdf5_path: str, prefer_layer: Optional[str]) -> Tuple[int, str]:
    """Pick an annotation index that has both fundamentals and a non-heterodyne
    reference layer, and return (index, reference_layer_name)."""
    with hv.HDF5SpectrogramLoader(hdf5_path) as loader:
        class_names = loader.get_class_names()
        n_ann = loader.get_num_annotations()
        best = None  # (pixels, idx, name)
        for idx in range(n_ann):
            hfc = loader.get_class_mask("f0_HFC", idx)
            lfc = loader.get_class_mask("f0_LFC", idx)
            if hfc is None or hfc.sum() == 0 or lfc is None or lfc.sum() == 0:
                continue
            candidates = ([prefer_layer] if prefer_layer else
                          [n for n in class_names
                           if any(n.startswith(p) for p in REF_PREFIXES)])
            for name in candidates:
                if name not in class_names:
                    continue
                m = loader.get_class_mask(name, idx)
                if m is None or m.sum() == 0:
                    continue
                px = int(m.sum())
                if best is None or px > best[0]:
                    best = (px, idx, name)
    if best is None:
        raise SystemExit(f"{hdf5_path}: no annotation index with fundamentals + a "
                         f"non-heterodyne reference layer found.")
    return best[1], best[2]


# =============================================================================
# Drawing primitives (pixel-coordinate, mask-registered -- mirrors your safe
# approach in make_publication_figure.py: never rescale a mask)
# =============================================================================

def show_spec(ax, b: ClipBundle, cmap="gray", aspect="auto"):
    ax.imshow(b.spec, cmap=cmap, origin="upper", aspect=aspect,
              extent=[0, b.W, b.H, 0], interpolation="nearest", zorder=0)


def set_crop(ax, b: ClipBundle, box, n=5, xlabel=True, ylabel=True):
    r0, r1, c0, c1 = box
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)  # origin upper: larger row lower on screen
    rows = np.linspace(r0, r1, n)
    ax.set_yticks(rows)
    ax.set_yticklabels([f"{(b.max_freq - r * b.fpb) / 1000:.1f}" for r in rows])
    cols = np.linspace(c0, c1, n)
    ax.set_xticks(cols)
    ax.set_xticklabels([f"{c * b.tpp:.2f}" for c in cols])
    ax.tick_params(labelsize=8)
    if xlabel:
        ax.set_xlabel("Time (s)", fontsize=9)
    if ylabel:
        ax.set_ylabel("Frequency (kHz)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def overlay_mask(ax, mask, color, alpha=1.0, zorder=3):
    if mask is None or mask.sum() == 0:
        return
    H, W = mask.shape
    rgba = np.zeros((H, W, 4))
    rgba[mask.astype(bool), :3] = to_rgb(color)
    rgba[mask.astype(bool), 3] = alpha
    ax.imshow(rgba, origin="upper", aspect="auto",
              extent=[0, W, H, 0], interpolation="nearest", zorder=zorder)


def plot_freq(ax, b: ClipBundle, fa, **kw):
    valid = ~np.isnan(fa)
    if valid.sum() == 0:
        return
    cols = np.where(valid)[0]
    rows = (b.max_freq - fa[valid]) / b.fpb
    ax.plot(cols, rows, **kw)


def bbox_of(mask, pad, H, W) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, H, 0, W
    r0 = max(0, ys.min() - pad); r1 = min(H, ys.max() + pad)
    c0 = max(0, xs.min() - pad); c1 = min(W, xs.max() + pad)
    return r0, r1, c0, c1


def panel_letter(ax, letter, size=14):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="right")


# =============================================================================
# FIGURE 1 -- the metric cascade on a positive clip
# =============================================================================

def _draw_zoom_leaders(fig, ax_src, box, ax_dst):
    """Draw two leader lines fanning RIGHTWARD from the zoom box in ``ax_src``
    (full spectrogram A) to the magnified crop panel ``ax_dst`` (A'), in figure
    coordinates. Lines run from the box's two RIGHT corners to A''s two LEFT
    corners, crossing only the (already-seen) spectrogram region between them.
    """
    r0, r1, c0, c1 = box
    # source: the two RIGHT corners of the zoom box (data coords)
    src_top = ax_src.transData.transform((c1, r0))   # box upper-right (low row)
    src_bot = ax_src.transData.transform((c1, r1))   # box lower-right (high row)
    inv = fig.transFigure.inverted()
    src_top_f = inv.transform(src_top)
    src_bot_f = inv.transform(src_bot)
    # destination: the two LEFT corners of A'.
    pos = ax_dst.get_position()
    dst_top_f = (pos.x0, pos.y1)   # A' top-left
    dst_bot_f = (pos.x0, pos.y0)   # A' bottom-left
    for (sf, df) in ((src_top_f, dst_top_f), (src_bot_f, dst_bot_f)):
        line = Line2D([sf[0], df[0]], [sf[1], df[1]], transform=fig.transFigure,
                      color="#ff3b30", lw=1.2, ls="-", alpha=0.9, zorder=10,
                      clip_on=False)
        fig.add_artist(line)


def make_pipeline_figure(b: ClipBundle, kernel_size: int, tol_hz: float,
                         box, out_base: str, dpi: int) -> dict:
    morph = MaskMorphology()
    tol_px = int(round(tol_hz / b.fpb))

    # --- metrics, straight from the validated functions -----------------
    raw_inter = int(np.logical_and(b.scored_mask, b.label_mask).sum())
    raw_union = int(np.logical_or(b.scored_mask, b.label_mask).sum())
    iou_raw = raw_inter / raw_union if raw_union else float("nan")

    iou = hv.compute_iou(b.scored_mask, b.label_mask, kernel_size)["iou"]
    node = hv.compute_node_level_metrics(b.scored_mask, b.label_mask, kernel_size)
    ba = hv.compute_band_aware_metrics(b.pred_set, b.label_mask, b.max_freq,
                                       tolerances_hz=[int(tol_hz)])
    cl = hv.compute_contour_level_metrics(b.pred_set, b.label_mask, b.max_freq,
                                          freq_tolerance_hz=tol_hz)
    mae = ba["ba_mae_hz"]
    acc = ba[hv.acc_key(int(tol_hz))]
    # cov / frag are still computed by hv (cl dict) but DELIBERATELY NOT shown:
    # not reported in the validation table, so they stay out of the figure.

    # ---------------------------------------------------------------- layout
    # Four equal-height rows of three columns. The TOP row holds the full
    # spectrogram (A, over column 1) and its magnified crop (A', over column 3,
    # the SAME size as every metric panel and aligned above C). Column 2 of the
    # top row is left empty so the zoom leader lines can fan rightward from A's
    # box into A' while only crossing the spectrogram region already shown.
    #   Row 0:  A (full spec)   [empty]        A' (magnified crop)
    #   Row 1:  B               C              D
    #   Row 2:  E               F              G
    #   Row 3:  H (stems)       I (histogram)  [empty]
    fig = plt.figure(figsize=(13.5, 14.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.0],
                          hspace=0.40, wspace=0.28,
                          left=0.06, right=0.985, top=0.95, bottom=0.05)

    r0, r1, c0, c1 = box

    # ---- A: full spectrogram (over column 1) + fundamentals + zoom box --
    # aspect="auto" (like make_publication_figure.py): a spectrogram's time vs
    # frequency scaling is conventional, not a natural-image aspect to preserve.
    axA = fig.add_subplot(gs[0, 0])
    show_spec(axA, b, aspect="auto")
    plot_freq(axA, b, b.f0_hfc, color=F0_HFC_COLOR, lw=1.5, label="f0_HFC")
    plot_freq(axA, b, b.f0_lfc, color=F0_LFC_COLOR, lw=1.5, label="f0_LFC")
    axA.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                            edgecolor="#ff3b30", lw=1.8, zorder=5))
    axA.set_xlim(0, b.W); axA.set_ylim(b.H, 0)
    yt = np.linspace(0, b.H, 6)
    axA.set_yticks(yt)
    axA.set_yticklabels([f"{(b.max_freq - r * b.fpb) / 1000:.0f}" for r in yt])
    xt = np.linspace(0, b.W, 6)
    axA.set_xticks(xt)
    axA.set_xticklabels([f"{c * b.tpp:.1f}" for c in xt])
    axA.set_xlabel("Time (s)", fontsize=9); axA.set_ylabel("Frequency (kHz)", fontsize=9)
    axA.tick_params(labelsize=8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.set_title("Full spectrogram + biphonic inputs", fontsize=9)
    axA.legend(loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axA, "A")

    # ---- A': magnified crop (over column 3) — the view B-G all show ------
    axAz = fig.add_subplot(gs[0, 2])
    show_spec(axAz, b)
    plot_freq(axAz, b, b.f0_hfc, color=F0_HFC_COLOR, lw=1.4)
    plot_freq(axAz, b, b.f0_lfc, color=F0_LFC_COLOR, lw=1.4)
    # border in the same red as the source box, to tie them together
    set_crop(axAz, b, box, ylabel=True)
    for s in ("top", "right", "bottom", "left"):
        axAz.spines[s].set_visible(True)
        axAz.spines[s].set_color("#ff3b30")
        axAz.spines[s].set_linewidth(1.6)
    axAz.set_title(f"Magnified region (Heterodynes/{b.order})", fontsize=9)
    panel_letter(axAz, "A\u2032")

    # ---- B: the target (label + (n+1)*HFC reference) -------------------
    axB = fig.add_subplot(gs[1, 0])
    show_spec(axB, b)
    overlay_mask(axB, b.label_mask, LABEL_COLOR, alpha=1.0, zorder=4)
    hfc_harm = (b.order + 1) * b.f0_hfc
    plot_freq(axB, b, hfc_harm, color=HFC_REF_COLOR, ls="--", lw=1.2, alpha=0.7,
              zorder=3)
    set_crop(axB, b, box, ylabel=True)
    axB.set_title(f"Target: hand-labelled Heterodynes/{b.order}\n"
                  f"(dashed = {b.order+1}\u00d7 f0_HFC reference)", fontsize=9)
    panel_letter(axB, "B")

    # ---- C: full sub-band fan, ONE COLOR PER k -------------------------
    # +k = solid line; -k = dotted markers riding the same colored curve, so
    # the sign is legible at print size without a separate solid/dashed key.
    axC = fig.add_subplot(gs[1, 1])
    show_spec(axC, b)
    fan = b.pred_freqs
    max_k = (len(fan) + 1) // 2
    for i, fa in enumerate(fan):
        k = i // 2 + 1
        is_plus = (i % 2 == 0)
        if is_plus:
            plot_freq(axC, b, fa, color=k_color(k), lw=1.5, ls="-", alpha=0.95)
        else:
            # dotted-marker rendering for the -k sideband
            plot_freq(axC, b, fa, color=k_color(k), lw=0, marker="o",
                      markersize=1.6, markevery=3, alpha=0.95)
    k_handles = [Line2D([0], [0], color=k_color(k), lw=2, label=f"k={k}")
                 for k in range(1, max_k + 1)]
    # +k / -k shown by BLACK example swatches (solid line vs dots) with no
    # words; the contrast against the colored k-entries makes the key obvious.
    sign_handles = [
        Line2D([0], [0], color="#000000", lw=2.2, ls="-", label="+k"),
        Line2D([0], [0], color="#000000", lw=0, marker="o", markersize=3.2,
               label="\u2212k"),
    ]
    axC.set_title(f"Sub-band fan: (n+1)\u00b7f_HFC \u00b1 k\u00b7f_LFC\n"
                  f"{len(fan)} candidate curves (k=1..{max_k})", fontsize=9)
    leg1 = axC.legend(handles=k_handles + sign_handles, loc="upper right",
                      fontsize=6, ncol=2, framealpha=0.9)
    axC.add_artist(leg1)
    set_crop(axC, b, box, ylabel=False)
    panel_letter(axC, "C")

    # ---- D: per-segment committed fit ----------------------------------
    axD = fig.add_subplot(gs[1, 2])
    show_spec(axD, b)
    overlay_mask(axD, b.label_mask, LABEL_COLOR, alpha=0.55, zorder=3)
    for fa in b.fitted_freqs:
        plot_freq(axD, b, fa, color=FITTED_COLOR, lw=1.6, zorder=4)
    set_crop(axD, b, box, ylabel=False)
    n_seg = len(b.fitted_freqs)
    axD.set_title(f"Per-segment fit (committed prediction)\n"
                  f"{n_seg} segment(s) \u2192 1 best sub-band each", fontsize=9)
    axD.legend(handles=[Patch(color=LABEL_COLOR, label="label"),
                        Line2D([0], [0], color=FITTED_COLOR, lw=2, label="fitted")],
               loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axD, "D")

    # ---- E: raw 1-px overlap -------------------------------------------
    axE = fig.add_subplot(gs[2, 0])
    show_spec(axE, b)
    # raw (un-dilated) TP/FP/FN, same color scheme as F
    pe = b.scored_mask.astype(bool)
    le = b.label_mask.astype(bool)
    tp_e = pe & le
    fp_e = pe & ~le
    fn_e = ~pe & le
    overlay_mask(axE, fn_e.astype(np.uint8), LABEL_COLOR, alpha=0.85, zorder=3)
    overlay_mask(axE, fp_e.astype(np.uint8), FITTED_COLOR, alpha=0.85, zorder=3)
    overlay_mask(axE, tp_e.astype(np.uint8), "#ffffff", alpha=1.0, zorder=5)
    set_crop(axE, b, box, ylabel=True)
    axE.set_title(f"Raw overlap (1-px masks)\nIoU_raw = {iou_raw:.3f}", fontsize=9)
    axE.legend(handles=[Patch(color="#ffffff", label="TP"),
                        Patch(color=FITTED_COLOR, label="FP"),
                        Patch(color=LABEL_COLOR, label="FN")],
               loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axE, "E")

    # ---- F: 5x5 dilation, node-level overlap (IoU only in title) -------
    axF = fig.add_subplot(gs[2, 1])
    show_spec(axF, b)
    pred_d = morph.dilate(b.scored_mask, kernel_size).astype(bool) \
        if b.scored_mask.sum() else b.scored_mask.astype(bool)
    lab_d = morph.dilate(b.label_mask, kernel_size).astype(bool) \
        if b.label_mask.sum() else b.label_mask.astype(bool)
    tp = pred_d & lab_d
    fp = pred_d & ~lab_d
    fn = ~pred_d & lab_d
    overlay_mask(axF, fn.astype(np.uint8), LABEL_COLOR, alpha=0.6, zorder=3)
    overlay_mask(axF, fp.astype(np.uint8), FITTED_COLOR, alpha=0.6, zorder=3)
    overlay_mask(axF, tp.astype(np.uint8), "#ffffff", alpha=0.95, zorder=4)
    set_crop(axF, b, box, ylabel=False)
    axF.set_title(f"{kernel_size}\u00d7{kernel_size} dilation (both masks)\n"
                  f"IoU = {iou:.3f}", fontsize=9)
    axF.legend(handles=[Patch(color="#ffffff", label="TP"),
                        Patch(color=FITTED_COLOR, label="FP"),
                        Patch(color=LABEL_COLOR, label="FN")],
               loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axF, "F")

    # ---- G: Acc@tol explainer -- teal fitted curves vs label + tol ribbon
    # G is dedicated to explaining Acc@250: the committed (fitted) curves (teal,
    # matching D) drawn against the label inside the +/-tol window. The reader
    # sees Acc@250 as "how much of the teal sits inside the amber band".
    axG = fig.add_subplot(gs[2, 2])
    show_spec(axG, b)
    struct = np.ones((2 * tol_px + 1, 1), dtype=bool)
    ribbon = binary_dilation(b.label_mask.astype(bool), structure=struct)
    overlay_mask(axG, ribbon.astype(np.uint8), RIBBON_COLOR, alpha=0.30, zorder=2)
    overlay_mask(axG, (ribbon ^ binary_dilation(ribbon, iterations=1)
                       ).astype(np.uint8), RIBBON_EDGE, alpha=0.45, zorder=2)
    overlay_mask(axG, b.label_mask, LABEL_COLOR, alpha=1.0, zorder=4)
    lab_bands = hv._labelled_bands_per_column(b.label_mask, b.max_freq)
    for fa in b.pred_set:
        plot_freq(axG, b, fa, color=FITTED_COLOR, lw=1.6, zorder=5)
    set_crop(axG, b, box, ylabel=False)
    axG.set_title(f"Acc@{int(tol_hz)} explainer: fitted curves vs label\n"
                  f"(\u00b1{int(tol_hz)} Hz window shaded)   Acc@{int(tol_hz)} = {acc:.0%}",
                  fontsize=9)
    axG.legend(handles=[Patch(color=LABEL_COLOR, label="label"),
                        Patch(color=RIBBON_COLOR, label=f"\u00b1{int(tol_hz)} Hz"),
                        Line2D([0], [0], color=FITTED_COLOR, lw=2, label="fitted")],
               loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axG, "G")

    # ---- H: band-aware error stems (the original G stem view) ----------
    axH = fig.add_subplot(gs[3, 0])
    show_spec(axH, b)
    overlay_mask(axH, b.label_mask, LABEL_COLOR, alpha=1.0, zorder=4)
    # teal fitted curves (matching D/G) ...
    for fa in b.pred_set:
        plot_freq(axH, b, fa, color=FITTED_COLOR, lw=1.2, alpha=0.85, zorder=3)
    # ... and a stem per (column, labelled band) to the nearest prediction,
    # colored by whether the error is BELOW the MAE (green = better than the
    # mean) or ABOVE it (red = worse). This shows the spread around the MAE.
    errs = []
    for t, lf, pf, e in _band_error_stems(b.label_mask, b.pred_set, b.max_freq):
        errs.append(e)
        r_lab = (b.max_freq - lf) / b.fpb
        r_pred = (b.max_freq - pf) / b.fpb
        c = MAE_BELOW_COLOR if e <= mae else MAE_ABOVE_COLOR
        axH.plot([t, t], [r_lab, r_pred], color=c, lw=1.3,
                 alpha=0.95, zorder=5, solid_capstyle="butt")
    set_crop(axH, b, box, ylabel=True)
    axH.set_title(f"Per-band error stems vs fitted curve\n"
                  f"MAE = {mae:.0f} Hz", fontsize=9)
    axH.legend(handles=[Line2D([0], [0], color=FITTED_COLOR, lw=2, label="fitted"),
                        Patch(color=MAE_BELOW_COLOR, label="\u2264 MAE"),
                        Patch(color=MAE_ABOVE_COLOR, label="> MAE")],
               loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axH, "H")

    # ---- I: combined MAE histogram, split BELOW/ABOVE the MAE ----------
    axI = fig.add_subplot(gs[3, 1])
    if errs:
        errs_arr = np.asarray(errs)
        hi = float(errs_arr.max()) if errs_arr.size else 1.0
        bins = np.linspace(0, max(hi, mae * 1.5), 24)
        below = errs_arr <= mae
        axI.hist(errs_arr[below], bins=bins, color=MAE_BELOW_COLOR,
                 edgecolor="white", linewidth=0.3, label="\u2264 MAE")
        axI.hist(errs_arr[~below], bins=bins, color=MAE_ABOVE_COLOR,
                 edgecolor="white", linewidth=0.3, label="> MAE")
        axI.axvline(mae, color="#222222", lw=1.8, label=f"MAE = {mae:.0f} Hz")
        axI.set_xlabel("Absolute error (Hz)", fontsize=9)
        axI.set_ylabel("Count (column\u00d7band samples)", fontsize=9)
        axI.set_title("Error distribution underlying the MAE", fontsize=9)
        axI.legend(loc="upper right", fontsize=7, framealpha=0.9)
        axI.tick_params(labelsize=8)
        for s in ("top", "right"):
            axI.spines[s].set_visible(False)
    else:
        axI.axis("off")
        axI.text(0.5, 0.5, "no error samples", ha="center", va="center")
    panel_letter(axI, "I")

    # third cell of the bottom row intentionally left empty for breathing room

    # --- zoom leader lines: fan rightward from A's box into A' (the crop) ---
    _draw_zoom_leaders(fig, axA, box, axAz)

    fig.suptitle(f"Heterodyne validation pipeline \u2014 Heterodynes/{b.order}",
                 fontsize=13, fontweight="bold", y=0.985)

    _save(fig, out_base, dpi)
    return {"iou_raw": iou_raw, "iou": iou, "mae": mae, "acc": acc, **node}


def _band_error_stems(label_mask, pred_set, max_freq):
    """(column, label_freq, nearest_pred_freq, abs_error) per labelled band.

    Same matching rule as hv.compute_band_aware_metrics, exposed for drawing.
    """
    bands = hv._labelled_bands_per_column(label_mask, max_freq)
    out = []
    for t, freqs in bands.items():
        preds = [fa[t] for fa in pred_set if not np.isnan(fa[t])]
        if not preds:
            continue
        for lf in freqs:
            pf = min(preds, key=lambda p: abs(lf - p))
            out.append((t, lf, pf, abs(lf - pf)))
    return out


# =============================================================================
# FIGURE 2 -- the same machinery on a null-control clip
# =============================================================================

def make_negative_figure(b: ClipBundle, ref_layer: str, kernel_size: int,
                          tol_hz: float, box, pos_metrics: Optional[dict],
                          out_base: str, dpi: int):
    tol_px = int(round(tol_hz / b.fpb))
    ba = hv.compute_band_aware_metrics(b.pred_set, b.label_mask, b.max_freq,
                                       tolerances_hz=[int(tol_hz)])
    neg_mae = ba["ba_mae_hz"]
    neg_acc = ba[hv.acc_key(int(tol_hz))]

    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.1, 0.7],
                          wspace=0.3, left=0.06, right=0.98, top=0.88, bottom=0.14)

    # ---- H: full null spectrogram + fundamentals + reference + fan -----
    axH = fig.add_subplot(gs[0, 0])
    show_spec(axH, b)
    plot_freq(axH, b, b.f0_hfc, color=F0_HFC_COLOR, lw=1.4, label="f0_HFC")
    plot_freq(axH, b, b.f0_lfc, color=F0_LFC_COLOR, lw=1.4, label="f0_LFC")
    overlay_mask(axH, b.label_mask, LABEL_COLOR, alpha=0.9, zorder=4)
    r0, r1, c0, c1 = box
    axH.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                            edgecolor="#ff3b30", lw=1.6, zorder=5))
    axH.set_xlim(0, b.W); axH.set_ylim(b.H, 0)
    yt = np.linspace(0, b.H, 6)
    axH.set_yticks(yt)
    axH.set_yticklabels([f"{(b.max_freq - r * b.fpb) / 1000:.0f}" for r in yt])
    xt = np.linspace(0, b.W, 6)
    axH.set_xticks(xt); axH.set_xticklabels([f"{c * b.tpp:.1f}" for c in xt])
    axH.tick_params(labelsize=8)
    axH.set_xlabel("Time (s)", fontsize=9); axH.set_ylabel("Frequency (kHz)", fontsize=9)
    for s in ("top", "right"):
        axH.spines[s].set_visible(False)
    axH.set_title(f"Null control: no heterodynes labelled\nreference layer = "
                  f"{ref_layer}", fontsize=10)
    axH.legend(loc="upper right", fontsize=7, framealpha=0.85)
    panel_letter(axH, "K")

    # ---- I: zoom, prediction vs reference, in/out-of-tol stems + ribbon -
    axI = fig.add_subplot(gs[0, 1])
    show_spec(axI, b)
    struct = np.ones((2 * tol_px + 1, 1), dtype=bool)
    ribbon = binary_dilation(b.label_mask.astype(bool), structure=struct)
    overlay_mask(axI, ribbon.astype(np.uint8), RIBBON_COLOR, alpha=0.30, zorder=2)
    overlay_mask(axI, (ribbon ^ binary_dilation(ribbon, iterations=1)
                       ).astype(np.uint8), RIBBON_EDGE, alpha=0.45, zorder=2)
    overlay_mask(axI, b.label_mask, LABEL_COLOR, alpha=1.0, zorder=4)
    for i, fa in enumerate(b.pred_freqs):
        plot_freq(axI, b, fa, color=FITTED_COLOR, lw=1.0, alpha=0.7, zorder=3)
    for t, lf, pf, e in _band_error_stems(b.label_mask, b.pred_set, b.max_freq):
        r_lab = (b.max_freq - lf) / b.fpb
        r_pred = (b.max_freq - pf) / b.fpb
        axI.plot([t, t], [r_lab, r_pred], color=stem_color(e, tol_hz), lw=1.1,
                 alpha=0.9, zorder=3, solid_capstyle="butt")
    set_crop(axI, b, box, ylabel=False)
    axI.set_title(f"Predictions vs reference contour\n"
                  f"MAE = {neg_mae:.0f} Hz   Acc@{int(tol_hz)} = {neg_acc:.0%}",
                  fontsize=9)
    axI.legend(handles=[Patch(color=TOL_IN_COLOR, label=f"\u2264{int(tol_hz)} Hz"),
                        Patch(color=TOL_OUT_COLOR, label=f">{int(tol_hz)} Hz")],
               loc="upper right", fontsize=6, framealpha=0.85)
    panel_letter(axI, "L")

    # ---- J: positive vs negative bar comparison ------------------------
    axJ = fig.add_subplot(gs[0, 2])
    if pos_metrics is not None:
        pos_vals = [pos_metrics["acc"], pos_metrics["mae"]]
        neg_vals = [neg_acc, neg_mae]
        axJ.bar([0, 0.35], [pos_vals[0], neg_vals[0]], width=0.32,
                color=[TOL_IN_COLOR, TOL_OUT_COLOR], edgecolor="black")
        axJ.set_ylim(0, 1.05)
        axJ.set_xticks([0, 0.35]); axJ.set_xticklabels(["positive", "negative"],
                                                       fontsize=8)
        axJ.set_ylabel(f"Acc@{int(tol_hz)} Hz", fontsize=9)
        axJ.set_title("Specificity check", fontsize=10)
        axt = axJ.twinx()
        axt.plot([0, 0.35], [pos_vals[1], neg_vals[1]], "o-", color="#34495e",
                 lw=1.5, label="MAE")
        axt.set_ylabel("MAE (Hz)", fontsize=9)
        axt.legend(loc="upper center", fontsize=7)
    else:
        axJ.axis("off")
        axJ.text(0.5, 0.5, "pass a positive clip\nto draw comparison",
                 ha="center", va="center", fontsize=9)
    panel_letter(axJ, "M")

    fig.suptitle("Negative control \u2014 biphonic formula vs non-heterodyne contours",
                 fontsize=12, fontweight="bold", y=0.97)
    _save(fig, out_base, dpi)


# =============================================================================
# Save / CLI
# =============================================================================

def _save(fig, out_base, dpi):
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for ext in ("pdf", "svg", "png"):
        print(f"  wrote {out_base}.{ext}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, help="positive clip (has heterodynes)")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--annotation-index", type=int, default=None,
                   help="override; default = first valid index with this order")
    p.add_argument("--max-k", type=int, default=5,
                   help="sub-bands per side (paper uses 5; do NOT leave at 1)")
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=float(hv.PRIMARY_TOLERANCE_HZ))
    p.add_argument("--pad-px", type=int, default=8, help="padding around auto crop box")
    p.add_argument("--box", type=int, nargs=4, default=None,
                   metavar=("R0", "R1", "C0", "C1"),
                   help="manual crop in pixels (rows then cols); default = auto from label")
    p.add_argument("--neg-hdf5", default=None, help="null-control clip (no heterodynes)")
    p.add_argument("--neg-reference-layer", default=None,
                   help="force a reference layer for the negative control")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--dpi", type=int, default=400)
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)

    ann_idx = (a.annotation_index if a.annotation_index is not None
               else resolve_positive_index(a.hdf5, a.order))
    pos = build_bundle(a.hdf5, ann_idx, a.order, a.max_k, f"Heterodynes/{a.order}")
    if pos.label_mask.sum() == 0:
        raise SystemExit(f"Heterodynes/{a.order} is empty in {a.hdf5} index {ann_idx}.")

    box = tuple(a.box) if a.box else bbox_of(pos.label_mask, a.pad_px, pos.H, pos.W)
    print(f"Positive: {Path(a.hdf5).name} index {ann_idx} order {a.order}  "
          f"crop(rows {box[0]}:{box[1]}, cols {box[2]}:{box[3]})")
    pos_metrics = make_pipeline_figure(
        pos, a.kernel_size, a.tolerance, box,
        os.path.join(a.output_dir, "metric_pipeline"), a.dpi)
    print(f"  metrics: {pos_metrics}")

    if a.neg_hdf5:
        nidx, ref = resolve_negative(a.neg_hdf5, a.neg_reference_layer)
        print(f"Negative: {Path(a.neg_hdf5).name} index {nidx} ref '{ref}'")
        neg = build_bundle(a.neg_hdf5, nidx, a.order, a.max_k, ref)
        nbox = bbox_of(neg.label_mask, a.pad_px, neg.H, neg.W)
        make_negative_figure(
            neg, ref, a.kernel_size, a.tolerance, nbox, pos_metrics,
            os.path.join(a.output_dir, "negative_control"), a.dpi)


if __name__ == "__main__":
    main()