#!/usr/bin/env python3
"""
make_publication_figure.py
===========================

Build a publication / poster quality figure for the Spectrace orca-annotation
dataset.

Layout
------
    Row 1 (large):   [A] clean spectrogram   |   [B] spectrogram + all traces
    Row 2 (small):   [C] index 0  [D] index 1  [E] index 2   (same clip,
                     each project's own annotation set overlaid)
    Shared legend on the right.

It REUSES your existing mask-extraction code from utils.py
(`extract_layers_from_xcf`) so masks are read exactly as in your proof of
concept, but it REPLACES all rendering with vector-friendly, print-grade code
and recomputes the spectrogram from each WAV for a crisp, axis-correct image.

Outputs (into --output-dir):
    <clip_basename>.pdf   <- vector, for the manuscript
    <clip_basename>.svg   <- vector, hand-editable in Illustrator / Inkscape
    <clip_basename>.png   <- high-DPI raster, for posters

Run
---
    python make_publication_figure.py \
        --project-folder tests/data/xcf_project_data \
        --clip-basename "2023-12-03--10-15-30--00-15-00--Ct-Dt--03-52--D" \
        --template templates/orca_template.xcf \
        --output-dir figures

All knobs have sensible defaults; see `build_config()` / argparse below.
"""

from __future__ import annotations

import argparse

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for headless / scripted runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from PIL import Image
from scipy.ndimage import binary_dilation
# NOTE: librosa is intentionally NOT imported -- we display the saved PNG the
# masks were drawn on, rather than recomputing the STFT (which de-registered
# the masks from the image). Axes are calibrated from metadata instead.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils import (
    extract_layers_from_xcf,
    find_all_project_indices,
    load_metadata,
    get_or_create_color_mapping,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FigureConfig:
    project_folder: str
    clip_basename: str
    output_dir: str = "figures"
    layer_group_name: str = "OrcinusOrca_FrequencyContours"
    template_xcf_path: Optional[str] = None

    # spectrogram rendering
    spectrogram_cmap: str = "gray"          # config flag; default grayscale
    db_floor: float = -80.0                 # dynamic-range floor (dB below max)
    fmax_hz: Optional[float] = None         # None -> Nyquist (full 0-50 kHz)

    # trace rendering
    mask_alpha: float = 0.95
    dilate_px: int = 1                      # thicken 1px contours for print
    legend_label_overrides: Dict[str, str] = field(default_factory=dict)

    # which indices to show in the small-multiple strip (None -> auto first 3)
    strip_indices: Optional[List[int]] = None
    strip_max: int = 3

    # output
    dpi_png: int = 400
    figure_width_in: float = 16.0
    panel_letter_size: int = 16

    # a semantically ordered palette (overrides auto color mapping where names
    # match). Edit freely. Anything not listed falls back to the auto mapping.
    palette_overrides: Dict[str, str] = field(default_factory=lambda: {
        # fundamental frequencies — strong, distinct
        "f0_LFC":        "#1f78ff",   # blue
        "f0_HFC":        "#6a3d9a",   # purple
        # harmonic stacks — warm
        "harmonics_LFC": "#e31a8c",   # magenta-pink
        "harmonics_HFC": "#ff7f00",   # orange
        # heterodyne series — a perceptual yellow->green->cyan gradient (0..6)
        "Heterodynes/0": "#fde725",
        "Heterodynes/1": "#a0da39",
        "Heterodynes/2": "#4ac16d",
        "Heterodynes/3": "#1fa187",
        "Heterodynes/4": "#277f8e",
        "Heterodynes/5": "#365c8d",
        "Heterodynes/6": "#46327e",
    })


# =============================================================================
# Spectrogram (recomputed from WAV for sharpness + correct axes)
# =============================================================================

def _find_file(project_path: str, suffix: str) -> Optional[str]:
    for f in sorted(os.listdir(project_path)):
        if f.endswith(suffix):
            return os.path.join(project_path, f)
    return None


def merge_layers_across_projects(
    project_paths: List[str], cfg: FigureConfig, target_hw: Tuple[int, int]
) -> dict:
    """
    Combine annotation layers from several projects into ONE layer dict for a
    single overlay. Masks sharing a class name are OR-ed together so panel B
    shows the union of every annotation set drawn on the clip.

    All masks are forced onto `target_hw` (the reference PNG grid) so the
    combined overlay registers exactly with the displayed spectrogram.
    """
    H, W = target_hw
    merged: dict = {}
    for pp in project_paths:
        try:
            layers = extract_layers_for_project(pp, cfg)
        except FileNotFoundError:
            continue
        for name, info in layers.items():
            mask = info.get("mask")
            if mask is None or mask.sum() == 0:
                continue
            if mask.shape != (H, W):  # crop/pad onto reference grid
                fixed = np.zeros((H, W), dtype=mask.dtype)
                h = min(H, mask.shape[0]); w = min(W, mask.shape[1])
                fixed[:h, :w] = mask[:h, :w]
                mask = fixed
            if name in merged:
                merged[name]["mask"] = (
                    merged[name]["mask"].astype(bool) | mask.astype(bool)
                ).astype(mask.dtype)
            else:
                merged[name] = {"mask": mask.copy(), "visible": True,
                                "size": (W, H)}
    return merged


def extract_layers_for_project(project_path: str, cfg: FigureConfig) -> dict:
    """
    Resolve the .xcf file *inside* a project folder, then call the user's
    extractor. `extract_layers_from_xcf` expects a path to the .xcf FILE,
    not the containing directory.
    """
    xcf = _find_file(project_path, ".xcf")
    if xcf is None:
        raise FileNotFoundError(f"No .xcf found in {project_path}")
    return extract_layers_from_xcf(
        xcf, cfg.layer_group_name, verbose=False)["layers"]


def load_spectrogram(project_path: str, cfg: FigureConfig
                     ) -> Tuple[np.ndarray, float, float, dict]:
    """
    Load the *saved* spectrogram PNG -- the exact image the masks were drawn
    on -- so the mask grid and image grid are identical by construction.

    This is the only safe choice: the masks live in the PNG's pixel
    coordinates. Recomputing the STFT yields a different number of columns
    (matplotlib's tight-bbox crop changes the width), which de-registers the
    masks from the image and is what produced the smeared / mis-spanned
    traces. We derive the time/frequency axes from metadata instead.

    Returns (img [H,W], duration_sec, nyquist_hz, metadata).
    """
    meta = load_metadata(project_path)

    png = _find_file(project_path, "_spectrogram.png")
    if png is None:
        raise FileNotFoundError(f"No *_spectrogram.png found in {project_path}")

    img = np.array(Image.open(png).convert("L"))  # grayscale, row0 = top
    H, W = img.shape

    # --- axis calibration from metadata --------------------------------------
    sr = float(meta.get("sample_rate", 96000))
    nyquist = sr / 2.0
    # time_per_pixel is seconds per STFT frame (column). Prefer the stored
    # value; fall back to deriving it from nfft/sr.
    tpp = meta.get("time_per_pixel")
    if tpp is None:
        nfft = int(meta.get("nfft", 2048))
        noverlap = int(meta.get("noverlap", nfft // 2))
        hop = nfft - noverlap
        tpp = hop / sr
    duration = float(tpp) * W

    return img, duration, nyquist, meta


# =============================================================================
# Mask compositing  (predictable overlaps, crisp thin traces)
# =============================================================================

def composite_masks(
    layers: Dict[str, dict],
    target_hw: Tuple[int, int],
    color_mapping: Dict[str, str],
    cfg: FigureConfig,
    only_layers: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Composite all (visible, non-empty) masks into ONE RGBA array.

    Painting order = color_mapping insertion order, so later classes sit on
    top. We resolve overlaps by last-writer-wins on a per-pixel basis rather
    than stacking translucent imshows (which muddies crossings).

    Returns (rgba [H,W,4] float 0-1, drawn_layer_names).
    """
    H, W = target_hw
    rgba = np.zeros((H, W, 4), dtype=float)
    drawn: List[str] = []

    names = list(layers.keys())
    if only_layers is not None:
        names = [n for n in names if n in only_layers]
    # paint in palette order for stable z-stacking
    names.sort(key=lambda n: list(color_mapping.keys()).index(n)
               if n in color_mapping else 1e9)

    for name in names:
        info = layers[name]
        if not info.get("visible", True):
            continue
        mask = info["mask"]
        if mask.sum() == 0:
            continue

        # The mask and the PNG come from the same project and SHOULD share a
        # grid. If they don't, something is off (wrong PNG, edited XCF) -- warn
        # loudly instead of silently stretching, which is what smeared the
        # traces before. As a last resort we crop/pad rather than rescale.
        if mask.shape != (H, W):
            print(f"  ⚠️  mask '{name}' shape {mask.shape} != image {(H, W)}; "
                  f"cropping/padding (check that this XCF matches its PNG)")
            fixed = np.zeros((H, W), dtype=mask.dtype)
            h = min(H, mask.shape[0]); w = min(W, mask.shape[1])
            fixed[:h, :w] = mask[:h, :w]
            mask = fixed

        if cfg.dilate_px > 0:
            mask = binary_dilation(mask, iterations=cfg.dilate_px)

        m = mask.astype(bool)
        rgb = np.array(to_rgb(color_mapping.get(name, "#ffffff")))
        rgba[m, :3] = rgb
        rgba[m, 3] = cfg.mask_alpha
        drawn.append(name)

    return rgba, drawn


# =============================================================================
# Panel drawing
# =============================================================================

def _draw_spectrogram(ax, img, duration, fmax, cfg, vmax_freq):
    # PNG row 0 = top = HIGH frequency, so origin='upper' with a 0->fmax
    # extent makes the displayed axis read low (bottom) to high (top) while
    # keeping pixel rows aligned with the masks.
    extent = [0, duration, 0, fmax]
    ax.imshow(img, aspect="auto", origin="upper", cmap=cfg.spectrogram_cmap,
              extent=extent, interpolation="nearest")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, vmax_freq)


def _draw_overlay(ax, rgba, duration, fmax, vmax_freq):
    ax.imshow(rgba, aspect="auto", origin="upper",
              extent=[0, duration, 0, fmax], interpolation="nearest")
    ax.set_ylim(0, vmax_freq)


def _panel_letter(ax, letter, size):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="bottom", ha="right")


def _style_axis(ax, xlabel=True, ylabel=True):
    ax.tick_params(labelsize=10)
    if xlabel:
        ax.set_xlabel("Time (s)", fontsize=11)
    if ylabel:
        ax.set_ylabel("Frequency (kHz)", fontsize=11)
    # show frequency in kHz for readability
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =============================================================================
# Main figure assembly
# =============================================================================

def build_figure(cfg: FigureConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    indices = find_all_project_indices(cfg.project_folder, cfg.clip_basename)
    if not indices:
        raise SystemExit(f"No projects found for clip '{cfg.clip_basename}'")
    strip = cfg.strip_indices or indices[: cfg.strip_max]
    primary_idx = strip[0]

    # color mapping: auto-discover, then apply palette overrides
    color_mapping = get_or_create_color_mapping(
        cfg.project_folder, cfg.layer_group_name, cfg.template_xcf_path)
    color_mapping = dict(color_mapping)  # copy
    for k, v in cfg.palette_overrides.items():
        if k in color_mapping:
            color_mapping[k] = v

    def project_path(idx):
        return os.path.join(cfg.project_folder,
                            f"{cfg.clip_basename}_{idx}")

    # --- primary spectrogram (the SINGLE reference image for ALL panels) ----
    img, duration, nyquist, meta = load_spectrogram(
        project_path(primary_idx), cfg)
    fmax = nyquist
    vmax_freq = cfg.fmax_hz or nyquist  # full 0-50 kHz by default
    H, W = img.shape

    # Panel B = union of masks from EVERY project index, on the reference grid.
    all_paths = [project_path(i) for i in indices]
    merged_layers = merge_layers_across_projects(all_paths, cfg, (H, W))
    primary_rgba, drawn = composite_masks(
        merged_layers, (H, W), color_mapping, cfg)

    # ------------------------------------------------------------------ layout
    # Every panel is placed manually at an explicit rectangle so ALL FIVE are
    # exactly the same size and aspect. Spanning-gridspec couples panel width
    # to column count (which is what made B wider than A); explicit rectangles
    # avoid that entirely.
    #
    # Bottom row: 3 panels. Top row: 2 panels, centered over the bottom 3.
    n_bottom = len(strip)

    fig = plt.figure(figsize=(cfg.figure_width_in,
                              cfg.figure_width_in * 0.78))

    # ---- horizontal geometry (figure fraction) ----
    # Legend now lives INSIDE the grid (empty third top cell), so panels can
    # use nearly the full width.
    fig_left, fig_right = 0.055, 0.965
    gap_x = 0.035                          # horizontal gap between panels
    avail_w = fig_right - fig_left
    panel_w = (avail_w - (n_bottom - 1) * gap_x) / n_bottom

    # ---- vertical geometry (figure fraction) ----
    top_edge, bottom_edge = 0.93, 0.075
    gap_y = 0.135                          # vertical gap between the two rows
    # both rows get panels of identical HEIGHT
    total_h = top_edge - bottom_edge
    panel_h = (total_h - gap_y) / 2.0

    row_top_b = top_edge - panel_h          # bottom of the TOP row
    row_bot_b = bottom_edge                 # bottom of the BOTTOM row

    def bottom_left_x(col):
        return fig_left + col * (panel_w + gap_x)

    # Top-row x positions: LEFT-ALIGN over the first two bottom panels, so
    # A sits above C and B above D. The third bottom column (above E) is left
    # empty for the legend.
    top_xs = [bottom_left_x(0), bottom_left_x(1)]
    third_col_x = bottom_left_x(2)   # legend goes here, in the top row

    ax_before = fig.add_axes([top_xs[0], row_top_b, panel_w, panel_h])
    ax_after = fig.add_axes([top_xs[1], row_top_b, panel_w, panel_h])

    _draw_spectrogram(ax_before, img, duration, fmax, cfg, vmax_freq)
    _style_axis(ax_before)
    ax_before.set_title("Spectrogram", fontsize=12)
    _panel_letter(ax_before, "A", cfg.panel_letter_size)

    _draw_spectrogram(ax_after, img, duration, fmax, cfg, vmax_freq)
    _draw_overlay(ax_after, primary_rgba, duration, fmax, vmax_freq)
    _style_axis(ax_after, ylabel=False)
    ax_after.set_title("Annotated frequency contours", fontsize=12)
    _panel_letter(ax_after, "B", cfg.panel_letter_size)

    # Bottom strip: one panel per project index. Every panel reuses the SAME
    # reference image and the SAME extent as panels A/B, so all five panels are
    # pixel-for-pixel identical in dimension. Only the overlaid masks differ.
    strip_letters = list("CDEFGH")
    for col, idx in enumerate(strip):
        ax = fig.add_axes([bottom_left_x(col), row_bot_b, panel_w, panel_h])
        lyr = extract_layers_for_project(project_path(idx), cfg)
        rgba, _ = composite_masks(lyr, (H, W), color_mapping, cfg)
        _draw_spectrogram(ax, img, duration, fmax, cfg, vmax_freq)
        _draw_overlay(ax, rgba, duration, fmax, vmax_freq)
        _style_axis(ax, ylabel=(col == 0))
        ax.set_title(f"Annotation index {idx}", fontsize=10)
        _panel_letter(ax, strip_letters[col], cfg.panel_letter_size - 2)

    # ------------------------------------------------------------------ legend
    # Required order: f0_LFC, harmonics_LFC, f0_HFC, harmonics_HFC,
    # then Heterodynes/0..N ascending, then any other classes that were drawn.
    drawn_set = set(drawn)

    def heterodyne_key(name):
        # sort 'Heterodynes/<n>' numerically; non-numeric tails go last
        tail = name.split("/", 1)[1] if "/" in name else ""
        try:
            return (0, int(tail))
        except ValueError:
            return (1, tail)

    preferred = ["f0_LFC", "harmonics_LFC", "f0_HFC", "harmonics_HFC"]
    heterodynes = sorted([n for n in drawn_set if n.startswith("Heterodynes")],
                         key=heterodyne_key)
    accounted = set(preferred) | set(heterodynes)
    others = sorted(n for n in drawn_set if n not in accounted)

    legend_order = ([n for n in preferred if n in drawn_set]
                    + heterodynes + others)

    handles = []
    for name in legend_order:
        label = cfg.legend_label_overrides.get(name, name)  # one line; wrap in Inkscape
        handles.append(mpatches.Patch(color=color_mapping.get(name, "#ffffff"),
                                      label=label))
    if handles:
        # Anchor the legend inside the empty third top-row cell (above panel E).
        legend_cx = third_col_x + panel_w / 2.0
        legend_cy = row_top_b + panel_h / 2.0
        fig.legend(handles=handles, loc="center",
                   bbox_to_anchor=(legend_cx, legend_cy),
                   frameon=True, framealpha=0.95, fontsize=10,
                   title="Contour class", title_fontsize=11,
                   borderaxespad=0.0)

    # ------------------------------------------------------------------ save
    base = os.path.join(cfg.output_dir, cfg.clip_basename)
    fig.savefig(base + ".pdf", bbox_inches="tight")          # vector
    fig.savefig(base + ".svg", bbox_inches="tight")          # editable vector
    fig.savefig(base + ".png", dpi=cfg.dpi_png, bbox_inches="tight")  # poster
    plt.close(fig)

    print("Wrote:")
    for ext in ("pdf", "svg", "png"):
        print(f"  {base}.{ext}")
    print(f"Indices shown in strip: {strip}")
    print(f"Classes drawn: {drawn}")


# =============================================================================
# CLI
# =============================================================================

def build_config() -> FigureConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-folder", required=True)
    p.add_argument("--clip-basename", required=True)
    p.add_argument("--template", default=None,
                   help="master template XCF (optional)")
    p.add_argument("--output-dir", default="../figures")
    p.add_argument("--layer-group", default="OrcinusOrca_FrequencyContours")
    p.add_argument("--cmap", default="gray",
                   help="spectrogram colormap (gray, magma, inferno, viridis)")
    p.add_argument("--db-floor", type=float, default=-80.0)
    p.add_argument("--fmax", type=float, default=None,
                   help="max frequency Hz to display (default: Nyquist)")
    p.add_argument("--dilate", type=int, default=1,
                   help="px to thicken masks (0 to disable)")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--strip-indices", type=int, nargs="*", default=None)
    a = p.parse_args()

    return FigureConfig(
        project_folder=a.project_folder,
        clip_basename=a.clip_basename,
        template_xcf_path=a.template,
        output_dir=a.output_dir,
        layer_group_name=a.layer_group,
        spectrogram_cmap=a.cmap,
        db_floor=a.db_floor,
        fmax_hz=a.fmax,
        dilate_px=a.dilate,
        dpi_png=a.dpi,
        strip_indices=a.strip_indices,
    )


if __name__ == "__main__":
    build_figure(build_config())