"""
Quick viewer: render each HDF5 spectrogram with annotation masks overlaid.
No processing — just the raw spectrogram + colored contour overlays.

Usage:
    conda activate spectrace
    python view_hdf5_annotations.py                          # all files
    python view_hdf5_annotations.py --file <path.hdf5>       # single file
    python view_hdf5_annotations.py --limit 5                # first N files
"""

import os
import json
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path


# Distinct colors for annotation classes (skip classes with no pixels)
CLASS_COLORS = {
    "f0_HFC":       "#ff0000",   # red
    "f0_LFC":       "#00ccff",   # cyan
    "harmonics_HFC": "#ff8800",  # orange
    "harmonics_LFC": "#00ff88",  # green
    "Heterodynes/0": "#ffff00",
    "Heterodynes/1": "#ff00ff",
    "Heterodynes/2": "#8800ff",
    "Heterodynes/3": "#00ff00",
    "Heterodynes/4": "#ff4488",
    "Heterodynes/5": "#44ff88",
    "Heterodynes/6": "#8844ff",
    "Heterodynes/7": "#ffaa00",
    "Heterodynes/8": "#00aaff",
    "Heterodynes/9": "#ff0088",
    "Heterodynes/10": "#88ff00",
    "Heterodynes/11": "#0088ff",
    "Heterodynes/12": "#aa00ff",
    "Heterodynes/unsure": "#888888",
    "Subharmonics/subharmonics_HFC": "#ff6666",
    "Subharmonics/subharmonics_LFC": "#6666ff",
    "Cetacean_AdditionalContours/f0_CetaceanAdditionalContours": "#ffcc00",
    "Cetacean_AdditionalContours/harmonics_CetaceanAdditionalContours": "#cc00ff",
    "Cetacean_AdditionalContours/unsure_CetaceanAdditionalContours": "#999999",
    "unsure_HFC": "#cc6666",
    "unsure_LFC": "#6666cc",
    "heterodyne_or_subharmonic_or_other": "#ffaaff",
}

FALLBACK_COLOR = "#ffffff"


def render_hdf5(hdf5_path, output_dir):
    """Render spectrogram + annotation overlays for one HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        spec = f["spectrogram"][:]  # uint8 HxW
        class_names = json.loads(f.attrs["class_names"])
        ann_keys = sorted(f["annotations"].keys(), key=int)

        # Read metadata for axis labels
        meta = f["metadata"]
        sr = meta.attrs.get("sample_rate", None)
        max_freq = meta.attrs.get("max_freq_hz", None)
        duration = meta.attrs.get("duration_sec", None)

        for ann_key in ann_keys:
            masks = f["annotations"][ann_key]["masks"][:]  # CxHxW

            # Find which channels have annotations
            active = [(i, class_names[i]) for i in range(masks.shape[0]) if np.any(masks[i])]

            if not active:
                continue

            # Build figure
            fig, ax = plt.subplots(1, 1, figsize=(max(12, spec.shape[1] / 40), max(5, spec.shape[0] / 80)))

            # Show spectrogram (origin lower so low freq at bottom)
            extent = None
            if duration and max_freq:
                extent = [0, duration, 0, max_freq]

            ax.imshow(spec, aspect="auto", cmap="gray", origin="lower", extent=extent)

            # Overlay each active mask
            for ch_idx, ch_name in active:
                mask = masks[ch_idx]  # HxW, uint8
                color_hex = CLASS_COLORS.get(ch_name, FALLBACK_COLOR)
                rgba = list(to_rgba(color_hex))
                rgba[3] = 0.85  # overlay alpha

                # Create RGBA overlay: transparent where mask==0
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[mask > 0] = rgba

                ax.imshow(overlay, aspect="auto", origin="lower", extent=extent)

            # Legend
            handles = []
            for _, ch_name in active:
                color_hex = CLASS_COLORS.get(ch_name, FALLBACK_COLOR)
                handles.append(plt.Line2D([0], [0], color=color_hex, linewidth=3, label=ch_name))
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)

            # Labels
            basename = Path(hdf5_path).stem
            ax.set_title("%s  (annotation set %s)" % (basename, ann_key), fontsize=9)
            if extent:
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")

            # Save
            out_name = "%s_ann%s.png" % (basename, ann_key)
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
            plt.close(fig)
            print("  -> %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description="Render HDF5 spectrograms with annotation overlays")
    parser.add_argument("--hdf5-dir", default="current_spectrace_data/hdf5_data",
                        help="Directory containing HDF5 files")
    parser.add_argument("--file", default=None, help="Single HDF5 file to render")
    parser.add_argument("--output-dir", default="annotations", help="Output directory for PNGs")
    parser.add_argument("--limit", type=int, default=0, help="Max number of files to process (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.file:
        files = [args.file]
    else:
        files = sorted([
            os.path.join(args.hdf5_dir, f)
            for f in os.listdir(args.hdf5_dir)
            if f.endswith(".hdf5")
        ])

    if args.limit > 0:
        files = files[:args.limit]

    print("Rendering %d HDF5 files -> %s/" % (len(files), args.output_dir))
    for fp in files:
        print(Path(fp).name)
        render_hdf5(fp, args.output_dir)

    print("\nDone. %d files rendered." % len(files))


if __name__ == "__main__":
    main()
