#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge script: called by the GIMP plugin to perform Python 3.11 operations.

Runs in the spectrace conda environment (Python 3.11 + librosa).
Outputs JSON to stdout; status messages go to stderr.

Modes:
    spectrogram          Generate spectrogram PNG from a full WAV file (default, existing behavior)
    parse-callmark       Parse a CallMark Excel export and return vocalization metadata as JSON
    segment-spectrogram  Generate spectrogram PNG for a time-delimited WAV segment (CallMark)

Usage:
    # Normal mode (unchanged)
    python spectrace_wav_bridge.py --wav file.wav --output-dir ./projects --nfft 2048 --grayscale

    # Parse CallMark Excel
    python spectrace_wav_bridge.py --mode parse-callmark --callmark-excel export.xlsx

    # Generate segment spectrogram
    python spectrace_wav_bridge.py --mode segment-spectrogram --wav file.wav --output-dir ./projects \\
        --nfft 2048 --grayscale --offset 0.2067 --duration 0.164 --individual R3277 \\
        --voc-index 0 --callmark-meta '{"onset_sec": 0.2067, "offset_sec": 0.3705, ...}'
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before any other matplotlib import

import sys
import os
import json
import argparse

# Add project root to path so we can import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def mode_spectrogram(args):
    """Original behavior: generate spectrogram for a full WAV file."""
    from utils import process_audio_project

    if not os.path.isfile(args.wav):
        print(json.dumps({"error": "WAV file not found: %s" % args.wav}), file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    audio_info = {
        "clip_path": args.wav,
        "nfft": args.nfft,
        "grayscale": args.grayscale,
    }

    # Redirect status prints to stderr so stdout stays clean for JSON
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = process_audio_project(args.output_dir, audio_info)
    finally:
        sys.stdout = real_stdout

    output = {
        "spectrogram_path": os.path.abspath(result["spectrogram_path"]),
        "project_folder": os.path.abspath(result["project_folder"]),
    }
    print(json.dumps(output))


def mode_parse_callmark(args):
    """Parse a CallMark Excel export and return vocalization metadata as JSON."""
    from callmark_utils import parse_callmark_excel, get_unique_individuals, get_unique_clusternames

    if not os.path.isfile(args.callmark_excel):
        print(json.dumps({"error": "Excel file not found: %s" % args.callmark_excel}), file=sys.stderr)
        sys.exit(1)

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        vocalizations = parse_callmark_excel(args.callmark_excel)
        individuals = get_unique_individuals(vocalizations)
        clusternames = get_unique_clusternames(vocalizations)
    finally:
        sys.stdout = real_stdout

    output = {
        "vocalizations": vocalizations,
        "individuals": individuals,
        "clusternames": clusternames,
        "total_count": len(vocalizations),
    }
    print(json.dumps(output))


def mode_segment_spectrogram(args):
    """Generate spectrogram for a time-delimited WAV segment."""
    from utils import create_callmark_project

    if not os.path.isfile(args.wav):
        print(json.dumps({"error": "WAV file not found: %s" % args.wav}), file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    callmark_meta = json.loads(args.callmark_meta) if args.callmark_meta else {}

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        subfolder = args.subfolder or args.individual or "unknown"
        result = create_callmark_project(
            project_root=args.output_dir,
            clip_path=args.wav,
            subfolder=subfolder,
            voc_index=args.voc_index,
            callmark_meta=callmark_meta,
            nfft=args.nfft,
            grayscale=args.grayscale,
        )
    finally:
        sys.stdout = real_stdout

    output = {
        "spectrogram_path": os.path.abspath(result["spectrogram_path"]),
        "project_folder": os.path.abspath(result["project_folder"]),
    }
    print(json.dumps(output))


def main():
    parser = argparse.ArgumentParser(
        description="Spectrace bridge: spectrogram generation and CallMark parsing"
    )
    parser.add_argument("--mode", default="spectrogram",
                        choices=["spectrogram", "parse-callmark", "segment-spectrogram"],
                        help="Operating mode (default: spectrogram)")

    # Common args
    parser.add_argument("--wav", help="Path to WAV audio file")
    parser.add_argument("--output-dir", help="Project output directory")
    parser.add_argument("--nfft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="Generate grayscale spectrogram")

    # CallMark parse args
    parser.add_argument("--callmark-excel", help="Path to CallMark .xlsx export file")

    # Segment spectrogram args
    parser.add_argument("--offset", type=float, help="Segment start time in seconds")
    parser.add_argument("--duration", type=float, help="Segment duration in seconds")
    parser.add_argument("--individual", help="Individual ID (e.g., R3277)")
    parser.add_argument("--subfolder", help="Subfolder name under clip dir (e.g., R3277, vocal, R3277_vocal)")
    parser.add_argument("--voc-index", type=int, default=0,
                        help="Vocalization index within filtered list")
    parser.add_argument("--callmark-meta", help="JSON string with CallMark metadata for this segment")

    args = parser.parse_args()

    if args.mode == "spectrogram":
        if not args.wav or not args.output_dir:
            parser.error("--wav and --output-dir are required for spectrogram mode")
        mode_spectrogram(args)

    elif args.mode == "parse-callmark":
        if not args.callmark_excel:
            parser.error("--callmark-excel is required for parse-callmark mode")
        mode_parse_callmark(args)

    elif args.mode == "segment-spectrogram":
        if not args.wav or not args.output_dir:
            parser.error("--wav and --output-dir are required for segment-spectrogram mode")
        if not args.subfolder and not args.individual:
            parser.error("--subfolder or --individual is required for segment-spectrogram mode")
        mode_segment_spectrogram(args)


if __name__ == "__main__":
    main()
