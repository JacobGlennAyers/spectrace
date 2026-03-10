#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge script: called by the GIMP plugin to generate a spectrogram from a WAV file.

Runs in the spectrace conda environment (Python 3.11 + librosa).
Outputs JSON to stdout with the spectrogram path for the GIMP plugin to load.

Usage:
    python spectrace_wav_bridge.py --wav /path/to/file.wav --output-dir /path/to/projects --nfft 2048 --grayscale
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

from utils import process_audio_project


def main():
    parser = argparse.ArgumentParser(description="Generate spectrogram PNG from WAV file")
    parser.add_argument("--wav", required=True, help="Path to the WAV audio file")
    parser.add_argument("--output-dir", required=True, help="Project output directory")
    parser.add_argument("--nfft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="Generate grayscale spectrogram")
    args = parser.parse_args()

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

    # Output result as JSON on stdout
    output = {
        "spectrogram_path": os.path.abspath(result["spectrogram_path"]),
        "project_folder": os.path.abspath(result["project_folder"]),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
