"""
CallMark export file parsing and filtering utilities.

Reads CallMark annotation exports (.xlsx) and converts them into
a list of vocalization dicts with onset/offset in seconds, ready
for Spectrace segment extraction.

CallMark column format (from ZFVocalizations.xlsx):
    onset, offset        — spectrogram column indices (NOT seconds)
    minFrequency         — Hz
    maxFrequency         — Hz
    species              — e.g. "zebrafinch"
    individual           — e.g. "R3277"
    clustername          — e.g. "vocal"
    filename             — e.g. "ZF.wav"
    channelIndex         — audio channel (0 = mono)
    age                  — days post-hatch
    category             — "Adults" or "Juveniles"
    publication info     — citation string

Time conversion: T_k = (k * h + n/2) / sr
    where n = FFT size (default 256), h = hop size (default 28), sr = sample rate (default 44100)
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def callmark_index_to_seconds(k: int, n: int = 256, h: int = 28, sr: int = 44100) -> float:
    """
    Convert a CallMark spectrogram column index to time in seconds.

    Formula from CallMark documentation:
        T_k = (k * h + n/2) / sr

    Args:
        k:  Spectrogram column index from CallMark export.
        n:  FFT window size used by CallMark (default 256).
        h:  Hop size used by CallMark (default 28).
        sr: Sample rate in Hz (default 44100).

    Returns:
        Time in seconds.
    """
    return (k * h + n / 2) / sr


def parse_callmark_excel(
    excel_path: str,
    callmark_n: int = 256,
    callmark_h: int = 28,
    callmark_sr: int = 44100,
) -> List[Dict]:
    """
    Parse a CallMark Excel export file into a list of vocalization dicts.

    Each dict contains the original columns plus computed onset_sec, offset_sec,
    and duration_sec fields.

    Args:
        excel_path:   Path to .xlsx file.
        callmark_n:   FFT size used by CallMark for column index conversion.
        callmark_h:   Hop size used by CallMark for column index conversion.
        callmark_sr:  Sample rate used by CallMark for column index conversion.

    Returns:
        List of dicts sorted by onset, each with keys:
            index, onset, offset, onset_sec, offset_sec, duration_sec,
            min_frequency, max_frequency, species, individual, clustername,
            filename, channel_index, age, category, publication_info
    """
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Normalize column names: strip whitespace, lowercase
    df.columns = [c.strip() for c in df.columns]

    # Map expected columns (handle both camelCase from Excel and snake_case)
    column_map = {
        "onset": "onset",
        "offset": "offset",
        "minFrequency": "min_frequency",
        "minfrequency": "min_frequency",
        "maxFrequency": "max_frequency",
        "maxfrequency": "max_frequency",
        "species": "species",
        "individual": "individual",
        "clustername": "clustername",
        "filename": "filename",
        "channelIndex": "channel_index",
        "channelindex": "channel_index",
        "age": "age",
        "category": "category",
        "publication info": "publication_info",
    }

    renamed = {}
    for col in df.columns:
        key = col.strip()
        if key in column_map:
            renamed[col] = column_map[key]
    df = df.rename(columns=renamed)

    # Sort by onset
    df = df.sort_values("onset").reset_index(drop=True)

    vocalizations = []
    for i, row in df.iterrows():
        onset = int(row["onset"])
        offset = int(row["offset"])
        onset_sec = callmark_index_to_seconds(onset, callmark_n, callmark_h, callmark_sr)
        offset_sec = callmark_index_to_seconds(offset, callmark_n, callmark_h, callmark_sr)

        vocalizations.append({
            "index": i,
            "onset": onset,
            "offset": offset,
            "onset_sec": round(onset_sec, 6),
            "offset_sec": round(offset_sec, 6),
            "duration_sec": round(offset_sec - onset_sec, 6),
            "min_frequency": int(row.get("min_frequency", 0)),
            "max_frequency": int(row.get("max_frequency", 0)),
            "species": str(row.get("species", "")),
            "individual": str(row.get("individual", "")),
            "clustername": str(row.get("clustername", "")),
            "filename": str(row.get("filename", "")),
            "channel_index": int(row.get("channel_index", 0)),
            "age": int(row.get("age", 0)),
            "category": str(row.get("category", "")),
            "publication_info": str(row.get("publication_info", "")),
        })

    return vocalizations


def get_unique_individuals(vocalizations: List[Dict]) -> List[str]:
    """Return sorted list of unique individual IDs from vocalizations."""
    individuals = sorted(set(v["individual"] for v in vocalizations if v["individual"]))
    return individuals


def filter_vocalizations(vocalizations: List[Dict], individual: str = "All") -> List[Dict]:
    """
    Filter vocalization list by individual.

    Args:
        vocalizations: List of vocalization dicts from parse_callmark_excel().
        individual:    Individual ID to filter by, or "All" for no filtering.

    Returns:
        Filtered list sorted by onset_sec.
    """
    if individual == "All":
        return list(vocalizations)
    return [v for v in vocalizations if v["individual"] == individual]


def add_padding(
    onset_sec: float,
    offset_sec: float,
    padding_sec: float = 0.05,
    wav_duration_sec: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Add padding around a vocalization segment for visual context.

    Args:
        onset_sec:        Segment start time in seconds.
        offset_sec:       Segment end time in seconds.
        padding_sec:      Padding to add on each side (default 50ms).
        wav_duration_sec: Total WAV duration for clamping. None = no upper clamp.

    Returns:
        (padded_onset_sec, padded_offset_sec) clamped to [0, wav_duration_sec].
    """
    padded_onset = max(0.0, onset_sec - padding_sec)
    padded_offset = offset_sec + padding_sec
    if wav_duration_sec is not None:
        padded_offset = min(padded_offset, wav_duration_sec)
    return padded_onset, padded_offset


def build_callmark_manifest(
    vocalizations: List[Dict],
    excel_path: str,
    wav_path: str,
    individual_filter: str = "All",
) -> Dict:
    """
    Build a manifest dict that records the CallMark import session.

    Saved as callmark_manifest.json in the recording-level project folder
    for traceability back to the original CallMark export.
    """
    return {
        "callmark_excel": str(excel_path),
        "wav_file": str(wav_path),
        "individual_filter": individual_filter,
        "total_vocalizations": len(vocalizations),
        "individuals": get_unique_individuals(vocalizations),
        "vocalizations": vocalizations,
    }
