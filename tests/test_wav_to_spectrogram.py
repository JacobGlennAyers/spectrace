import sys
import numpy as np
import pytest
import librosa
from PIL import Image
import pandas as pd
from pathlib import Path

# Make sure spectrace root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import process_audio_project
from conftest import wav_files

NFFT     = 2048
NOVERLAP = 1024

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_spectrogram_png_exists(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    process_audio_project(str(tmp_path), audio_dict)
    pngs = list(tmp_path.rglob("*_spectrogram.png"))
    assert len(pngs) == 1, f"Expected 1 spectrogram PNG, got {len(pngs)}"
    img = Image.open(pngs[0])
    assert img.mode == "L", f"Expected grayscale image, got mode {img.mode}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_spectrogram_dimensions(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    process_audio_project(str(tmp_path), audio_dict)
    png = next(tmp_path.rglob("*_spectrogram.png"))
    y, sr           = librosa.load(wav_path, sr=None)
    hop             = NFFT - NOVERLAP
    expected_height = NFFT // 2 + 1
    expected_width  = (len(y) - NFFT) // hop + 1
    img = np.array(Image.open(png))
    assert img.shape[0] == expected_height, \
        f"Height mismatch for {wav_path.name}: got {img.shape[0]}, expected {expected_height}"
    assert abs(img.shape[1] - expected_width) <= 2, \
        f"Width mismatch for {wav_path.name}: got {img.shape[1]}, expected ~{expected_width}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_spectrogram_pixel_values(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    process_audio_project(str(tmp_path), audio_dict)
    arr = np.array(Image.open(next(tmp_path.rglob("*_spectrogram.png"))))
    assert arr.max()  >  10,  f"Spectrogram looks all-black for {wav_path.name}"
    assert arr.min()  < 245,  f"Spectrogram looks all-white for {wav_path.name}"
    assert 20 < arr.mean() < 220, \
        f"Suspicious mean pixel value {arr.mean():.1f} for {wav_path.name}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_metadata_files_created(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    process_audio_project(str(tmp_path), audio_dict)
    assert any(tmp_path.rglob("metadata.pkl")), \
        f"No metadata.pkl created for {wav_path.name}"
    assert any(tmp_path.rglob("metadata.csv")), \
        f"No metadata.csv created for {wav_path.name}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_metadata_values_consistent_with_wav(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    result = process_audio_project(str(tmp_path), audio_dict)
    y, sr = librosa.load(wav_path, sr=None)
    assert result["sample_rate"] == sr, \
        f"Sample rate mismatch for {wav_path.name}"
    assert result["nfft"]     == NFFT
    assert result["noverlap"] == NOVERLAP
    assert abs(result["spectrogram_shape"][0] - (NFFT // 2 + 1)) <= 1, \
        f"Unexpected spectrogram height for {wav_path.name}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_wav_copied_to_project_folder(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    result = process_audio_project(str(tmp_path), audio_dict)
    copied = Path(result["copied_audio_path"])
    assert copied.exists(), \
        f"WAV not copied to project folder for {wav_path.name}"