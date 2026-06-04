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
    assert img.mode in ("L", "RGBA", "RGB"), \
        f"Unexpected image mode {img.mode}"
    # When grayscale=True, all RGB channels should be equal
    arr = np.array(img.convert("RGB"))
    assert np.allclose(arr[:,:,0], arr[:,:,1], atol=2) and \
           np.allclose(arr[:,:,1], arr[:,:,2], atol=2), \
        f"grayscale=True produced a non-grayscale image for {wav_path.name}"

@pytest.mark.parametrize("wav_path", wav_files, ids=[w.stem for w in wav_files])
def test_spectrogram_dimensions(wav_path, tmp_path):
    audio_dict = {
        "clip_path": str(wav_path),
        "nfft": NFFT,
        "grayscale": True,
    }
    result = process_audio_project(str(tmp_path), audio_dict)
    png = next(tmp_path.rglob("*_spectrogram.png"))
    img = np.array(Image.open(png))

    # Height should always be nfft//2 + 1 frequency bins
    expected_height = NFFT // 2 + 1
    assert img.shape[0] == expected_height, \
        f"Height mismatch for {wav_path.name}: got {img.shape[0]}, expected {expected_height}"

    # Width: the PNG pixel width should be proportional to the number of
    # STFT frames reported by the function — allow for a 2x render scale
    reported_frames = result["spectrogram_shape"][1]
    scale = img.shape[1] / reported_frames
    assert 0.9 <= scale <= 2.1, \
        f"PNG width {img.shape[1]} is not proportional to " \
        f"reported frames {reported_frames} (scale={scale:.2f}) for {wav_path.name}"

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