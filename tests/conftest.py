from pathlib import Path
from collections import Counter
import pytest

TESTS_DIR = Path(__file__).parent
DATA_DIR  = TESTS_DIR / "data"
HDF5_DIR  = DATA_DIR / "hdf5_data"
XCF_DIR   = DATA_DIR / "xcf_project_data"

# All HDF5 files in the dataset
hdf5_files = sorted(HDF5_DIR.glob("*.hdf5"))

# One WAV per clip — found inside the _0 folder for each clip
wav_files = [
    wav
    for folder in sorted(XCF_DIR.iterdir())
    if folder.is_dir() and folder.name.endswith("_0")
    for wav in folder.glob("*.wav")
]

# XCF project folders — all of them, every index
xcf_project_dirs = [f for f in sorted(XCF_DIR.iterdir()) if f.is_dir()]

# Clips that have more than one annotation pass
def _clip_stem(folder_name):
    return folder_name.rsplit("_", 1)[0]

_pass_counts = Counter(_clip_stem(f.name) for f in XCF_DIR.iterdir() if f.is_dir())
multi_pass_hdf5 = [
    HDF5_DIR / f"{stem}.hdf5"
    for stem, count in _pass_counts.items()
    if count > 1 and (HDF5_DIR / f"{stem}.hdf5").exists()
]

def pytest_collection_modifyitems(config, items):
    """Skip all tests if LFS files have not been pulled."""
    lfs_missing = not any(HDF5_DIR.glob("*.hdf5"))
    if lfs_missing:
        skip = pytest.mark.skip(
            reason="Test data not found — run `git lfs pull` to download."
        )
        for item in items:
            item.add_marker(skip)