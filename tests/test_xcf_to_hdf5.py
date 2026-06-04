import sys
import h5py
import json
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hdf5_utils import XCFToHDF5Converter, HDF5SpectrogramLoader
from conftest import hdf5_files, multi_pass_hdf5, XCF_DIR, HDF5_DIR, DATA_DIR

# timing_drift is set outside spectrace after the fact via set_annotation_attrs
# and is not part of the conversion — exclude it from all structural checks
ROOT_ATTRS_TO_IGNORE = {"timing_drift"}

TEMPLATE_XCF = Path(__file__).parent.parent / "templates" / "orca_template.xcf"

EXPECTED_ROOT_ATTRS = {
    "class_names", "num_annotations", "num_classes",
    "registry_version", "schema_version",
}

EXPECTED_CLASSES = [
    "f0_LFC", "f0_HFC",
    "harmonics_LFC", "harmonics_HFC",
    "unsure_LFC", "unsure_HFC",
    "Subharmonics/subharmonics_LFC", "Subharmonics/subharmonics_HFC",
    "heterodyne_or_subharmonic_or_other",
    "Heterodynes/0", "Heterodynes/1",
    "Cetacean_AdditionalContours/f0_CetaceanAdditionalContours",
]

EXPECTED_METADATA_KEYS = {
    "sample_rate", "nfft", "noverlap", "duration_sec",
    "freq_per_pixel", "time_per_pixel", "max_freq_hz", "audio_path",
}


# ---------------------------------------------------------------------------
# Structural tests — run against the committed HDF5 files in tests/data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_required_keys(hdf5_path):
    with h5py.File(hdf5_path) as f:
        assert "spectrogram"  in f, "Missing spectrogram"
        assert "annotations"  in f, "Missing annotations group"
        assert "metadata"     in f, "Missing metadata group"
        assert "audio_wav"    in f, "Missing embedded audio_wav"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_root_attributes(hdf5_path):
    with h5py.File(hdf5_path) as f:
        present = set(f.attrs.keys()) - ROOT_ATTRS_TO_IGNORE
        missing = EXPECTED_ROOT_ATTRS - present
        assert not missing, f"Missing root attributes: {missing}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_schema_version(hdf5_path):
    with h5py.File(hdf5_path) as f:
        assert str(f.attrs["schema_version"]) == "2.0"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_class_names(hdf5_path):
    with h5py.File(hdf5_path) as f:
        class_names = json.loads(f.attrs["class_names"])
        assert len(class_names) == 26, \
            f"Expected 26 classes, got {len(class_names)} in {hdf5_path.name}"
        for c in EXPECTED_CLASSES:
            assert c in class_names, \
                f"Missing class '{c}' in {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_num_annotations_consistent(hdf5_path):
    with h5py.File(hdf5_path) as f:
        declared = int(f.attrs["num_annotations"])
        actual   = len(list(f["annotations"].keys()))
        assert declared == actual, \
            f"{hdf5_path.name}: num_annotations={declared} but {actual} groups found"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_spectrogram_shape(hdf5_path):
    with h5py.File(hdf5_path) as f:
        spec = f["spectrogram"][:]
        assert spec.ndim == 2, "Spectrogram should be 2D (H, W)"
        assert spec.shape[0] == 1025, \
            f"Expected height 1025 (nfft//2+1), got {spec.shape[0]} in {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_masks_shape_and_binary(hdf5_path):
    with h5py.File(hdf5_path) as f:
        spec = f["spectrogram"][:]
        for idx in f["annotations"].keys():
            masks = f[f"annotations/{idx}/masks"][:]
            assert masks.ndim == 3, \
                f"Masks should be 3D (C,H,W) — annotation {idx} in {hdf5_path.name}"
            assert masks.shape == (26, spec.shape[0], spec.shape[1]), \
                f"Mask shape {masks.shape} != (26, {spec.shape[0]}, {spec.shape[1]}) in {hdf5_path.name}"
            unique = set(np.unique(masks))
            assert unique.issubset({0, 1}), \
                f"Non-binary values {unique} in annotation {idx} of {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_metadata_keys(hdf5_path):
    with h5py.File(hdf5_path) as f:
        present = set(f["metadata"].attrs.keys())
        missing = EXPECTED_METADATA_KEYS - present
        assert not missing, \
            f"Missing metadata keys {missing} in {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_metadata_values_sensible(hdf5_path):
    with h5py.File(hdf5_path) as f:
        m = dict(f["metadata"].attrs)
        assert int(m["nfft"])        == 2048,    f"Unexpected nfft in {hdf5_path.name}"
        assert int(m["noverlap"])    == 1024,    f"Unexpected noverlap in {hdf5_path.name}"
        assert int(m["sample_rate"]) in {102400, 10000} ,  f"Unexpected sample_rate in {hdf5_path.name}"
        assert float(m["max_freq_hz"]) == 51200.0
        assert float(m["duration_sec"]) > 0

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_annotation_has_notes_and_timing_drift_attrs(hdf5_path):
    """Every annotation group should have notes and timing_drift attributes."""
    with h5py.File(hdf5_path) as f:
        for idx in f["annotations"].keys():
            grp = f[f"annotations/{idx}"]
            assert "notes"        in grp.attrs, \
                f"Missing 'notes' attr on annotation {idx} in {hdf5_path.name}"
            assert "timing_drift" in grp.attrs, \
                f"Missing 'timing_drift' attr on annotation {idx} in {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", hdf5_files, ids=[f.stem for f in hdf5_files])
def test_hdf5_audio_wav_embedded(hdf5_path):
    with HDF5SpectrogramLoader(str(hdf5_path)) as loader:
        assert loader.has_audio(), \
            f"No embedded WAV in {hdf5_path.name}"

@pytest.mark.parametrize("hdf5_path", multi_pass_hdf5,
                         ids=[f.stem for f in multi_pass_hdf5])
def test_hdf5_multi_pass_all_indices_present(hdf5_path):
    with h5py.File(hdf5_path) as f:
        n = int(f.attrs["num_annotations"])
        assert n > 1, \
            f"Expected >1 annotation passes in {hdf5_path.name}, got {n}"
        for i in range(n):
            assert f"annotations/{i}/masks" in f, \
                f"Missing annotations/{i}/masks in {hdf5_path.name}"


# ---------------------------------------------------------------------------
# Conversion test — re-run xcf_to_hdf5 and compare output to committed files
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEMPLATE_XCF.exists(),
                    reason="Template XCF not found — skipping conversion test")
def test_conversion_matches_committed_hdf5(tmp_path):
    """
    Re-run XCFToHDF5Converter on the full test dataset and verify that
    the output matches the committed HDF5 files structurally.
    timing_drift is excluded since it is set after conversion.
    """
    converter = XCFToHDF5Converter(
        project_folder=str(XCF_DIR),
        output_folder=str(tmp_path),
        template_xcf=str(TEMPLATE_XCF),
    )
    index_df = converter.convert_all()

    assert len(index_df) > 0, "Conversion produced no output rows"

    for committed_hdf5 in hdf5_files:
        converted = tmp_path / committed_hdf5.name
        assert converted.exists(), \
            f"Conversion did not produce {committed_hdf5.name}"

        with h5py.File(converted) as new, h5py.File(committed_hdf5) as old:
            # Class names must match exactly
            assert new.attrs["class_names"] == old.attrs["class_names"], \
                f"class_names mismatch for {committed_hdf5.name}"

            # Number of annotation passes must match
            assert int(new.attrs["num_annotations"]) == int(old.attrs["num_annotations"]), \
                f"num_annotations mismatch for {committed_hdf5.name}"

            # Spectrogram must be identical
            np.testing.assert_array_equal(
                new["spectrogram"][:],
                old["spectrogram"][:],
                err_msg=f"Spectrogram changed for {committed_hdf5.name}",
            )

            # Masks must be identical for every annotation pass
            for idx in old["annotations"].keys():
                np.testing.assert_array_equal(
                    new[f"annotations/{idx}/masks"][:],
                    old[f"annotations/{idx}/masks"][:],
                    err_msg=f"Masks changed for annotation {idx} in {committed_hdf5.name}",
                )

            # Metadata must match (excluding timing_drift which is post-hoc)
            new_meta = {k: v for k, v in new["metadata"].attrs.items()}
            old_meta = {k: v for k, v in old["metadata"].attrs.items()}
            for key in EXPECTED_METADATA_KEYS - {"audio_path"}:
                assert new_meta[key] == old_meta[key], \
                    f"Metadata key '{key}' changed for {committed_hdf5.name}"