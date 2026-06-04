import sys
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_contours_to_excel import export_to_excel
from conftest import DATA_DIR, HDF5_DIR

COMMITTED_EXCEL  = DATA_DIR / "whale_contours_export.xlsx"
EXPECTED_SHEETS  = {"Summary", "Contours", "Statistics", "Class_Summary"}
CONTOUR_METHODS  = ["centroid", "min_max", "all_points"]

EXPECTED_CLASSES = [
    "f0_LFC", "f0_HFC",
    "harmonics_LFC", "harmonics_HFC",
    "Heterodynes/0", "Heterodynes/1",
]


# ---------------------------------------------------------------------------
# Structural tests against the committed Excel file
# ---------------------------------------------------------------------------

def test_committed_excel_exists():
    assert COMMITTED_EXCEL.exists(), \
        "whale_contours_export.xlsx not found in tests/data"

def test_committed_excel_sheets():
    sheets = set(pd.ExcelFile(COMMITTED_EXCEL).sheet_names)
    assert sheets == EXPECTED_SHEETS, \
        f"Sheet mismatch: got {sheets}, expected {EXPECTED_SHEETS}"

def test_contours_required_columns():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    for col in ["class", "clip_basename", "annotation_index",
                "time_sec", "freq_hz"]:
        assert col in df.columns, f"Missing column '{col}' in Contours sheet"

def test_contours_no_null_time_or_frequency():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    assert df["time_sec"].notna().all(),  "Null values in time_sec"
    assert df["freq_hz"].notna().all(),   "Null values in freq_hz"

def test_contours_not_empty():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    assert len(df) > 0, "Contours sheet is empty"

def test_frequency_values_within_recording_range():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    assert (df["freq_hz"] >= 0).all(),      "Negative frequency values found"
    assert (df["freq_hz"] <= 51200).all(),  "Frequency values exceed max_freq_hz"

def test_time_values_non_negative():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    assert (df["time_sec"] >= 0).all(), "Negative time values found"

def test_statistics_no_duplicate_project_class():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Statistics")
    dupes = df.duplicated(subset=["clip_basename", "annotation_index", "class"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate clip+annotation+class rows in Statistics"

def test_class_summary_expected_classes_present():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Class_Summary")
    present = set(df["class"].tolist())
    missing = set(EXPECTED_CLASSES) - present
    assert not missing, f"Missing classes in Class_Summary: {missing}"

def test_class_summary_pixel_counts_positive():
    df = pd.read_excel(COMMITTED_EXCEL, sheet_name="Class_Summary")
    assert (df["pixel_count_total"] > 0).all(), \
        "Some classes have zero total pixels in Class_Summary"


# ---------------------------------------------------------------------------
# Regression tests — re-export and compare to committed file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", CONTOUR_METHODS)
def test_export_produces_valid_output(method, tmp_path):
    out = str(tmp_path / f"export_{method}.xlsx")
    export_to_excel(
        ml_data_folder=str(HDF5_DIR),
        output_excel=out,
        contour_method=method,
    )
    assert Path(out).exists(), f"No output file produced for method '{method}'"
    df = pd.read_excel(out, sheet_name="Contours")
    assert len(df) > 0, f"Empty Contours sheet for method '{method}'"
    assert df.notna().any().any(), \
        f"All-null Contours sheet for method '{method}'"

def test_export_regression_row_counts(tmp_path):
    """Re-exporting with centroid should produce the same row counts."""
    out = str(tmp_path / "regression.xlsx")
    export_to_excel(
        ml_data_folder=str(HDF5_DIR),
        output_excel=out,
        contour_method="centroid",
    )
    for sheet in ["Contours", "Statistics"]:
        actual = pd.read_excel(out,              sheet_name=sheet)
        gold   = pd.read_excel(COMMITTED_EXCEL,  sheet_name=sheet)
        assert len(actual) == len(gold), \
            f"Row count changed in {sheet}: got {len(actual)}, expected {len(gold)}"

def test_export_regression_frequencies(tmp_path):
    """Frequency values should not change between runs."""
    out = str(tmp_path / "regression_freq.xlsx")
    export_to_excel(
        ml_data_folder=str(HDF5_DIR),
        output_excel=out,
        contour_method="centroid",
    )
    actual = pd.read_excel(out,             sheet_name="Contours")
    gold   = pd.read_excel(COMMITTED_EXCEL, sheet_name="Contours")
    np.testing.assert_allclose(
        actual["freq_hz"].values,
        gold["freq_hz"].values,
        rtol=1e-5,
        err_msg="Frequency values changed — check extract_contours_from_mask",
    )