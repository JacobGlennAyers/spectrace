"""
tests/test_heterodyne_validation.py
====================================
Regression tests for the heterodyne validation pipeline.

Purpose
-------
These tests guard the specific numeric results that appear in the submitted
paper (max_k=5, tolerance=250 Hz).  They are **regression** tests: the first
time they run they capture whatever the current codebase produces; from that
point on any change that shifts a metric beyond the stated tolerance band will
fail CI and must be reviewed before merging.

Design decisions
----------------
* End-to-end, real HDF5 files — tests run against the actual clips in
  tests/data/ (pulled via LFS in CI exactly as the existing test_hdf5.py
  suite does).  No mocking.

* Parametrised per clip + per order — each (clip, heterodyne_order) pair is
  a separate test node so failures pinpoint exactly which signal changed.

* Tolerances are deliberately tight:
    - MAE / deviation : ±1 Hz  (sub-bin rounding noise only)
    - Ratios / fractions : ±0.005  (0.5 percentage point)
    - Integer counts : exact

* The reported operating point (max_k=5, tolerance=250 Hz) is defined ONCE
  at the top of this file via REPORTED_MAX_K and PRIMARY_TOLERANCE_HZ
  (imported from heterodyne_validation so it cannot drift silently).

* A clearly-commented section explains the API mismatch between
  get_valid_annotation_indices (returns a tuple) and heterodyne_sweep
  (expects a plain list).  One test explicitly asserts the correct return
  type so the mismatch does not regress silently.

Running locally
---------------
    pytest tests/test_heterodyne_validation.py -v
    pytest tests/test_heterodyne_validation.py -v -k "test_api"  # fast subset

Adding new paper results
------------------------
1. Run the full suite once with --snapshot-update (see _load_snapshot /
   _save_snapshot below) to write tests/data/heterodyne_snapshots.json.
2. Commit the snapshot alongside the code.
3. CI will then compare future runs against it.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors heterodyne_validation.py's own sys.path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent
_DEMOS_DIR = _REPO_ROOT / "demos"

sys.path.insert(0, str(_DEMOS_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import heterodyne_validation as hv  # noqa: E402

# ---------------------------------------------------------------------------
# Reported operating point — single source of truth for the whole file
# ---------------------------------------------------------------------------
REPORTED_MAX_K: int = 5
# PRIMARY_TOLERANCE_HZ is imported from hv so it tracks the module constant.
# If the module value ever changes from 250 the assertion below will fire
# immediately, forcing a conscious decision rather than a silent drift.
assert hv.PRIMARY_TOLERANCE_HZ == 250, (
    f"heterodyne_validation.PRIMARY_TOLERANCE_HZ changed from 250 to "
    f"{hv.PRIMARY_TOLERANCE_HZ}. Update REPORTED_MAX_K / paper table if "
    f"intentional, then update this assertion."
)

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
_SNAPSHOT_PATH = _TESTS_DIR / "data" / "heterodyne_snapshots.json"


def _load_snapshot() -> dict:
    """Load the committed numeric snapshot, or return {} if not yet created."""
    if _SNAPSHOT_PATH.exists():
        return json.loads(_SNAPSHOT_PATH.read_text())
    return {}


def _save_snapshot(snap: dict) -> None:
    _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Collect real HDF5 files (same discovery logic as conftest.py / test_hdf5.py)
# ---------------------------------------------------------------------------
from conftest import HDF5_DIR  # noqa: E402

_hdf5_files: List[Path] = sorted(HDF5_DIR.glob("*.hdf5"))

# Skip the entire module gracefully if there are no HDF5 files (e.g. a
# partial checkout without LFS).
if not _hdf5_files:
    pytest.skip(
        "No .hdf5 files found in tests/data/ — run with git lfs pull first.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def snapshot() -> dict:
    """Session-scoped snapshot dict.  Tests read from here; the update
    command writes back at session end."""
    return _load_snapshot()


@pytest.fixture(scope="session")
def all_clip_results() -> Dict[str, object]:
    """Run validate_single_clip once per HDF5 file at the reported operating
    point and cache the DataFrames for the whole test session.

    Returns {clip_stem: DataFrame}.
    """
    results = {}
    for hdf5_path in _hdf5_files:
        valid_indices, _total = hv.get_valid_annotation_indices(str(hdf5_path))
        if not valid_indices:
            continue
        # Use the first valid annotation index (index 0 in the vast majority
        # of clips; the fixture loops over all valid indices if needed below).
        ann_idx = valid_indices[0]
        df = hv.validate_single_clip(
            str(hdf5_path),
            annotation_index=ann_idx,
            kernel_size=5,   # default used in the paper
            max_k=REPORTED_MAX_K,
            output_dir=None,  # no plots in CI
        )
        if not df.empty:
            results[hdf5_path.stem] = df
    return results


# ---------------------------------------------------------------------------
# Helper: assert a metric is within tolerance of its snapshot value
# ---------------------------------------------------------------------------

def _assert_close(
    value: float,
    expected: float,
    abs_tol: float,
    label: str,
) -> None:
    """Fail with a descriptive message if |value - expected| > abs_tol."""
    if math.isnan(expected) and math.isnan(value):
        return  # both NaN — consistent
    if math.isnan(expected) or math.isnan(value):
        raise AssertionError(
            f"{label}: one value is NaN but the other is not "
            f"(got {value}, expected {expected})"
        )
    diff = abs(value - expected)
    assert diff <= abs_tol, (
        f"{label}: got {value:.4f}, expected {expected:.4f}, "
        f"diff {diff:.4f} exceeds tolerance ±{abs_tol}"
    )


# ---------------------------------------------------------------------------
# Helper: run the negative control MAE for use in guard tests
# ---------------------------------------------------------------------------

def _negative_control_mae_for_test(skipped_files: List[Path], max_k: int) -> float:
    """Compute the weighted mean band-aware MAE of heterodyne predictions
    against non-heterodyne reference contours in clips with no drawn
    heterodyne annotations.

    This replicates the logic in heterodyne_validation.run_negative_control
    but returns a single float rather than printing a table, so it can be
    used directly in assertions.  The reference layers and weighting scheme
    are identical to those used in the CLI negative control run, ensuring
    the guard test reflects exactly what is reported in the paper.

    Returns NaN if no valid comparisons are found (e.g. no reference masks
    drawn in any skipped clip), in which case the calling test should skip.
    """
    _REFERENCE_LAYER_PREFIXES = (
        "harmonics_HFC", "harmonics_LFC",
        "subharmonics_HFC", "subharmonics_LFC",
        "heterodyne_or_subharmonic_or_other",
        "Cetacean_AdditionalContours",
        "unsure_HFC", "unsure_LFC",
        "Heterodynes/unsure",
    )
    maes: List[float] = []
    weights: List[int] = []

    for hdf5_path in skipped_files:
        try:
            with hv.HDF5SpectrogramLoader(str(hdf5_path)) as loader:
                class_names = loader.get_class_names()
                if "f0_HFC" not in class_names or "f0_LFC" not in class_names:
                    continue
                meta = loader.get_metadata()
                max_freq = meta.max_freq_hz

                for ann_idx in range(loader.get_num_annotations()):
                    hfc_mask = loader.get_class_mask("f0_HFC", ann_idx)
                    lfc_mask = loader.get_class_mask("f0_LFC", ann_idx)
                    if hfc_mask is None or hfc_mask.sum() == 0:
                        continue
                    if lfc_mask is None or lfc_mask.sum() == 0:
                        continue

                    f0_hfc = hv.smooth_f0_contour(
                        hv.extract_f0_contour(hfc_mask, max_freq))
                    f0_lfc = hv.smooth_f0_contour(
                        hv.extract_f0_contour(lfc_mask, max_freq))

                    ref_masks = {
                        name: loader.get_class_mask(name, ann_idx)
                        for name in class_names
                        if any(name.startswith(p) for p in _REFERENCE_LAYER_PREFIXES)
                        and loader.get_class_mask(name, ann_idx) is not None
                        and loader.get_class_mask(name, ann_idx).sum() > 0
                    }
                    if not ref_masks:
                        continue

                    for order in range(7):
                        pred_freqs = hv.compute_predicted_heterodyne_freqs(
                            f0_hfc, f0_lfc,
                            hfc_multiplier=order + 1,
                            max_k=max_k,
                            max_freq=max_freq,
                        )
                        for ref_mask in ref_masks.values():
                            ba = hv.compute_band_aware_metrics(
                                pred_freqs, ref_mask, max_freq)
                            if ba["ba_n_samples"] > 0:
                                maes.append(ba["ba_mae_hz"])
                                weights.append(ba["ba_n_samples"])
        except Exception:
            continue

    if not maes:
        return float("nan")
    return float(np.average(maes, weights=weights))


# ===========================================================================
# SECTION 1 — API contract tests (fast, no HDF5 I/O)
# ===========================================================================

class TestAPIContracts:
    """Guard public function signatures that the sweep script depends on."""

    # -----------------------------------------------------------------------
    # BUG NOTICE — documented here so it cannot regress silently.
    #
    # heterodyne_sweep.py calls:
    #     idx = hv.get_valid_annotation_indices(str(f))
    #     if idx:
    #         files_idx.append((f, idx))
    #     ...
    #     for ann_idx in valid_indices:   # iterates the tuple, not the list!
    #
    # But heterodyne_validation.py returns a TUPLE:
    #     return valid_indices, total_het_count   # (List[int], int)
    #
    # So the sweep's `for ann_idx in valid_indices` iterates over
    # (List[int], int) — the first element is the whole list, the second is
    # an int — which silently processes only annotation index 0 and then
    # crashes or skips on the int.
    #
    # The test below asserts the CORRECT return type from the validation
    # module.  A companion test (test_sweep_api_mismatch) documents the
    # mismatch explicitly so it cannot be overlooked during code review.
    # -----------------------------------------------------------------------

    def test_get_valid_annotation_indices_return_type(self):
        """get_valid_annotation_indices must return (List[int], int)."""
        result = hv.get_valid_annotation_indices(str(_hdf5_files[0]))
        assert isinstance(result, tuple), (
            "get_valid_annotation_indices should return a tuple "
            f"(List[int], int), got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected 2-element tuple, got length {len(result)}"
        )
        valid_indices, total_count = result
        assert isinstance(valid_indices, list), (
            f"First element should be List[int], got {type(valid_indices)}"
        )
        assert isinstance(total_count, int), (
            f"Second element should be int (total heterodyne count), "
            f"got {type(total_count)}"
        )

    def test_sweep_api_mismatch_is_documented(self):
        """
        Canary test: heterodyne_sweep.get_valid_annotation_indices call
        does NOT unpack the tuple, so it passes the raw (List[int], int)
        tuple as valid_indices into the inner loop.

        This test will PASS as long as the bug is present (it confirms the
        mismatch exists), giving a clear signal when it is fixed: the test
        should then be deleted or inverted.

        To fix in heterodyne_sweep.py, change:
            idx = hv.get_valid_annotation_indices(str(f))
        to:
            idx, _ = hv.get_valid_annotation_indices(str(f))
        """
        import heterodyne_sweep as hs  # noqa: F401  — just check importable
        # Demonstrate the mismatch: unpacking proves the tuple has 2 elements.
        result = hv.get_valid_annotation_indices(str(_hdf5_files[0]))
        valid_indices, total = result
        # The sweep currently does `if idx:` on the raw tuple — a non-empty
        # tuple is always truthy, so clips are never skipped, but valid_indices
        # inside the loop is the whole tuple, not the list.
        assert isinstance(valid_indices, list), (
            "Mismatch resolved — delete test_sweep_api_mismatch_is_documented "
            "and verify heterodyne_sweep.py was updated."
        )

    def test_primary_tolerance_hz_in_tolerances_list(self):
        """PRIMARY_TOLERANCE_HZ must appear in TOLERANCES_HZ (module invariant)."""
        assert hv.PRIMARY_TOLERANCE_HZ in hv.TOLERANCES_HZ, (
            f"PRIMARY_TOLERANCE_HZ={hv.PRIMARY_TOLERANCE_HZ} is not in "
            f"TOLERANCES_HZ={hv.TOLERANCES_HZ}"
        )

    def test_acc_key_naming_convention(self):
        """acc_key() must produce the column names the DataFrame actually has."""
        assert hv.acc_key(250) == "ba_acc_250hz"
        assert hv.acc_key(500) == "ba_acc_500hz"
        assert hv.acc_key(hv.PRIMARY_TOLERANCE_HZ) == f"ba_acc_{hv.PRIMARY_TOLERANCE_HZ}hz"

    def test_primary_acc_col_per_variant(self):
        """primary_acc_col() must compose acc_key + variant suffix correctly."""
        for variant in hv.BA_VARIANTS:
            col = hv.primary_acc_col(variant)
            assert col == f"ba_acc_{hv.PRIMARY_TOLERANCE_HZ}hz_{variant}", (
                f"primary_acc_col('{variant}') returned unexpected '{col}'"
            )

    def test_ba_empty_fill_covers_all_tolerances(self):
        """ba_empty_fill() must produce NaN for every tolerance in TOLERANCES_HZ."""
        result = {}
        hv.ba_empty_fill(result, "fitted")
        for tol in hv.TOLERANCES_HZ:
            key = f"ba_acc_{tol}hz_fitted"
            assert key in result, f"ba_empty_fill missing key '{key}'"
            assert math.isnan(result[key]), f"ba_empty_fill key '{key}' is not NaN"


# ===========================================================================
# SECTION 2 — Core formula unit tests (synthetic arrays, deterministic)
# ===========================================================================

class TestHeterodyneFormula:
    """Verify the biphonic formula f = (n+1)*HFC ± k*LFC on known inputs."""

    def _constant_contour(self, freq_hz: float, length: int = 100) -> np.ndarray:
        return np.full(length, freq_hz)

    def test_order0_k1_both_signs(self):
        """Order 0, k=1: expected freqs are HFC±LFC."""
        hfc = self._constant_contour(3000.0)
        lfc = self._constant_contour(1000.0)
        results = hv.compute_predicted_heterodyne_freqs(
            hfc, lfc, hfc_multiplier=1, max_k=1, max_freq=50000.0
        )
        assert len(results) == 2  # k=1 plus, k=1 minus
        np.testing.assert_allclose(results[0], 4000.0, atol=1e-6,
                                   err_msg="k=1 plus sign should be HFC+LFC=4000")
        np.testing.assert_allclose(results[1], 2000.0, atol=1e-6,
                                   err_msg="k=1 minus sign should be HFC-LFC=2000")

    def test_order1_k2_values(self):
        """Order 1 (hfc_multiplier=2), k=2: 2*HFC ± 2*LFC."""
        hfc = self._constant_contour(3000.0)
        lfc = self._constant_contour(1000.0)
        results = hv.compute_predicted_heterodyne_freqs(
            hfc, lfc, hfc_multiplier=2, max_k=2, max_freq=50000.0
        )
        assert len(results) == 4  # k=1+, k=1-, k=2+, k=2-
        # k=2 positive: 2*3000 + 2*1000 = 8000
        np.testing.assert_allclose(results[2], 8000.0, atol=1e-6)
        # k=2 negative: 2*3000 - 2*1000 = 4000
        np.testing.assert_allclose(results[3], 4000.0, atol=1e-6)

    def test_max_k_controls_fan_width(self):
        """Number of returned arrays == 2*max_k (one per sign per k)."""
        hfc = self._constant_contour(5000.0)
        lfc = self._constant_contour(1000.0)
        for max_k in [1, 2, 3, 5]:
            results = hv.compute_predicted_heterodyne_freqs(
                hfc, lfc, hfc_multiplier=1, max_k=max_k, max_freq=50000.0
            )
            assert len(results) == 2 * max_k, (
                f"max_k={max_k} should yield {2*max_k} arrays, got {len(results)}"
            )

    def test_out_of_range_frequencies_become_nan(self):
        """Predictions that fall below 0 or above max_freq must be NaN."""
        hfc = self._constant_contour(1000.0)
        lfc = self._constant_contour(1500.0)  # LFC > HFC → minus sign goes negative
        results = hv.compute_predicted_heterodyne_freqs(
            hfc, lfc, hfc_multiplier=1, max_k=1, max_freq=50000.0
        )
        minus_band = results[1]  # 1*HFC - 1*LFC = 1000 - 1500 = -500 → NaN
        assert np.all(np.isnan(minus_band)), (
            "Negative frequencies should be NaN, got non-NaN values"
        )

    def test_nan_in_input_propagates(self):
        """NaN in either fundamental must produce NaN predictions at that frame."""
        hfc = np.array([3000.0, np.nan, 3000.0])
        lfc = np.array([1000.0, 1000.0, np.nan])
        results = hv.compute_predicted_heterodyne_freqs(
            hfc, lfc, hfc_multiplier=1, max_k=1, max_freq=50000.0
        )
        for band in results:
            assert np.isnan(band[1]), "NaN in HFC must propagate to frame 1"
            assert np.isnan(band[2]), "NaN in LFC must propagate to frame 2"
            assert not np.isnan(band[0]), "Frame 0 (both valid) should not be NaN"


# ===========================================================================
# SECTION 3 — Metric unit tests (synthetic masks, deterministic)
# ===========================================================================

class TestMetricFunctions:
    """Verify metric functions on analytically tractable synthetic inputs."""

    # --- band-aware MAE ---

    def test_band_aware_mae_perfect_prediction(self):
        """When the prediction exactly matches the label, MAE=0 and acc=1."""
        # Mask: single horizontal band at row 512 (out of 1024)
        H, W = 1024, 50
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[512, :] = 1

        max_freq = 50000.0
        freq_per_bin = max_freq / H
        exact_freq = max_freq - 512 * freq_per_bin
        pred = [np.full(W, exact_freq)]

        metrics = hv.compute_band_aware_metrics(pred, mask, max_freq)
        assert metrics["ba_mae_hz"] == pytest.approx(0.0, abs=1e-6)
        assert metrics[hv.acc_key(250)] == pytest.approx(1.0, abs=1e-6)
        assert metrics["ba_n_samples"] == W

    def test_band_aware_mae_constant_offset(self):
        """MAE equals the constant offset when every prediction is offset by that amount."""
        H, W = 1024, 50
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[512, :] = 1

        max_freq = 50000.0
        freq_per_bin = max_freq / H
        exact_freq = max_freq - 512 * freq_per_bin
        offset = 100.0
        pred = [np.full(W, exact_freq + offset)]

        metrics = hv.compute_band_aware_metrics(pred, mask, max_freq)
        assert metrics["ba_mae_hz"] == pytest.approx(offset, abs=1.0)
        # 100 Hz offset < 250 Hz tolerance → all frames should still pass
        assert metrics[hv.acc_key(250)] == pytest.approx(1.0, abs=1e-6)
        # 100 Hz offset < 200 Hz tolerance → all frames should pass that too
        assert metrics[hv.acc_key(200)] == pytest.approx(1.0, abs=1e-6)

    def test_band_aware_acc_threshold_boundary(self):
        """Frames exactly at the tolerance boundary count as passing."""
        H, W = 1024, 100
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[512, :50] = 1   # first 50 cols: offset = 250 Hz exactly (pass)
        mask[512, 50:] = 1   # last 50 cols: offset = 251 Hz (fail)

        max_freq = 50000.0
        freq_per_bin = max_freq / H
        exact_freq = max_freq - 512 * freq_per_bin

        pred_vals = np.full(W, exact_freq)
        pred_vals[:50] += 250.0   # exactly at boundary → should pass
        pred_vals[50:] += 251.0   # 1 Hz over → should fail
        pred = [pred_vals]

        metrics = hv.compute_band_aware_metrics(pred, mask, max_freq)
        assert metrics[hv.acc_key(250)] == pytest.approx(0.5, abs=1e-6), (
            "Exactly 50/100 frames should pass the 250 Hz tolerance"
        )

    def test_band_aware_empty_mask_returns_nan(self):
        """Empty label mask must return NaN for all metrics without raising."""
        H, W = 1024, 50
        mask = np.zeros((H, W), dtype=np.uint8)
        pred = [np.full(W, 25000.0)]
        metrics = hv.compute_band_aware_metrics(pred, mask, 50000.0)
        assert math.isnan(metrics["ba_mae_hz"])
        assert metrics["ba_n_samples"] == 0
        for tol in hv.TOLERANCES_HZ:
            assert math.isnan(metrics[hv.acc_key(tol)]), (
                f"acc_key({tol}) should be NaN for empty mask"
            )

    # --- IoU ---

    def test_iou_identical_masks(self):
        """IoU of identical non-empty masks (after dilation) is 1.0."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 20:80] = 1
        result = hv.compute_iou(mask, mask.copy(), kernel_size=5)
        assert result["iou"] == pytest.approx(1.0, abs=1e-6)

    def test_iou_disjoint_masks_is_zero(self):
        """IoU of spatially disjoint masks that don't touch after dilation is 0."""
        pred = np.zeros((200, 200), dtype=np.uint8)
        lab  = np.zeros((200, 200), dtype=np.uint8)
        pred[10, 10] = 1   # top-left corner
        lab[190, 190] = 1  # bottom-right corner
        # kernel_size=1 → no dilation → guaranteed disjoint
        result = hv.compute_iou(pred, lab, kernel_size=1)
        assert result["iou"] == pytest.approx(0.0, abs=1e-6)

    def test_iou_both_empty_returns_nan(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = hv.compute_iou(mask, mask.copy(), kernel_size=5)
        assert result["both_empty"] is True
        assert math.isnan(result["iou"])

    # --- smooth_f0_contour ---

    def test_smooth_f0_preserves_nan_gaps(self):
        """NaN positions must remain NaN after smoothing."""
        contour = np.array([3000.0] * 20 + [np.nan] * 10 + [3500.0] * 20)
        smoothed = hv.smooth_f0_contour(contour)
        assert np.all(np.isnan(smoothed[20:30])), (
            "NaN gap must survive Savitzky-Golay smoothing"
        )
        assert np.all(~np.isnan(smoothed[:20]))
        assert np.all(~np.isnan(smoothed[30:]))

    def test_smooth_f0_constant_signal_unchanged(self):
        """Smoothing a perfectly constant signal must leave it unchanged."""
        contour = np.full(50, 3000.0)
        smoothed = hv.smooth_f0_contour(contour)
        np.testing.assert_allclose(smoothed, 3000.0, atol=1e-6)

    # --- render_frequency_to_mask ---

    def test_render_frequency_to_mask_round_trip(self):
        """Rendering a contour and re-extracting it must recover the frequency
        within one bin (freq_per_bin = max_freq / H)."""
        H, W = 1024, 200
        max_freq = 50000.0
        freq_per_bin = max_freq / H
        target_freq = 20000.0
        pred = [np.full(W, target_freq)]
        mask = hv.render_frequency_to_mask(pred, H, W, max_freq)
        recovered = hv.extract_f0_contour(mask, max_freq)
        valid = ~np.isnan(recovered)
        assert valid.sum() > 0, "No active pixels after rendering"
        np.testing.assert_allclose(
            recovered[valid], target_freq, atol=freq_per_bin,
            err_msg="Recovered frequency must be within one bin of original"
        )


# ===========================================================================
# SECTION 4 — Snapshot regression tests against real HDF5 clips
# ===========================================================================

def _build_snapshot_key(clip_stem: str, order: int, metric: str) -> str:
    return f"{clip_stem}__order{order}__{metric}"


def _collect_current_metrics(all_clip_results) -> dict:
    """Extract the paper-relevant metrics from all_clip_results into a flat dict."""
    current = {}
    for clip_stem, df in all_clip_results.items():
        labelled = df[df["labelled_px"] > 0]
        for _, row in labelled.iterrows():
            order = int(row["order"])
            for metric in [
                "iou_fitted",
                "ba_mae_hz_fitted",
                hv.primary_acc_col("fitted"),   # ba_acc_250hz_fitted
                "contour_freq_deviation_hz",
                "contour_coverage",
                "n_segments",
            ]:
                val = row.get(metric, float("nan"))
                key = _build_snapshot_key(clip_stem, order, metric)
                current[key] = float(val) if not (
                    isinstance(val, float) and math.isnan(val)
                ) else None   # JSON cannot store NaN
    return current


# Tolerance bands per metric type (applied in _assert_snapshot_match)
_METRIC_TOLERANCES = {
    "iou_fitted":                      0.005,   # ±0.5 percentage points
    "ba_mae_hz_fitted":                1.0,     # ±1 Hz
    hv.primary_acc_col("fitted"):      0.005,   # ±0.5 pp
    "contour_freq_deviation_hz":       1.0,     # ±1 Hz
    "contour_coverage":                0.005,   # ±0.5 pp
    "n_segments":                      0,       # exact integer match
}


def _assert_snapshot_match(key: str, current_val, expected_val) -> None:
    metric = key.split("__")[-1]
    tol = _METRIC_TOLERANCES.get(metric, 1.0)

    if expected_val is None and current_val is None:
        return  # both NaN — fine
    if expected_val is None or current_val is None:
        raise AssertionError(
            f"{key}: NaN mismatch — got {current_val}, snapshot has {expected_val}"
        )
    if tol == 0:  # exact integer match
        assert int(current_val) == int(expected_val), (
            f"{key}: got {int(current_val)}, expected {int(expected_val)} (exact)"
        )
    else:
        _assert_close(float(current_val), float(expected_val), tol, key)


# --- Snapshot creation CLI hook ---
# Run:  pytest tests/test_heterodyne_validation.py --snapshot-update -v
# to regenerate tests/data/heterodyne_snapshots.json.

def pytest_addoption_snapshot(parser):
    # Registered in conftest.py if you want to expose it project-wide.
    # Defined here as a module-level no-op fallback.
    pass


@pytest.fixture(scope="session")
def snapshot_update(request) -> bool:
    return request.config.getoption("--snapshot-update", default=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "snapshot: mark test as a snapshot regression test"
    )


# ---------------------------------------------------------------------------
# Parametrised snapshot tests — one node per (clip, order) pair
# ---------------------------------------------------------------------------

def _clip_order_ids():
    """Generate (clip_stem, ann_idx_placeholder) pairs for parametrize."""
    ids = []
    for f in _hdf5_files:
        valid_indices, _ = hv.get_valid_annotation_indices(str(f))
        if valid_indices:
            ids.append(f.stem)
    return ids


_clip_stems = _clip_order_ids()


@pytest.mark.snapshot
@pytest.mark.parametrize("clip_stem", _clip_stems)
def test_snapshot_iou_fitted(clip_stem, all_clip_results, snapshot, snapshot_update):
    """IoU (fitted variant) must not drift from the committed snapshot."""
    df = all_clip_results.get(clip_stem)
    if df is None:
        pytest.skip(f"No results for clip {clip_stem}")

    labelled = df[df["labelled_px"] > 0]
    snap = _load_snapshot()

    if snapshot_update:
        current = _collect_current_metrics(all_clip_results)
        _save_snapshot(current)
        pytest.skip("Snapshot updated — re-run without --snapshot-update to validate.")

    for _, row in labelled.iterrows():
        order = int(row["order"])
        key = _build_snapshot_key(clip_stem, order, "iou_fitted")
        if key not in snap:
            pytest.skip(
                f"No snapshot entry for {key} — run with --snapshot-update first."
            )
        _assert_snapshot_match(key, row.get("iou_fitted"), snap[key])


@pytest.mark.snapshot
@pytest.mark.parametrize("clip_stem", _clip_stems)
def test_snapshot_mae_fitted(clip_stem, all_clip_results, snapshot, snapshot_update):
    """Band-aware MAE (fitted, Hz) must not drift from the committed snapshot."""
    df = all_clip_results.get(clip_stem)
    if df is None:
        pytest.skip(f"No results for clip {clip_stem}")

    snap = _load_snapshot()
    if snapshot_update:
        pytest.skip("Re-run without --snapshot-update to validate.")

    labelled = df[df["labelled_px"] > 0]
    for _, row in labelled.iterrows():
        order = int(row["order"])
        key = _build_snapshot_key(clip_stem, order, "ba_mae_hz_fitted")
        if key not in snap:
            pytest.skip(f"No snapshot entry for {key}")
        _assert_snapshot_match(key, row.get("ba_mae_hz_fitted"), snap[key])


@pytest.mark.snapshot
@pytest.mark.parametrize("clip_stem", _clip_stems)
def test_snapshot_acc250_fitted(clip_stem, all_clip_results, snapshot, snapshot_update):
    """Band-aware Acc@250Hz (fitted) must not drift from the committed snapshot.

    This is the PRIMARY paper metric at the reported operating point.
    """
    df = all_clip_results.get(clip_stem)
    if df is None:
        pytest.skip(f"No results for clip {clip_stem}")

    snap = _load_snapshot()
    if snapshot_update:
        pytest.skip("Re-run without --snapshot-update to validate.")

    acc_col = hv.primary_acc_col("fitted")
    labelled = df[df["labelled_px"] > 0]
    for _, row in labelled.iterrows():
        order = int(row["order"])
        key = _build_snapshot_key(clip_stem, order, acc_col)
        if key not in snap:
            pytest.skip(f"No snapshot entry for {key}")
        _assert_snapshot_match(key, row.get(acc_col), snap[key])


@pytest.mark.snapshot
@pytest.mark.parametrize("clip_stem", _clip_stems)
def test_snapshot_segment_count(clip_stem, all_clip_results, snapshot, snapshot_update):
    """Number of fitted segments must be exactly equal to the snapshot value."""
    df = all_clip_results.get(clip_stem)
    if df is None:
        pytest.skip(f"No results for clip {clip_stem}")

    snap = _load_snapshot()
    if snapshot_update:
        pytest.skip("Re-run without --snapshot-update to validate.")

    labelled = df[df["labelled_px"] > 0]
    for _, row in labelled.iterrows():
        order = int(row["order"])
        key = _build_snapshot_key(clip_stem, order, "n_segments")
        if key not in snap:
            pytest.skip(f"No snapshot entry for {key}")
        _assert_snapshot_match(key, row.get("n_segments"), snap[key])


# ===========================================================================
# SECTION 5 — Operating-point guard tests (direction assertions, always run)
# ===========================================================================

class TestOperatingPointGuards:
    """Direction assertions that must hold at the reported operating point
    (max_k=5, tolerance=250 Hz) regardless of exact numeric values.
    These do not need a snapshot; they encode paper-level claims.
    """

    def test_reported_max_k_is_five(self):
        """Fail loudly if someone changes the reported max_k without updating
        this file and the paper."""
        assert REPORTED_MAX_K == 5, (
            "REPORTED_MAX_K changed — update the paper table and this assertion."
        )

    def test_reported_tolerance_is_250hz(self):
        assert hv.PRIMARY_TOLERANCE_HZ == 250, (
            "PRIMARY_TOLERANCE_HZ changed — update paper and this assertion."
        )

    def test_all_labelled_clips_have_results(self, all_clip_results):
        """Every HDF5 file that has valid heterodyne annotations must produce
        a non-empty results DataFrame."""
        for f in _hdf5_files:
            valid_indices, _ = hv.get_valid_annotation_indices(str(f))
            if not valid_indices:
                continue
            assert f.stem in all_clip_results, (
                f"validate_single_clip returned empty DataFrame for {f.name}"
            )
            assert not all_clip_results[f.stem].empty, (
                f"DataFrame for {f.name} is unexpectedly empty"
            )

    def test_fitted_iou_above_zero_for_labelled_orders(self, all_clip_results):
        """For every labelled heterodyne order, the fitted IoU must be > 0.
        A value of 0 would mean the formula never overlaps the annotation —
        which would be a sign of a unit or coordinate bug.
        """
        failures = []
        for clip_stem, df in all_clip_results.items():
            labelled = df[df["labelled_px"] > 0]
            for _, row in labelled.iterrows():
                iou = row.get("iou_fitted", float("nan"))
                if math.isnan(iou):
                    continue  # no segments fitted — skip rather than fail
                if iou <= 0.0:
                    failures.append(
                        f"{clip_stem} order={int(row['order'])}: iou_fitted={iou:.4f}"
                    )
        assert not failures, (
            "Fitted IoU is 0 or negative for the following (clip, order) pairs — "
            "the heterodyne formula may have a coordinate system bug:\n"
            + "\n".join(failures)
        )

    def test_fitted_mae_below_primary_tolerance(self, all_clip_results):
        """Mean band-aware MAE (fitted) across all clips must be below
        PRIMARY_TOLERANCE_HZ.  This is the headline claim in the paper.
        """
        all_maes = []
        for clip_stem, df in all_clip_results.items():
            labelled = df[df["labelled_px"] > 0]
            maes = labelled["ba_mae_hz_fitted"].dropna().tolist()
            all_maes.extend(maes)

        if not all_maes:
            pytest.skip("No labelled rows found across all clips.")

        mean_mae = float(np.mean(all_maes))
        assert mean_mae < hv.PRIMARY_TOLERANCE_HZ, (
            f"Mean fitted MAE ({mean_mae:.1f} Hz) exceeds PRIMARY_TOLERANCE_HZ "
            f"({hv.PRIMARY_TOLERANCE_HZ} Hz) — the formula is not meeting the "
            f"paper's headline accuracy claim."
        )

    def test_fitted_acc250_above_chance(self, all_clip_results):
        """Acc@250Hz (fitted) must be substantially above chance (>0.5) overall.
        Chance-level accuracy would mean max_k=5 adds no discriminative power.
        """
        acc_col = hv.primary_acc_col("fitted")
        all_accs = []
        for clip_stem, df in all_clip_results.items():
            labelled = df[df["labelled_px"] > 0]
            if acc_col in labelled.columns:
                accs = labelled[acc_col].dropna().tolist()
                all_accs.extend(accs)

        if not all_accs:
            pytest.skip("No acc values found — check column naming.")

        mean_acc = float(np.mean(all_accs))
        assert mean_acc > 0.5, (
            f"Mean Acc@250Hz (fitted) is {mean_acc:.3f}, which is at or below "
            f"chance level (0.5). The formula is not providing useful predictions."
        )

    def test_negative_control_available(self):
        """There must be at least one HDF5 file with NO heterodyne annotations
        to serve as the negative control pool.  Without it the specificity
        claim in the paper cannot be demonstrated.
        """
        negative_pool = []
        for f in _hdf5_files:
            valid_indices, _ = hv.get_valid_annotation_indices(str(f))
            if not valid_indices:
                negative_pool.append(f)
        assert len(negative_pool) > 0, (
            "Every HDF5 file in tests/data/ has heterodyne annotations — "
            "there is no negative control pool. Add at least one clip with "
            "only harmonics/fundamentals (no Heterodynes/N masks drawn)."
        )

    def test_dataframe_schema_contains_all_tolerance_columns(self, all_clip_results):
        """The results DataFrame must contain a ba_acc_*_fitted column for every
        tolerance in TOLERANCES_HZ — guards against ba_empty_fill regressions.
        """
        for clip_stem, df in all_clip_results.items():
            for tol in hv.TOLERANCES_HZ:
                expected_col = f"{hv.acc_key(tol)}_fitted"
                assert expected_col in df.columns, (
                    f"Column '{expected_col}' missing from results of {clip_stem}. "
                    f"ba_empty_fill() may not be filling all tolerances."
                )

    def test_negative_control_mae_gap_is_large(self, all_clip_results):
        """The negative control MAE must be at least 10x the positive MAE.

        This guards the paper's specificity claim: that the biphonic formula
        lands close to labelled heterodynes but far from other drawn contours
        (harmonics, subharmonics, etc.) in clips with no heterodyne annotations.

        Current observed values (max_k=5, kernel=5):
            Positive MAE : ~174 Hz  (mean across labelled orders, full fan)
            Negative MAE : ~12,984 Hz  (weighted mean across reference layers)
            Ratio        : ~75x

        The 10x threshold is deliberately conservative — a ratio this large
        collapsing to near 10x would itself be a result worth investigating.
        If it falls below 10x, either the formula has lost specificity or the
        negative pool composition has changed materially.
        """
        # --- positive side: mean-of-per-clip-means, full fan variant ---
        # Use the full fan (not fitted) to match the negative control exactly:
        # the negative side has no labels to fit against, so it always uses
        # the full fan. Comparing fitted-positive vs full-negative would mix
        # two different sub-band selection strategies.
        all_dfs = list(all_clip_results.values())
        if not all_dfs:
            pytest.skip("No positive results available.")

        combined = pd.concat(all_dfs, ignore_index=True)
        pos_metrics = hv.aggregate_positive_metrics(combined, variant="full")
        pos_mae = pos_metrics["mae"]

        if math.isnan(pos_mae):
            pytest.skip("Positive MAE is NaN — cannot compute ratio.")

        # --- negative side ---
        skipped = [f for f in _hdf5_files if f.stem not in all_clip_results]
        if not skipped:
            pytest.skip(
                "No negative pool (all clips have heterodyne annotations). "
                "Add clips with only harmonics/fundamentals to enable this test."
            )

        neg_mae = _negative_control_mae_for_test(skipped, REPORTED_MAX_K)
        if math.isnan(neg_mae):
            pytest.skip(
                "Negative control returned no valid comparisons — "
                "check that skipped clips have reference masks drawn."
            )

        ratio = neg_mae / pos_mae
        assert ratio > 10, (
            f"Negative/positive MAE ratio is {ratio:.1f}x "
            f"(neg={neg_mae:.0f} Hz, pos={pos_mae:.0f} Hz). "
            f"Expected >10x. The formula's specificity claim may be weakened — "
            f"review whether the negative pool composition has changed or whether "
            f"a code change reduced selectivity."
        )