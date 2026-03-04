---
phase: 1
slug: core-foundations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (to be installed in Wave 0) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest tests/test_templates.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_templates.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | FOUND-01 | setup | `pytest --version` | No — W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | FOUND-01a | unit | `pytest tests/test_templates.py::test_load_orca_template -x` | No — W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | FOUND-01b | unit | `pytest tests/test_templates.py::test_modified_template_produces_different_layers -x` | No — W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | FOUND-01c | unit | `pytest tests/test_templates.py::test_malformed_template_fails_loud -x` | No — W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | FOUND-01d | unit | `pytest tests/test_templates.py::test_orca_template_matches_legacy_constants -x` | No — W0 | ⬜ pending |
| 01-01-06 | 01 | 1 | FOUND-01e | unit | `pytest tests/test_templates.py::test_discover_templates -x` | No — W0 | ⬜ pending |
| 01-01-07 | 01 | 1 | FOUND-01f | unit | `pytest tests/test_templates.py::test_hex_to_rgb -x` | No — W0 | ⬜ pending |
| 01-01-08 | 01 | 2 | FOUND-01g | integration | `pytest tests/test_templates.py::test_template_to_hdf5_pipeline -x` | No — W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — empty init for test package
- [ ] `tests/test_templates.py` — stubs for FOUND-01a through FOUND-01g
- [ ] `pytest.ini` or `pyproject.toml` [tool.pytest.ini_options] — pytest config
- [ ] Framework install: `pip install pytest` (add to environment.yml)
- [ ] `spectrace/__init__.py` and `spectrace/core/__init__.py` — package init files

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GIMP plugin loads template and creates correct layers | FOUND-01 | Requires GIMP runtime with Python-Fu | Open GIMP > Filters > Spectrace > Setup Annotation > verify layers match template |
| Layer colors applied correctly in GIMP | FOUND-01 | Color rendering requires GIMP display | Draw on layers, verify foreground color switches per LAYER_COLORS |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
