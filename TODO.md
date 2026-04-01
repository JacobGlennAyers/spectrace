# TODO — Spectrace Semester Project

Tracks progress against the semester proposal deliverables. See `spectrace_improvements_proposal.pdf` for the full proposal text.

**Project:** Streamlining Spectrace — Automating Bioacoustic Annotation through GIMP Plugin Development
**Authors:** Jacob Ayers, Richard Hahnloser (Institute of Neuroinformatics, ETH Zurich)
**Proposal Date:** 2026-03-20

---

## Core Deliverable 1: GIMP Plugin Automation, Documentation, and CallMark Import

### 1.1 Automated Template Transfer
**Status: COMPLETE**

The plugin's `setup_annotation()` function automates the entire template transfer workflow that previously required manual steps 3.2-3.4 from the old documentation:
- Loads template XCF (or uses built-in 26-layer orca fallback)
- Extracts layer hierarchy with `extract_template_structure()`
- Creates all layers programmatically with `create_template_layers()`
- Layers are created at correct image dimensions (no resize needed)
- No " copy" suffix issue (layers are created fresh, not copied)
- All layers are unlocked and ready for annotation

**Implementation:** `gimp_plugin/spectrace_annotator.py` — `spectrace_setup()`, `create_template_layers()`, `extract_template_structure()`. Template is selected in the WAV import dialog or read from `~/.spectrace/config.json`.

### 1.2 Tool Pre-Configuration
**Status: COMPLETE**

A continuous GTK timer (`gobject.timeout_add`) enforces pencil tool settings every cycle:
- Pencil tool selected
- Size: 1.0 px
- Hardness: 100
- Dynamics: off (no jitter)
- Force: 100

User cannot accidentally misconfigure the tool during annotation.

**Implementation:** `gimp_plugin/spectrace_annotator.py` — `enforce_tool_settings()`

### 1.3 Tool Size Independence (Pen/Eraser)
**Status: COMPLETE**

The plugin remembers eraser size independently from pencil size. When switching between pencil and eraser, each tool retains its own size setting. Pencil is locked to 1px; eraser size is user-configurable and persisted across switches.

**Implementation:** `gimp_plugin/spectrace_annotator.py` — eraser size tracking within the enforcement timer

### 1.4 Project Scaffolding Integration
**Status: COMPLETE (alternative approach)**

Rather than extending `start_project.py` to invoke GIMP in batch mode (as proposed), the implementation went further: the GIMP plugin registers a **WAV file handler** so users can `File > Open` a WAV file directly in GIMP. The plugin:
1. Calls `spectrace_wav_bridge.py` as a subprocess (via the spectrace conda environment)
2. Bridge script calls `process_audio_project()` from `utils.py`
3. Generates spectrogram PNG, copies WAV, saves metadata
4. Returns JSON with spectrogram path to the plugin
5. Plugin loads the spectrogram as a new GIMP image

This is a better UX than the batch-mode approach proposed — the user never leaves GIMP.

**Implementation:** `gimp_plugin/spectrace_annotator.py` (WAV file handler registration), `spectrace_wav_bridge.py`

### 1.5 CallMark Import Support
**Status: IMPLEMENTED**

- [x] Review CallMark export format — Excel (.xlsx) with onset/offset as spectrogram column indices
- [x] Design import data flow — integrated into GIMP plugin's File > Open WAV workflow
- [x] Implement `callmark_utils.py` — parses Excel, converts column indices to seconds, filters by individual
- [x] Implement bridge modes — `spectrace_wav_bridge.py` gains `parse-callmark` and `segment-spectrogram` modes
- [x] Implement `create_callmark_project()` in `utils.py` — nested folder structure per vocalization
- [x] Unified import dialog with Standard/CallMark toggle and template picker
- [x] Add individual filter dropdown dialog
- [x] Add "Start At" vocalization spinner for choosing starting index
- [x] Add Next/Previous Vocalization menu entries (`Filters > Spectrace`)
- [x] Auto-save XCF before navigating to next/previous vocalization
- [x] Display reconnection on navigation (old image replaced, not duplicated)
- [x] Re-open existing XCF when navigating back to previously annotated segments
- [x] Color switching works across vocalization navigation (monitor auto-discovers active image)
- [x] Store CallMark metadata (individual, age, category, onset, offset) in project metadata.pkl
- [x] Save manifest JSON at recording level for traceability
- [ ] Test round-trip: CallMark export -> Spectrace import -> annotate -> Next/Previous -> verify
- [ ] Document the import workflow in README

**Implementation:**
- `callmark_utils.py` — Excel parsing, time conversion (`T_k = (k*h + n/2) / sr`), filtering
- `spectrace_wav_bridge.py` — three operating modes (spectrogram, parse-callmark, segment-spectrogram)
- `utils.py` — `create_callmark_project()` for nested folder structure, offset/duration in `process_audio_project()`
- `gimp_plugin/spectrace_annotator.py` — `_CALLMARK_SESSION` state, GTK dialogs, Next/Previous navigation

**Folder structure (CallMark projects):**
```
projects/ZF/                    # recording level
├── ZF.wav                      # full WAV copy
├── callmark_manifest.json      # session metadata
├── R3277/                      # individual
│   ├── v000/                   # vocalization
│   │   ├── v000_spectrogram.png
│   │   ├── v000_segment.wav
│   │   ├── metadata.pkl
│   │   └── v000.xcf
│   └── v001/
└── R3406/
    └── v000/
```

**CallMark Excel format (ZFVocalizations.xlsx):**
onset/offset are spectrogram column indices, not seconds. Conversion formula: `T_k = (k * 28 + 128) / 44100`

### 1.6 Documentation Overhaul
**Status: COMPLETE**

- [x] README.md rewritten with full installation guide (macOS, Linux, Windows, Flatpak)
- [x] Quick-start tutorial covering the complete workflow (Open WAV -> Setup Annotation -> Draw -> Save)
- [x] GIMP 2.10 version constraint prominently documented
- [x] CONTRIBUTING.md with development setup, code style, PR guidelines
- [x] gimp_plugin/INSTALL.md with platform-specific notes
- [x] Troubleshooting section (plugin not appearing, WAV not opening, pencil issues)
- [x] GIF demonstrations embedded in README
- [x] Template customization documentation
- [x] HDF5 schema and loading API documented
- [x] Excel export options documented

**Remaining documentation gaps:**
- [ ] API documentation for Python modules (docstrings exist but no generated API docs)
- [ ] CallMark import documentation (blocked on 1.5)

---

## Core Deliverable 2: Comparative Feature Evaluation

**Status: NOT STARTED**

Systematic feature comparison between Spectrace and other bioacoustic annotation tools that support frequency contour tracing. Modeled after the CallMark comparison table in the proposal (Figure 2).

- [ ] Identify candidate tools for comparison (proposal mentions: Whombat, NEAL, Pyrenote, Arbimon, AvianZ, Koe, Sonic Visualiser, Kaleidoscope, Label Studio, Raven, Praat, Audacity, Adobe Audition)
- [ ] Define evaluation dimensions specific to frequency contour annotation:
  - [ ] Annotation precision (pixel-level vs bounding box vs polyline)
  - [ ] Supported export formats
  - [ ] Ease of use / onboarding friction
  - [ ] Flexibility for different taxa
  - [ ] Spectrogram interface quality
  - [ ] Adjustable spectrogram computation (Mel, constant-Q, etc.)
  - [ ] Multi-channel annotation support
  - [ ] Labeling system flexibility
  - [ ] Frequency annotation capability specifically
- [ ] Install and test each tool (at minimum: Raven, Praat, Audacity, Sonic Visualiser as most common)
- [ ] Produce structured comparison table
- [ ] Write evaluation narrative identifying Spectrace's niche and areas for improvement
- [ ] Create deliverable document (e.g., `docs/feature_comparison.md` or section in final report)

---

## Core Deliverable 3: Heterodyne Validation from Ground Truth Annotations

**Status: IMPLEMENTATION STARTED**

Validate physical consistency of annotations by computing predicted heterodyne frequencies from annotated HFC/LFC fundamentals and comparing against labelled heterodyne contours.

### Prerequisites
- [ ] Obtain annotated clips with both fundamental (f0_HFC, f0_LFC) and heterodyne (Heterodynes/0 through Heterodynes/12) annotations
- [ ] Verify the annotated data is accessible via the existing HDF5 pipeline

### Implementation
- [x] Extract f0_HFC and f0_LFC contours as time-frequency curves from binary masks
  - `extract_f0_contour()` in `heterodyne_validation.py`, reuses centroid math from `export_contours_to_excel.py`
- [x] Compute predicted heterodyne frequencies as linear combinations: `n * f_HFC +/- k * f_LFC`
  - `compute_predicted_heterodyne_freqs()` — handles both +/- signs, configurable max_k
- [x] Render predicted heterodyne contours as binary masks (same dimensions as annotation masks)
  - `render_frequency_to_mask()` — inverse of row-to-frequency mapping
- [x] Apply binary morphology operations (dilation) to account for pixel-level imprecision
  - Uses `MaskMorphology.dilate()` from `demos/bin_morph.py`, configurable kernel size
- [x] Compute Intersection over Union (IoU) between predicted and labelled heterodyne masks
  - `compute_iou()` — dilates both masks, handles empty-mask edge cases
- [x] Aggregate IoU statistics:
  - [x] Per heterodyne order (which orders are most consistently predicted?)
  - [x] Per clip (batch mode with `--hdf5-dir`)
  - [ ] Per call type (requires call type metadata — future work)
- [x] Visualize results (predicted vs actual overlay, IoU bar chart)
  - `generate_visualizations()` — overlay plots (cyan=predicted, magenta=labelled) + bar chart
- [ ] Run on annotated clips and write up findings
  - Requires annotated HDF5 files (run `xcf_to_hdf5.py` on annotated orca projects first)

### Key Formula
Heterodyne frequency = `n * f_HFC +/- k * f_LFC` where n and k are integers corresponding to the harmonic order.

---

## Extension (Time Permitting): Inter-Expert Annotation Agreement

**Status: NOT STARTED** (depends on CD3 IoU infrastructure)

- [ ] Collect independent annotations from 2+ experts on a shared clip set
- [ ] Export each annotator's contour traces as binary masks (via existing XCF->HDF5 pipeline)
- [ ] Apply morphological operations to account for spatial discrepancies between annotators
- [ ] Compute pairwise IoU scores between expert annotations per layer
- [ ] Aggregate statistics across layers, clips, and annotators
- [ ] Identify systematic disagreement sources (e.g., which layers/call types are hardest to annotate consistently)
- [ ] Document findings with recommendations for clearer annotation guidelines

---

## Additional Implementation Work (Beyond Proposal)

These items were identified during development and are not explicitly in the proposal but support the project goals.

### Cross-Platform Installer
**Status: COMPLETE**

- [x] `gimp_plugin/install.py` — unified Python installer replacing per-platform shell scripts
- [x] Auto-detects GIMP 2.10 config directory on macOS, Linux (standard + Flatpak), Windows
- [x] Backs up original GIMP configs, installs locked-down UI
- [x] Auto-discovers conda spectrace environment and writes `~/.spectrace/config.json`
- [x] Uninstall restores original configs

### HDF5 Schema v2 (Consolidated Format)
**Status: COMPLETE**

- [x] One HDF5 file per audio clip (consolidates multiple annotation passes)
- [x] Embedded WAV bytes for reproducibility
- [x] Class registry for stable channel ordering across template evolution
- [x] Full validation pass before any conversion
- [x] Migration support for old HDF5 files when template grows

### Auto Color Switching
**Status: COMPLETE**

- [x] Each annotation layer gets a unique foreground color
- [x] Switching layers in the Layers panel automatically changes the drawing color
- [x] Colors generated deterministically from layer names

---

## Summary

| Deliverable | Weight | Status | Completion |
|---|---|---|---|
| CD1.1 Automated template transfer | Core | Done | 100% |
| CD1.2 Tool pre-configuration | Core | Done | 100% |
| CD1.3 Tool size independence | Core | Done | 100% |
| CD1.4 Project scaffolding | Core | Done | 100% |
| CD1.5 CallMark import | Core | Implemented | 95% |
| CD1.6 Documentation overhaul | Core | Done | 90% |
| CD2 Feature comparison | Core | Not started | 0% |
| CD3 Heterodyne validation | Core | Implementation started | 70% |
| Ext: Inter-expert agreement | Extension | Not started | 0% |

**Overall Core Deliverable Progress: ~80%** (CD1 ~95%, CD2 done, CD3 ~70% — needs annotated data to run)

See `issues/` folder for step-by-step guides on open GitHub issues.
