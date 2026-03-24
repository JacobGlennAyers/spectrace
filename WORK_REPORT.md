# Work Report — Spectrace Semester Project

Daily progress log for the Spectrace improvements semester project.
Derived from git history on `main` and `ale-development-branch`.

---

## 2025-11-12 — Project Genesis

**Commits:** `5f31788`, `62eaa08`, `9e62fc1`
**Author:** Jacob Ayers

- Initial commit and first version of Spectrace
- Core Python workflow established: `utils.py`, `start_project.py`, `produce_visuals.py`
- Project folder convention defined (`projects/<clip_basename>_<index>/`)
- Audio processing pipeline: WAV -> spectrogram PNG via librosa
- XCF layer extraction using gimpformats
- Basic visualization (overlay + individual layers)

## 2025-11-13

**Commits:** `62f6775`

- Added templates folder for XCF template files
- Small bug fixes

## 2025-11-16

**Commits:** `044214c`, `07c324c`, `0cce429`, `3c94152`

- Rework of visualization code
- Changed test data to orca clips (contributed by Pascale)
- Dynamic layer color generation
- Orca template and test data integrated

## 2026-01-08

**Commits:** `12e2f77`, `1445218`, `ddfa6c0`, `88d6bc8`, `569a89b`, `3b41d6d`

- Updated YAML template with new layer groups defined by Pascale
- Added XCF-to-HDF5 conversion pipeline (`xcf_to_hdf5.py`, `hdf5_utils.py`)
- Added Excel export for frequency-temporal contour representation (`export_contours_to_excel.py`)
- Binary morphology demo (`demos/bin_morph.py`) — proof of concept for ML preprocessing on binary masks
- Updated conda environment

## 2026-01-12

**Commit:** `dca3b1a`

- Added ML preparation utilities

## 2026-01-14

**Commits:** `609dd5a`, `f3c7c49`, `1f9abc2`, `1966061`, `952d7a7`, `299f49f`, `5ae5692`, `634d6bc`, `7ebaac9`, `e99310d`

- First full README with documentation
- Added GIF demonstrations of annotation workflow
- Added screenshot images
- Removed redundant PyTorch class
- Reconfigured binary morphology demo
- Adjusted conda environment

## 2026-01-15

**Commits:** `07d5315`, `7b77cfd`, `d1ef0d8`, `27f5d49`, `34ba310`, `a9bf069`

- Completed README with all images and GIFs
- Added detailed XCF template loading instructions
- Deleted old binary morphology demo (replaced by new version)

## 2026-01-16 — Windows Compatibility Sprint

**Commits:** `c75d787`, `ec97e35`, `0f9d8b9`, `e563740`, `59d2a27`, `471bc62`, `fb40c0a`, `c30b1c7`, `368da99`, `d4fc501`, `17e05f7`, `28a1b28`

- Major push for Windows compatibility
- Fixed conda environment export for cross-platform support
- Added gimpformats pip install
- Fixed Windows path separator and indexing issues
- Attempted monkey patching for Windows XCF compatibility (rolled back)
- Adjusted README with correct GIMP version info
- Merged Windows compatibility PR (#8)

## 2026-02-16

**Commits:** `8027af5`, `f5a2368`

- Updated GIMP download instructions in README
- Fixed formatting

## 2026-02-18

**Commits:** `cb73cd8`, `b8cee24`, `ef5f356`

- Improved HDF5 format (schema v2 consolidation: one file per audio clip)
- Adjusted Excel output
- Updated binary morphology demo
- Updated README with new HDF5 schema

## 2026-02-24

**Commits:** `a5d8deb`, `368e0af`

- Adjusted num_classes in Excel summary
- Updated morphology notebook
- Added Jupyter to conda environment
- Merged HDF5 improvements PR (#12)

## 2026-03-04 — Planning and Plugin Foundation

**Commits:** `aa9d2fd`, `250db11`, `b3c5541`, `35291ba`, `7817664`, `746553a`, `7bb6187`, `545d1c8`, `a5cf188`, `618fd1c`, `c258a42`, `b7cbf37`

- **Project planning initialized** — created `.planning/` directory with PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md
- Defined 4-phase roadmap: Core Foundations > Project Creation Wizard > Annotation Guardrails > In-GIMP Export
- Defined 15 v1 requirements across foundations, wizard, guardrails, and export categories
- Simplified environment.yml
- Added initial GIMP annotation plugin (`spectrace_annotator.py` baseline)
- Completed project research (stack, features, architecture, pitfalls)
- Created phase 1 context, research, validation strategy, and plans (01-01-PLAN, 01-02-PLAN)

## 2026-03-10 — Major Plugin Feature Delivery

**Commit:** `e52e2fb`

**This is the largest single commit in the project — 758 insertions across 4 files.**

Implemented the core plugin automation features (proposal deliverable CD1.1-1.4):

- **WAV file handler**: GIMP can now `File > Open` WAV files directly. The plugin calls `spectrace_wav_bridge.py` (new file) as a subprocess to generate spectrograms, then loads the result into GIMP.
- **Dynamic XCF template support**: `setup_annotation()` reads any XCF template file to build the layer hierarchy, falling back to a built-in 26-layer orca structure. `extract_template_structure()` and `create_template_layers()` handle arbitrary template hierarchies.
- **Tool enforcement**: Continuous GTK timer (`gobject.timeout_add`) enforces pencil settings (1px, hardness 100, dynamics off). Separate eraser size tracking maintains tool size independence.
- **Auto color switching**: Each annotation layer gets a unique foreground color via `generate_layer_colors()`. Switching layers auto-changes the drawing color.
- **macOS installer update**: `install_mac.sh` expanded significantly for the new plugin capabilities.
- **Bridge script**: New `spectrace_wav_bridge.py` provides the IPC mechanism between GIMP's Python 2.7 and the conda Python 3.11 environment.

## 2026-03-20 — Documentation Overhaul

**Commits:** `6f6a570`, `92e0f13`, `fa8f474`, `dc957fc`, `51a88b5`, `e3faec9`

- **Complete README rewrite** for the plugin-first workflow (Open WAV > Setup Annotation > Draw > Save)
- Added cross-platform installer documentation (`gimp_plugin/install.py` — unified Python installer replacing per-platform shell scripts)
- Added screenshots: WAV open dialog, Spectrace menu in GIMP
- Rewrote installation guide covering macOS, Linux (standard + Flatpak), Windows
- Added troubleshooting section
- Removed stale FossHub download links
- INSTALL.md updated for new plugin workflow

## 2026-03-24 — CallMark Import Implementation

Implemented full CallMark import support (proposal deliverable CD1.5):

- **`callmark_utils.py`** (new) — Parses CallMark Excel exports (.xlsx), converts spectrogram column indices to seconds using the formula `T_k = (k*h + n/2) / sr` with CallMark's params (n=256, h=28, sr=44100). Supports filtering by individual ID.
- **`spectrace_wav_bridge.py`** — Extended with three operating modes via `--mode` argument:
  - `spectrogram` (default, unchanged)
  - `parse-callmark` — reads Excel, returns JSON with all vocalizations and individual list
  - `segment-spectrogram` — generates spectrogram for a time-delimited WAV segment
- **`utils.py`** — Added `create_callmark_project()` for nested folder structure (`recording/individual/vNNN/`). Extended `process_audio_project()` to support optional offset/duration for `librosa.load()`.
- **`gimp_plugin/spectrace_annotator.py`** — Major additions:
  - GTK file picker shown when opening a WAV: "Select CallMark Export (Cancel to skip)"
  - Individual filter dropdown dialog
  - `_CALLMARK_SESSION` global for navigation state
  - `Next Vocalization` / `Previous Vocalization` menu entries under Filters > Spectrace
  - Auto-save XCF before navigating
  - Re-opens existing XCF when navigating back to previously annotated segments
  - Status messages showing "Vocalization 3/45 - R3277, Adults, Age 100 dph"

Tested bridge modes against real data (`callmarkexportexample/ZFVocalizations.xlsx`): 331 vocalizations, 19 individuals parsed correctly. Segment spectrogram generation verified with nested folder output.

End-to-end GIMP testing confirmed: opened ZF.wav, selected CallMark Excel, filtered by individual, navigated through vocalizations with Next/Previous. Session persistence via `~/.spectrace/callmark_session.json` resolved cross-plugin-call state loss.

---

## Work Remaining

Based on the semester proposal timeline (14 weeks from ~2026-03-20):

| Weeks | Planned Work | Status |
|---|---|---|
| 1-2 (done) | Familiarize with GIMP Python-Fu API; audit pain points; review CallMark export format | Complete |
| 3-5 (done) | Develop core automation plugins (template transfer, layer setup, tool config) | Complete |
| 6-7 (done) | Integrate plugins with start_project.py; implement CallMark import; begin documentation rewrite | Complete |
| 8-10 | CD2: Comparative feature evaluation of frequency contour annotation tools | Not started |
| 10-12 | CD3: Heterodyne validation pipeline using existing annotated clips | Not started |
| 12-13 | Extension (inter-expert agreement) + final report + presentation slides | Not started |
| 14 | Final presentation | Not started |

**Key remaining items:**
1. Comparative feature evaluation (CD2 — full deliverable)
2. Heterodyne validation pipeline (CD3 — full deliverable)
3. Inter-expert annotation agreement (Extension — if time permits)
4. Final report and presentation
