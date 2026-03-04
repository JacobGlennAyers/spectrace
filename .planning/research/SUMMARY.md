# Project Research Summary

**Project:** Spectrace — GIMP-based bioacoustic spectrogram annotation toolkit
**Domain:** Scientific annotation tooling / bioacoustics ML data creation
**Researched:** 2026-03-04
**Confidence:** HIGH

## Executive Summary

Spectrace is a GIMP plugin that enables biologists to produce pixel-precise binary annotation masks of cetacean vocalizations for downstream ML training pipelines. The core innovation — using GIMP layers as semantically named annotation classes — is fundamentally sound and has no viable analog in competing tools (Raven Pro, Whombat, Audacity, Sonic Visualiser all use bounding boxes or time-frequency rectangles, not per-pixel contours). The existing codebase is more mature than it may appear: spectrogram generation, layer structure injection, HDF5/Excel export, class registry, and annotation color switching are all working. The critical gap is the integration layer: researchers currently must run external Python scripts, manually copy template layers, resize canvases, and unlock layers — a 6+ step error-prone workflow that the plugin should collapse into two or three menu clicks.

The recommended approach is a phased build that starts by hardening the foundation (pixel-exact spectrogram output, JSON metadata replacing pickle, external config replacing hardcoded layer constants), then builds the high-impact GIMP wizard (WAV → annotatable canvas in one dialog), then adds annotation guardrails, and finally delivers the in-GIMP export panel that eliminates the terminal entirely. This order is forced by dependency: the wizard requires metadata persistence (parasites), guardrails require knowing the correct state (parasites + template sync), and the export panel requires all of the above. The existing two-tier architecture (GIMP plugin in Python 2.7, scientific core in Python 3) is correct and must be maintained — the plugin communicates with the core exclusively through files on disk and subprocess calls, never via shared memory or direct imports.

The primary technical risks are spectrogram pixel-dimension drift (matplotlib's `bbox_inches='tight'` can silently alter PNG dimensions, corrupting frequency mappings), layer dimension mismatches being silently repaired rather than rejected (the `scipy.ndimage.zoom` fallback corrupts scientific data), and the looming GIMP 3.0 migration (the current plugin is completely incompatible with GIMP 3.0's GObject Introspection API). All three risks are preventable if addressed in Phase 1, before any annotation data is generated in the new system. The GIMP 3.0 migration is a future milestone, not an immediate crisis, but the two-tier boundary must be kept strict so that only the GIMP adapter layer (a single file) needs rewriting when the time comes.

## Key Findings

### Recommended Stack

The stack is already largely settled and working. GIMP 2.10 + Python-Fu (Python 2.7) is the required plugin environment and should not be changed until the lab migrates to GIMP 3.0. The scientific core runs on Python 3.11 with NumPy, SciPy, librosa, matplotlib, Pillow, h5py, pandas, openpyxl, and gimpformats — all installed via Conda from environment.yml. The only notable dependency risk is `gimpformats` (single-maintainer library, no conda-forge package, calendar-versioned on PyPI), which is nonetheless the only viable option for headless XCF parsing.

**Core technologies:**
- GIMP 2.10 + Python-Fu (Python 2.7): Annotation interface — only viable option; GIMP 3.0 requires full plugin rewrite, defer migration
- Python 3.11 + conda-forge stack: Scientific core — stable, all dependencies tested, pin major.minor versions in environment.yml
- librosa 0.11.x: Audio loading and STFT — standard library, deeply integrated, well-maintained
- h5py 3.15.x + HDF5 schema v2.0: ML-ready dataset storage — rock solid, schema already implemented and working
- gimpformats 2025: Headless XCF parsing — MEDIUM risk (single maintainer), no viable alternative; pin to exact calendar version
- GIMP image parasites (JSON): Metadata persistence — replaces fragile metadata.pkl; parasites travel with the XCF file

**What to avoid:**
- pickle for metadata (security risk, not portable across Python versions/environments)
- `bbox_inches='tight'` in matplotlib savefig (silently changes pixel dimensions)
- `scipy.ndimage.zoom` on annotation masks (nearest-neighbor resize corrupts binary annotations)
- TensorFlow/PyTorch (Spectrace creates training data, it does not train models)
- GIMP 3.0 (full plugin rewrite required; wait for lab adoption)

### Expected Features

The MVP closes the gap between "collection of scripts" and "integrated tool." Everything on the already-implemented list is working; the three P1 gaps are what transform Spectrace from a prototype into something researchers can adopt without GIMP expertise.

**Must have — already implemented:**
- Spectrogram generation from WAV (librosa STFT)
- Template layer structure injection (orca template, ~26 layers)
- Tool enforcement (pencil, 1px, hardness 100, dynamics off)
- Layer color auto-switching via background polling (200ms timer)
- HDF5 export (schema v2.0, C×H×W masks)
- Excel export (contour statistics)
- Visualization generation (overlay and per-layer PNGs)
- Append-only class registry with HDF5 migration

**Must have — not yet implemented (P1 gaps):**
- Project Creation Wizard (collapse 6+ manual steps into one GIMP dialog)
- Layer-to-image-size auto-enforcement (automate the "CRITICAL STEP" currently skipped by users)
- Metadata persistence via image parasites (required by wizard and export panel; replaces metadata.pkl)

**Should have (v1.x, after wizard is validated):**
- One-click in-GIMP export panel (eliminate command-line export)
- Annotation guardrails (wrong-layer warnings, dimension validation, locked-layer detection)
- Species presets (pre-baked FFT/frequency profiles beyond orca)
- Batch processing (multiple WAV files through the wizard in sequence)

**Defer (v2+):**
- Custom template builder (GUI for new annotation templates)
- Inter-annotator comparison utility (IoU/Dice metrics)
- GIMP 3.0 compatibility (blocked on gimpformats upstream + lab adoption timeline)
- Bounding box derivation from masks
- Automated detection / ML-assisted annotation (out of scope; Spectrace creates ground truth)

### Architecture Approach

The two-tier architecture (GIMP plugin in Python 2.7 inside the GIMP process; scientific core in Python 3 outside) is the correct and only viable design given GIMP 2.10's constraints. Communication between tiers is exclusively file-based (XCF files, PNG files, JSON metadata) with no runtime coupling. The plugin invokes core scripts via subprocess and polls for completion using `gobject.timeout_add` to avoid blocking the GTK main loop. The XCF file is the single source of truth: it carries pixel data (annotation layers), semantic structure (layer names and groups), and metadata (image parasites). All downstream artifacts (HDF5, Excel, visualizations) are derived from XCF and can be regenerated.

**Major components:**
1. `spectrace_annotator.py` (GIMP plugin, Python 2.7) — layer creation, tool enforcement, color switching, GTK lifecycle; currently lacks wizard, export panel, and guardrails
2. `utils.py` / `hdf5_utils.py` / `class_registry.py` (Python 3 core) — spectrogram generation, XCF→HDF5 conversion, class vocabulary management; mature and functional
3. `export_contours_to_excel.py` / `produce_visuals.py` (Python 3 export) — HDF5→Excel, PNG overlay generation; mature
4. `start_project.py` (Python 3 CLI) — project creation entry point; functional but exposes the manual workflow gap
5. Templates (`orca_template.xcf`, `orca_template.yaml`) — authoritative layer structure; currently replicated as hardcoded constants in the plugin (known fragility)

**Key patterns to follow:**
- Fire-and-forget subprocess with `gobject.timeout_add` polling (never block the GTK main loop)
- XCF as single source of truth (never store annotation state outside XCF that cannot be reconstructed)
- Layer names as semantic encoding (layer name = class identity; no sidecar files needed)
- Append-only class registry (existing channel indices never change; new classes zero-fill old files)
- JSON parasites for metadata (replace metadata.pkl; parasites travel with the XCF file)

### Critical Pitfalls

1. **Spectrogram pixel-dimension drift** — matplotlib's `bbox_inches='tight'` can shift output dimensions by 1-3 pixels, causing annotation masks to map to wrong frequencies. Fix: use explicit DPI, save with `bbox_inches=None`, and assert `Image.open(png).size == (S_db.shape[1], S_db.shape[0])` after every save. Address in Phase 1.

2. **Silent mask corruption via resize** — `scipy.ndimage.zoom` is currently called as a fallback when layer dimensions mismatch the canvas. Nearest-neighbor interpolation on binary masks adds or removes annotation pixels at boundaries, corrupting scientific data. Fix: the export pipeline must reject (raise an error), not silently fix, mismatched layers. Address in Phase 1.

3. **Pickle metadata fragility** — `metadata.pkl` is non-portable across Python versions, non-human-readable, and a security risk. It breaks silently on module renames or class refactors. Fix: replace with JSON (all metadata values are JSON-serializable primitives); keep pickle deserialization as a read-only migration path for existing projects. Address in Phase 1.

4. **Hardcoded orca-specific layer structure** — `LAYER_STRUCTURE`, root group name, section definitions, and color mapping are all hardcoded in `spectrace_annotator.py`. Adding a second species requires source code edits in multiple files. Fix: externalize to a JSON/YAML template file that the plugin reads at startup via `json` (available in Python 2.7 stdlib). Address in Phase 1 (required before the wizard, which must offer template selection).

5. **`gtk.main()` blocking and no graceful shutdown** — the plugin enters a GTK main loop that blocks the Python-Fu interpreter; there is no clean shutdown path when GIMP or the image is closed. Fix: register an image destroy handler, add a try/finally around `gtk.main()` that calls `monitor.stop()` and `gtk.main_quit()`. Address in Phase 2 (wizard lifecycle management).

## Implications for Roadmap

Based on the combined research, the dependency graph forces a clear four-phase structure. Each phase unblocks the next: metadata persistence enables the wizard; the wizard patterns (subprocess, GTK dialogs) enable the export panel; guardrails require the template sync and parasites from Phase 1.

### Phase 1: Core Foundations

**Rationale:** Three critical issues must be resolved before any new annotation data is generated in the system: spectrogram pixel accuracy, metadata format, and template externalization. These are not user-visible features, but every subsequent phase depends on them being correct. Getting them wrong now means corrupted datasets and code that is impossible to extend.

**Delivers:**
- Pixel-exact spectrogram PNG generation with automated dimension assertions
- `metadata.json` replacing `metadata.pkl` (with migration script for existing projects)
- External JSON template config replacing hardcoded `LAYER_STRUCTURE` constants in the plugin
- Plugin reads template from a JSON file at startup via Python 2.7's stdlib `json`
- Export pipeline rejects (errors, does not resize) dimension-mismatched layers
- Core modules refactored into `spectrace/` package structure (core/, export/, registry/, cli/)
- Automated test: core modules import cleanly without any GIMP-related imports (enforces two-tier boundary)

**Addresses features:** Layer-to-image-size enforcement (export side), metadata persistence foundation
**Avoids pitfalls:** Spectrogram pixel drift, mask resize corruption, pickle fragility, hardcoded species config

### Phase 2: Project Creation Wizard

**Rationale:** The Project Creation Wizard is the highest-impact unbuilt feature. It replaces a 6+ step error-prone manual workflow (run external script, open PNG, copy-paste template layers, resize canvas, rename layers, unlock layers) with a single Filters > Spectrace > New Project dialog. It is the feature that determines whether Spectrace is adoptable by a biologist who has never used GIMP. The wizard also establishes the subprocess and GTK dialog patterns that Phase 4 (export panel) will reuse.

**Delivers:**
- Filters > Spectrace > New Project dialog (GTK file chooser + parameter inputs)
- Dialog invokes `spectrace.cli.new_project` via subprocess, polls via `gobject.timeout_add`
- Spectrogram loaded directly into GIMP via PDB after generation
- Template layers injected automatically at canvas size
- All metadata written as image parasites (FFT params, audio path, pixel-to-Hz mapping, template version)
- Clean plugin lifecycle: destroy handler, graceful `gtk.main()` shutdown on image close

**Addresses features:** Project Creation Wizard (P1), metadata persistence via parasites (P1), layer-to-image-size auto-enforcement (plugin side)
**Avoids pitfalls:** `gtk.main()` shutdown hang, layer dimension mismatch (inject at correct size from the start)
**Uses:** subprocess + `gobject.timeout_add` pattern, GIMP parasite API, Python 2.7 GTK file chooser

### Phase 3: Annotation Guardrails

**Rationale:** Guardrails prevent the wrong-layer and dimension-mismatch annotation errors that are undetectable until export (or never). Wrong-layer annotations require manual expert review to find — there is no automated recovery. Guardrails are the investment that keeps data quality high as the user base grows and as researchers become faster and less careful. They build on Phase 1 (template sync, parasite metadata) and Phase 2 (layer polling infrastructure).

**Delivers:**
- Layer dimension polling: warn immediately if any annotation layer diverges from canvas size; auto-correct via `pdb.gimp_layer_resize_to_image_size()`
- Wrong-layer detection: if user draws pixels and active layer is a group (not a leaf layer), warn in status bar
- Layer name validation: warn if a layer is renamed to something not in the template config
- Pre-export annotation check: "Check Annotations" menu item reporting pixel counts per layer, empty layers, and validation warnings
- Reduced polling overhead: only call `set_foreground_color` and `gimp_displays_flush` when active layer actually changes (not unconditionally every 200ms)
- Replace osascript keystroke hack with correct GIMP PDB call for pencil tool activation

**Addresses features:** Annotation guardrails (P2), layer-to-image-size auto-enforcement (active enforcement during annotation)
**Avoids pitfalls:** Layer dimension mismatch (active prevention), color/layer identity confusion, wrong-layer annotations

### Phase 4: In-GIMP Export Panel

**Rationale:** The export panel is the second highest-impact unbuilt feature: it eliminates the terminal entirely from the annotation workflow. Currently, researchers must switch to a terminal, navigate to the project directory, and run multiple Python scripts to produce HDF5, Excel, and PNG outputs. The export panel delivers this in one Filters > Spectrace > Export dialog. It reuses the subprocess patterns from Phase 2 and requires the parasite metadata from Phase 1 (to know FFT params and audio path without user re-entry).

**Delivers:**
- Filters > Spectrace > Export dialog (checkboxes for HDF5, Excel, visualizations)
- Reads all export parameters from image parasites (no re-entry of FFT params)
- Invokes `spectrace.cli.export_hdf5` and `spectrace.cli.export_excel` via subprocess
- Progress bar via `pdb.gimp_progress_init/update` polled with `gobject.timeout_add`
- Export validation: refuses to export if parasite metadata is missing or if layer dimensions fail validation
- Batch export: option to process all XCF files in a project folder from within GIMP

**Addresses features:** One-click export panel (P2), batch processing (P3 starter)
**Uses:** Reuses subprocess runner pattern from Phase 2, parasite reader from Phase 1

### Phase 5: Species Presets and Multi-Species Support (v1.x)

**Rationale:** Once the wizard, guardrails, and export panel are validated with orca annotations, extending to additional species (humpback, blue whale, dolphins) requires only new JSON template files. This phase adds a template selector to the wizard dialog and ships 2-3 additional species templates. It is deferred until Phase 1-4 are validated because a multi-species template system built before the wizard exists has no delivery vehicle.

**Delivers:**
- Template selection in the New Project wizard (dropdown showing available .json templates)
- Humpback whale template (nfft=4096, freq_range=0-4kHz, adjusted layer hierarchy)
- Template validation: CI check that YAML and JSON templates stay synchronized
- Documentation for researchers to create new species templates without editing Python

**Addresses features:** Species presets (P2), config-driven layer structure (architectural maturity)

### Phase Ordering Rationale

- Phase 1 must be first: pixel accuracy, metadata format, and template externalization are load-bearing. Building anything on top of pixel-drifted spectrograms or pickle metadata creates technical debt that corrupts scientific data.
- Phase 2 must precede Phase 4: the wizard establishes the subprocess runner and GTK dialog patterns that the export panel reuses. Building both independently duplicates code and creates inconsistent UI patterns.
- Phase 3 fits between wizard and export panel: guardrails require the polling infrastructure (from Phase 2) and parasite metadata (from Phase 1). They also make Phase 4 safer — an export panel without guardrails would let researchers export corrupted data silently.
- Phase 5 is last: the template system built in Phase 1 makes Phase 5 low effort; it is correctly deferred until the core workflow is validated with real users.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Wizard):** GTK file chooser patterns in GIMP Python-Fu 2.7 are sparsely documented. The `gobject.timeout_add` + subprocess pattern is known but the specific GTK dialog widgets available in GIMP 2.10's bundled GTK need verification against actual GIMP plugin examples.
- **Phase 3 (Guardrails):** Whether GIMP 2.10 exposes layer-change events (vs. requiring polling) is unclear from available documentation. Event-driven layer change detection would be preferable to 200ms polling but may not exist in the Python-Fu API.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Core Foundations):** JSON serialization, matplotlib figure saving, h5py attribute writing — all well-documented with clear correct approaches identified in PITFALLS.md.
- **Phase 4 (Export Panel):** Reuses subprocess patterns validated in Phase 2. No new integration territory.
- **Phase 5 (Species Presets):** Configuration file reading and template management are standard Python patterns; no novel integration required.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing codebase confirms all technology choices work in practice. GIMP 2.10 / Python-Fu constraint is verified. gimpformats is the only risk (single maintainer). |
| Features | HIGH | Competitor analysis is thorough (Raven, Whombat, Audacity, Sonic Visualiser, Koe). Table stakes and differentiators are well-defined. Prioritization matrix reflects clear user value judgment. |
| Architecture | HIGH | Based on direct codebase analysis plus GIMP developer documentation. The two-tier boundary constraint is verified. Parasite API is documented and working. |
| Pitfalls | HIGH | Most pitfalls are derived from existing code issues (observable, not hypothetical). GIMP 3.0 risk is documented against official release notes. matplotlib dimension issue referenced against confirmed upstream bug report. |

**Overall confidence:** HIGH

### Gaps to Address

- **gimpformats parasite reading:** ARCHITECTURE.md documents that gimpformats can read parasites from XCF files but this was not verified against a specific version. Verify `doc.parasites` works in gimpformats `2025` before relying on it in the export pipeline.
- **GTK dialog capabilities in GIMP 2.10 Python-Fu:** The exact set of GTK widgets available in GIMP 2.10's bundled Python-Fu runtime needs hands-on verification during Phase 2 planning. The research assumes standard GTK 2 file chooser dialogs are accessible — confirm against an actual running GIMP 2.10 instance.
- **Layer-change event API:** Whether GIMP 2.10's Python-Fu exposes an event or callback for active layer changes (as opposed to requiring polling) should be investigated at the start of Phase 3. If events exist, the polling architecture can be simplified significantly.
- **osascript pencil tool activation:** The current osascript keystroke hack for tool switching is platform-specific and requires accessibility permissions. The correct GIMP PDB call to activate a named tool should be identified before Phase 3 ships.

## Sources

### Primary (HIGH confidence)
- [GIMP Developer: Python Plug-Ins Tutorial](https://developer.gimp.org/resource/writing-a-plug-in/tutorial-python/) — plugin registration, PDB API, lifecycle
- [GIMP Developer: Parasite Registry](https://developer.gimp.org/core/specifications/parasites/) — metadata persistence format and flags
- [GIMP 3.0 Release Notes](https://www.gimp.org/release-notes/gimp-3.0.html) — breaking API changes, migration scope
- [GIMP 3.0.8 Release](https://www.gimp.org/news/2026/01/24/gimp-3-0-8-released/) — API stability status for 3.x releases
- [GimpFormats on PyPI](https://pypi.org/project/gimpformats/) — version 2025, pure Python XCF parser
- [FHPythonUtils/GimpFormats on GitHub](https://github.com/FHPythonUtils/GimpFormats) — active fork, issue tracker
- [librosa 0.11.0 documentation](https://librosa.org/doc/main/index.html) — confirmed current stable
- [h5py 3.15.1 documentation](https://docs.h5py.org/) — confirmed current stable
- [matplotlib savefig bbox_inches issue #11681](https://github.com/matplotlib/matplotlib/issues/11681) — confirmed bug: bbox_inches='tight' does not respect figsize
- Codebase analysis: spectrace_annotator.py, utils.py, hdf5_utils.py, class_registry.py, start_project.py, export_contours_to_excel.py

### Secondary (MEDIUM confidence)
- [Whombat: Bioacoustic Annotation Tool](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14468) — competitor feature comparison
- [Raven Pro](https://www.ravensoundsoftware.com/software/raven-pro/) — industry-standard comparison, annotation paradigm contrast
- [OpenSoundscape (Lapp 2023)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14196) — bioacoustics ML tooling landscape
- [Annotated Orcinus orca acoustic signals dataset (Nature 2025)](https://www.nature.com/articles/s41597-025-05281-5) — domain validation for orca annotation requirements
- [GIMP 3.0 Python Plugin Migration Guide](https://gist.github.com/hnbdr/d4aa13f830b104b23694a5ac275958f8) — practical porting guide for future migration
- [Interpolation Artifacts in Segmentation Masks](https://github.com/albumentations-team/albumentations/issues/1294) — why nearest-neighbor resize corrupts binary masks

### Tertiary (MEDIUM-LOW confidence)
- [XDA: Reasons to stick with GIMP 2.10](https://www.xda-developers.com/reasons-still-use-gimp-2-10-instead-of-3-0/) — user adoption context for GIMP 2.10 vs 3.0
- [GIMP Forum: Python-Fu to Launch Python 3](https://www.gimp-forum.net/Thread-Gimp-Python-fu-2-7-to-launch-Python-3-for-editing-images) — subprocess pattern validation
- [Writing GIMP 3.0 Plugins](https://schoenitzer.de/blog/2025/Gimp%203.0%20Plugin%20Ressources.html) — community migration resources

---
*Research completed: 2026-03-04*
*Ready for roadmap: yes*
