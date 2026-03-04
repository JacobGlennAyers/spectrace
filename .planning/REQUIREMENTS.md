# Requirements: Spectrace

**Defined:** 2026-03-04
**Core Value:** A biologist who has never used GIMP can produce correctly formatted annotation masks without making a single configuration mistake.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundations

- [ ] **FOUND-01**: Layer schema externalized from hardcoded orca constants to template config files, enabling species-agnostic annotation

### Wizard

- [ ] **WIZ-01**: User can access annotation workflow via Filters → Bioacoustics → New Spectrogram Annotation menu item in GIMP
- [ ] **WIZ-02**: User can select a WAV file and optionally specify start/stop time via file chooser dialog
- [ ] **WIZ-03**: User can select a species preset (orca default) from dropdown that configures spectrogram parameters
- [ ] **WIZ-04**: Spectrogram is generated from WAV file by calling spectrace core Python (not inside GIMP)
- [ ] **WIZ-05**: Template layers are automatically selected and injected into the workspace with correct names (no " copy" suffix)
- [ ] **WIZ-06**: All layers are automatically resized to match image dimensions with zero tolerance for mismatches
- [ ] **WIZ-07**: Drawing tool is auto-configured (pencil, size 1, hardness 100, dynamics off) on wizard completion
- [ ] **WIZ-08**: FFT params, audio path, template version, and pixel↔Hz mapping are stored as GIMP image parasites

### Export

- [ ] **EXP-01**: User can export annotations via Filters → Bioacoustics → Export Annotations menu item in GIMP
- [ ] **EXP-02**: Export validates layer names and dimensions before proceeding, rejecting invalid state
- [ ] **EXP-03**: Export delegates to spectrace core Python for HDF5, Excel, and visualization generation

### Guardrails

- [ ] **GUARD-01**: User is warned when drawing on a layer group or wrong layer type
- [ ] **GUARD-02**: Dimension mismatches between layers and image are detected and rejected (not silently resized)
- [ ] **GUARD-03**: User can reset drawing tool to correct state via toolbar button ("Reset drawing tool")

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Annotation Quality

- **GUARD-04**: User is warned when drawing outside time/frequency bounds
- **GUARD-05**: Pre-export validation report showing annotation statistics per layer

### Batch Operations

- **BATCH-01**: User can process multiple WAV files through the wizard in sequence
- **BATCH-02**: User can export all open annotation projects in one operation

### Species Support

- **SPEC-01**: Humpback whale spectrogram presets and annotation template
- **SPEC-02**: Blue whale spectrogram presets and annotation template
- **SPEC-03**: Custom template builder GUI for creating new species templates

### Compatibility

- **COMPAT-01**: GIMP 3.0 compatibility (Python 3 + GObject Introspection migration)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time audio playback | GIMP is a renderer/editor, not a media player. Researchers preview audio in Audacity before annotating. |
| Bounding box annotations | Spectrace's differentiator is pixel-level masks. Bounding boxes can be derived from masks in export if needed. |
| ML-assisted annotation | Spectrace creates training data for ML models, it doesn't run them. |
| Web-based interface | Would require reimplementing GIMP's drawing tools in the browser. |
| Non-WAV audio formats | Lab standardized on WAV. Convert with ffmpeg before import. |
| Configurable colormaps | Grayscale maximizes contrast for annotation. Different colormaps are aesthetic, not scientific. |
| Real-time spectrogram rendering | Contradicts two-tier architecture. Wizard generates once with presets. |
| Pickle-to-JSON metadata migration | Current pickle approach works for the lab's workflow. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | — | Pending |
| WIZ-01 | — | Pending |
| WIZ-02 | — | Pending |
| WIZ-03 | — | Pending |
| WIZ-04 | — | Pending |
| WIZ-05 | — | Pending |
| WIZ-06 | — | Pending |
| WIZ-07 | — | Pending |
| WIZ-08 | — | Pending |
| EXP-01 | — | Pending |
| EXP-02 | — | Pending |
| EXP-03 | — | Pending |
| GUARD-01 | — | Pending |
| GUARD-02 | — | Pending |
| GUARD-03 | — | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 0
- Unmapped: 15 ⚠️

---
*Requirements defined: 2026-03-04*
*Last updated: 2026-03-04 after initial definition*
