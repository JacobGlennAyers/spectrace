# Spectrace

## What This Is

A GIMP plugin and Python toolkit that lets bioacoustics researchers create precise binary mask annotations on spectrograms without leaving GIMP. Biologists click one menu item, point at a WAV file, and get a fully configured annotation workspace — then export results in one click. The plugin is opinionated by design: it eliminates decisions so researchers focus on drawing, not configuring tools.

## Core Value

A biologist who has never used GIMP can open it, click Filters → Bioacoustics → New Spectrogram Annotation, and produce correctly formatted annotation masks without making a single configuration mistake.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] GIMP plugin wizard that guides users from WAV file to ready-to-annotate workspace
- [ ] Spectrogram generation from WAV files with species-appropriate presets (whales)
- [ ] Template-based layer structure injection (e.g., frequency contours, harmonics, heterodynes)
- [ ] Automatic tool enforcement (pencil, size 1, hardness 100, no dynamics)
- [ ] One-click export panel: HDF5, Excel, visualizations — all from within GIMP
- [ ] Annotation guardrails: wrong-layer warnings, bounds checking, dimension validation
- [ ] Image parasites for metadata persistence (FFT params, audio path, template version, pixel↔Hz mapping)
- [ ] Layer resize enforcement — zero tolerance for dimension mismatches
- [ ] Batch visualization generation (overlays and individual layers)
- [ ] Color management with automatic class-based color assignment

### Out of Scope

- Real-time audio playback in GIMP — GIMP is a renderer/editor, not a media player
- Non-WAV audio format support — lab standardized on WAV
- Mobile or web interface — GIMP desktop only
- ML model training — Spectrace creates training data, doesn't consume it
- Fancy interactive plotting inside GIMP — DSP stays in Python core

## Context

- **Existing code:** Partial implementation exists — core Python modules for spectrogram generation, metadata handling, templates, and I/O. GIMP plugin structure started but wizard flow, export panel, and guardrails not yet built.
- **Architecture:** Two-tier design. `spectrace/core/` handles all DSP and data processing in standard Python. `gimp_plugin/` is a thin GIMP Python-Fu layer that calls core modules and manages the GIMP UI. Heavy computation never runs inside GIMP.
- **Target users:** Small bioacoustics lab team studying whale vocalizations. Users are biologists, not programmers. The plugin must be foolproof — if a user can accidentally do the wrong thing, the plugin has failed.
- **GIMP version:** Python-Fu (Script-Fu not used). GTK dialogs styled like Audacity/ImageJ/Raven — clinical, not creative.
- **Species focus:** Whales (orcas and other cetaceans). Frequency ranges and spectrogram parameters need species-appropriate defaults.
- **Annotation workflow:** Layer groups encode semantic meaning (fundamental frequency, harmonics, heterodynes). Layer names carry semantics parsed by export pipeline. Templates ensure consistency across projects and researchers.

## Constraints

- **Tech stack**: Python + GIMP Python-Fu — no external GUI frameworks beyond GTK (which GIMP provides)
- **DSP boundary**: All spectrogram generation and signal processing happens outside GIMP, in spectrace core Python modules
- **GIMP limitations**: No long-running blocking jobs inside GIMP; no fancy plotting; plugin communicates results back via file I/O or direct module import
- **UX philosophy**: Zero exposed options that biologists shouldn't touch. Defaults are mandatory. If it requires a "CRITICAL STEP" in documentation, it should be automated instead.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GIMP as annotation interface (not custom UI) | Familiar layer-based drawing tools, existing ecosystem, no GUI development burden | — Pending |
| Two-tier architecture (core Python + thin GIMP plugin) | GIMP Python-Fu is bad at DSP; Python is bad at interactive drawing. Each does what it's good at. | — Pending |
| Opinionated defaults over user configuration | Target users are biologists who shouldn't make tool configuration decisions | — Pending |
| Layer names as semantic encoding | Works with GIMP's existing model, parseable by export pipeline, survives XCF saves | — Pending |
| Image parasites for metadata | Persist FFT params, mappings through saves; criminally underused GIMP feature | — Pending |

---
*Last updated: 2026-03-04 after initialization*
