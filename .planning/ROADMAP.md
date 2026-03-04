# Roadmap: Spectrace

## Overview

Spectrace transforms from a collection of working-but-disconnected Python scripts into an integrated GIMP plugin that biologists use without leaving GIMP. The dependency chain is strict: template externalization enables the wizard, the wizard establishes subprocess/GTK patterns reused by export, and guardrails require both template config and parasite metadata. Four phases deliver the full v1 scope: harden the foundation, build the wizard, add guardrails, ship the export panel.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Foundations** - Externalize template config so the plugin is species-agnostic and metadata-ready
- [ ] **Phase 2: Project Creation Wizard** - One-dialog WAV-to-annotatable-workspace flow inside GIMP
- [ ] **Phase 3: Annotation Guardrails** - Prevent wrong-layer and dimension-mismatch errors during annotation
- [ ] **Phase 4: In-GIMP Export** - One-click HDF5, Excel, and visualization export without leaving GIMP

## Phase Details

### Phase 1: Core Foundations
**Goal**: The plugin reads layer structure from external config files instead of hardcoded constants, making Spectrace species-agnostic and ready for the wizard
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01
**Success Criteria** (what must be TRUE):
  1. Plugin loads layer structure (group names, layer names, colors) from an external JSON/YAML template file -- not from hardcoded Python constants
  2. Changing the template file changes the layers that get created -- no code edits required to define a new species template
  3. Existing orca annotation workflow still works end-to-end (spectrogram generation, layer injection, HDF5 export) after the refactor
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD

### Phase 2: Project Creation Wizard
**Goal**: A biologist opens GIMP, clicks one menu item, points at a WAV file, and gets a fully configured annotation workspace with correct layers, tool settings, and metadata -- no manual steps
**Depends on**: Phase 1
**Requirements**: WIZ-01, WIZ-02, WIZ-03, WIZ-04, WIZ-05, WIZ-06, WIZ-07, WIZ-08
**Success Criteria** (what must be TRUE):
  1. User can access the wizard via Filters > Bioacoustics > New Spectrogram Annotation in GIMP
  2. User can select a WAV file, choose a species preset, and get a spectrogram image with all template layers injected -- all matching image dimensions exactly
  3. After wizard completes, the drawing tool is pencil/size-1/hardness-100/dynamics-off with no user intervention
  4. FFT parameters, audio file path, template version, and pixel-to-Hz mapping are persisted as GIMP image parasites (survive save/reopen)
  5. User can optionally specify start/stop time for the WAV segment
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD
- [ ] 02-03: TBD

### Phase 3: Annotation Guardrails
**Goal**: The plugin actively prevents annotation mistakes that are undetectable until export -- wrong-layer drawing, dimension mismatches, and tool misconfiguration
**Depends on**: Phase 2
**Requirements**: GUARD-01, GUARD-02, GUARD-03
**Success Criteria** (what must be TRUE):
  1. User sees a warning when drawing on a layer group or wrong layer type
  2. Dimension mismatches between any annotation layer and the image canvas are detected and reported as errors (not silently resized)
  3. User can reset the drawing tool to the correct annotation state (pencil, size 1, hardness 100) via a single toolbar action
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: In-GIMP Export
**Goal**: Researchers export annotation data (HDF5, Excel, visualizations) from within GIMP without touching a terminal
**Depends on**: Phase 3
**Requirements**: EXP-01, EXP-02, EXP-03
**Success Criteria** (what must be TRUE):
  1. User can trigger export via Filters > Bioacoustics > Export Annotations menu item in GIMP
  2. Export refuses to proceed if layer names or dimensions are invalid, showing the user what is wrong
  3. HDF5 dataset, Excel summary, and overlay visualizations are generated in the project folder by calling spectrace core Python (not inside GIMP process)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 > 2 > 3 > 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Foundations | 0/TBD | Not started | - |
| 2. Project Creation Wizard | 0/TBD | Not started | - |
| 3. Annotation Guardrails | 0/TBD | Not started | - |
| 4. In-GIMP Export | 0/TBD | Not started | - |
