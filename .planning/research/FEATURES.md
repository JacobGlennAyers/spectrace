# Feature Research

**Domain:** Bioacoustic spectrogram annotation (GIMP-based plugin for whale vocalization mask creation)
**Researched:** 2026-03-04
**Confidence:** MEDIUM-HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these means the product feels incomplete or unusable compared to even basic alternatives like manual GIMP workflows.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Spectrogram generation from WAV | Every tool (Raven, Audacity, Whombat) generates spectrograms from audio. Without it, there is no image to annotate. | LOW | Already implemented in `utils.py`. Uses librosa STFT. |
| Template-based layer structure | Researchers need consistent annotation schemas across recordings. Raven uses selection tables; Spectrace uses layer groups. Consistency is non-negotiable for ML training data. | MEDIUM | Already implemented. Orca template exists in `templates/orca_template.xcf` and is replicated programmatically in the GIMP plugin. |
| Tool enforcement (pencil, 1px, hard) | Binary mask annotation requires exact pixel drawing. Any anti-aliasing, brush softness, or wrong tool destroys mask integrity. Users WILL forget to configure this. | LOW | Already implemented in `spectrace_annotator.py`. Enforces pencil, 1px, hardness 100, dynamics off. |
| Automatic layer color assignment | Annotators must visually distinguish which contour class they are drawing on. Raven color-codes selections; Spectrace must color-code layers. | LOW | Already implemented. `LAYER_COLORS` dict in plugin and `layer_color_mapping.json` in core. |
| Export to ML-ready format (HDF5) | The entire point of annotation is downstream ML consumption. Raven exports selection tables; Spectrace exports binary mask arrays. HDF5 with (C, H, W) mask arrays is the standard for segmentation pipelines. | MEDIUM | Already implemented in `xcf_to_hdf5.py` and `hdf5_utils.py`. |
| Export to spreadsheet (Excel) | Biologists need tabular data for statistical analysis outside of ML. Raven exports selection tables to text; researchers expect frequency/time data in familiar formats. | MEDIUM | Already implemented in `export_contours_to_excel.py`. |
| Visualization of annotations | Researchers need to verify annotations visually. Overlay views showing all classes on the spectrogram are standard in every annotation tool. | LOW | Already implemented in `produce_visuals.py`. |
| Layer-to-image-size enforcement | Template layers must match spectrogram pixel dimensions exactly. Mismatched dimensions silently corrupt all downstream data. Currently a manual step the README calls "CRITICAL." | MEDIUM | Not yet automated. The README documents this as a manual step that users frequently skip. This MUST be automated by the plugin -- it is explicitly called out in PROJECT.md as a guardrail. |
| Active layer switching with color sync | When a user clicks a different annotation layer, the foreground color must change automatically. Without this, users draw on the right layer in the wrong color (or worse, the wrong layer entirely). | MEDIUM | Already implemented via background polling in `SpectraceBackgroundMonitor`. Polls every 200ms. |
| Project creation wizard | Users need a guided path from "I have a WAV file" to "I am drawing on a correctly configured canvas." Raven does this automatically; the manual Spectrace workflow has 6+ critical steps. | HIGH | Not yet implemented as a GIMP wizard. Currently requires running `start_project.py` externally, manually opening files in GIMP, copy-pasting template layers, resizing, renaming, and unlocking. The wizard should collapse all of this into a single menu action. |
| Undo/redo support | Standard drawing capability. GIMP provides this natively. | NONE | Free from GIMP. No implementation needed. |
| Save/load annotation state (XCF) | Researchers work across sessions. XCF format preserves layer structure, pixel data, and metadata through saves. | NONE | Free from GIMP. No implementation needed. |
| Metadata persistence | FFT parameters, audio path, pixel-to-Hz mapping, template version must survive save/reload cycles. Without this, exported data cannot be linked back to source audio or calibrated. | MEDIUM | Planned via GIMP image parasites. Not yet implemented. |

### Differentiators (Competitive Advantage)

Features that set Spectrace apart from Raven, Audacity, Whombat, and manual workflows. These are not expected but create significant value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pixel-level binary mask annotation | Every other bioacoustic tool annotates with bounding boxes or time-frequency selections (rectangles). Spectrace is the only tool that produces pixel-precise contour masks. This matters enormously for: (a) ML segmentation models (U-Net, etc.) that need pixel masks not boxes, (b) capturing the actual shape of frequency contours, harmonics, and heterodynes rather than approximating with rectangles. | LOW | Core differentiator. Already the fundamental design decision. No other bioacoustic annotation tool produces per-pixel binary masks. |
| Semantic layer hierarchy | Raven uses flat label columns. Audacity uses simple text labels. Spectrace uses hierarchical layer groups that encode biological meaning (f0_LFC, harmonics_HFC, Heterodynes/0..12). This structure captures the complex multi-component nature of cetacean calls (biphonic components, subharmonics) in a way flat labels cannot. | MEDIUM | Already implemented in template structure. The orca template has ~25 semantically meaningful layers organized into groups. |
| Class registry with append-only migration | As annotation schemas evolve (new layer types discovered), old HDF5 files must be migrated to match new schemas without breaking existing channel indices. No other annotation tool handles schema evolution this gracefully. | LOW | Already implemented in `class_registry.py`. New classes are always appended; existing indices never change; migration adds zero-filled channels. |
| One-click export panel inside GIMP | Other tools require switching between annotation GUI and command-line scripts. Spectrace should let users export HDF5 + Excel + visualizations from within GIMP without touching a terminal. | HIGH | Not yet implemented. Currently requires running separate Python scripts outside GIMP. This is the highest-impact unbuilt differentiator. |
| Annotation guardrails and validation | Wrong-layer warnings, bounds checking, dimension validation. No other annotation tool actively prevents annotation mistakes. Raven silently accepts invalid selections. Audacity has no concept of annotation validity. | MEDIUM | Not yet implemented. Planned per PROJECT.md. Should warn when drawing on a group instead of a layer, when layers are locked, when dimensions mismatch. |
| Species-appropriate spectrogram presets | Orca calls have specific frequency ranges and optimal FFT parameters. Pre-baked presets eliminate the "what nfft should I use?" question. Other tools require manual configuration. | LOW | Partially implemented -- nfft is configurable but not preset per species. Could add preset profiles (orca: nfft=2048, freq_range=0-10kHz; humpback: nfft=4096, freq_range=0-4kHz). |
| Opinionated zero-configuration UX | Deliberately hide every parameter biologists should not touch. Raven exposes 70+ measurements and endless configuration. Spectrace's value is that a biologist who has never used GIMP can produce correct annotations. | MEDIUM | Partially implemented in the plugin (tool enforcement). The wizard and export panel are the remaining pieces. |
| Multiple annotation passes per recording | Researchers can create multiple annotation sets for the same audio (different interpretations, different vocalizations). HDF5 schema supports multiple annotation indices per clip. | LOW | Already implemented. `start_project.py` auto-increments project index; HDF5 schema has `annotations/<index>/masks`. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create complexity, scope creep, or architectural problems for a GIMP-based plugin.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time audio playback synced to spectrogram | Raven, Sonic Visualiser, and Audacity all play audio while highlighting the cursor position on the spectrogram. Researchers expect this. | GIMP is an image editor, not a media player. Adding audio playback requires GTK audio backends, synchronization with canvas scroll position, and platform-specific audio code. This would double project complexity for a feature that does not improve annotation quality -- researchers already know what the call sounds like before they annotate. | Recommend researchers preview audio in Audacity or a media player before annotating in GIMP. Spectrace stores the audio file path in metadata so it is easy to locate. |
| Bounding box annotations | Raven's primary annotation mode. Some ML pipelines expect bounding boxes, not masks. | Spectrace's entire value proposition is pixel-level masks. Adding bounding boxes would dilute the differentiator and create a "which format?" decision for users. Any bounding box can be derived from a mask, but not vice versa. | Export pipeline can compute bounding boxes from masks if needed (min/max row/col per class). This is a data transformation, not an annotation feature. |
| Automated detection / ML-assisted annotation | Whombat and modern tools offer ML-assisted pre-labeling. Researchers may ask "can the tool suggest where contours are?" | This is an entire ML system (model training, inference, model management) that would dwarf the annotation tool itself. It also requires GPU infrastructure and model versioning. | Spectrace creates training data for these models. Keep the tool focused on producing high-quality ground truth. Automated detection is a downstream consumer, not a feature of the annotation tool. |
| Web-based interface | Whombat and Koe are browser-based for collaboration and remote access. | GIMP is the annotation engine. A web interface would mean reimplementing all of GIMP's drawing tools in the browser, which is a separate multi-year project. | GIMP is installed locally. Collaboration happens through shared file systems or version-controlled XCF/HDF5 files. |
| Non-WAV audio format support | Researchers sometimes have MP3, FLAC, or other formats. | The lab has standardized on WAV. Adding format support adds dependencies (ffmpeg), edge cases (lossy compression artifacts), and codec compatibility issues across platforms. | Use a one-liner ffmpeg command to convert before import. Document this in the README. |
| Configurable spectrogram colormaps | Researchers may want viridis, plasma, magma, etc. instead of grayscale. | Colormap choice is aesthetic, not scientific. Binary mask annotation does not benefit from different colormaps. More options means more "which should I pick?" decisions for biologists. Grayscale maximizes contrast for annotation overlays. | Keep grayscale as the only option. If a researcher needs a different colormap for publication figures, they can generate it from the HDF5 data outside Spectrace. |
| Inter-annotator agreement metrics | Research projects often need to measure consistency between annotators. | This is a quality assurance process that happens after annotation, not during it. It requires comparing masks from multiple annotators, which is an analysis task better handled by a separate script. | Provide a utility script that computes IoU/Dice between two HDF5 annotation sets. Keep it outside the GIMP plugin. |
| Real-time spectrogram rendering inside GIMP | Dynamically generate and update spectrograms as users adjust FFT parameters. | GIMP cannot efficiently re-render large spectrograms in real time. This would require heavy DSP computation inside the plugin, which contradicts the two-tier architecture (DSP stays in Python core). | Wizard generates the spectrogram once with preset parameters. If different parameters are needed, re-run the wizard. |

## Feature Dependencies

```
[Project Creation Wizard]
    |--requires--> [Spectrogram Generation]
    |--requires--> [Template Layer Structure]
    |--requires--> [Layer-to-Image-Size Enforcement]
    |--requires--> [Metadata Persistence (parasites)]
    |--requires--> [Tool Enforcement]

[One-Click Export Panel]
    |--requires--> [Metadata Persistence (parasites)]
    |--requires--> [HDF5 Export]
    |--requires--> [Excel Export]
    |--requires--> [Visualization Generation]

[Annotation Guardrails]
    |--requires--> [Template Layer Structure]
    |--requires--> [Active Layer Switching]
    |--enhances--> [Project Creation Wizard]

[HDF5 Export]
    |--requires--> [Class Registry]
    |--requires--> [Metadata Persistence (parasites)]

[Active Layer Switching] --enhances--> [Annotation Guardrails]

[Species Presets] --enhances--> [Project Creation Wizard]
```

### Dependency Notes

- **Project Creation Wizard requires Metadata Persistence:** The wizard must write FFT params, audio path, and pixel-Hz mapping as image parasites so the export panel can read them later without user input.
- **One-Click Export requires Metadata Persistence:** Export needs to know FFT params, sample rate, and frequency axis mapping to produce calibrated HDF5/Excel output. Without parasites, users would have to re-enter this information.
- **Annotation Guardrails enhance the Wizard:** Guardrails are most valuable during annotation (warning about wrong layers), but the wizard should also validate the workspace on setup.
- **HDF5 Export requires Class Registry:** The registry ensures consistent channel ordering across files. Without it, mask arrays from different recordings would have incompatible class orderings.

## MVP Definition

### Launch With (v1)

The minimum set to replace the current 6+ step manual workflow with a single-click experience.

- [x] Spectrogram generation from WAV -- already implemented
- [x] Template layer structure injection -- already implemented
- [x] Tool enforcement (pencil settings) -- already implemented
- [x] Layer color auto-switching -- already implemented
- [x] HDF5 export -- already implemented
- [x] Excel export -- already implemented
- [x] Visualization generation -- already implemented
- [x] Class registry -- already implemented
- [ ] Project Creation Wizard (GIMP menu) -- collapse the manual workflow into one dialog
- [ ] Layer-to-image-size auto-enforcement -- automate the "CRITICAL STEP" users skip
- [ ] Metadata persistence via image parasites -- required for export panel

### Add After Validation (v1.x)

Features to add once the wizard-to-export workflow is validated with lab users.

- [ ] One-click export panel inside GIMP -- eliminate command-line export steps
- [ ] Annotation guardrails -- wrong-layer warnings, dimension validation, locked-layer detection
- [ ] Species presets -- pre-baked FFT/frequency profiles beyond orca (humpback, blue whale, etc.)
- [ ] Batch processing -- process multiple WAV files through the wizard in sequence

### Future Consideration (v2+)

Features to defer until the tool is stable and adopted.

- [ ] Custom template builder -- GUI for creating new annotation templates without editing XCF manually
- [ ] Inter-annotator comparison utility -- IoU/Dice metrics between annotation sets
- [ ] Bounding box derivation from masks -- export computed bounding boxes alongside pixel masks
- [ ] GIMP 3.0 compatibility -- when gimpformats supports GIMP 3.0 XCF format (blocked on upstream library)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Project Creation Wizard | HIGH | HIGH | P1 |
| Layer-to-image-size auto-enforcement | HIGH | LOW | P1 |
| Metadata persistence (parasites) | HIGH | MEDIUM | P1 |
| One-click export panel | HIGH | HIGH | P2 |
| Annotation guardrails | MEDIUM | MEDIUM | P2 |
| Species presets | MEDIUM | LOW | P2 |
| Batch processing | MEDIUM | MEDIUM | P3 |
| Custom template builder | LOW | HIGH | P3 |
| Inter-annotator comparison | LOW | MEDIUM | P3 |
| Bounding box derivation | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for launch -- these close the gap between "collection of scripts" and "integrated tool"
- P2: Should have -- these make the tool genuinely better than alternatives
- P3: Nice to have -- these serve edge cases or future growth

## Competitor Feature Analysis

| Feature | Raven Pro | Audacity | Whombat | Koe | Sonic Visualiser | Spectrace |
|---------|-----------|----------|---------|-----|-----------------|-----------|
| Annotation type | Bounding boxes (time-freq selections) | Time-only labels | Bounding boxes, time intervals, vertical lines | Acoustic unit segmentation | Time points, curves, labels | Pixel-level binary masks |
| Audio playback | Yes, synced cursor | Yes, full DAW | Yes, browser-based | Yes, web-based | Yes, synced with annotation layers | No (by design) |
| Spectrogram generation | Built-in, highly configurable | Built-in | Built-in, dynamic rendering | Built-in | Built-in, plugin-extensible | External (Python core) rendered as PNG |
| Export formats | Selection tables (text), correlation matrices | Labels (text) | JSON, CSV | CSV, feature matrices | Annotation layers (CSV, XML) | HDF5 (C,H,W masks), Excel, PNG overlays |
| ML integration | Limited (text export for external tools) | None | Active learning, model-assisted labeling | Feature extraction, ordination | Plugin-based feature extraction | Direct ML training data (HDF5 masks) |
| Annotation schema | Flat columns in selection table | Flat text labels | Key-value tag pairs | Multi-level classification | Generic annotation layers | Hierarchical semantic layer groups |
| Platform | Desktop (proprietary, paid) | Desktop (open source) | Web (open source) | Web (open source) | Desktop (open source) | GIMP plugin (open source) |
| Target audience | General bioacoustics researchers | Audio enthusiasts, some researchers | ML-focused bioacoustics | Vocalization classification | Music/audio analysis | Cetacean vocalization researchers creating ML training data |
| Schema evolution | Manual column management | N/A | Tag vocabulary management | Manual class management | Manual layer management | Append-only class registry with HDF5 migration |
| Configuration | 70+ measurements, many options | Moderate | Moderate, sensible defaults | Moderate | Plugin-dependent | Opinionated: minimal configuration by design |

## Sources

- [Raven Pro - Cornell Lab of Ornithology](https://www.ravensoundsoftware.com/software/raven-pro/)
- [Raven Selection Review and Annotation](https://www.ravensoundsoftware.com/knowledge-base/selection-review-and-annotation/)
- [Koe: Web-based software to classify acoustic units (Fukuzawa 2020)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13336)
- [Koe GitHub](https://github.com/fzyukio/koe)
- [Whombat: An open-source audio annotation tool for ML-assisted bioacoustics (2025)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14468)
- [Whombat GitHub](https://github.com/mbsantiago/whombat)
- [Sonic Visualiser Features](https://www.sonicvisualiser.org/features.html)
- [Audacity Spectrogram View](https://manual.audacityteam.org/man/spectrogram_view.html)
- [Audacity Spectral Selection](https://manual.audacityteam.org/man/spectral_selection.html)
- [NEAL: an open-source tool for audio annotation (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10461540/)
- [Automatic detection for bioacoustic research: a practical guide (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11885706/)
- [OpenSoundscape (Lapp 2023)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14196)
- [An image processing based paradigm for extraction of tonal sounds in cetacean communications (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3874055/)
- [Annotated Orcinus orca acoustic signals dataset (Nature Scientific Data 2025)](https://www.nature.com/articles/s41597-025-05281-5)

---
*Feature research for: Bioacoustic spectrogram annotation (GIMP plugin)*
*Researched: 2026-03-04*
