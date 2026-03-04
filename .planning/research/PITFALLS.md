# Pitfalls Research

**Domain:** GIMP-based bioacoustic spectrogram annotation plugin (Spectrace)
**Researched:** 2026-03-04
**Confidence:** HIGH (derived from codebase analysis, GIMP developer docs, bioacoustics literature, and known library issues)

## Critical Pitfalls

### Pitfall 1: Spectrogram Pixel-to-Physical Dimension Drift

**What goes wrong:**
The spectrogram PNG saved by matplotlib does not have the exact pixel dimensions expected by the annotation pipeline. `bbox_inches='tight'` in `plt.savefig()` recalculates the bounding box and can add or remove pixels compared to `S_db.shape`. The current code sets `fig.set_size_inches(S_db.shape[1] / 100, S_db.shape[0] / 100)` then uses `bbox_inches="tight", pad_inches=0` -- but `bbox_inches='tight'` overrides the figure size to fit visible content, which can silently alter output dimensions by a few pixels. This means the pixel-to-Hz and pixel-to-seconds mappings stored in metadata become slightly wrong. Annotation masks drawn at pixel coordinates then map to the wrong frequencies when exported.

**Why it happens:**
Matplotlib's `bbox_inches='tight'` is designed to trim whitespace, but it does not guarantee the output has the same pixel dimensions as the data array. DPI rounding, anti-aliasing, and tight-bbox recalculation all contribute. The current code relies on a division by 100 (implicit DPI assumption) that may not survive across platforms or matplotlib versions.

**How to avoid:**
Use `fig.set_size_inches(width_px / dpi, height_px / dpi)` with an explicit DPI, then save with `bbox_inches=None` (not 'tight') and `pad_inches=0`. After saving, verify the output PNG dimensions match `S_db.shape` exactly. Better yet, skip matplotlib rendering entirely and write the spectrogram array directly to PNG using PIL/Pillow with a known colormap, guaranteeing 1:1 pixel correspondence. Store the exact pixel dimensions, DPI, and physical-unit mappings as image parasites in the XCF so they can never fall out of sync.

**Warning signs:**
- Spectrogram PNG dimensions differ from `S_db.shape` by 1-3 pixels
- Overlay visualizations show annotations shifted by a thin line at the edges
- `mask.shape != spec_array.shape[:2]` triggers the resize path in `visualize_overlay()`
- The `scipy.ndimage.zoom` fallback in the visualization code is being called at all -- if masks need resizing, something upstream is already wrong

**Phase to address:**
Phase 1 (Core Foundations) -- spectrogram generation must produce pixel-exact images before any annotation work begins.

---

### Pitfall 2: GIMP 3.0 Migration Will Break Everything

**What goes wrong:**
GIMP 3.0 (released 2024-2025) completely replaced the Python-Fu plugin API. The `gimpfu` module, `pdb` calls, `gimp` module, `gtk`/`gobject` imports, and the `register()` function all work differently or do not exist in GIMP 3.0. Every line of `spectrace_annotator.py` will fail. The project is locked to GIMP 2.10, which ships Python 2.7 on many platforms (macOS Homebrew GIMP 2.10 bundles Python 2.7). GIMP 2.10 will stop receiving updates, and new installs will increasingly default to GIMP 3.0.

**Why it happens:**
GIMP 3.0 switched from the `gimpfu` registration model to GObject Introspection (`gi.repository` imports). Properties like `image.width` became `image.get_width()`. The entire GTK layer moved from GTK2 to GTK3. Script-Fu registration also changed. This is not a minor version bump -- it is a full API rewrite.

**How to avoid:**
Pin the project to GIMP 2.10 explicitly in all documentation and install scripts (already partially done). Isolate every GIMP API call into a thin adapter module so that a future GIMP 3.0 port only requires rewriting the adapter, not the entire plugin. Keep the two-tier architecture strict: core Python modules must never import `gimpfu`, `gimp`, `gtk`, or `gobject`. Test that core modules run independently in standard Python 3 without GIMP. Document the GIMP 3.0 migration path so a future developer knows exactly which adapter functions need rewriting.

**Warning signs:**
- Lab machines start shipping GIMP 3.0 by default
- Users report "No module named gimpfu" errors
- macOS Homebrew drops GIMP 2.10 from the formula
- Core Python modules accidentally import GIMP-specific types

**Phase to address:**
Phase 1 (Architecture) -- enforce the two-tier boundary from the start. The GIMP adapter layer should be a single file with a clearly defined interface. Phase N (Future) -- GIMP 3.0 port is a dedicated milestone.

---

### Pitfall 3: gtk.main() Blocks GIMP and Prevents Graceful Shutdown

**What goes wrong:**
The current plugin calls `gtk.main()` at the end of `spectrace_setup()`. This enters a GTK main loop that keeps the PDB wire alive for the background polling timer (`gobject.timeout_add`). But `gtk.main()` blocks the GIMP Script-Fu/Python-Fu thread. While this loop is running, GIMP cannot process other plugin requests. If the user closes the image or GIMP itself without the loop terminating, GIMP may hang, requiring a force-quit. There is no clean shutdown path -- the `SpectraceBackgroundMonitor.stop()` method exists but is never called.

**Why it happens:**
GIMP Python-Fu plugins run in a single-threaded Script-Fu interpreter. Calling `gtk.main()` is the only way to keep a timer alive across PDB calls, but it monopolizes the interpreter. The plugin has no signal handler for image close or GIMP quit events.

**How to avoid:**
Register a `gimp.Image` destroy handler (or use `pdb.gimp_image_list()` polling) to detect when the image is closed, then call `gtk.main_quit()` to exit the loop cleanly. Add a try/finally block around `gtk.main()` that calls `monitor.stop()`. Consider whether the 200ms polling timer is even necessary -- GIMP 2.10 has `gimp.register_load_handler` and `gimp.register_save_handler` hooks that may provide layer-change callbacks without polling. If polling is needed, increase the interval to 500ms-1000ms to reduce CPU overhead. Document that the plugin occupies the Python-Fu interpreter while running and that no other Python-Fu plugins can execute concurrently.

**Warning signs:**
- GIMP hangs on quit (must be force-killed)
- "gimp_wire_read(): error" messages in terminal
- CPU usage stays elevated (200ms polling = 5 PDB calls/second indefinitely)
- Users cannot run other Python-Fu plugins while Spectrace is active

**Phase to address:**
Phase 2 (Plugin Wizard) -- the setup flow must include graceful lifecycle management from the beginning.

---

### Pitfall 4: Layer Dimension Mismatch Silently Corrupts Annotation Data

**What goes wrong:**
When a GIMP layer is resized, moved, or has its canvas adjusted (e.g., via Canvas Size, Scale Image, or Flatten operations), the layer pixel dimensions can diverge from the image canvas dimensions. The export pipeline (`extract_layers_from_xcf`) then gets a mask with different dimensions than the spectrogram. The current code silently resizes masks with `scipy.ndimage.zoom(mask, scale, order=0)` which uses nearest-neighbor interpolation -- this can introduce or remove annotation pixels at boundaries, corrupting the scientific data. A single extra row or column of pixels maps to real frequency/time values that were never annotated.

**Why it happens:**
GIMP allows layers to be any size independent of the canvas. Users can accidentally resize a layer (not the image) via menus, or GIMP may auto-resize layers during paste operations. The `gimpformats` library reads each layer at its actual size, which may differ from the image canvas. The project documentation mentions "zero tolerance for dimension mismatches" but the current code tolerates them silently.

**How to avoid:**
In the GIMP plugin: intercept layer resize operations using a polling check (compare `layer.width/height` to `image.width/height` every cycle) and warn the user immediately. Better: use `pdb.gimp_layer_resize_to_image_size()` to force all annotation layers to canvas size after every operation. In the export pipeline: refuse to export (raise an error) if any layer dimensions do not match the spectrogram dimensions exactly, rather than silently resizing. Log the mismatch with exact dimensions so the user can fix the XCF file. The resize-and-hope approach should only exist as a last-resort recovery tool, never as the default path.

**Warning signs:**
- Any layer's `size` field in `extract_layers_from_xcf()` output differs from `(width, height)` of the image
- The zoom/resize code path is hit during normal conversion (not just recovery)
- Overlay visualizations show annotations that don't quite line up with spectrogram features
- Annotation pixel counts change between the XCF view and the HDF5 export

**Phase to address:**
Phase 2 (Guardrails) -- layer dimension enforcement must be active before users start annotating. Phase 3 (Export) -- export pipeline must validate, not silently fix.

---

### Pitfall 5: Pickle-Based Metadata Is Fragile, Insecure, and Non-Portable

**What goes wrong:**
Project metadata is stored as `metadata.pkl` (Python pickle). Pickle files are: (a) not human-readable, so researchers cannot inspect or edit metadata without Python; (b) tied to the exact Python version and module structure -- renaming `utils.py` or changing class definitions breaks deserialization; (c) a known security vulnerability -- loading a malicious pickle file executes arbitrary code; (d) not portable across machines/environments if custom objects are pickled. The current code also writes `metadata.csv` but uses `pd.DataFrame.from_dict(audio_dict)` which will fail or produce mangled output if dict values are not uniform-length sequences.

**Why it happens:**
Pickle is the easiest Python serialization format -- one line to save, one line to load. But it is designed for temporary caching between trusted processes, not for long-term data storage in a scientific workflow where files move between machines and survive across years.

**How to avoid:**
Replace pickle with JSON for metadata storage. All metadata values (sample rate, FFT parameters, file paths, physical mappings) are JSON-serializable primitives. Store metadata as `metadata.json` alongside the spectrogram PNG. For the HDF5 export path, metadata is already stored as HDF5 attributes (which is correct) -- ensure the JSON source of truth and the HDF5 attributes stay in sync. Remove the `metadata.csv` file entirely (it is redundant with JSON and currently broken for non-tabular data). Keep pickle deserialization as a read-only migration path for existing projects.

**Warning signs:**
- `UnpicklingError` or `ModuleNotFoundError` when loading old metadata files
- Researchers ask "how do I see what FFT parameters were used?" and have no answer without Python
- `metadata.csv` contains mangled or partial data
- The project gets shared to a collaborator and pickle files fail on their machine

**Phase to address:**
Phase 1 (Core Foundations) -- metadata format must be settled before any data is generated in the new system.

---

### Pitfall 6: Color Assignment Assumes Drawing Color = Annotation Identity

**What goes wrong:**
The plugin assigns a unique foreground color per layer and enforces it via polling. The export pipeline, however, extracts binary masks by alpha channel (`alpha > 0`), not by color. This creates a dangerous disconnect: the user thinks color matters (they see different colors per layer), but the actual data capture is layer-based. If a user draws on the wrong layer with the "right" color, the annotation is captured under the wrong class. The 200ms polling interval means there is a window where the user switches layers in GIMP's layer panel, starts drawing, and the monitor has not yet updated the foreground color. In that window, the user draws in the previous layer's color on the new layer -- visually confusing but the export still captures the annotation correctly (by alpha). However, the user may then "fix" what appears to be a color mistake by erasing and redrawing, wasting time.

**Why it happens:**
The color is a visual cue, not a data encoding. But the user has no way to know this. The system uses two different identity mechanisms (color for display, layer for data) without communicating this to the user.

**How to avoid:**
Make the layer-based identity system explicit in the UI. When the monitor detects a layer switch, flash the layer name briefly on the canvas or in GIMP's status bar. Add a "wrong layer" guardrail: if the user draws pixels that match another layer's assigned color (by checking the foreground color on tool-down), warn immediately. Consider reducing the polling interval for the critical moment of layer switch, or switch to an event-driven approach if GIMP's API supports it. In the export documentation, explicitly state that annotations are captured by layer membership, not by pixel color.

**Warning signs:**
- Users report "the color was wrong" after switching layers quickly
- Users erase and redraw annotations that were actually correct
- Quality control reveals annotations on wrong layers that the user didn't notice because the color appeared correct on a different layer

**Phase to address:**
Phase 2 (Plugin UX and Guardrails) -- the annotation workflow must make layer identity unambiguous.

---

### Pitfall 7: Hardcoded Orca-Specific Layer Structure Prevents Reuse

**What goes wrong:**
The layer structure (`LAYER_STRUCTURE`), root group name (`OrcinusOrca_FrequencyContours`), layer sections, and color mapping are all hardcoded in `spectrace_annotator.py`. The `class_registry.py` and `hdf5_utils.py` also default to `OrcinusOrca_FrequencyContours`. This means adding a new species (e.g., humpback whales, dolphins) or a new annotation schema (e.g., noise classification, call type taxonomy) requires modifying source code in multiple files. The "template" concept exists in the project docs but the template is actually the hardcoded constant -- there is no external template file that drives the layer structure.

**Why it happens:**
The project started with one species and one lab. Hardcoding was faster than building a template system. The orca-specific names are deeply embedded in constants, function signatures, and default arguments across the codebase.

**How to avoid:**
Extract the layer structure, root group name, color mapping, and section definitions into an external configuration file (JSON or YAML). The GIMP plugin reads this config at startup. The export pipeline reads the same config. A "template" is a config file, not a source code edit. The config file should live in the project directory (not globally) so different projects can have different schemas. Provide an `orca.json` template as the default. Make the `layer_group_name` parameter required (not defaulted) in core functions to force callers to be explicit about which template they are using.

**Warning signs:**
- A collaborator asks "can I use this for dolphin clicks?" and the answer is "you need to edit the Python source"
- The project description says "template-based" but the templates are constants in `.py` files
- Default arguments like `layer_group_name="OrcinusOrca_FrequencyContours"` appear in 10+ function signatures

**Phase to address:**
Phase 1 (Core Foundations) -- externalize configuration before building the wizard, because the wizard needs to present template choices.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Pickle for metadata | One-line save/load | Non-portable, insecure, breaks on refactor | Never for production data. Acceptable only as temporary cache during a single session. |
| `scipy.ndimage.zoom` for mask resize | Prevents crash on dimension mismatch | Silently corrupts scientific annotations | Never for annotation data. Acceptable only for preview/visualization (not export). |
| 200ms GTK polling for layer detection | Works without GIMP event API | CPU overhead, race conditions on fast layer switches, blocks Python-Fu interpreter | Acceptable in prototype. Must add graceful shutdown and consider reducing frequency before production. |
| Hardcoded layer structure | Fast initial development | Cannot add species, cannot change schema without code edits | Only in early prototype for single-lab use. Must externalize before first release. |
| `from utils import *` across modules | Quick access to functions | Circular import risk, namespace pollution, makes dependency graph unclear | Never. Use explicit imports. |
| osascript keystroke hack for tool switching | Activates pencil tool on macOS | Fragile, platform-specific, requires accessibility permissions, fails if keyboard layout differs | Never in production. Find the GIMP PDB call that activates a specific tool. |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| gimpformats (XCF parsing) | Assuming layer pixel data always matches image canvas dimensions | Always compare `layer.size` to `(doc.width, doc.height)`. Layers in GIMP can have arbitrary offsets and dimensions. Account for layer offsets (`layer.xOffset`, `layer.yOffset`) when extracting mask positions. |
| librosa (spectrogram generation) | Using `librosa.display.specshow()` for file output (it is a display function that adds axes/labels) | For pixel-exact spectrograms, compute the STFT with `librosa.stft()`, convert to dB, then save the array directly via PIL. Only use `specshow()` for interactive visualization. |
| matplotlib (figure saving) | Using `bbox_inches='tight'` and expecting exact pixel dimensions | Use explicit figure size and DPI, save with `bbox_inches=None`. Verify output dimensions match expected values programmatically after save. |
| h5py (HDF5 writing) | Storing class names as a list attribute (HDF5 attributes cannot store Python lists directly) | Serialize class names to JSON string before storing as attribute (already done correctly in the codebase). |
| GIMP Python-Fu (PDB calls) | Calling PDB functions without undo grouping | Always wrap multi-step operations in `gimp_image_undo_group_start/end`. Already done for layer creation, but must also be done for any future guardrail operations that modify the image. |
| GIMP foreground color | Assuming `gimpcolor.RGB()` takes the same argument format across GIMP builds | The current code already tries 4 different methods (float 0-1, int 0-255, `gimp.set_foreground`, tuple). Keep this defensive approach. |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading entire WAV into HDF5 as opaque blob | HDF5 files become very large (WAV files are uncompressed); read times slow down when you only need the spectrogram | Store audio path reference in metadata; embed WAV only for archival/portable datasets. Consider embedding compressed audio (FLAC) instead of raw WAV. | When dataset exceeds ~100 clips (each WAV = 10-100MB). A 500-clip dataset becomes 5-50GB of mostly audio data. |
| `extract_layers_from_xcf()` calls `forceFullyLoaded()` on every layer | Loads all pixel data into memory even for layers with no annotations | Add an early-exit check: if the layer has no visible pixels (all alpha = 0), skip full loading. Check layer visibility flag first. | When XCF files have 25+ layers and processing hundreds of files in batch. Memory usage scales as layers x pixels x projects. |
| 200ms polling timer in GIMP plugin | 5 PDB calls per second per open image. Each call queries the active layer, sets foreground color, and flushes displays. | Increase poll interval to 500-1000ms. Only call `set_foreground_color` and `gimp_displays_flush` when the active layer actually changes (already partially done -- but `enforce_pencil_settings()` is called unconditionally). | Immediately on low-powered laptops. The timer runs continuously even when the user is not interacting with GIMP. |
| `generate_distinct_colors()` uses evenly spaced HSV hues | With 25 classes, adjacent colors become indistinguishable (hue separation = 14.4 degrees) | Use a perceptually distinct palette (e.g., Tableau 20, ColorBrewer qualitative) for up to 25 classes. For >25, accept that some colors will be similar and rely on layer names for disambiguation. | At 15+ classes. Human color discrimination fails below ~20 degrees of hue separation for similar saturation/value. |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No visual feedback when switching layers | User is unsure which layer they are drawing on. The only indicator is GIMP's layer panel (which may be collapsed or scrolled away from the active layer). | Flash the layer name on the canvas or in a persistent status indicator. Change the cursor color to match the layer's assigned color. |
| Plugin requires right-click -> Filters -> Spectrace -> Setup Annotation | Biologists unfamiliar with GIMP will not discover this menu path. The "right-click for menus" paradigm (hidden menubar) is non-obvious. | Add a GIMP toolbar button or make the plugin auto-run on image open. At minimum, show a startup dialog the first time GIMP launches after install. |
| No undo feedback for annotation operations | GIMP's undo works (ctrl+Z), but the Spectrace color/tool enforcement re-applies settings after undo, potentially confusing the user about what was undone. | Ensure the polling monitor does not interfere with undo operations. Only re-enforce settings when the user explicitly switches layers, not on every poll cycle. |
| Install script replaces gimprc, menurc, toolrc, sessionrc | User loses all their GIMP customizations. If they use GIMP for other purposes (photo editing), the locked-down configuration is hostile. | Create a separate GIMP profile or use `--new-system` flag. Alternatively, use GIMP's `--session` flag to load Spectrace-specific session config without replacing the default. Provide clear instructions for switching between annotation mode and normal GIMP mode. |
| No indication of export readiness | User finishes drawing but has no way to know if annotations are complete, if any layers are empty, or if the project is ready for export. | Add a "Check Annotations" menu item that reports which layers have content, total pixel counts per layer, and any validation warnings before export. |

## "Looks Done But Isn't" Checklist

- [ ] **Spectrogram generation:** Often produces PNG with dimensions 1-3 pixels different from the STFT array -- verify `Image.open(png_path).size == (S_db.shape[1], S_db.shape[0])` after every save
- [ ] **Layer structure creation:** Layers exist but may not all be at canvas size -- verify every layer's width/height matches image canvas dimensions
- [ ] **Tool enforcement:** Pencil is selected but brush/dynamics may not have changed -- verify by reading back `pdb.gimp_context_get_brush()` and `pdb.gimp_context_get_dynamics()` after setting them
- [ ] **Color assignment:** Foreground color is set but the fallback path may have been hit -- check `/tmp/spectrace_debug.log` for "ALL METHODS FAILED" entries
- [ ] **HDF5 export:** File is created and non-empty, but masks may have been resized (lossy) -- verify `masks.shape[1:] == spectrogram.shape` in the output HDF5
- [ ] **Class registry sync:** Registry file exists but may not include all template classes -- verify `len(registry.get_class_names()) == len(template_layers)` after sync
- [ ] **Metadata persistence:** Metadata is saved but pixel-to-physical mappings may be stale (from a different spectrogram generation run) -- verify metadata timestamps match spectrogram file timestamps
- [ ] **Install script:** Plugin file is copied but may not be executable -- verify `os.access(plugin_path, os.X_OK)` on the installed file

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Spectrogram dimension mismatch | LOW | Regenerate the spectrogram PNG with pixel-exact settings. Existing XCF annotations remain valid as long as canvas dimensions do not change. |
| GIMP 3.0 migration | HIGH | Rewrite the GIMP adapter layer. Core Python modules should not need changes if the two-tier boundary was maintained. Estimate 2-4 weeks for a developer familiar with both APIs. |
| gtk.main() hang on shutdown | LOW | Force-quit GIMP. No data loss because XCF saves are explicit (not auto-save). Fix the plugin to register a shutdown handler. |
| Layer dimension mismatch in exported data | MEDIUM | Re-export from the original XCF files after fixing the export pipeline to reject (not resize) mismatched layers. Requires access to original XCF files. If XCF files were deleted, the HDF5 data with resized masks is the only copy -- partially corrupted. |
| Pickle metadata deserialization failure | MEDIUM | Write a migration script that tries to unpickle with the original module structure, extracts the primitive values, and re-saves as JSON. If the pickle is truly unreadable, reconstruct metadata from the WAV file (sample rate, duration) and the spectrogram PNG (dimensions). FFT parameters may be lost. |
| Wrong-layer annotations | HIGH | Requires manual review of every annotation by a domain expert. There is no automated way to determine if an annotation "belongs" on a different layer. Prevention is the only viable strategy. |
| Hardcoded species config prevents reuse | MEDIUM | Refactor hardcoded constants into external config files. The data format (HDF5 schema, XCF layer structure) does not change -- only where the configuration lives. Existing HDF5 files remain valid. |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Spectrogram pixel-dimension drift | Phase 1: Core spectrogram generation | Automated test: assert saved PNG dimensions == STFT array dimensions |
| GIMP 3.0 API break | Phase 1: Architecture (enforce two-tier boundary) | CI test: core modules import and run without any GIMP-related imports |
| gtk.main() blocking / no shutdown | Phase 2: Plugin wizard and lifecycle | Manual test: close image, close GIMP -- both should exit cleanly without force-quit |
| Layer dimension mismatch | Phase 2: Annotation guardrails | Automated check in export: reject any layer where `size != canvas_size` |
| Pickle metadata fragility | Phase 1: Core metadata format | Migration test: old pickle files produce identical JSON output |
| Color/layer identity confusion | Phase 2: Plugin UX guardrails | User testing: observe 3 users annotating and check for wrong-layer errors |
| Hardcoded species configuration | Phase 1: Template/config system | Test: create a non-orca config file and run the full pipeline with it |

## Sources

- [GIMP Python Plugin Documentation](https://developer.gimp.org/resource/writing-a-plug-in/tutorial-python/) -- plugin registration and API patterns
- [GIMP 3.0 Release Notes](https://www.gimp.org/release-notes/gimp-3.0.html) -- breaking API changes
- [GIMP 3.0 Python Migration Guide](https://gist.github.com/hnbdr/d4aa13f830b104b23694a5ac275958f8) -- gimpfu to gi.repository migration details
- [GIMP Parasite Registry](https://developer.gimp.org/core/specifications/parasites/) -- metadata persistence format and limitations
- [GIMP Metadata Parasite Corruption](https://gimpchat.com/viewtopic.php?f=28&t=15386) -- known XMP/parasite corruption issues
- [matplotlib savefig bbox_inches issue #11681](https://github.com/matplotlib/matplotlib/issues/11681) -- bbox_inches='tight' does not respect figsize
- [Whombat: Bioacoustic Annotation Tool](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14468) -- design lessons from modern bioacoustic annotation
- [Bioacoustic Research Practical Guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC11885706/) -- annotation standardization challenges
- [Interpolation Artifacts in Segmentation Masks](https://github.com/albumentations-team/albumentations/issues/1294) -- why nearest-neighbor resize corrupts binary masks
- [GimpFormats Library](https://github.com/FHPythonUtils/GimpFormats) -- XCF parsing limitations and known issues
- Codebase analysis: `spectrace_annotator.py`, `utils.py`, `hdf5_utils.py`, `class_registry.py`, `start_project.py`

---
*Pitfalls research for: GIMP-based bioacoustic spectrogram annotation (Spectrace)*
*Researched: 2026-03-04*
