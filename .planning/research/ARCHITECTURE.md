# Architecture Research

**Domain:** GIMP plugin + Python backend for bioacoustic spectrogram annotation
**Researched:** 2026-03-04
**Confidence:** HIGH (based on existing codebase analysis + GIMP developer documentation)

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        GIMP 2.10 Process                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              GIMP Python-Fu Runtime (Python 2.7)           │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐ │  │
│  │  │  spectrace_annotator│  │   gimpfu / gimp / gtk       │ │  │
│  │  │  (Plugin Layer)     │──│   (GIMP-provided modules)   │ │  │
│  │  │  - Layer creation   │  │   - PDB procedures          │ │  │
│  │  │  - Tool enforcement │  │   - Image manipulation      │ │  │
│  │  │  - Color management │  │   - GTK event loop          │ │  │
│  │  │  - Background poll  │  │   - Parasite API            │ │  │
│  │  └─────────────────────┘  └─────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  XCF File (persistence layer)                                    │
│  ├── Pixel layers (annotation masks)                             │
│  ├── Layer groups (semantic hierarchy)                            │
│  └── Image parasites (metadata: FFT params, audio path, mapping) │
└──────────────────────┬───────────────────────────────────────────┘
                       │ XCF files on disk
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Spectrace Core (Python 3 / conda)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │  utils   │  │ hdf5_    │  │ class_   │  │ export_contours_ ││
│  │  .py     │  │ utils.py │  │registry  │  │ to_excel.py      ││
│  │ -audio   │  │ -XCF→HDF5│  │ .py      │  │ -HDF5→Excel      ││
│  │  process │  │ -loader  │  │ -append  │  │ -contour extract ││
│  │ -XCF read│  │ -schema  │  │  only    │  │ -statistics      ││
│  │ -visuals │  │  v2.0    │  │ -migrate │  │                  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
│  ┌──────────┐  ┌──────────────────────────────────────────────┐ │
│  │produce_  │  │         Output Artifacts                     │ │
│  │visuals.py│  │  ├── HDF5 files (ML-ready, schema v2.0)     │ │
│  │          │  │  ├── Excel spreadsheets (contours/stats)     │ │
│  │          │  │  ├── PNG visualizations (overlay/individual) │ │
│  │          │  │  └── dataset_index.csv                       │ │
│  └──────────┘  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Current State |
|-----------|----------------|---------------|
| `spectrace_annotator.py` | GIMP plugin: layer creation, tool enforcement, color auto-switch, GTK background monitor | Working but lacks wizard, export panel, guardrails |
| `utils.py` | Core library: audio processing, spectrogram generation, XCF layer extraction, visualization, color mapping | Mature, functional |
| `hdf5_utils.py` | XCF-to-HDF5 conversion, HDF5 loader, schema v2.0 management | Mature, well-structured |
| `class_registry.py` | Append-only class vocabulary management, HDF5 migration when classes grow | Mature |
| `export_contours_to_excel.py` | HDF5-to-Excel export with contour extraction and statistics | Mature |
| `produce_visuals.py` | Batch visualization orchestration | Thin wrapper, functional |
| `start_project.py` | CLI entry point for new project creation from WAV | Thin wrapper, functional |
| `templates/` | Master XCF template + YAML class descriptions | Defined, authoritative |
| `gimp_plugin/config/` | Locked-down GIMP config (gimprc, menurc, toolrc, sessionrc) | Working |

## The Two-Tier Architecture: Why It Works This Way

### The Core Constraint: Two Python Runtimes

GIMP 2.10 ships its own embedded Python 2.7 interpreter. The spectrace core modules (utils.py, hdf5_utils.py, etc.) run on Python 3 with scientific libraries (numpy, scipy, librosa, h5py, pandas, PIL, gimpformats). These two runtimes cannot share a process.

This is the fundamental architectural decision that shapes everything:

| Concern | GIMP Plugin (Python 2.7) | Spectrace Core (Python 3) |
|---------|--------------------------|---------------------------|
| Runtime | GIMP's embedded Python 2.7 | System/conda Python 3.x |
| Available libraries | gimpfu, gimp, gtk, gobject, os, sys | numpy, scipy, librosa, h5py, pandas, PIL, gimpformats |
| Can access | GIMP PDB, image data, layers, parasites, GTK | File system, audio files, XCF files (via gimpformats), HDF5 |
| Cannot access | numpy, h5py, librosa, pandas | GIMP PDB, live image state, layer pixels in memory |

### Communication Pattern: File I/O (Not Subprocess)

The existing architecture uses **file-based communication**. The plugin and core do not talk to each other at runtime. Instead:

1. **Core creates project** (start_project.py): WAV -> spectrogram PNG + metadata.pkl + project folder
2. **User opens PNG in GIMP**: Manual step
3. **Plugin creates layers**: Layer hierarchy injected programmatically
4. **User annotates**: Drawing happens entirely in GIMP
5. **User saves XCF**: GIMP's native save
6. **Core reads XCF** (utils.py): Uses gimpformats library to parse XCF outside GIMP
7. **Core exports** (hdf5_utils.py, export_contours_to_excel.py): XCF -> HDF5 -> Excel

```
Time ──────────────────────────────────────────────────────────►

Core Python 3          GIMP Python 2.7           Core Python 3
┌──────────┐          ┌──────────────┐          ┌──────────────┐
│ WAV→PNG  │──file──►│ Open PNG     │          │              │
│ +metadata│          │ Create layers│          │              │
│          │          │ Annotate     │          │              │
│          │          │ Save XCF  ───┼──file──►│ Read XCF     │
│          │          └──────────────┘          │ Export HDF5  │
│          │                                    │ Export Excel  │
└──────────┘                                    └──────────────┘
```

**This is the correct pattern.** The alternatives were considered and rejected:

| Alternative | Why Rejected |
|-------------|-------------|
| subprocess.Popen from plugin | GIMP's Python 2.7 calling Python 3 via subprocess works, but the PDB wire protocol blocks during plugin execution. Long-running DSP in a subprocess would freeze GIMP or require complex async handling via GTK idle callbacks. |
| sys.path manipulation to import core | Python 2.7 cannot import Python 3 modules. Even if versions matched, scientific libraries (numpy C extensions) compiled for one Python version crash in another. |
| GIMP 3.0 migration (Python 3 native) | GIMP 3.0 uses Python 3 via GObject Introspection (gi), which would unify runtimes. However: (a) GIMP 3.0 API is substantially different, (b) PyGTK is replaced by GTK3/GI, (c) the target lab is standardized on GIMP 2.10, and (d) gimpfu registration is completely different. Not viable for current deployment. |
| Script-Fu bridge | Script-Fu (Scheme) has even fewer capabilities than Python-Fu for this use case. |

### Where Subprocess IS Appropriate

The existing code already uses subprocess for one thing: sending an osascript keystroke on macOS to switch to the Pencil tool (line 441-450 in spectrace_annotator.py). This is appropriate because it is fire-and-forget, non-blocking, and does not require return values.

For the planned wizard flow (WAV -> spectrogram), subprocess is the right pattern:

```python
# Inside GIMP plugin (Python 2.7)
import subprocess
result = subprocess.Popen(
    ["/path/to/conda/python3", "-m", "spectrace.core.generate_spectrogram",
     "--wav", wav_path, "--output", project_dir, "--nfft", "2048"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
# Use gobject.timeout_add() to poll result.poll() without blocking GTK
```

Key constraints for subprocess from within the plugin:
- Must not block the GTK main loop (use `gobject.timeout_add` to poll)
- Must use absolute path to the correct Python 3 interpreter
- Must pass all arguments via CLI flags (no shared memory)
- Results come back via files written to disk, not stdout

## Metadata Persistence: Image Parasites

### What Parasites Are

GIMP parasites are named binary blobs attached to images, layers, or channels. They persist through XCF save/load cycles. This is the correct mechanism for storing annotation metadata.

### Parasite API (Python-Fu 2.10)

```python
# Attach a persistent parasite to an image
parasite = gimp.Parasite(
    "spectrace-metadata",      # name (string, use plugin prefix)
    PARASITE_PERSISTENT,       # flags (survives save/load)
    json.dumps(metadata_dict)  # data (bytes/string)
)
image.attach_parasite(parasite)

# Read it back
parasite = image.get_parasite("spectrace-metadata")
if parasite:
    metadata = json.loads(parasite.data)

# Remove it
image.detach_parasite("spectrace-metadata")
```

### Recommended Parasite Strategy

Store metadata as JSON strings in named parasites with a `spectrace-` prefix:

| Parasite Name | Content | When Written |
|---------------|---------|--------------|
| `spectrace-fft-params` | `{"nfft": 2048, "noverlap": 1024, "sample_rate": 44100}` | Project creation / wizard |
| `spectrace-audio-path` | `{"wav_path": "/path/to/original.wav", "clip_basename": "orca"}` | Project creation / wizard |
| `spectrace-pixel-mapping` | `{"time_per_pixel": 0.01, "freq_per_pixel": 10.0, "max_freq_hz": 22050}` | Project creation / wizard |
| `spectrace-template-version` | `{"template": "orca_v1", "plugin_version": "1.0.0"}` | Layer creation |
| `spectrace-annotation-notes` | `{"notes": "", "timing_drift": false}` | User-editable via export panel |

**Flags:**
- `PARASITE_PERSISTENT` (1): Survives XCF save/load. Use for all spectrace metadata.
- `PARASITE_UNDOABLE` (2): Can be combined with PERSISTENT. Use if metadata changes should be undoable.

**Size limits:** No documented hard limit. Parasites are stored inline in XCF. JSON metadata strings of a few KB are well within safe bounds.

**Why parasites over the current metadata.pkl approach:** Parasites travel with the XCF file. If someone moves or renames the project folder, the metadata is still attached to the image. The current metadata.pkl approach requires the pickle file to be co-located with the XCF, which is fragile.

### Reading Parasites from Outside GIMP

The gimpformats library (already used in utils.py) can read parasites from XCF files:

```python
from gimpformats.gimpXcfDocument import GimpDocument
doc = GimpDocument("annotation.xcf")
# Access parasites via doc.parasites (list of GimpParasite objects)
for p in doc.parasites:
    if p.name.startswith("spectrace-"):
        data = p.data.decode('utf-8')
```

This means the core Python 3 pipeline can read metadata written by the plugin without any runtime communication.

## Recommended Project Structure

```
spectrace/
├── gimp_plugin/
│   ├── spectrace_annotator.py    # Main GIMP plugin (Python 2.7)
│   ├── config/                    # Locked-down GIMP configuration
│   │   ├── gimprc
│   │   ├── menurc
│   │   ├── toolrc
│   │   └── sessionrc
│   ├── install_mac.sh
│   ├── uninstall_mac.sh
│   └── INSTALL.md
├── spectrace/                     # Python 3 package (future refactor target)
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py               # WAV processing, spectrogram generation
│   │   ├── metadata.py            # Parasite read/write, metadata.pkl compat
│   │   ├── templates.py           # Layer structure definitions, YAML parsing
│   │   └── xcf_reader.py          # XCF parsing via gimpformats
│   ├── export/
│   │   ├── __init__.py
│   │   ├── hdf5.py                # XCF→HDF5 conversion, loader, schema
│   │   ├── excel.py               # HDF5→Excel contour/stats export
│   │   └── visualizations.py      # Overlay and individual layer visuals
│   ├── registry/
│   │   ├── __init__.py
│   │   └── class_registry.py      # Append-only class vocabulary
│   └── cli/
│       ├── __init__.py
│       ├── new_project.py         # CLI: WAV → project folder
│       ├── export_hdf5.py         # CLI: XCF projects → HDF5
│       └── export_excel.py        # CLI: HDF5 → Excel
├── templates/
│   ├── orca_template.xcf
│   └── orca_template.yaml
├── environment.yml
└── README.md
```

### Structure Rationale

- **gimp_plugin/:** Completely isolated from the Python 3 package. Deployed by copying a single .py file to GIMP's plug-ins directory. Must have zero imports from spectrace core (different Python runtime).
- **spectrace/core/:** Pure computation modules. No GIMP dependencies. Testable with pytest.
- **spectrace/export/:** Output formatters. Depend on core but not on GIMP.
- **spectrace/cli/:** Entry points that the GIMP plugin can invoke via subprocess, or that users run directly from the terminal.

## Architectural Patterns

### Pattern 1: Fire-and-Forget Subprocess for Heavy Computation

**What:** The GIMP plugin invokes spectrace core CLI commands via `subprocess.Popen`, polls for completion with `gobject.timeout_add`, and loads results from disk when done.

**When to use:** Wizard flow (WAV -> spectrogram), batch export from within GIMP.

**Trade-offs:** Simple, no shared state bugs. Latency is acceptable (spectrogram generation takes seconds, user expects to wait). Error handling must be via exit codes and stderr.

```python
# In GIMP plugin (Python 2.7)
import subprocess
import gobject

class SubprocessRunner(object):
    def __init__(self, cmd, on_success, on_error):
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.on_success = on_success
        self.on_error = on_error
        gobject.timeout_add(200, self._poll)

    def _poll(self):
        ret = self.proc.poll()
        if ret is None:
            return True  # Keep polling
        if ret == 0:
            self.on_success(self.proc.stdout.read())
        else:
            self.on_error(self.proc.stderr.read())
        return False  # Stop polling
```

### Pattern 2: XCF as the Single Source of Truth

**What:** The XCF file is the canonical artifact. It contains pixel data (layers), semantic structure (layer names/groups), and metadata (parasites). Everything downstream (HDF5, Excel, visuals) is derived from XCF.

**When to use:** Always. Never store annotation state outside the XCF that cannot be reconstructed from it.

**Trade-offs:** XCF files are large (uncompressed pixel data). Parsing them outside GIMP requires gimpformats (pure Python, slower than GIMP's native C reader). But the benefit is that a biologist can hand someone an XCF file and it contains everything needed to reproduce the export.

### Pattern 3: Layer Names as Semantic Encoding

**What:** Layer names carry meaning parsed by the export pipeline. `f0_LFC` means "fundamental frequency of the low-frequency component." `Heterodynes/5` means "heterodyne affiliated with the 5th harmonic of the HFC." The layer hierarchy (groups) adds further semantics.

**When to use:** For all annotation class identity. No sidecar files needed.

**Trade-offs:** Renaming a layer in GIMP breaks the pipeline. Mitigation: the plugin should warn on layer rename (guardrail), and the export pipeline validates layer names against the template before conversion.

### Pattern 4: Append-Only Class Registry

**What:** `ClassRegistry` maintains a stable mapping from class names to channel indices. New classes are only ever appended; existing indices never change. Old HDF5 files can be migrated to the new schema by zero-filling new channels.

**When to use:** Whenever the template evolves (new layer types added for new species or annotation categories).

**Trade-offs:** Channel indices grow monotonically, which means HDF5 files may have sparse channels for classes that are never annotated. The overhead is negligible (zero-filled channels compress to nearly nothing in gzip).

## Data Flow

### Annotation Workflow (End to End)

```
1. WAV Audio File
       │
       ▼ [start_project.py / future wizard]
2. Project Folder
   ├── clip_spectrogram.png
   ├── clip.wav (copy)
   └── metadata.pkl
       │
       ▼ [User opens PNG in GIMP]
3. GIMP with Spectrogram Image
       │
       ▼ [Filters > Spectrace > Setup Annotation]
4. Plugin Creates Layers
   ├── 26 annotation layers in hierarchy
   ├── Tool enforcement (pencil, 1px, hard)
   ├── Background color monitor
   └── Parasites attached (future: FFT params, audio path)
       │
       ▼ [User draws annotations, saves XCF]
5. XCF File with Annotations
   ├── Spectrogram (flattened background)
   ├── Binary mask layers (drawn annotations)
   ├── Layer groups (semantic hierarchy)
   └── Parasites (metadata)
       │
       ▼ [xcf_to_hdf5.py]
6. HDF5 Files (Schema v2.0)
   ├── Spectrogram (uint8, HxW)
   ├── Masks (uint8, CxHxW per annotation set)
   ├── Metadata (sample_rate, nfft, pixel mappings)
   └── Embedded WAV (opaque bytes)
       │
       ├──▼ [export_contours_to_excel.py]
       │  7a. Excel: Contours, Statistics, Class_Summary
       │
       └──▼ [produce_visuals.py]
          7b. PNG Visualizations: Overlay, Individual Layers
```

### Plugin Internal Data Flow

```
spectrace_setup() called by GIMP PDB
    │
    ├── find_existing_root_group(image)
    │   └── Returns existing group or None
    │
    ├── create_template_layers(image)  [if no existing group]
    │   ├── Creates root layer group
    │   ├── Creates subgroups (Heterodynes, Subharmonics, etc.)
    │   └── Creates 26 RGBA layers in correct hierarchy
    │
    ├── build_layer_index(image, root_group)
    │   └── Returns {path_name: gimp_layer_object} mapping
    │
    ├── enforce_pencil_settings()
    │   ├── Set paint mode, opacity, brush, size, dynamics
    │   └── Find hard brush from candidates list
    │
    ├── set_foreground_color(r, g, b)
    │   └── Try 4 methods (gimpcolor.RGB float, int, gimp.set_foreground, tuple)
    │
    └── SpectraceBackgroundMonitor(image, layer_map)
        ├── gobject.timeout_add(200ms, poll)
        ├── On active layer change: switch color + enforce pencil
        └── gtk.main() keeps PDB wire alive for callbacks
```

## Key Integration Points

### Internal Boundaries

| Boundary | Communication | Direction | Notes |
|----------|---------------|-----------|-------|
| Plugin <-> Core | File I/O (XCF, PNG, metadata.pkl) | Async, both write to shared filesystem | No runtime coupling. Plugin writes XCF, core reads XCF. Core writes PNG, user opens in GIMP. |
| Plugin <-> GIMP | PDB procedure calls | Synchronous, within same process | All GIMP manipulation via `pdb.*` calls. Must not block long. |
| Plugin <-> GTK | Event loop + timer callbacks | Event-driven | `gobject.timeout_add` for polling. `gtk.main()` keeps plugin alive. |
| Core <-> gimpformats | Direct import | Synchronous | Pure Python XCF parser. Slower than GIMP native but works outside GIMP process. |
| Core <-> HDF5 | h5py file I/O | Synchronous | Schema v2.0. One file per audio clip. |
| Template <-> Plugin | Hardcoded constants | Build-time | LAYER_STRUCTURE, LAYER_SECTIONS, LAYER_COLORS are hardcoded in plugin. Must match template XCF. |
| Template <-> Core | XCF parsing at runtime | Runtime | Core reads template XCF to validate project layers and build class registry. |

### The Template Synchronization Problem

Currently, the layer structure is defined in three places:
1. `templates/orca_template.xcf` (authoritative XCF)
2. `templates/orca_template.yaml` (human-readable descriptions)
3. `spectrace_annotator.py` LAYER_STRUCTURE constant (hardcoded in plugin)

These must stay synchronized. The YAML and XCF are already read by the core pipeline. The plugin hardcodes the structure because it cannot import core modules. This is a known fragility point.

**Recommended mitigation:** Generate the LAYER_STRUCTURE constant from the YAML template as a build step, or have the plugin read a simple JSON file at startup (GIMP Python 2.7 has `json` in stdlib).

## Build Order (Dependency-Driven)

Based on component dependencies, this is the recommended implementation order:

```
Phase 1: Foundation
├── Parasite metadata system (no dependencies, enables everything else)
├── Template sync mechanism (JSON config read by plugin)
└── Core package reorganization (spectrace/ package structure)

Phase 2: Wizard Flow
├── CLI entry point for spectrogram generation (spectrace.cli.new_project)
├── Subprocess runner in plugin (depends on Phase 1 for metadata)
└── GTK wizard dialog (depends on subprocess runner)

Phase 3: Guardrails
├── Layer rename detection (depends on Phase 1 template sync)
├── Dimension validation (depends on parasite metadata)
├── Wrong-layer warnings (depends on background monitor, already exists)
└── Bounds checking (depends on parasite pixel mapping)

Phase 4: Export Panel
├── In-GIMP export trigger (depends on Phase 1 parasites for metadata)
├── Subprocess call to core export pipeline (depends on Phase 2 subprocess)
└── Progress feedback via GTK (depends on Phase 2 GTK patterns)
```

**Rationale for this order:**
- Parasites must come first because the wizard, guardrails, and export panel all need metadata that currently only exists in metadata.pkl (which the plugin cannot read).
- The wizard depends on subprocess patterns that the export panel also needs; building subprocess infrastructure once serves both.
- Guardrails require knowing what the correct state should be (template sync + parasite metadata).
- The export panel is last because it depends on all other pieces and biologists can use the existing CLI export workflow in the interim.

## Anti-Patterns

### Anti-Pattern 1: Blocking the GTK Main Loop

**What people do:** Call subprocess.Popen().wait() or run long computations synchronously inside a plugin function.
**Why it is wrong:** GIMP's UI freezes. The PDB wire protocol times out. On macOS, the spinning beachball appears and the OS may offer to force-quit GIMP.
**Do this instead:** Use `gobject.timeout_add()` to poll subprocess.poll() in 200ms intervals. Show a progress bar via `pdb.gimp_progress_init/update`.

### Anti-Pattern 2: Importing Core Modules into the Plugin

**What people do:** Try `sys.path.append("/path/to/spectrace")` and `from utils import process_audio_project` inside the GIMP plugin.
**Why it is wrong:** GIMP 2.10 runs Python 2.7. Core modules are Python 3. Even if the syntax were compatible, numpy/scipy/h5py C extensions compiled for Python 3 will segfault in Python 2.7.
**Do this instead:** Use subprocess to call Python 3 scripts. Communicate via files on disk.

### Anti-Pattern 3: Storing Metadata Only in Sidecar Files

**What people do:** Store FFT params, audio path, and pixel mappings only in metadata.pkl alongside the XCF.
**Why it is wrong:** If someone moves the XCF to a different folder, emails it, or renames the project directory, the metadata is lost. The export pipeline then has no way to compute correct time/frequency values.
**Do this instead:** Store critical metadata in GIMP image parasites (which travel with the XCF) AND in a sidecar file (for backward compatibility and for tools that cannot read XCF parasites).

### Anti-Pattern 4: Hardcoding Layer Structure Without Validation

**What people do:** Define LAYER_STRUCTURE as a constant and trust that it matches the template.
**Why it is wrong:** Template evolves independently (new species, new annotation categories). Plugin and template drift apart silently. Export pipeline then encounters unknown layers or missing layers.
**Do this instead:** Read layer structure from a shared JSON/YAML config at plugin startup. Validate at export time that project layers are a subset of the template (already implemented in XCFToHDF5Converter.validate()).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1 researcher, 1 species | Current architecture. Single template, manual CLI workflow. |
| 5 researchers, 1 species | Add parasite metadata (avoid data loss from file moves). Add wizard to reduce setup errors. Locked-down GIMP config prevents configuration drift between machines. |
| 5 researchers, 3 species | Multiple templates. Plugin reads species from config file. Class registry handles cross-species vocabulary. |
| 20 researchers, batch processing | CLI pipeline for batch spectrogram generation. CI/CD for template validation. Centralized HDF5 data store. |

### Scaling Priorities

1. **First bottleneck:** Annotation setup errors (biologists misconfigure GIMP, wrong layer names, missing metadata). Fix: wizard + parasites + guardrails.
2. **Second bottleneck:** Export friction (switching between GIMP and terminal). Fix: in-GIMP export panel.
3. **Third bottleneck:** Multi-species template management. Fix: config-driven layer structure.

## Sources

- [GIMP Developer: Parasite Registry](https://developer.gimp.org/core/specifications/parasites/)
- [GIMP Developer: Python Plug-Ins Tutorial](https://developer.gimp.org/resource/writing-a-plug-in/tutorial-python/)
- [GIMP Python Documentation](https://www.gimp.org/docs/python/)
- [GIMP Developer: About Plug-ins](https://developer.gimp.org/resource/about-plugins/)
- [GIMP Developer Wiki: Hacking Plugins](https://wiki.gimp.org/wiki/Hacking:Plugins)
- [GIMP Forum: Imports in a Python Plugin](https://www.gimp-forum.net/Thread-Imports-in-a-Python-plugin)
- [GIMP Forum: Python-Fu to Launch Python 3](https://www.gimp-forum.net/Thread-Gimp-Python-fu-2-7-to-launch-Python-3-for-editing-images)
- [GNOME Discourse: Image Parasites in Python 3 Plugin](https://discourse.gnome.org/t/gimp2-99-how-to-use-image-attach-parasite-in-a-python3-plugin/12806)
- Codebase analysis: spectrace_annotator.py, utils.py, hdf5_utils.py, class_registry.py, export_contours_to_excel.py

---
*Architecture research for: GIMP plugin + Python backend bioacoustic spectrogram annotation*
*Researched: 2026-03-04*
