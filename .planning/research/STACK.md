# Technology Stack

**Project:** Spectrace -- GIMP-based bioacoustic spectrogram annotation toolkit
**Researched:** 2026-03-04

## Recommended Stack

### GIMP Plugin Layer (runs inside GIMP's embedded Python)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| GIMP | 2.10.x | Annotation interface | Project is built on Python-Fu (Python 2.7, `gimpfu` API). GIMP 3.0 shipped in early 2025 with a completely incompatible Python 3 + GObject Introspection API. Migrating now would require a full rewrite of the plugin. Stick with 2.10 until the user base moves. | HIGH |
| Python-Fu (Python 2.7) | bundled with GIMP 2.10 | Plugin scripting | Only option for GIMP 2.10 plugins. Python 2.7 is EOL but GIMP 2.10 bundles its own interpreter -- the system Python version is irrelevant. | HIGH |
| GTK 2 (via `gtk`/`gobject`) | bundled with GIMP 2.10 | Plugin UI (dialogs, polling timers) | GIMP 2.10 exposes GTK 2 to Python-Fu plugins. No other UI toolkit option inside the GIMP process. The existing `gobject.timeout_add()` polling pattern is the correct approach for background layer monitoring. | HIGH |

### Core Python Backend (runs outside GIMP)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | 3.11 | Core runtime | Stable, well-supported. Python 3.12/3.13 work but 3.11 avoids edge-case issues with numba JIT compilation. Already in environment.yml. | HIGH |
| NumPy | >=2.3,<2.5 | Array operations, mask manipulation | Foundation for all spectrogram and mask array work. Pin to 2.3.x for stability (2.4.x is latest but 2.3 is battle-tested). Already in use. | HIGH |
| SciPy | >=1.16,<1.18 | Signal processing, image resampling (`zoom`) | Used for mask resampling during XCF-to-HDF5 conversion. The `scipy.ndimage.zoom` call is critical for dimension-mismatch handling. | HIGH |
| librosa | 0.11.x | Audio loading, STFT, spectrogram generation | The standard Python audio analysis library. 0.11.0 is the current stable release. Provides `librosa.stft`, `librosa.amplitude_to_db`, `librosa.display.specshow`. Already deeply integrated. | HIGH |
| matplotlib | >=3.10,<3.12 | Spectrogram rendering, visualization overlays | Used to render spectrograms as pixel-perfect PNGs and to produce overlay/individual layer visualizations. Already in use. | HIGH |
| Pillow (PIL) | >=12.0 | Image I/O, RGBA manipulation | Used for loading spectrogram PNGs and converting layer images to numpy arrays. Already in use. | HIGH |
| h5py | >=3.15,<3.16 | HDF5 read/write for ML-ready datasets | Stores spectrograms + multi-class binary masks in compressed HDF5. Schema v2.0 is well-designed and already implemented. 3.15.1 is current. | HIGH |
| pandas | >=2.3,<2.4 | Dataset indexing, metadata CSV export, Excel output | Used for dataset_index.csv and Excel summary exports via openpyxl. Already in use. | HIGH |
| openpyxl | >=3.1 | Excel export | Pandas backend for `.xlsx` writes. Lightweight, stable. Already in use. | HIGH |
| gimpformats | 2025 | XCF file parsing (outside GIMP) | Pure-Python XCF reader. Enables the core pipeline to extract layer masks from `.xcf` files without requiring a running GIMP instance. The FHPythonUtils fork is the actively maintained version. Version `2025` is current on PyPI. Critical dependency -- no viable alternative exists for headless XCF parsing. | MEDIUM |

### Development & Notebook Environment

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| JupyterLab | 4.5 | Interactive analysis, demos | Lab users run notebooks for visualization and data exploration. Already in environment.yml. | HIGH |
| ipywidgets | 8.1 | Interactive notebook widgets | Enables interactive spectrogram/mask viewers in notebooks. Already in use. | HIGH |
| scikit-learn | >=1.7 | (Future) ML utilities | Listed in environment.yml. Not yet used in core pipeline but available for downstream ML tasks. Keep for future use. | MEDIUM |
| numba | >=0.62 | JIT compilation for numeric hotspots | Listed in environment.yml. Useful if spectrogram computation becomes a bottleneck. Requires Python 3.11 (not 3.13). | MEDIUM |

## Critical Architecture Decision: GIMP 2.10 vs GIMP 3.0

**Recommendation: Stay on GIMP 2.10 for now. Plan migration path.**

| Factor | GIMP 2.10 | GIMP 3.0 |
|--------|-----------|----------|
| Python version | 2.7 (bundled) | 3.x (bundled) |
| Plugin API | `gimpfu.register()`, `pdb.*` | GObject Introspection (`Gimp.*`, `GimpUi.*`) |
| UI toolkit | GTK 2 | GTK 3 |
| XCF format | Fully supported by gimpformats | Supported (format is backward-compatible) |
| Plugin compatibility | Existing plugin works | Full rewrite required |
| Ecosystem maturity | Stable, well-documented | API stable for 3.x but plugin ecosystem still rebuilding |
| User adoption | Still widely installed | Growing but not universal |

**Migration strategy:** The GIMP 3.0 API is now stable across 3.x releases (3.0.8 released Jan 2026, 3.2 RC in progress). When the lab is ready to move, the plugin rewrite involves:
1. Replace `gimpfu` imports with `gi.repository.Gimp` / `gi.repository.GimpUi`
2. Replace `pdb.*` calls with `Gimp.get_pdb().run_procedure()`
3. Replace `register()` with `GimpPlugIn` class + `GimpProcedureDialog`
4. Port GTK 2 widget code to GTK 3
5. Core Python modules (utils.py, hdf5_utils.py, etc.) need zero changes -- they already run on Python 3.11

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Audio loading | librosa | soundfile, scipy.io.wavfile | librosa provides the full pipeline (load + STFT + display). soundfile is fine for raw loading but lacks DSP. scipy.io.wavfile is bare-bones. |
| Spectrogram rendering | matplotlib | cv2, PIL-only | matplotlib's `specshow` handles axes, colormaps, and DPI-correct rendering. OpenCV is overkill for static image generation. |
| XCF parsing | gimpformats | GIMP batch mode (`gimp -i -b`) | gimpformats is pure Python, no GIMP installation needed on the processing machine. GIMP batch mode is slow and fragile for headless pipelines. |
| ML data format | HDF5 (h5py) | TFRecord, LMDB, plain NumPy | HDF5 supports compression, hierarchical structure, metadata attributes, and random access. Schema v2.0 is already implemented and working. TFRecord is TensorFlow-specific. |
| Annotation tool | GIMP (layers) | Raven Pro, Whombat, Audacity | GIMP provides pixel-level control via layers with semantic naming. Raven Pro uses bounding boxes (too coarse for frequency contours). Whombat is web-based and box-oriented. Audacity has no layer concept. The layer-as-class paradigm is Spectrace's core innovation. |
| Environment management | Conda (conda-forge) | pip + venv | Conda handles the librosa/numba/scipy binary dependency chain better than pip on macOS. Already established. |

## What NOT to Use

| Technology | Why Avoid |
|------------|-----------|
| GIMP 3.0 (for now) | Full plugin rewrite required. Wait until lab users have migrated their GIMP installations. |
| sounddevice / pyaudio | Real-time audio playback is explicitly out of scope. GIMP is not a media player. |
| OpenCV (cv2) | Adds a heavy dependency for image operations already handled by PIL + numpy. |
| TensorFlow / PyTorch | Spectrace creates training data, it does not train models. Keep the dependency tree clean. |
| pickle for metadata | Already used in `utils.py` for `metadata.pkl`. This is a security and portability risk. Migrate to JSON or include metadata in HDF5 only. Pickle is fragile across Python versions. |
| Script-Fu | Python-Fu is the right choice for this plugin -- Script-Fu's TinyScheme lacks the data structures needed for layer management and color mapping. |

## Installation

```bash
# Create environment from existing environment.yml
conda env create -f environment.yml
conda activate spectrace

# Or manual install (core only)
conda install -c conda-forge python=3.11 numpy=2.3 scipy=1.16 \
  librosa=0.11 matplotlib=3.10 pillow=12.0 h5py=3.15 \
  pandas=2.3 openpyxl=3.1 numba=0.62

# gimpformats (not on conda-forge)
pip install gimpformats==2025

# GIMP plugin install (macOS)
cp gimp_plugin/spectrace_annotator.py \
  ~/Library/Application\ Support/GIMP/2.10/plug-ins/
chmod +x ~/Library/Application\ Support/GIMP/2.10/plug-ins/spectrace_annotator.py
```

## Version Pinning Strategy

The environment.yml already pins major.minor versions, which is correct. Do NOT pin to exact patch versions in environment.yml -- let conda resolve compatible patches. The `=2.3` syntax in conda means `>=2.3.0,<2.4.0`, which is the right granularity.

Exception: `gimpformats==2025` uses calendar versioning and should be pinned exactly since the library's API stability between calendar versions is unknown.

## Dependency Risk Assessment

| Dependency | Risk Level | Concern |
|------------|------------|---------|
| gimpformats | MEDIUM | Single maintainer (FHPythonUtils). If XCF format changes in GIMP 3.x, this library may lag. The project already works with current XCF files, so the risk is manageable. |
| GIMP 2.10 Python-Fu | LOW (short-term) / HIGH (long-term) | GIMP 2.10 is feature-frozen but will eventually lose distro support. Plan the 3.0 migration before 2.10 becomes hard to install. |
| librosa | LOW | Well-maintained, large community, stable API. |
| h5py | LOW | Rock-solid, backed by HDF Group. |
| numba | MEDIUM | Historically lags on new Python versions. Currently fine on 3.11. If upgrading Python, check numba compatibility first. |

## Sources

- [GIMP 3.0 Release Notes](https://www.gimp.org/release-notes/gimp-3.0.html) -- API changes, Python 3 migration
- [GIMP 3.0.8 Release](https://www.gimp.org/news/2026/01/24/gimp-3-0-8-released/) -- Latest stable 3.0 point release
- [GIMP 3.0 Python Plugin Migration Guide](https://gist.github.com/hnbdr/d4aa13f830b104b23694a5ac275958f8) -- Practical porting guide
- [GIMP Developer: Python Plugins](https://developer.gimp.org/resource/writing-a-plug-in/tutorial-python/) -- Official GIMP 3.0 Python plugin docs
- [GimpFormats on PyPI](https://pypi.org/project/gimpformats/) -- Version 2025, pure Python XCF parser
- [FHPythonUtils/GimpFormats on GitHub](https://github.com/FHPythonUtils/GimpFormats) -- Active fork
- [librosa 0.11.0 documentation](https://librosa.org/doc/main/index.html) -- Current stable
- [h5py 3.15.1 documentation](https://docs.h5py.org/) -- Current stable
- [NumPy releases](https://github.com/numpy/numpy/releases) -- 2.4.1 is latest, 2.3.x is stable
- [SciPy toolchain roadmap](https://docs.scipy.org/doc/scipy/dev/toolchain.html) -- 1.17.0 in docs
- [Whombat annotation tool](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14468) -- Bioacoustics annotation comparison
- [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) -- Alternative bioacoustics toolkit
- [Raven Pro](https://www.ravensoundsoftware.com/software/raven-pro/) -- Industry-standard comparison
- [XDA: Reasons to stick with GIMP 2.10](https://www.xda-developers.com/reasons-still-use-gimp-2-10-instead-of-3-0/) -- User adoption context
- [Writing GIMP 3.0 Plugins](https://schoenitzer.de/blog/2025/Gimp%203.0%20Plugin%20Ressources.html) -- Community migration resources
