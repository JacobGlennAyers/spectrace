# Spectrace

A Python-based workflow for creating precise binary mask annotations on spectrograms using GIMP, designed primarily for bioacoustic research applications.

## Overview

Spectrace streamlines the process of annotating audio spectrograms by combining Python's audio processing capabilities with GIMP's intuitive layer-based drawing interface. The workflow allows researchers to:

1. Generate spectrograms from audio files
2. Draw detailed binary masks on spectrograms using GIMP's tools
3. Organize annotations using layer groups (e.g., fundamental frequency, harmonics, heterodynes)
4. Export annotations to multiple formats (XCF, HDF5, Excel)
5. Visualize and validate annotations programmatically

This tool is particularly useful for creating training datasets for machine learning models, analyzing vocalizations, or documenting acoustic features with pixel-level precision.

## Key Features

- **Audio to Spectrogram**: Automatically generates spectrograms from audio files with customizable parameters
- **Template-Based Annotation**: Use predefined layer structures to ensure consistent labeling across projects
- **GIMP Integration**: Leverage GIMP's familiar interface and drawing tools for precise mask creation
- **Multiple Export Formats**: Convert annotations to HDF5 for ML pipelines or Excel for spreadsheet analysis
- **Batch Visualization**: Generate overlay and individual layer visualizations across all projects
- **Color Management**: Automatic color assignment to annotation classes for clear visualization
- **Binary Morphology Tools**: Included utilities for post-processing binary masks

## Installation

### Prerequisites

**Python Environment:**
- Python 3.8 or higher
- Conda or Miniconda (recommended for dependency management)

**GIMP Installation:**

*Linux (Ubuntu/Debian):*
```bash
sudo apt update
sudo apt install gimp
```

*Windows:*
1. Download GIMP from [https://www.gimp.org/downloads/](https://www.gimp.org/downloads/)
2. Run the installer and follow the installation wizard
3. Accept default settings unless you have specific preferences

### Setting Up Spectrace

1. **Clone or download this repository:**
```bash
git clone https://github.com/yourusername/spectrace.git
cd spectrace
```

2. **Create the conda environment:**
```bash
conda env create -f environment.yml
conda activate spectrace
```

This will install all required Python dependencies including:
- librosa (audio processing)
- numpy, pandas (data manipulation)
- matplotlib (visualization)
- gimpformats (reading GIMP XCF files)
- h5py (HDF5 file handling)
- pillow (image processing)

3. **Verify installation:**

The repository includes an example orca call (`audio/orca.wav`) and template (`templates/orca_template.xcf`). Test your installation by creating a project from this example:

```bash
python start_project.py
```

If successful, you should see output indicating a new project folder was created in `projects/`.

## Quick Start Guide

### 1. Prepare Your Audio File

Place your audio file (WAV format recommended) in the `audio/` directory.

### 2. Create a New Project

Open `start_project.py` in a text editor and modify the configuration:

```python
audio_info = {
    "clip_path": "audio/your_audio_file.wav",
    "nfft": 2048,  # FFT window size
    "grayscale": True
}
```

Parameters:
- `clip_path`: Path to your audio file
- `nfft`: FFT window length (larger values = better frequency resolution, worse time resolution)
- `grayscale`: Whether to generate grayscale spectrograms (recommended: True)

Run the script to create your project:
```bash
python start_project.py
```

This creates:
- A new folder in `projects/` named `your_audio_file_0`
- A spectrogram PNG image
- Metadata files (pkl, csv)
- A copy of the audio file

**Note:** You can run `start_project.py` multiple times for the same audio file. Each run creates a new project with an incremented index (`your_audio_file_0`, `your_audio_file_1`, etc.). This is useful when you want to annotate different portions of the same recording or create multiple sets of annotations for the same audio.

### 3. Set Up GIMP for Annotation


1. Open GIMP
2. Open the spectrogram PNG from your project folder: `projects/your_audio_file_0/your_audio_file_0_spectrogram.png`
3. Open the template file: `File > Open as Layers` → select `templates/orca_template.xcf`
4. In the Layers panel, ensure the template layer group is above the spectrogram image
5. Right-click the layer group → `Layer to Image Size` (ensures layers match spectrogram dimensions)
<video
  src="https://github.com/JacobGlennAyers/spectrace/releases/download/additional_materials/clip1.mp4"
  controls
  muted
  playsinline
  style="max-width: 100%; border-radius: 6px;">
</video>


**[GIF PLACEHOLDER: Loading template and adjusting layers]**

The template contains predefined layer groups for different annotation types. The included orca template (`orca_template.xcf`) is organized hierarchically to capture various acoustic features:

**For Biphonic Calls** (calls with two simultaneous frequency components):
- **High-Frequency Component (HFC)**: Fundamental frequency, harmonics, and subharmonics of the higher-pitched component
- **Low-Frequency Component (LFC)**: Fundamental frequency, harmonics, and subharmonics of the lower-pitched component
- **Heterodynes**: Intermodulation products that appear between harmonics (numbered layers 0-12 for different harmonic affiliations)

**For Monophonic Calls** (single fundamental frequency):
- **LFC layers**: Used for the fundamental frequency and harmonics
- **Subharmonics**: Frequencies at evenly spaced intervals below the fundamental (e.g., f0/2, f0/3)

**Additional Categories**:
- **Cetacean Additional Contours**: For non-orca cetacean vocalizations or ambiguous sources
- **Heterodyne/Subharmonic/Other**: For contours where classification is uncertain

Each major category includes an "unsure" layer for cases where you cannot definitively classify a contour. See `templates/orca_template.yaml` for detailed descriptions of each layer, including references to scientific papers with examples.

### 4. Draw Your Annotations



1. Zoom in for precision: `View > Zoom > 2:1 (200%)`
2. Select the Pencil tool (not paintbrush)
3. Set brush size to 1.0 pixel, hardness 100, force 100
4. Click on the layer where you want to draw (e.g., `f0_LFC` for fundamental frequency)
5. Draw along the frequency contour you wish to annotate
6. Use the Eraser tool (same hardness/force settings) to correct mistakes
7. Toggle layer visibility using the "eye" icon to check your work
<video
  src="https://github.com/JacobGlennAyers/spectrace/releases/download/additional_materials/clip2.mp4"
  controls
  muted
  playsinline
  style="max-width: 100%; border-radius: 6px;">
</video>

**Tips:**
- Use different layers for each annotation class
- Draw on the correct layer for each type of acoustic feature
- Keep annotations within the time boundaries of the vocalization
- Save frequently: `File > Save` or `Ctrl+S`

### 5. Save and Export

Save your work as an XCF file in the project folder:

```
File > Save As > projects/your_audio_file_0/your_audio_file_0.xcf
```
<video
  src="https://github.com/JacobGlennAyers/spectrace/releases/download/additional_materials/clip3.mp4"
  controls
  muted
  playsinline
  style="max-width: 100%; border-radius: 6px;">
</video>

The XCF filename should match the project folder name.

### 6. Visualize Your Annotations

Open `produce_visuals.py` and specify your audio file basename:

```python
clip_basename = "your_audio_file"  # without extension or index number
```

Generate visualization images:

```bash
python produce_visuals.py
```

This creates overlay and individual layer visualizations in the `visualizations/` folder, organized by audio file basename.

## Workflow Details

### Project Structure

Each project folder contains:
```
your_audio_file_0/
├── your_audio_file_0_spectrogram.png  # Spectrogram image
├── your_audio_file.wav                 # Copy of audio file
├── your_audio_file_0.xcf              # GIMP file with masks
├── metadata.pkl                        # Project metadata
└── metadata.csv                        # Human-readable metadata
```

### Layer Organization

The included orca template uses a hierarchical layer structure designed to capture the complexity of killer whale vocalizations:

```
OrcinusOrca_FrequencyContours/
├── Heterodynes/
│   ├── unsure
│   ├── 0 (affiliated with f0 of HFC)
│   ├── 1 (affiliated with 1st harmonic of HFC)
│   ├── 2 (affiliated with 2nd harmonic of HFC)
│   └── ... (up to 12)
├── Subharmonics/
│   ├── subharmonics_HFC
│   └── subharmonics_LFC
├── heterodyne_or_subharmonic_or_other
├── Cetacean_AdditionalContours/
│   ├── unsure_CetaceanAdditionalContours
│   ├── harmonics_CetaceanAdditionalContours
│   └── f0_CetaceanAdditionalContours
├── harmonics_HFC
├── f0_HFC
├── unsure_HFC
├── harmonics_LFC
├── f0_LFC
└── unsure_LFC
```

**Key abbreviations:**
- **HFC** = High-Frequency Component (for biphonic calls)
- **LFC** = Low-Frequency Component (for biphonic calls or monophonic calls)
- **f0** = Fundamental frequency

**Usage notes:**
- For **biphonic calls**, use both HFC and LFC layer sets
- For **monophonic calls**, use only LFC layer sets
- Heterodynes are numbered according to which harmonic of the HFC they're affiliated with
- Use "unsure" layers when classification is ambiguous

The template is specifically designed for killer whale (orca) vocalizations but can be adapted for other species. See `templates/orca_template.yaml` for complete documentation with scientific references.

### Multiple Projects per Audio File

You may want to create multiple projects from the same audio file for several reasons:
- Annotating different vocalizations within the same recording
- Creating alternative annotation sets with different interpretations
- Separating overlapping calls that require different layer configurations

To create additional projects, simply run `start_project.py` again with the same audio file configuration. The script automatically increments the project index, creating folders like `your_audio_file_0`, `your_audio_file_1`, `your_audio_file_2`, etc.

### Color Mapping

The first time you visualize a project, Spectrace automatically:
- Discovers all layer names from your template or existing projects
- Assigns a unique color to each annotation class
- Saves the mapping to `layer_color_mapping.json`

This ensures consistent colors across all visualizations. To use a master template for color assignment:

```python
template_xcf_path = "./templates/orca_template.xcf"
```

Or set to `None` to auto-discover from existing projects.

## Advanced Usage

### Converting to HDF5 Format

For machine learning pipelines, convert XCF annotations to HDF5:

Edit `xcf_to_hdf5.py` to set your input/output paths:
```python
project_folder = "./projects"
ml_data_folder = "./hdf5_files"
```

Run the conversion:
```bash
python xcf_to_hdf5.py
```

This creates:
- One HDF5 file per project
- `dataset_index.csv` with metadata for all samples

Each HDF5 file contains:
- `spectrogram`: Grayscale spectrogram array (H, W)
- `masks`: Binary masks array (C, H, W) where C is number of classes
- `metadata`: Audio parameters and project information
- `class_names`: List of annotation class names

### Loading HDF5 Data

```python
from ml_prep import HDF5SpectrogramLoader

with HDF5SpectrogramLoader("hdf5_files/orca_0.hdf5") as loader:
    spec, masks, metadata = loader.load()
    class_names = loader.get_class_names()
    
    # Get specific class mask
    f0_mask = loader.get_class_mask("f0_LFC")
```

### Exporting to Excel

Convert annotations to Excel spreadsheets for analysis in programs like Microsoft Excel or Google Sheets.

Edit `export_contours_to_excel.py` to configure:
```python
ml_data_folder = "./hdf5_files"
output_excel = "whale_contours_export.xlsx"
contour_method = "centroid"  # or "min_max" or "all_points"
```

Run the export:
```bash
python export_contours_to_excel.py
```

The Excel file contains multiple sheets:
- **Summary**: Overview of all samples
- **Contours**: Time-frequency points for each annotation
- **Statistics**: Per-annotation metrics (duration, bandwidth, etc.)
- **Class_Summary**: Aggregate statistics per class

Extraction methods:
- `"centroid"`: One frequency value per time frame (smoothest contours)
- `"min_max"`: Minimum and maximum frequency per time frame (captures bandwidth)
- `"all_points"`: Every pixel (most detailed, largest file)

### Binary Morphology Operations

The `demos/` folder includes examples of common binary morphology operations (erosion, dilation, opening, closing) that are frequently used for post-processing binary masks. These operations can help clean up annotations, connect nearby regions, or extract specific features. See `demos/binary_morphology_interactive.ipynb` for interactive examples and `demos/bin_morph.py` for a standalone demonstration script.

## Template Customization

To create your own annotation template:

1. Create a new XCF file in GIMP
2. Set up your layer groups with descriptive names
3. Save as `templates/your_template.xcf`
4. Create a corresponding YAML file documenting each layer's purpose
5. Update scripts to reference your template:
```python
layer_group_name = "YourSpecies_FrequencyContours"
template_xcf_path = "./templates/your_template.xcf"
```

## Troubleshooting

**Issue: Layers don't match spectrogram size**
- Solution: Right-click layer group → `Layer to Image Size`

**Issue: Pencil not drawing**
- Check tool settings: Size=1.0, Hardness=100, Force=100
- Ensure you've selected the Pencil tool (not Paintbrush)
- Verify you've clicked on a layer (not layer group)

**Issue: Colors look wrong in visualizations**
- Delete `layer_color_mapping.json` to regenerate color assignments
- Specify a master template XCF for consistent colors

**Issue: Python packages not found**
- Ensure conda environment is activated: `conda activate spectrace`
- Reinstall environment: `conda env remove -n spectrace` then `conda env create -f environment.yml`


## File Formats

- **XCF**: GIMP's native format, stores all layers and metadata
- **HDF5**: Hierarchical data format for ML pipelines
- **PNG**: Visualization outputs
- **Excel**: Tabular export of contour data
- **PKL/CSV**: Project metadata


## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.


