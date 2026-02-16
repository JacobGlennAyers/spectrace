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

⚠️ **IMPORTANT: You must install GIMP version 2.10.x** - The `gimpformats` Python library used by Spectrace is only compatible with GIMP 2.10 and will not work with GIMP 3.0 or later versions.

*Linux (Ubuntu/Debian):*
```bash
sudo apt update
sudo apt install gimp=2.10.*
```

If GIMP 2.10 is not available in your distribution's repositories, you can use Flatpak:
```bash
flatpak install flathub org.gimp.GIMP//2.10
flatpak run org.gimp.GIMP//2.10
```

*Windows:*
1. Download GIMP 2.10.30 from this [release](https://github.com/JacobGlennAyers/spectrace/releases/tag/correct_gimp_version) (includes both Windows installer and source code)
   - Alternative: Download from [FossHub GIMP archive](https://www.fosshub.com/GIMP-old.html)
2. Run the installer (`gimp-2.10.30-setup.exe`) and follow the installation wizard
3. Accept default settings unless you have specific preferences
4. **Do not upgrade to GIMP 3.0** if prompted - this will break compatibility with Spectrace

*macOS:*
```bash
# Using Homebrew with version pinning
brew install gimp@2.10
```

If Homebrew doesn't have GIMP 2.10 available, download from the .dmg from this release [release](https://github.com/JacobGlennAyers/spectrace/releases/tag/correct_gimp_version) or [FossHub archive](https://www.fosshub.com/GIMP-old.html).

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
- gimpformats (reading GIMP XCF files - **v2.10 only**)
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

#### 3.1 Open Your Files in GIMP

1. **Open GIMP 2.10**

2. **Open the template file first:**
   - Click `File > Open` in the top left corner of the GIMP window
   - Navigate to `templates/orca_template.xcf` (or your own template) in your spectrace directory
   - Click "Open"
   
   This template contains all the layers you'll need to draw frequency contours.
<img width="482" height="629" alt="image" src="https://github.com/user-attachments/assets/f2c71e38-8a01-41a1-aaff-ad39ed36e7e5" />


3. **Open your spectrogram image:**
   - Click `File > Open` again (keep the template open)
   - Navigate to `projects/your_audio_file_0/your_audio_file_0_spectrogram.png`
   - Click "Open"
   
   You should now have two tabs open at the top of your GIMP window: one for the template, one for your spectrogram image.

#### 3.2 Copy the Template Layers to Your Spectrogram

Now you need to transfer the layer structure from the template to your spectrogram image:

1. **Switch to the template tab** by clicking on it in the top row of the GIMP window
<img width="1096" height="217" alt="image" src="https://github.com/user-attachments/assets/29c53dce-a83c-4075-9193-c2906cd6e68a" />


2. **Locate the Layers panel** (usually on the right side of the window):
   - The Layers panel is typically in the bottom-right or top-right corner
   - If you don't see it: `Windows > Dockable Dialogs > Layers`

3. **Select the layer group** in the Layers panel:
   - Look for the layer group named `OrcinusOrca_FrequencyContours`
   - **Click on this layer group name** to select it
   - The selected layer will be highlighted (usually with a white or blue background)
   - **Important:** Select the layer GROUP (with a folder icon and +/- sign), not an individual layer within it
<img width="1096" height="639" alt="image" src="https://github.com/user-attachments/assets/c0180a80-be3f-49bd-9edd-b50273dd92b6" />


4. **Copy the layer group:**
   - Press `Ctrl+C` (Windows/Linux) or `Cmd+C` (Mac)
   - OR: Click `Edit > Copy` from the menu
   - The entire layer group structure is now copied to your clipboard

5. **Switch to your spectrogram image** by clicking on its tab in the top row of the GIMP window
<img width="1096" height="217" alt="image" src="https://github.com/user-attachments/assets/29c53dce-a83c-4075-9193-c2906cd6e68a" />

6. **Paste the template layers:**
   - Click `Edit > Paste as > New Layer` from the menu at the top left
   - This adds the entire layer group to your spectrogram image
<img width="678" height="805" alt="image" src="https://github.com/user-attachments/assets/a2da295f-39bb-4ed6-b124-93e27817bea4" />


   You should now see a new layer group `OrcinusOrca_FrequencyContours copy` in your Layers panel, above your spectrogram image.

#### 3.3 Resize Layers to Match Your Spectrogram

**⚠️ CRITICAL STEP - DO NOT SKIP!**

The template layers must be resized to match your spectrogram's exact dimensions, or your annotations will be cropped, misaligned, or the wrong size entirely.

1. **Make sure the pasted layer group is still selected** (it should be highlighted in the Layers panel)

2. **Resize the layers:**
   - Click `Layer > Layer to Image Size` from the menu at the top left
<img width="751" height="714" alt="image" src="https://github.com/user-attachments/assets/519c23a7-d2cf-473a-9b3b-526a57c4f24d" />


3. **Verify the resize worked:**
   - You should see a colorful dashed line (marching ants border) around the entire spectrogram image
   - **Before resizing:** This dashed line might have been smaller, offset, or not covering the whole image
   - **After resizing:** The dashed line should lie exactly on the boundaries of your spectrogram
   
   The resize operation adjusts all layers in the group to match your spectrogram's exact pixel dimensions.
<img width="987" height="495" alt="image" src="https://github.com/user-attachments/assets/4ee6ac6c-8b60-4c44-85a4-74b3d61d4e6c" />

### GIF Demonstration
![clip1](https://github.com/user-attachments/assets/12a65892-cdad-4b6c-988e-4aae47054d05)

#### 3.4 Rename the Layer Group and Unlock Layers

When you pasted the template, GIMP automatically added " copy" to the layer group name and locked the layers. You need to fix both issues:

1. **Right-click on the layer group** `OrcinusOrca_FrequencyContours copy` in the Layers panel

2. **Select `Edit Layer Attributes`** from the context menu
<img width="734" height="1194" alt="image" src="https://github.com/user-attachments/assets/8c9e53a0-9a84-4a0c-92fc-2853245f87a0" />


3. **In the dialog window that opens:**
   
   a. **Remove " copy" from the Layer name:**
      - You should be on the "Properties" tab (default)
      - In the **Layer name** field, delete " copy" (including the space before it)
      - The field should now show: `OrcinusOrca_FrequencyContours`
      - **Why this matters:** The Python scripts expect an exact layer name match. The " copy" suffix will cause the scripts to fail.
   
   b. **Click on the "Switches" tab** at the top of the dialog
   
   c. **Uncheck all three lock options:**
      - [ ] Lock pixels
      - [ ] Lock position and size
      - [ ] Lock alpha
      - All three checkboxes should be empty
      - **Why this matters:** Locked layers prevent you from drawing, erasing, or making any modifications.
   
   d. **Click "OK"** to apply the changes
<img width="711" height="629" alt="image" src="https://github.com/user-attachments/assets/c4aeeec7-b346-4920-bc0c-fbc8ee47b39f" />


#### 3.5 Verify Your Setup

Your Layers panel should now show (from top to bottom):

```
👁 OrcinusOrca_FrequencyContours  ← (no " copy"!)
    Click the + to expand and see sublayers:
    👁 Heterodynes
    👁 Subharmonics  
    👁 heterodyne_or_subharmonic_or_other
    👁 Cetacean_AdditionalContours
    👁 harmonics_HFC
    👁 f0_HFC
    👁 unsure_HFC
    👁 harmonics_LFC
    👁 f0_LFC
    👁 unsure_LFC
👁 your_audio_file_0_spectrogram.png  ← (your spectrogram)
```

- The `OrcinusOrca_FrequencyContours` group should be **above** your spectrogram layer
- If it's below, drag and drop it to move it above the spectrogram
- You should **not** see any lock icons next to the layer name

**You're now ready to start drawing!**

### 4. Draw Your Annotations

**Initial Setup:**

1. **Zoom in for precision:** `View > Zoom > 2:1 (200%)`
<img width="767" height="882" alt="image" src="https://github.com/user-attachments/assets/daec8555-bf92-48b1-9f40-22a1358a952c" />

2. **Select the Pencil tool** (not paintbrush - this is critical!)
   - Click the pencil icon in the toolbox, OR
   - `Tools > Paint Tools > Pencil`

3. **Configure tool settings:**
   - Size: `1.0` pixel
   - Hardness: `100`
   - Force: `100`
   - **Important:** Expand "Dynamics Options"
   - Check "Apply Jitter"
   - Set Amount: `0.00`
<img width="432" height="1112" alt="image" src="https://github.com/user-attachments/assets/ae1a5a84-03a0-4410-a1f9-a6e19e6b2bf0" />

4. **Select the layer** where you want to draw (e.g., `f0_LFC` for fundamental frequency)
   - Layer groups have +/- icons - click the + to expand and see individual layers
   - Click on a specific layer to select it (not the group itself)
<img width="1095" height="733" alt="image" src="https://github.com/user-attachments/assets/204212e2-3f17-4795-a480-9a5fabff0b70" />

5. **Choose a drawing color:**
   - Click the foreground color square (upper rectangle in toolbox)
   - Select a color using the color picker, OR
   - Enter HTML notation directly (recommended for consistency)
   - Use different colors for different layers to make it easier to verify your work later
<img width="1025" height="765" alt="image" src="https://github.com/user-attachments/assets/3e3b03ef-8255-4a87-aeae-2531b3e9f32a" />

**Drawing Tips:**

- Draw along the frequency contour you wish to annotate
- You don't need to press any buttons on your pen/mouse - just draw
- Use `Ctrl+Z` to undo mistakes
- Use the Eraser tool for corrections (same hardness/force settings: 100/100)
<img width="391" height="709" alt="image" src="https://github.com/user-attachments/assets/f6b68cdd-4ed4-479b-b042-eafdbfdcb085" />

- Make sure you're on the correct layer before erasing
- Toggle layer visibility using the "eye" icon to check your work
- When drawing, start with the bottom layer in the list and work upward to avoid forgetting layers

**What to Draw:**

Draw all contours that are within the onset and offset boundaries you entered in your spreadsheet:
- If multiple calls from the **same vocalization** are present → draw them all in one project
- If calls from **different individuals/vocalizations** are present → create separate projects for each

### 5. Managing Layers and Corrections

**If you drew on the wrong layer:**

1. Click on the layer with incorrect contours
2. Zoom out: `View > Zoom > 1:1 (100%)`
3. Select the Rectangle Select tool
4. Draw a rectangle around the contours to copy
5. Press `Ctrl+C` to copy
6. Click on the correct destination layer
7. Press `Ctrl+V` to paste
8. A "Floating Selection (Pasted Layer)" will appear - position it above the target layer if needed (drag and drop)
9. Right-click "Floating Selection (Pasted Layer)" → `Anchor Layer`
10. Erase any unwanted contours from the original or destination layer

**Checking your work:**

Click the "eye" icons next to layers to toggle visibility - this helps verify each contour is on the correct layer.
<img width="1102" height="741" alt="image" src="https://github.com/user-attachments/assets/d4ad6337-a49b-45ec-91a8-76f3c9dd4a88" />
### GIF Demonstration
![clip2](https://github.com/user-attachments/assets/0b40cdfb-3127-425b-9b9e-8ce0af64f9a4)

### 6. Save Your Work

**Save frequently during annotation:**

1. `File > Save As...` (first time) or `File > Save` / `Ctrl+S` (subsequent saves)
2. Save the XCF file in your project folder: `projects/your_audio_file_0/your_audio_file_0.xcf`
3. The XCF filename should match the project folder name

### GIF Demonstration
![clip3](https://github.com/user-attachments/assets/c4f07c9f-a1de-4ef9-933a-60ca48df8a0c)


**After completing all annotations for an audio file:**

- If you need to annotate more calls from the same audio file → return to Step 2 and create a new project
- If all annotations are complete → proceed to visualization (Step 7)

### 7. Visualize Your Annotations

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

1. Create a new XCF file in GIMP 2.10
2. Set up your layer groups with descriptive names
3. Save as `templates/your_template.xcf`
4. Create a corresponding YAML file documenting each layer's purpose
5. Update scripts to reference your template:
```python
layer_group_name = "YourSpecies_FrequencyContours"
template_xcf_path = "./templates/your_template.xcf"
```

**Note:** Templates created in GIMP 2.10 must be used exclusively with GIMP 2.10. Do not open or save them in GIMP 3.0, as this will make them incompatible with the `gimpformats` library.

## Troubleshooting

**Issue: "gimpformats library can't read my XCF file"**
- **Most common cause:** You're using GIMP 3.0 or have opened/saved the file in GIMP 3.0
- Solution: Use GIMP 2.10.x exclusively. If a file was saved in GIMP 3.0, you may need to recreate it from scratch in GIMP 2.10
- Verification: Check your GIMP version with `Help > About` in GIMP

**Issue: Template layer group has " copy" suffix and layers are locked**
- Solution: Follow Step 3.4 carefully - rename the layer group to remove " copy" and uncheck all three lock options in the Switches tab

**Issue: Annotations are cropped, misaligned, or not appearing correctly**
- **This is the most common issue!** 
- Solution: You likely skipped Step 3.3 - Right-click layer group → `Layer to Image Size`
- This step is **mandatory** for every project - layers must match spectrogram dimensions exactly
- Verify: The colorful dashed border should perfectly outline your entire spectrogram

**Issue: Pencil not drawing**
- Check tool settings: Size=1.0, Hardness=100, Force=100
- Verify Apply Jitter is checked with Amount=0.00
- Ensure you've selected the Pencil tool (not Paintbrush)
- Verify you've clicked on a layer (not layer group)
- Check that layers are unlocked (no lock icons in Layers panel)

**Issue: "I don't see the dashed border around my image"**
- The layer group might not be selected. Click on `OrcinusOrca_FrequencyContours` in the Layers panel
- The border (marching ants) appears when a layer is selected

**Issue: "Where is the Layers panel?"**
- Go to `Windows > Dockable Dialogs > Layers` to make it visible

**Issue: Colors look wrong in visualizations**
- Delete `layer_color_mapping.json` to regenerate color assignments
- Specify a master template XCF for consistent colors

**Issue: Python packages not found**
- Ensure conda environment is activated: `conda activate spectrace`
- Reinstall environment: `conda env remove -n spectrace` then `conda env create -f environment.yml`

## Compatibility Notes

- **GIMP Version:** Spectrace requires GIMP 2.10.x and is **not compatible** with GIMP 3.0 or later
- **Python:** Tested with Python 3.8-3.11
- **Operating Systems:** Linux, Windows, and macOS
- **gimpformats library:** Only supports GIMP 2.10 XCF file format

## File Formats

- **XCF**: GIMP's native format (v2.10), stores all layers and metadata
- **HDF5**: Hierarchical data format for ML pipelines
- **PNG**: Visualization outputs
- **Excel**: Tabular export of contour data
- **PKL/CSV**: Project metadata

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
