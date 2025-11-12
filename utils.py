import os
import shutil
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image
from gimpformats.gimpXcfDocument import GimpDocument, GimpGroup
import matplotlib.patches as mpatches
from typing import List, Dict, Optional


def process_audio_project(project_folder: str, audio_dict: dict):
    """
    Processes an audio file: copies it to a project folder,
    generates and saves a perfectly cropped, linearly spaced
    spectrogram image, and pickles metadata.
    Args:
        project_folder (str): Path to the root project directory.
        audio_dict (dict): Must contain:
            - "clip_path" (str): path to audio file
            - "nfft" (int): FFT window length
            - "grayscale" (bool): whether to save spectrogram in grayscale
    """
    clip_path = audio_dict.get("clip_path")
    nfft = audio_dict.get("nfft", 2048)
    grayscale = audio_dict.get("grayscale", True)
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Audio file not found: {clip_path}")
    # Derive project folder name: clipname_timestamp
    clip_basename = os.path.splitext(os.path.basename(clip_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #out_dir = os.path.join(project_folder, f"{clip_basename}_{timestamp}")
    clip_ndx = find_index(project_folder, clip_basename)
    out_dir = os.path.join(project_folder, clip_basename + '_' + str(clip_ndx))
    os.makedirs(out_dir, exist_ok=True)
    # Copy audio file
    audio_filename = os.path.basename(clip_path)
    copied_audio_path = os.path.join(out_dir, audio_filename)
    shutil.copy2(clip_path, copied_audio_path)
    # Load audio
    y, sr = librosa.load(clip_path, sr=None)
    # Compute linear-frequency magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=nfft, window="hann"))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    # Plot spectrogram: perfectly cropped, uniform frequency spacing
    fig = plt.figure(frameon=False)
    fig.set_size_inches(S_db.shape[1] / 100, S_db.shape[0] / 100)  # size to pixel resolution
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis=None,
        y_axis=None,       # ensures linear frequency axis (uniform spacing)
        cmap="gray" if grayscale else "magma"
    )
    spectrogram_path = os.path.join(out_dir, clip_basename + '_' + str(clip_ndx) + "_spectrogram"  + ".png")
    plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # Save all relevant info
    audio_dict.update({
        "sample_rate": sr,
        "spectrogram_path": spectrogram_path,
        "copied_audio_path": copied_audio_path,
        "project_folder": out_dir,
        "spectrogram_shape": S_db.shape,
        "frequency_spacing": sr / 2 / (S_db.shape[0] - 1),  # Hz per pixel row (approx.)
        "time_per_pixel": librosa.frames_to_time(1, sr=sr, n_fft=nfft)  # seconds per column
    })
    # Pickle metadata
    pickle_path = os.path.join(out_dir, "metadata.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(audio_dict, f)
    audio_df = pd.DataFrame.from_dict(audio_dict, index=[0])
    audio_df.to_csv(os.path.join(out_dir, "metadata.csv"))
    print(f"✅ Project created at: {out_dir}")
    return audio_dict


def find_index(folder, clip_basename):
    ndx = 0
    for project in os.listdir(folder):
        if project.startswith(clip_basename):
            ndx += 1
    return ndx


# =============================================================================
# GIMP Layer Visualization Functions
# =============================================================================

def find_group_by_name(group: GimpGroup, target_name: str) -> Optional[GimpGroup]:
    """Find a group by name, searching recursively."""
    for child in group.children:
        if isinstance(child, GimpGroup):
            if child.name == target_name:
                return child
            # Recursively search in subgroups
            result = find_group_by_name(child, target_name)
            if result is not None:
                return result
    return None


def binarize_image_by_alpha(img: Image.Image) -> np.ndarray:
    """
    Binarize an image based on its alpha channel.
    Alpha = 0 -> 0 (transparent)
    Alpha > 0 -> 1 (opaque)
    
    Returns a binary numpy array with values 0 or 1.
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get the alpha channel
    alpha = np.array(img.getchannel('A'))
    
    # Binarize: 0 where alpha is 0, 1 where alpha > 0
    binary = (alpha > 0).astype(np.uint8)
    
    return binary


def extract_layers_from_xcf(xcf_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours") -> Dict:
    """
    Extract layers from XCF file within a specified layer group.
    
    Args:
        xcf_path (str): Path to the XCF file
        layer_group_name (str): Name of the layer group to extract
    
    Returns:
        Dict containing layer data and metadata
    """
    print(f"\n🔍 Loading XCF file: {xcf_path}")
    xcf_doc = GimpDocument(xcf_path)
    
    print(f"📄 XCF dimensions: {xcf_doc.width}x{xcf_doc.height}")
    
    # Get the root group
    root_group = xcf_doc.walkTree()
    
    # Find the target group
    target_group = find_group_by_name(root_group, layer_group_name)
    
    if target_group is None:
        print(f"❌ Error: Group '{layer_group_name}' not found!")
        return {
            'layers': {},
            'width': xcf_doc.width,
            'height': xcf_doc.height
        }
    
    print(f"✅ Found layer group: {layer_group_name}")
    
    # Extract layers from the group
    layers_data = {}
    
    for child in target_group.children:
        if isinstance(child, GimpGroup):
            continue  # Skip subgroups
        
        layer_name = child.name
        
        # Force the layer to be fully loaded
        child.forceFullyLoaded()
        
        # Get the image
        img = child.image
        
        if img is not None:
            # Convert to binary mask based on alpha channel
            binary_mask = binarize_image_by_alpha(img)
            
            # Check if layer is visible
            visible = child.visible
            
            layers_data[layer_name] = {
                'mask': binary_mask,
                'visible': visible,
                'size': img.size
            }
            
            pixel_count = np.sum(binary_mask)
            print(f"  📝 Layer '{layer_name}': {img.size}, visible={visible}, pixels={pixel_count}")
        else:
            print(f"  ⚠️  Warning: No image found for layer: {layer_name}")
    
    print(f"📚 Total layers extracted: {len(layers_data)}")
    
    return {
        'layers': layers_data,
        'width': xcf_doc.width,
        'height': xcf_doc.height
    }


def find_all_project_indices(project_folder: str, clip_basename: str) -> List[int]:
    """
    Find all project indices for a given clip basename.
    
    Args:
        project_folder (str): Root project directory
        clip_basename (str): Base name of the audio clip (without extension)
    
    Returns:
        List[int]: Sorted list of project indices
    """
    indices = []
    for project in os.listdir(project_folder):
        if project.startswith(clip_basename + '_'):
            try:
                idx = int(project.split('_')[-1])
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)


def load_metadata(project_path: str) -> dict:
    """
    Load metadata from a project folder.
    
    Args:
        project_path (str): Path to the project folder
    
    Returns:
        dict: Metadata dictionary
    """
    pickle_path = os.path.join(project_path, "metadata.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    return {}


def visualize_overlay(project_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours", 
                      project_index: Optional[int] = None, output_path: Optional[str] = None):
    """
    Create an overlay visualization with all layers on the spectrogram with time/frequency axes.
    
    Args:
        project_path (str): Path to the project folder
        layer_group_name (str): Name of the layer group in GIMP
        project_index (int, optional): Index to display in title
        output_path (str, optional): Where to save the output image
    """
    # Find spectrogram and XCF files
    files = os.listdir(project_path)
    spectrogram_file = [f for f in files if f.endswith('_spectrogram.png')][0]
    xcf_file = [f for f in files if f.endswith('.xcf')][0]
    
    spectrogram_path = os.path.join(project_path, spectrogram_file)
    xcf_path = os.path.join(project_path, xcf_file)
    
    # Load metadata for time/frequency info
    metadata = load_metadata(project_path)
    
    print(f"\n📊 Loading spectrogram: {spectrogram_file}")
    
    # Load spectrogram
    spectrogram = Image.open(spectrogram_path)
    spec_array = np.array(spectrogram)
    print(f"Spectrogram shape: {spec_array.shape}")
    
    # Extract layers
    layer_data = extract_layers_from_xcf(xcf_path, layer_group_name)
    layers = layer_data['layers']
    
    if not layers:
        print("⚠️  No layers found! Cannot create visualization.")
        return
    
    # Define colors for each layer
    layer_colors = {
        'heterodynes': '#FF00FF',      # Magenta
        'harmonics_HFC': '#00FFFF',    # Cyan
        'f0_HFC': '#FF0000',           # Red
        'harmonics_LFC': '#00FF00',    # Green
        'f0_LFC': '#FFFF00'            # Yellow
    }
    
    # Create figure with proper axes
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get audio parameters from metadata
    sr = metadata.get('sample_rate', 44100)
    nfft = metadata.get('nfft', 2048)
    
    # Compute time and frequency parameters
    duration = spec_array.shape[1] * librosa.frames_to_time(1, sr=sr, n_fft=nfft)
    max_freq = sr / 2
    
    # Display spectrogram with proper extent
    ax.imshow(spec_array, aspect='auto', cmap='gray', origin='upper',
              extent=[0, duration, 0, max_freq])
    
    # Overlay each layer
    legend_handles = []
    layers_drawn = 0
    
    for layer_name, layer_info in layers.items():
        if not layer_info['visible']:
            print(f"⏭️  Skipping invisible layer: {layer_name}")
            continue
        
        # Get the binary mask
        mask = layer_info['mask']
        
        if np.sum(mask) == 0:
            print(f"⚠️  Layer '{layer_name}' has no drawn pixels, skipping")
            continue
        
        color = layer_colors.get(layer_name, '#FFFFFF')
        
        # Ensure mask matches spectrogram dimensions
        if mask.shape != spec_array.shape[:2]:
            print(f"⚠️  Layer mask shape {mask.shape} doesn't match spectrogram {spec_array.shape[:2]}")
            from scipy.ndimage import zoom
            scale_y = spec_array.shape[0] / mask.shape[0]
            scale_x = spec_array.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
            print(f"    Resized mask to {mask.shape}")
        
        # Create colored overlay from mask
        color_rgb = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
        
        # Create RGBA overlay: RGB from color, Alpha from mask
        overlay = np.zeros((*mask.shape, 4))
        overlay[:, :, :3] = color_rgb
        overlay[:, :, 3] = mask.astype(float) * 0.8
        
        ax.imshow(overlay, aspect='auto', origin='upper', 
                  extent=[0, duration, 0, max_freq])
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=color, label=layer_name))
        layers_drawn += 1
        print(f"✅ Drew layer '{layer_name}' ({np.sum(mask)} pixels)")
    
    print(f"\n✅ Successfully drew {layers_drawn} layers")
    
    # Add legend outside plot area
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  framealpha=0.9, fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    
    title = f"Overlaid Layers ({layers_drawn} layers)"
    if project_index is not None:
        title += f" (Project Index: {project_index})"
    ax.set_title(title, fontsize=14, pad=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Overlay visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_individual_layers(project_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours",
                                output_path: Optional[str] = None):
    """
    Create a row visualization showing original spectrogram and each layer individually with time/frequency axes.
    
    Args:
        project_path (str): Path to the project folder
        layer_group_name (str): Name of the layer group in GIMP
        output_path (str, optional): Where to save the output image
    """
    # Find spectrogram and XCF files
    files = os.listdir(project_path)
    spectrogram_file = [f for f in files if f.endswith('_spectrogram.png')][0]
    xcf_file = [f for f in files if f.endswith('.xcf')][0]
    
    spectrogram_path = os.path.join(project_path, spectrogram_file)
    xcf_path = os.path.join(project_path, xcf_file)
    
    # Load metadata for time/frequency info
    metadata = load_metadata(project_path)
    
    print(f"\n📊 Loading spectrogram: {spectrogram_file}")
    
    # Load spectrogram
    spectrogram = Image.open(spectrogram_path)
    spec_array = np.array(spectrogram)
    
    # Extract layers
    layer_data = extract_layers_from_xcf(xcf_path, layer_group_name)
    layers = layer_data['layers']
    
    if not layers:
        print("⚠️  No layers found! Cannot create visualization.")
        return
    
    # Define colors
    layer_colors = {
        'heterodynes': '#FF00FF',
        'harmonics_HFC': '#00FFFF',
        'f0_HFC': '#FF0000',
        'harmonics_LFC': '#00FF00',
        'f0_LFC': '#FFFF00'
    }
    
    # Filter to only visible layers with data
    visible_layers = {name: info for name, info in layers.items() 
                     if info['visible'] and np.sum(info['mask']) > 0}
    
    # Get audio parameters from metadata
    sr = metadata.get('sample_rate', 44100)
    nfft = metadata.get('nfft', 2048)
    
    # Compute time and frequency parameters
    duration = spec_array.shape[1] * librosa.frames_to_time(1, sr=sr, n_fft=nfft)
    max_freq = sr / 2
    
    # Create figure with subplots
    n_layers = len(visible_layers)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 5))
    
    # Ensure axes is always iterable
    if n_layers == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # First subplot: original spectrogram
    axes[0].imshow(spec_array, aspect='auto', cmap='gray', origin='upper',
                   extent=[0, duration, 0, max_freq])
    axes[0].set_title("Original", fontsize=12, weight='bold')
    axes[0].set_xlabel('Time (s)', fontsize=9)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=9)
    
    # Subsequent subplots: individual layers
    for idx, (layer_name, layer_info) in enumerate(visible_layers.items(), start=1):
        ax = axes[idx]
        
        # Display spectrogram as background
        ax.imshow(spec_array, aspect='auto', cmap='gray', origin='upper',
                  extent=[0, duration, 0, max_freq])
        
        # Overlay single layer
        color = layer_colors.get(layer_name, '#FFFFFF')
        mask = layer_info['mask']
        
        # Ensure mask matches spectrogram dimensions
        if mask.shape != spec_array.shape[:2]:
            from scipy.ndimage import zoom
            scale_y = spec_array.shape[0] / mask.shape[0]
            scale_x = spec_array.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
        
        # Create colored overlay
        color_rgb = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
        overlay = np.zeros((*mask.shape, 4))
        overlay[:, :, :3] = color_rgb
        overlay[:, :, 3] = mask.astype(float) * 0.8
        
        ax.imshow(overlay, aspect='auto', origin='upper',
                  extent=[0, duration, 0, max_freq])
        
        pixel_count = np.sum(mask)
        ax.set_title(f"{layer_name}\n({pixel_count} px)", 
                     fontsize=9, color=color, weight='bold')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Frequency (Hz)', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Individual layers visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_projects(project_folder: str, clip_basename: str, 
                           layer_group_name: str = "OrcinusOrca_FrequencyContours",
                           output_folder: Optional[str] = None):
    """
    Create visualizations for all project indices of a given clip.
    
    Args:
        project_folder (str): Root project directory
        clip_basename (str): Base name of the audio clip
        layer_group_name (str): Name of the layer group in GIMP
        output_folder (str, optional): Where to save output images
    """
    indices = find_all_project_indices(project_folder, clip_basename)
    
    if not indices:
        print(f"No projects found for clip: {clip_basename}")
        return
    
    print(f"Found {len(indices)} projects for {clip_basename}: {indices}")
    
    # Create output folder with subfolder for this clip if specified
    if output_folder:
        clip_output_folder = os.path.join(output_folder, clip_basename)
        os.makedirs(clip_output_folder, exist_ok=True)
        print(f"📁 Output folder: {clip_output_folder}")
    else:
        clip_output_folder = None
    
    # Process each project
    for idx in indices:
        project_path = os.path.join(project_folder, f"{clip_basename}_{idx}")
        
        print(f"\n{'='*60}")
        print(f"Processing project index {idx}...")
        print(f"{'='*60}")
        
        # Generate overlay visualization
        overlay_output = None
        if clip_output_folder:
            overlay_output = os.path.join(clip_output_folder, f"{clip_basename}_{idx}_overlay.png")
        
        try:
            visualize_overlay(project_path, layer_group_name, project_index=idx, output_path=overlay_output)
        except Exception as e:
            print(f"❌ Error creating overlay: {e}")
        
        # Generate individual layers visualization
        individual_output = None
        if clip_output_folder:
            individual_output = os.path.join(clip_output_folder, f"{clip_basename}_{idx}_individual.png")
        
        try:
            visualize_individual_layers(project_path, layer_group_name, output_path=individual_output)
        except Exception as e:
            print(f"❌ Error creating individual layers: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed visualization for all {len(indices)} projects")
    print(f"{'='*60}")