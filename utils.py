import os
import shutil
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Set
import json
import traceback

# Patch gimpformats for Windows compatibility BEFORE importing
def patch_gimpformats():
    """Monkey-patch gimpformats to handle unknown property values on Windows"""
    try:
        import gimpformats.GimpIOBase as GimpIOBase
        from gimpformats.GimpImageHierarchy import ImageProperties
        
        # Save the original function
        original_prop_cmp = GimpIOBase._prop_cmp
        
        def safe_prop_cmp(val, prop):
            """Safe version that handles out-of-range property values"""
            try:
                return original_prop_cmp(val, prop)
            except (IndexError, ValueError):
                # If property value is out of range, just return False
                # This allows the XCF to load even with unknown properties
                return False
        
        # Replace the function
        GimpIOBase._prop_cmp = safe_prop_cmp
        print("✅ Applied gimpformats compatibility patch for Windows")
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Could not apply gimpformats patch: {e}")
        print("    XCF files may not load correctly on Windows")
        return False

# Apply the patch before importing GimpDocument
patch_gimpformats()

from gimpformats.gimpXcfDocument import GimpDocument, GimpGroup


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
        "noverlap" : audio_dict["nfft"] // 2,
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
    audio_df = pd.DataFrame.from_dict(audio_dict)
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
# COLOR MANAGEMENT FUNCTIONS
# =============================================================================

def generate_distinct_colors(n: int, saturation: float = 0.8, value: float = 0.9) -> List[str]:
    """
    Generate n visually distinct colors using HSV color space.
    
    Args:
        n (int): Number of colors to generate
        saturation (float): Saturation value (0-1), default 0.8 for vibrant colors
        value (float): Value/brightness (0-1), default 0.9 for bright colors
    
    Returns:
        List[str]: List of hex color codes
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly space hues around the color wheel
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        hex_color = mcolors.to_hex(rgb)
        colors.append(hex_color)
    return colors


def discover_layer_names_from_template(template_xcf_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours") -> Set[str]:
    """
    Discover all layer names from a master template XCF file.
    
    Args:
        template_xcf_path (str): Path to the master template XCF file
        layer_group_name (str): Name of the layer group to search in
    
    Returns:
        Set[str]: Set of all layer names defined in the template
    """
    print(f"\n🔍 Reading master template: {template_xcf_path}")
    
    if not os.path.isfile(template_xcf_path):
        raise FileNotFoundError(f"Template file not found: {template_xcf_path}")
    
    try:
        # Extract layer names from the template
        layer_data = extract_layers_from_xcf(template_xcf_path, layer_group_name, verbose=False)
        layer_names = set(layer_data['layers'].keys())
        
        print(f"✅ Found {len(layer_names)} layer definitions in template")
        print(f"   Layers: {sorted(layer_names)}")
        
        return layer_names
        
    except Exception as e:
        raise RuntimeError(f"Error reading template XCF: {e}")


def discover_all_layer_names(project_folder: str, layer_group_name: str = "OrcinusOrca_FrequencyContours") -> Set[str]:
    """
    Scan all project folders to discover all unique layer names.
    This is a fallback method when no master template is provided.
    
    Args:
        project_folder (str): Root project directory
        layer_group_name (str): Name of the layer group to search in
    
    Returns:
        Set[str]: Set of unique layer names found across all projects
    """
    all_layer_names = set()
    
    print(f"\n🔍 Scanning projects in: {project_folder}")
    
    # Iterate through all subdirectories
    for project_name in os.listdir(project_folder):
        project_path = os.path.join(project_folder, project_name)
        
        if not os.path.isdir(project_path):
            continue
        
        # Look for XCF files
        xcf_files = [f for f in os.listdir(project_path) if f.endswith('.xcf')]
        
        for xcf_file in xcf_files:
            xcf_path = os.path.join(project_path, xcf_file)
            
            try:
                # Extract layer names from this XCF
                layer_data = extract_layers_from_xcf(xcf_path, layer_group_name, verbose=False)
                layer_names = set(layer_data['layers'].keys())
                all_layer_names.update(layer_names)
                
                if layer_names:
                    print(f"  📁 {project_name}/{xcf_file}: {len(layer_names)} layers")
                
            except Exception as e:
                print(f"  ⚠️  Error reading {xcf_path}: {e}")
                continue
    
    print(f"\n✅ Found {len(all_layer_names)} unique layer names across all projects")
    print(f"   Layers: {sorted(all_layer_names)}")
    
    return all_layer_names


def create_color_mapping(layer_names: Set[str], existing_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Create a color mapping for layer names, preserving existing mappings if provided.
    
    Args:
        layer_names (Set[str]): Set of layer names to create colors for
        existing_mapping (Dict[str, str], optional): Existing color mapping to preserve
    
    Returns:
        Dict[str, str]: Mapping of layer names to hex colors
    """
    if existing_mapping is None:
        existing_mapping = {}
    
    # Separate layers into existing and new
    existing_layers = {name for name in layer_names if name in existing_mapping}
    new_layers = layer_names - existing_layers
    
    # Start with existing mappings
    color_mapping = {name: existing_mapping[name] for name in existing_layers}
    
    # Generate colors for new layers
    if new_layers:
        new_colors = generate_distinct_colors(len(new_layers))
        
        # Assign colors to new layers (sorted for consistency)
        for layer_name, color in zip(sorted(new_layers), new_colors):
            color_mapping[layer_name] = color
    
    return color_mapping


def save_color_mapping(color_mapping: Dict[str, str], project_folder: str, template_path: Optional[str] = None):
    """
    Save color mapping to a JSON file in the project folder.
    
    Args:
        color_mapping (Dict[str, str]): Color mapping to save
        project_folder (str): Root project directory
        template_path (str, optional): Path to the master template (for documentation)
    """
    mapping_path = os.path.join(project_folder, "layer_color_mapping.json")
    
    # Create a wrapper with metadata
    data = {
        "template_source": template_path if template_path else "auto-discovered",
        "last_updated": datetime.now().isoformat(),
        "color_mapping": color_mapping
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    
    print(f"💾 Color mapping saved to: {mapping_path}")
    if template_path:
        print(f"   Based on template: {template_path}")


def load_color_mapping(project_folder: str) -> Optional[Dict[str, str]]:
    """
    Load color mapping from a JSON file in the project folder.
    
    Args:
        project_folder (str): Root project directory
    
    Returns:
        Dict[str, str] or None: Color mapping if exists, None otherwise
    """
    mapping_path = os.path.join(project_folder, "layer_color_mapping.json")
    
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (direct dict) and new format (wrapped with metadata)
        if "color_mapping" in data:
            color_mapping = data["color_mapping"]
            template_source = data.get("template_source", "unknown")
            last_updated = data.get("last_updated", "unknown")
            print(f"📂 Loaded color mapping from: {mapping_path}")
            print(f"   Template source: {template_source}")
            print(f"   Last updated: {last_updated}")
        else:
            # Old format - direct color mapping
            color_mapping = data
            print(f"📂 Loaded color mapping from: {mapping_path} (old format)")
        
        return color_mapping
    
    return None


def get_or_create_color_mapping(project_folder: str, 
                               layer_group_name: str = "OrcinusOrca_FrequencyContours",
                               template_xcf_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get existing color mapping or create a new one by discovering all layers.
    
    Args:
        project_folder (str): Root project directory
        layer_group_name (str): Name of the layer group to search in
        template_xcf_path (str, optional): Path to master template XCF file. If provided,
            this will be used as the authoritative source for layer names. If not provided,
            the function will scan all project XCF files to discover layers.
    
    Returns:
        Dict[str, str]: Color mapping for all discovered layers
    """
    # Try to load existing mapping
    existing_mapping = load_color_mapping(project_folder)
    
    # Discover layer names from template or projects
    if template_xcf_path:
        print(f"\n📋 Using master template for layer discovery")
        all_layer_names = discover_layer_names_from_template(template_xcf_path, layer_group_name)
    else:
        print(f"\n📋 Scanning all projects for layer discovery")
        all_layer_names = discover_all_layer_names(project_folder, layer_group_name)
    
    if not all_layer_names:
        print("⚠️  No layers found!")
        return {}
    
    # Create or update color mapping
    color_mapping = create_color_mapping(all_layer_names, existing_mapping)
    
    # Save the updated mapping
    save_color_mapping(color_mapping, project_folder, template_xcf_path)
    
    # Display the mapping
    print("\n🎨 Color Mapping:")
    for layer_name in sorted(color_mapping.keys()):
        color = color_mapping[layer_name]
        print(f"   {layer_name}: {color}")
    
    return color_mapping


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


def extract_layers_from_xcf(xcf_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours", 
                           verbose: bool = True) -> Dict:
    """
    Extract layers from XCF file within a specified layer group.
    Now recursively processes subgroups and includes full group paths in layer names!
    
    Args:
        xcf_path (str): Path to the XCF file
        layer_group_name (str): Name of the layer group to extract (this is the ROOT group that will be excluded from paths)
        verbose (bool): Whether to print detailed information
    
    Returns:
        Dict containing layer data and metadata
    """
    if verbose:
        print(f"\n🔍 Loading XCF file: {xcf_path}")
    
    xcf_doc = GimpDocument(xcf_path)
    
    if verbose:
        print(f"📄 XCF dimensions: {xcf_doc.width}x{xcf_doc.height}")
    
    # Get the root group
    root_group = xcf_doc.walkTree()
    
    # Find the target group
    target_group = find_group_by_name(root_group, layer_group_name)
    
    if target_group is None:
        if verbose:
            print(f"❌ Error: Group '{layer_group_name}' not found!")
        return {
            'layers': {},
            'width': xcf_doc.width,
            'height': xcf_doc.height
        }
    
    if verbose:
        print(f"✅ Found layer group: {layer_group_name}")
        print(f"   (This root group name will be excluded from layer paths)")
    
    # Extract layers from the group (recursively)
    layers_data = {}
    
    def extract_layers_recursive(group, path_components=None, depth=0):
        """
        Recursively extract layers from a group and its subgroups.
        
        Args:
            group: The GIMP group to process
            path_components: List of group names forming the path (excluding root)
            depth: Current recursion depth
        """
        if path_components is None:
            path_components = []
        
        indent = "  " * depth
        
        for child in group.children:
            if isinstance(child, GimpGroup):
                # It's a subgroup - add to path and recurse into it
                if verbose:
                    current_path = "/".join(path_components + [child.name])
                    print(f"{indent}📂 Subgroup: {child.name} (path: {current_path if current_path else '(root)'})")
                
                # Recurse with updated path
                extract_layers_recursive(child, path_components + [child.name], depth + 1)
            else:
                # It's a layer - process it with full path
                layer_name = child.name
                
                # Build full layer name with path (excluding root group)
                if path_components:
                    full_layer_name = "/".join(path_components) + "/" + layer_name
                else:
                    # Layer is directly in the root group, no path prefix needed
                    full_layer_name = layer_name
                
                # Force the layer to be fully loaded
                child.forceFullyLoaded()
                
                # Get the image
                img = child.image
                
                if img is not None:
                    # Convert to binary mask based on alpha channel
                    binary_mask = binarize_image_by_alpha(img)
                    
                    # Check if layer is visible
                    visible = child.visible
                    
                    layers_data[full_layer_name] = {
                        'mask': binary_mask,
                        'visible': visible,
                        'size': img.size
                    }
                    
                    if verbose:
                        pixel_count = np.sum(binary_mask)
                        print(f"{indent}  📝 Layer '{full_layer_name}': {img.size}, visible={visible}, pixels={pixel_count}")
                else:
                    if verbose:
                        print(f"{indent}  ⚠️  Warning: No image found for layer: {full_layer_name}")
    
    # Start recursive extraction (path_components starts empty to exclude root group name)
    extract_layers_recursive(target_group, path_components=[])
    
    if verbose:
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
                      project_index: Optional[int] = None, output_path: Optional[str] = None,
                      color_mapping: Optional[Dict[str, str]] = None):
    """
    Create an overlay visualization with all layers on the spectrogram with time/frequency axes.
    
    Args:
        project_path (str): Path to the project folder
        layer_group_name (str): Name of the layer group in GIMP
        project_index (int, optional): Index to display in title
        output_path (str, optional): Where to save the output image
        color_mapping (Dict[str, str], optional): Mapping of layer names to colors
    """
    try:
        # Find spectrogram and XCF files with better error handling
        files = os.listdir(project_path)
        print(f"📁 Files in {project_path}:")
        for f in files:
            print(f"   - {f}")
        
        # Case-insensitive search for spectrogram
        spectrogram_files = [f for f in files if f.lower().endswith('_spectrogram.png')]
        if not spectrogram_files:
            raise FileNotFoundError(f"No spectrogram file (*_spectrogram.png) found in {project_path}")
        spectrogram_file = spectrogram_files[0]
        
        # Case-insensitive search for XCF
        xcf_files = [f for f in files if f.lower().endswith('.xcf')]
        if not xcf_files:
            raise FileNotFoundError(f"No XCF file (*.xcf) found in {project_path}")
        xcf_file = xcf_files[0]
        
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
        
        # Use provided color mapping or generate default colors
        if color_mapping is None:
            layer_names = list(layers.keys())
            colors = generate_distinct_colors(len(layer_names))
            color_mapping = dict(zip(layer_names, colors))
        
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
            
            # Get color from mapping, fallback to white if not found
            color = color_mapping.get(layer_name, '#FFFFFF')
            
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
        
    except Exception as e:
        print(f"❌ Detailed error in visualize_overlay:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Traceback:")
        traceback.print_exc()
        raise


def visualize_individual_layers(project_path: str, layer_group_name: str = "OrcinusOrca_FrequencyContours",
                                output_path: Optional[str] = None, color_mapping: Optional[Dict[str, str]] = None):
    """
    Create a row visualization showing original spectrogram and each layer individually with time/frequency axes.
    
    Args:
        project_path (str): Path to the project folder
        layer_group_name (str): Name of the layer group in GIMP
        output_path (str, optional): Where to save the output image
        color_mapping (Dict[str, str], optional): Mapping of layer names to colors
    """
    try:
        # Find spectrogram and XCF files with better error handling
        files = os.listdir(project_path)
        print(f"📁 Files in {project_path}:")
        for f in files:
            print(f"   - {f}")
        
        # Case-insensitive search for spectrogram
        spectrogram_files = [f for f in files if f.lower().endswith('_spectrogram.png')]
        if not spectrogram_files:
            raise FileNotFoundError(f"No spectrogram file (*_spectrogram.png) found in {project_path}")
        spectrogram_file = spectrogram_files[0]
        
        # Case-insensitive search for XCF
        xcf_files = [f for f in files if f.lower().endswith('.xcf')]
        if not xcf_files:
            raise FileNotFoundError(f"No XCF file (*.xcf) found in {project_path}")
        xcf_file = xcf_files[0]
        
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
        
        # Use provided color mapping or generate default colors
        if color_mapping is None:
            layer_names = list(layers.keys())
            colors = generate_distinct_colors(len(layer_names))
            color_mapping = dict(zip(layer_names, colors))
        
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
            color = color_mapping.get(layer_name, '#FFFFFF')
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
        
    except Exception as e:
        print(f"❌ Detailed error in visualize_individual_layers:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Traceback:")
        traceback.print_exc()
        raise


def visualize_all_projects(project_folder: str, clip_basename: str, 
                           layer_group_name: str = "OrcinusOrca_FrequencyContours",
                           output_folder: Optional[str] = None,
                           color_mapping: Optional[Dict[str, str]] = None,
                           template_xcf_path: Optional[str] = None):
    """
    Create visualizations for all project indices of a given clip.
    
    Args:
        project_folder (str): Root project directory
        clip_basename (str): Base name of the audio clip
        layer_group_name (str): Name of the layer group in GIMP
        output_folder (str, optional): Where to save output images
        color_mapping (Dict[str, str], optional): Mapping of layer names to colors
        template_xcf_path (str, optional): Path to master template (used if color_mapping not provided)
    """
    indices = find_all_project_indices(project_folder, clip_basename)
    
    if not indices:
        print(f"No projects found for clip: {clip_basename}")
        return
    
    print(f"Found {len(indices)} projects for {clip_basename}: {indices}")
    
    # If no color mapping provided, get or create one
    if color_mapping is None:
        color_mapping = get_or_create_color_mapping(project_folder, layer_group_name, template_xcf_path)
    
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
            visualize_overlay(project_path, layer_group_name, project_index=idx, 
                            output_path=overlay_output, color_mapping=color_mapping)
        except Exception as e:
            print(f"❌ Error creating overlay: {e}")
        
        # Generate individual layers visualization
        individual_output = None
        if clip_output_folder:
            individual_output = os.path.join(clip_output_folder, f"{clip_basename}_{idx}_individual.png")
        
        try:
            visualize_individual_layers(project_path, layer_group_name, 
                                       output_path=individual_output, color_mapping=color_mapping)
        except Exception as e:
            print(f"❌ Error creating individual layers: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed visualization for all {len(indices)} projects")
    print(f"{'='*60}")


def visualize_all_projects_in_folder(project_folder: str, 
                                     layer_group_name: str = "OrcinusOrca_FrequencyContours",
                                     output_folder: Optional[str] = None,
                                     template_xcf_path: Optional[str] = None):
    """
    Automatically discover and visualize all projects in a folder.
    This is the main entry point for processing an entire project folder.
    
    Args:
        project_folder (str): Root project directory
        layer_group_name (str): Name of the layer group in GIMP
        output_folder (str, optional): Where to save output images
        template_xcf_path (str, optional): Path to master template XCF file
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting automatic project visualization")
    print(f"{'='*60}")
    
    # Step 1: Discover all layers and create/load color mapping
    color_mapping = get_or_create_color_mapping(project_folder, layer_group_name, template_xcf_path)
    
    if not color_mapping:
        print("❌ No layers found. Exiting.")
        return
    
    # Step 2: Find all unique clip basenames
    clip_basenames = set()
    for project_name in os.listdir(project_folder):
        if os.path.isdir(os.path.join(project_folder, project_name)):
            # Extract clip basename (everything before the last underscore)
            parts = project_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                clip_basenames.add(parts[0])
    
    print(f"\n📁 Found {len(clip_basenames)} unique clips: {sorted(clip_basenames)}")
    
    # Step 3: Process each clip
    for clip_basename in sorted(clip_basenames):
        print(f"\n{'='*60}")
        print(f"Processing clip: {clip_basename}")
        print(f"{'='*60}")
        
        visualize_all_projects(project_folder, clip_basename, layer_group_name, 
                              output_folder, color_mapping, template_xcf_path)
    
    print(f"\n{'='*60}")
    print(f"✅ All visualizations complete!")
    print(f"{'='*60}")