#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrace Annotator - GIMP 2.10 Plugin

Constrains GIMP's interface for bioacoustics spectrogram annotation.
Automates template layer setup and provides a simplified control panel
so annotators only need to: pick a layer, draw, and save.

Features:
  - Open WAV files directly in GIMP (File > Open, select .wav)
  - Dynamic layer structure from any .xcf template
  - Auto foreground color switching per layer
  - Tool enforcement (pencil, 1px, hardness 100)

Install: Copy to GIMP's plug-ins directory and make executable.
  Linux:   ~/.config/GIMP/2.10/plug-ins/
  macOS:   ~/Library/Application Support/GIMP/2.10/plug-ins/
  Windows: %APPDATA%\\GIMP\\2.10\\plug-ins\\

Requires GIMP 2.10.x (NOT 3.0+).
"""

from gimpfu import *
import gimp
import gimpcolor
import gtk
import gobject
import os
import sys
import json
import subprocess
import colorsys

# ============================================================
# DEBUG LOGGING
# ============================================================

_DEBUG = True
_LOG_PATH = "/tmp/spectrace_debug.log"

def _log(msg):
    if not _DEBUG:
        return
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# ============================================================
# CONFIGURATION
# ============================================================

PLUGIN_VERSION = "2.0.0"

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".spectrace", "config.json")

DEFAULT_CONFIG = {
    "spectrace_root": "",
    "python3_path": "",
    "default_nfft": 2048,
    "default_grayscale": True,
    "default_project_dir": "projects",
}

def load_config():
    """Load spectrace configuration from ~/.spectrace/config.json."""
    config = dict(DEFAULT_CONFIG)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                user_config = json.load(f)
            config.update(user_config)
        except Exception as e:
            _log("Config load error: %s" % str(e))
    return config


def find_python3():
    """Find a Python 3 interpreter with the spectrace conda environment."""
    config = load_config()

    # 1. Explicit config
    if config.get("python3_path") and os.path.isfile(config["python3_path"]):
        return config["python3_path"]

    # 2. Common conda paths
    home = os.path.expanduser("~")
    conda_dirs = [
        os.path.join(home, "miniconda3"),
        os.path.join(home, "anaconda3"),
        os.path.join(home, "miniforge3"),
        os.path.join(home, "mambaforge"),
        "/opt/miniconda3",
        "/opt/anaconda3",
        "/opt/miniforge3",
        "/usr/local/miniconda3",
    ]
    for conda_dir in conda_dirs:
        p = os.path.join(conda_dir, "envs", "spectrace", "bin", "python")
        if os.path.isfile(p):
            return p

    # 3. Fallback
    return "python3"


def find_spectrace_root():
    """Find the spectrace project root directory."""
    config = load_config()

    if config.get("spectrace_root") and os.path.isdir(config["spectrace_root"]):
        return config["spectrace_root"]

    # Try common locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "spectrace"),
        os.path.join(home, "Documents", "spectrace"),
        os.path.join(home, "projects", "spectrace"),
    ]
    for c in candidates:
        bridge = os.path.join(c, "spectrace_wav_bridge.py")
        if os.path.isfile(bridge):
            return c

    return ""


# ============================================================
# FALLBACK CONSTANTS (orca template, used when no XCF selected)
# ============================================================

ROOT_GROUP_NAME = "OrcinusOrca_FrequencyContours"

LAYER_STRUCTURE = [
    ("Heterodynes", None, "group"),
    ("unsure", "Heterodynes", "layer"),
    ("12", "Heterodynes", "layer"),
    ("11", "Heterodynes", "layer"),
    ("10", "Heterodynes", "layer"),
    ("9", "Heterodynes", "layer"),
    ("8", "Heterodynes", "layer"),
    ("7", "Heterodynes", "layer"),
    ("6", "Heterodynes", "layer"),
    ("5", "Heterodynes", "layer"),
    ("4", "Heterodynes", "layer"),
    ("3", "Heterodynes", "layer"),
    ("2", "Heterodynes", "layer"),
    ("1", "Heterodynes", "layer"),
    ("0", "Heterodynes", "layer"),
    ("Subharmonics", None, "group"),
    ("subharmonics_HFC", "Subharmonics", "layer"),
    ("subharmonics_LFC", "Subharmonics", "layer"),
    ("heterodyne_or_subharmonic_or_other", None, "layer"),
    ("Cetacean_AdditionalContours", None, "group"),
    ("unsure_CetaceanAdditionalContours", "Cetacean_AdditionalContours", "layer"),
    ("harmonics_CetaceanAdditionalContours", "Cetacean_AdditionalContours", "layer"),
    ("f0_CetaceanAdditionalContours", "Cetacean_AdditionalContours", "layer"),
    ("harmonics_HFC", None, "layer"),
    ("f0_HFC", None, "layer"),
    ("unsure_HFC", None, "layer"),
    ("harmonics_LFC", None, "layer"),
    ("f0_LFC", None, "layer"),
    ("unsure_LFC", None, "layer"),
]

LAYER_SECTIONS = [
    ("Main Layers", [
        "f0_LFC",
        "f0_HFC",
        "harmonics_LFC",
        "harmonics_HFC",
        "unsure_LFC",
        "unsure_HFC",
        "heterodyne_or_subharmonic_or_other",
    ]),
    ("Heterodynes", [
        "Heterodynes/unsure",
        "Heterodynes/0",
        "Heterodynes/1",
        "Heterodynes/2",
        "Heterodynes/3",
        "Heterodynes/4",
        "Heterodynes/5",
        "Heterodynes/6",
        "Heterodynes/7",
        "Heterodynes/8",
        "Heterodynes/9",
        "Heterodynes/10",
        "Heterodynes/11",
        "Heterodynes/12",
    ]),
    ("Subharmonics", [
        "Subharmonics/subharmonics_HFC",
        "Subharmonics/subharmonics_LFC",
    ]),
    ("Cetacean Additional", [
        "Cetacean_AdditionalContours/f0_CetaceanAdditionalContours",
        "Cetacean_AdditionalContours/harmonics_CetaceanAdditionalContours",
        "Cetacean_AdditionalContours/unsure_CetaceanAdditionalContours",
    ]),
]

LAYER_COLORS = {
    "Cetacean_AdditionalContours/f0_CetaceanAdditionalContours": (229, 45, 45),
    "Cetacean_AdditionalContours/harmonics_CetaceanAdditionalContours": (229, 88, 45),
    "Cetacean_AdditionalContours/unsure_CetaceanAdditionalContours": (229, 130, 45),
    "Heterodynes/0": (229, 173, 45),
    "Heterodynes/1": (229, 215, 45),
    "Heterodynes/10": (201, 229, 45),
    "Heterodynes/11": (158, 229, 45),
    "Heterodynes/12": (116, 229, 45),
    "Heterodynes/2": (74, 229, 45),
    "Heterodynes/3": (45, 229, 60),
    "Heterodynes/4": (45, 229, 102),
    "Heterodynes/5": (45, 229, 144),
    "Heterodynes/6": (45, 229, 187),
    "Heterodynes/7": (45, 229, 229),
    "Heterodynes/8": (45, 187, 229),
    "Heterodynes/9": (45, 144, 229),
    "Heterodynes/unsure": (45, 102, 229),
    "Subharmonics/subharmonics_HFC": (45, 60, 229),
    "Subharmonics/subharmonics_LFC": (74, 45, 229),
    "f0_HFC": (0, 0, 255),
    "f0_LFC": (255, 0, 0),
    "harmonics_HFC": (201, 45, 229),
    "harmonics_LFC": (229, 45, 215),
    "heterodyne_or_subharmonic_or_other": (229, 45, 173),
    "unsure_HFC": (229, 45, 130),
    "unsure_LFC": (229, 45, 88),
}

# Brushes to try in order (varies by GIMP install/locale)
HARD_BRUSH_CANDIDATES = [
    "1. Pixel",
    "2. Hardness 100",
    "Hardness 100",
    "2. Hardness 100 (1x1)",
    "Circle (01)",
]


# ============================================================
# DYNAMIC TEMPLATE EXTRACTION
# ============================================================

def extract_template_structure(template_image):
    """
    Extract layer structure from a GIMP template image.

    Opens the first layer group found (the annotation root) and reads
    its hierarchy. Supports two levels: root group with subgroups
    containing layers.

    Returns:
        root_name:       Name of the root layer group
        layer_structure: List of (name, parent_name, "group"|"layer") tuples
        layer_sections:  List of (section_name, [qualified_layer_names]) for UI
        all_layer_names: Ordered list of path-qualified layer names
    """
    # Find root annotation group (first group in the template)
    root_group = None
    for layer in template_image.layers:
        if pdb.gimp_item_is_group(layer):
            root_group = layer
            break

    if root_group is None:
        return None, [], [], []

    root_name = root_group.name
    layer_structure = []
    all_layer_names = []
    layer_sections = []
    main_layers = []

    for child in root_group.children:
        name = child.name

        if pdb.gimp_item_is_group(child):
            layer_structure.append((name, None, "group"))
            section_layers = []

            for sub in child.children:
                sub_name = sub.name
                layer_structure.append((sub_name, name, "layer"))
                qualified = name + "/" + sub_name
                section_layers.append(qualified)
                all_layer_names.append(qualified)

            if section_layers:
                layer_sections.append((name, section_layers))
        else:
            layer_structure.append((name, None, "layer"))
            main_layers.append(name)
            all_layer_names.append(name)

    if main_layers:
        layer_sections.insert(0, ("Main Layers", main_layers))

    return root_name, layer_structure, layer_sections, all_layer_names


def generate_layer_colors(layer_names):
    """
    Generate visually distinct colors for each layer using HSV distribution.

    Args:
        layer_names: Ordered list of path-qualified layer names

    Returns:
        Dict mapping layer name -> (R, G, B) tuple (0-255)
    """
    n = len(layer_names)
    colors = {}
    for i, name in enumerate(layer_names):
        hue = float(i) / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[name] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


# ============================================================
# LAYER CREATION
# ============================================================

def find_existing_root_group(image, root_name):
    """Return the existing annotation root group, or None."""
    for layer in image.layers:
        if layer.name == root_name:
            return layer
    return None


def create_template_layers(image, root_name=ROOT_GROUP_NAME, structure=None):
    """
    Programmatically create the full annotation layer hierarchy.

    Args:
        image:      Target GIMP image
        root_name:  Name for the root layer group
        structure:  List of (name, parent_name, type) tuples.
                    Defaults to LAYER_STRUCTURE if None.

    Returns the root layer group.
    """
    if structure is None:
        structure = LAYER_STRUCTURE

    width = image.width
    height = image.height

    pdb.gimp_image_undo_group_start(image)

    try:
        # Create root group
        root_group = pdb.gimp_layer_group_new(image)
        pdb.gimp_layer_set_name(root_group, root_name)
        pdb.gimp_image_insert_layer(image, root_group, None, 0)

        # Track subgroups and their insertion positions
        groups = {}
        position_in_root = 0
        position_in_group = {}

        for name, parent_name, item_type in structure:
            if item_type == "group":
                subgroup = pdb.gimp_layer_group_new(image)
                pdb.gimp_layer_set_name(subgroup, name)
                pdb.gimp_image_insert_layer(image, subgroup, root_group, position_in_root)
                groups[name] = subgroup
                position_in_group[name] = 0
                position_in_root += 1
            else:
                # Create RGBA layer (transparent by default)
                layer = gimp.Layer(
                    image, name, width, height, RGBA_IMAGE, 100, LAYER_MODE_NORMAL
                )
                if parent_name:
                    parent = groups[parent_name]
                    pos = position_in_group[parent_name]
                    pdb.gimp_image_insert_layer(image, layer, parent, pos)
                    position_in_group[parent_name] += 1
                else:
                    pdb.gimp_image_insert_layer(image, layer, root_group, position_in_root)
                    position_in_root += 1

    finally:
        pdb.gimp_image_undo_group_end(image)

    pdb.gimp_displays_flush()
    return root_group


# ============================================================
# LAYER NAVIGATION
# ============================================================

def build_layer_index(image, root_group):
    """
    Build a mapping from path-qualified layer name to GIMP layer object.
    Matches the naming convention in utils.extract_layers_from_xcf().
    """
    layer_map = {}

    def walk(group, path_prefix=""):
        for child in group.children:
            if pdb.gimp_item_is_group(child):
                subpath = child.name + "/"
                walk(child, path_prefix + subpath)
            else:
                full_name = path_prefix + child.name
                layer_map[full_name] = child

    walk(root_group)
    return layer_map


# ============================================================
# TOOL ENFORCEMENT
# ============================================================

def find_hard_brush():
    """Find a suitable hard brush, trying known names then searching."""
    for candidate in HARD_BRUSH_CANDIDATES:
        try:
            pdb.gimp_context_set_brush(candidate)
            return candidate
        except Exception:
            pass

    # Fallback: search all brushes for one containing "pixel" or "hard"
    num_brushes, brush_list = pdb.gimp_brushes_get_list("")
    for brush_name in brush_list:
        lower = brush_name.lower()
        if "pixel" in lower or "hardness 100" in lower:
            try:
                pdb.gimp_context_set_brush(brush_name)
                return brush_name
            except Exception:
                pass

    return None


def set_foreground_color(r, g, b):
    """Set foreground color using gimpcolor.RGB (best wire serialization)."""
    _log("set_foreground_color(%d, %d, %d)" % (r, g, b))

    try:
        color = gimpcolor.RGB(r / 255.0, g / 255.0, b / 255.0)
        pdb.gimp_context_set_foreground(color)
        _log("  gimpcolor.RGB float succeeded")
        return
    except Exception as e:
        _log("  gimpcolor.RGB float failed: %s" % str(e))

    try:
        color = gimpcolor.RGB(r, g, b)
        pdb.gimp_context_set_foreground(color)
        _log("  gimpcolor.RGB int succeeded")
        return
    except Exception as e:
        _log("  gimpcolor.RGB int failed: %s" % str(e))

    try:
        gimp.set_foreground(r, g, b)
        _log("  gimp.set_foreground succeeded")
        return
    except Exception as e:
        _log("  gimp.set_foreground failed: %s" % str(e))

    try:
        pdb.gimp_context_set_foreground((r, g, b))
        _log("  tuple form succeeded")
        return
    except Exception as e:
        _log("  tuple form failed: %s" % str(e))

    _log("  ALL COLOR METHODS FAILED")


def enforce_pencil_settings():
    """Force pencil tool with the exact required settings."""
    pdb.gimp_context_set_paint_mode(LAYER_MODE_NORMAL)
    pdb.gimp_context_set_opacity(100.0)
    find_hard_brush()
    pdb.gimp_context_set_brush_size(1.0)
    pdb.gimp_context_set_dynamics("Dynamics Off")


def enforce_eraser_settings():
    """Force eraser tool with matching settings."""
    pdb.gimp_context_set_opacity(100.0)
    find_hard_brush()
    pdb.gimp_context_set_brush_size(1.0)
    pdb.gimp_context_set_dynamics("Dynamics Off")


# ============================================================
# GTK CONTROL PANEL
# ============================================================

class SpectraceBackgroundMonitor(object):
    """Hidden background monitor -- polls GIMP's active layer and
    continuously enforces brush/opacity/dynamics settings so tool
    switches (pencil, eraser, or accidental changes) are corrected."""

    def __init__(self, image, layer_map, layer_colors=None):
        self.image = image
        self.layer_map = layer_map
        self.layer_colors = layer_colors if layer_colors is not None else LAYER_COLORS
        self.current_layer_name = None
        self._last_tool = None        # track tool switches
        self._eraser_size = 10.0      # remembered eraser brush size

        # Poll every 200ms
        self._timer_id = gobject.timeout_add(200, self._poll)

    def _poll(self):
        """Check active layer and enforce tool settings every cycle."""
        try:
            self._enforce_settings()

            # --- Layer change detection ---
            gimp_active = self.image.active_layer
            if gimp_active is not None:
                gimp_lname = self._resolve_layer_path(gimp_active)
            else:
                gimp_lname = None

            if gimp_lname is not None and gimp_lname != self.current_layer_name:
                self._switch_to_layer(gimp_lname)
        except Exception:
            pass

        return True  # Keep polling

    def _enforce_settings(self):
        """Enforce tool settings. Pencil is locked to 1px. Eraser size is
        remembered and restored across tool switches."""
        try:
            is_pencil = True
            try:
                method = pdb.gimp_context_get_paint_method()
                is_pencil = (method == "gimp-pencil")
            except Exception:
                pass

            current_size = pdb.gimp_context_get_brush_size()

            # Detect tool switch
            tool_key = "pencil" if is_pencil else "eraser"
            if tool_key != self._last_tool:
                if self._last_tool == "eraser":
                    # Leaving eraser — save its size before we set 1px
                    self._eraser_size = current_size
                if tool_key == "eraser":
                    # Entering eraser — restore saved size
                    pdb.gimp_context_set_brush_size(self._eraser_size)
                self._last_tool = tool_key

            # --- Always enforced (pencil AND eraser) ---
            if pdb.gimp_context_get_opacity() != 100.0:
                pdb.gimp_context_set_opacity(100.0)

            try:
                if pdb.gimp_context_get_dynamics() != "Dynamics Off":
                    pdb.gimp_context_set_dynamics("Dynamics Off")
            except Exception:
                pass

            if pdb.gimp_context_get_paint_mode() != LAYER_MODE_NORMAL:
                pdb.gimp_context_set_paint_mode(LAYER_MODE_NORMAL)

            try:
                current_brush = pdb.gimp_context_get_brush()
                lower = current_brush.lower() if current_brush else ""
                if "pixel" not in lower and "hardness 100" not in lower:
                    find_hard_brush()
            except Exception:
                pass

            # --- Pencil only: lock brush size to 1px ---
            if is_pencil:
                if pdb.gimp_context_get_brush_size() != 1.0:
                    pdb.gimp_context_set_brush_size(1.0)
            else:
                # Eraser: track user's size changes for next restore
                self._eraser_size = pdb.gimp_context_get_brush_size()
        except Exception:
            pass

    def _switch_to_layer(self, lname):
        """Apply a layer switch: reset to pencil defaults and update color."""
        self.current_layer_name = lname
        _log("POLL: switching to %s" % lname)

        # Set this layer's foreground color
        color = self.layer_colors.get(lname, (255, 255, 255))
        r, g, b = color
        fg = gimpcolor.RGB(r / 255.0, g / 255.0, b / 255.0)
        pdb.gimp_context_set_foreground(fg)
        pdb.gimp_displays_flush()

    def _resolve_layer_path(self, layer):
        """Get path-qualified name for a GIMP layer (e.g. 'Heterodynes/5')."""
        parent = pdb.gimp_item_get_parent(layer)
        if parent is not None and parent.name != ROOT_GROUP_NAME:
            # Check if we're dealing with a dynamic root name
            grandparent = pdb.gimp_item_get_parent(parent)
            if grandparent is not None:
                # parent is a subgroup inside the root — use parent.name/layer.name
                return parent.name + "/" + layer.name
        return layer.name

    def stop(self):
        """Stop the polling timer."""
        if self._timer_id is not None:
            gobject.source_remove(self._timer_id)
            self._timer_id = None


# ============================================================
# CALLMARK SESSION STATE
# ============================================================

_CALLMARK_SESSION = {
    "active": False,
    "wav_path": "",
    "excel_path": "",
    "vocalizations": [],
    "current_index": 0,
    "individual_filter": "All",
    "project_folders": {},
    "nfft": 2048,
    "grayscale": True,
    "current_image": None,
    "spectrace_root": "",
    "project_dir": "",
}

_CALLMARK_SESSION_FILE = os.path.join(
    os.path.expanduser("~"), ".spectrace", "callmark_session.json"
)


def _save_callmark_session():
    """Persist CallMark session to disk so it survives across plugin calls."""
    session = _CALLMARK_SESSION
    # Only save serializable fields (not the GIMP image object)
    data = {
        "active": session["active"],
        "wav_path": session["wav_path"],
        "excel_path": session["excel_path"],
        "vocalizations": session["vocalizations"],
        "current_index": session["current_index"],
        "individual_filter": session["individual_filter"],
        "project_folders": {str(k): v for k, v in session["project_folders"].items()},
        "nfft": session["nfft"],
        "grayscale": session["grayscale"],
        "spectrace_root": session["spectrace_root"],
        "project_dir": session["project_dir"],
    }
    try:
        session_dir = os.path.dirname(_CALLMARK_SESSION_FILE)
        if not os.path.isdir(session_dir):
            os.makedirs(session_dir)
        with open(_CALLMARK_SESSION_FILE, "w") as f:
            json.dump(data, f)
        _log("CallMark session saved to %s" % _CALLMARK_SESSION_FILE)
    except Exception as e:
        _log("CallMark session save error: %s" % str(e))


def _load_callmark_session():
    """Restore CallMark session from disk."""
    global _CALLMARK_SESSION
    if not os.path.isfile(_CALLMARK_SESSION_FILE):
        return False
    try:
        with open(_CALLMARK_SESSION_FILE, "r") as f:
            data = json.load(f)
        _CALLMARK_SESSION["active"] = data.get("active", False)
        _CALLMARK_SESSION["wav_path"] = data.get("wav_path", "")
        _CALLMARK_SESSION["excel_path"] = data.get("excel_path", "")
        _CALLMARK_SESSION["vocalizations"] = data.get("vocalizations", [])
        _CALLMARK_SESSION["current_index"] = data.get("current_index", 0)
        _CALLMARK_SESSION["individual_filter"] = data.get("individual_filter", "All")
        _CALLMARK_SESSION["project_folders"] = {
            int(k): v for k, v in data.get("project_folders", {}).items()
        }
        _CALLMARK_SESSION["nfft"] = data.get("nfft", 2048)
        _CALLMARK_SESSION["grayscale"] = data.get("grayscale", True)
        _CALLMARK_SESSION["spectrace_root"] = data.get("spectrace_root", "")
        _CALLMARK_SESSION["project_dir"] = data.get("project_dir", "")
        _CALLMARK_SESSION["current_image"] = None  # Can't persist GIMP objects
        _log("CallMark session restored: active=%s, index=%d/%d" % (
            _CALLMARK_SESSION["active"],
            _CALLMARK_SESSION["current_index"],
            len(_CALLMARK_SESSION["vocalizations"]),
        ))
        return _CALLMARK_SESSION["active"]
    except Exception as e:
        _log("CallMark session load error: %s" % str(e))
        return False


def _clear_callmark_session():
    """Clear the persisted session file."""
    try:
        if os.path.isfile(_CALLMARK_SESSION_FILE):
            os.remove(_CALLMARK_SESSION_FILE)
    except Exception:
        pass


def _build_clean_env(python3):
    """Build a clean environment for Python 3 subprocess calls."""
    clean_env = dict(os.environ)
    for key in ["PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                "PYTHONSTARTUP", "PYTHONCASEOK"]:
        clean_env.pop(key, None)
    conda_bin = os.path.dirname(python3)
    if conda_bin not in clean_env.get("PATH", ""):
        clean_env["PATH"] = conda_bin + ":" + clean_env.get("PATH", "")
    return clean_env


def _run_bridge(spectrace_root, python3, args_list):
    """
    Run spectrace_wav_bridge.py with given arguments.
    Returns parsed JSON dict on success, or None on failure.
    """
    bridge_script = os.path.join(spectrace_root, "spectrace_wav_bridge.py")
    if not os.path.isfile(bridge_script):
        _log("Bridge script not found: %s" % bridge_script)
        return None

    cmd = [python3, bridge_script] + args_list
    _log("Running bridge: %s" % " ".join(cmd))

    clean_env = _build_clean_env(python3)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=spectrace_root,
            env=clean_env,
        )
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.strip() if stderr else "Unknown error"
            _log("Bridge failed (rc=%d): %s" % (proc.returncode, error_msg))
            return None

        result = json.loads(stdout.strip())
        return result

    except Exception as e:
        _log("Bridge exception: %s" % str(e))
        return None


def _show_callmark_picker():
    """
    Show a GTK file chooser asking the user to optionally select a CallMark
    Excel export file. Returns the file path string, or None if cancelled.
    """
    dialog = gtk.FileChooserDialog(
        "Select CallMark Export (Cancel to skip)",
        None,
        gtk.FILE_CHOOSER_ACTION_OPEN,
        (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
         gtk.STOCK_OK, gtk.RESPONSE_OK),
    )

    # Add filter for Excel files
    filt = gtk.FileFilter()
    filt.set_name("Excel files (*.xlsx)")
    filt.add_pattern("*.xlsx")
    filt.add_pattern("*.XLSX")
    dialog.add_filter(filt)

    filt_all = gtk.FileFilter()
    filt_all.set_name("All files")
    filt_all.add_pattern("*")
    dialog.add_filter(filt_all)

    response = dialog.run()
    filepath = None
    if response == gtk.RESPONSE_OK:
        filepath = dialog.get_filename()
    dialog.destroy()

    # Process pending GTK events so dialog fully closes
    while gtk.events_pending():
        gtk.main_iteration()

    return filepath


def _show_individual_filter(individuals):
    """
    Show a GTK dialog with a dropdown to filter by individual.
    Returns the selected individual string or "All".
    Returns None if cancelled.
    """
    dialog = gtk.Dialog(
        "Filter by Individual",
        None,
        gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
        (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
         gtk.STOCK_OK, gtk.RESPONSE_OK),
    )

    label = gtk.Label("Select individual (or 'All'):")
    dialog.vbox.pack_start(label, False, False, 8)

    combo = gtk.combo_box_new_text()
    combo.append_text("All")
    for ind in individuals:
        combo.append_text(str(ind))
    combo.set_active(0)
    dialog.vbox.pack_start(combo, False, False, 8)

    # Show count info
    count_label = gtk.Label("%d individuals available" % len(individuals))
    dialog.vbox.pack_start(count_label, False, False, 4)

    dialog.show_all()
    response = dialog.run()

    selected = None
    if response == gtk.RESPONSE_OK:
        selected = combo.get_active_text()
    dialog.destroy()

    while gtk.events_pending():
        gtk.main_iteration()

    return selected


def _load_png_into_gimp(png_path):
    """Load a PNG file into a new GIMP image. Returns the image object."""
    import struct
    with open(png_path, "rb") as f:
        f.read(16)
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    _log("Loading PNG %dx%d: %s" % (w, h, png_path))

    img = gimp.Image(w, h, RGB)
    layer = pdb.gimp_file_load_layer(img, png_path)
    pdb.gimp_image_insert_layer(img, layer, None, 0)
    img.flatten()
    pdb.gimp_image_set_filename(img, png_path)
    return img


def _load_wav_callmark_mode(filename, clip_basename, project_dir, callmark_data):
    """
    Handle WAV loading in CallMark mode.
    Shows individual filter, stores session state, generates first segment.
    Returns a GIMP image.
    """
    global _CALLMARK_SESSION

    vocalizations = callmark_data["vocalizations"]
    individuals = callmark_data["individuals"]

    # Show individual filter
    selected = _show_individual_filter(individuals)
    if selected is None:
        _log("CallMark: individual filter cancelled, falling back to normal mode")
        return None  # Fall through to normal mode

    # Filter vocalizations
    if selected != "All":
        filtered = [v for v in vocalizations if v.get("individual") == selected]
    else:
        filtered = list(vocalizations)

    if not filtered:
        gimp.message("Spectrace: No vocalizations found for individual '%s'." % selected)
        return None

    _log("CallMark: %d vocalizations for '%s'" % (len(filtered), selected))

    config = load_config()
    spectrace_root = find_spectrace_root()
    python3 = find_python3()

    # Store session state
    _CALLMARK_SESSION["active"] = True
    _CALLMARK_SESSION["wav_path"] = filename
    _CALLMARK_SESSION["excel_path"] = callmark_data.get("excel_path", "")
    _CALLMARK_SESSION["vocalizations"] = filtered
    _CALLMARK_SESSION["current_index"] = 0
    _CALLMARK_SESSION["individual_filter"] = selected
    _CALLMARK_SESSION["project_folders"] = {}
    _CALLMARK_SESSION["nfft"] = config.get("default_nfft", 2048)
    _CALLMARK_SESSION["grayscale"] = config.get("default_grayscale", True)
    _CALLMARK_SESSION["spectrace_root"] = spectrace_root
    _CALLMARK_SESSION["project_dir"] = project_dir

    # Save manifest at recording level
    recording_dir = os.path.join(project_dir, clip_basename)
    if not os.path.isdir(recording_dir):
        os.makedirs(recording_dir)
    manifest_path = os.path.join(recording_dir, "callmark_manifest.json")
    manifest = {
        "callmark_excel": callmark_data.get("excel_path", ""),
        "wav_file": filename,
        "individual_filter": selected,
        "total_vocalizations": len(filtered),
        "individuals": individuals,
    }
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _log("Saved manifest: %s" % manifest_path)
    except Exception as e:
        _log("Manifest save error: %s" % str(e))

    # Generate first segment
    result = _generate_callmark_segment(0)
    if result is None:
        gimp.message("Spectrace: Failed to generate first vocalization spectrogram.")
        _CALLMARK_SESSION["active"] = False
        return None

    png_path = result["spectrogram_path"]
    _CALLMARK_SESSION["project_folders"][0] = result["project_folder"]

    # Load into GIMP
    img = _load_png_into_gimp(png_path)
    _CALLMARK_SESSION["current_image"] = img

    # Persist session to disk
    _save_callmark_session()

    # Show status
    voc = filtered[0]
    gimp.message("CallMark: Vocalization 1/%d - %s, %s, Age %d dph" % (
        len(filtered),
        voc.get("individual", "?"),
        voc.get("category", "?"),
        voc.get("age", 0),
    ))

    return img


def _generate_callmark_segment(index):
    """
    Call the bridge to generate a spectrogram for the vocalization at index.
    Returns dict with spectrogram_path and project_folder, or None on failure.
    """
    session = _CALLMARK_SESSION
    voc = session["vocalizations"][index]
    python3 = find_python3()

    args_list = [
        "--mode", "segment-spectrogram",
        "--wav", session["wav_path"],
        "--output-dir", session["project_dir"],
        "--nfft", str(session["nfft"]),
        "--individual", voc.get("individual", "unknown"),
        "--voc-index", str(index),
        "--callmark-meta", json.dumps(voc),
    ]
    if session["grayscale"]:
        args_list.append("--grayscale")

    return _run_bridge(session["spectrace_root"], python3, args_list)


def _auto_save_xcf(image):
    """Auto-save the current image as XCF in the CallMark project folder."""
    session = _CALLMARK_SESSION
    idx = session["current_index"]
    project_folder = session["project_folders"].get(idx)
    if not project_folder:
        _log("CallMark: no project folder for index %d, skipping save" % idx)
        return

    voc_name = "v%03d" % idx
    xcf_path = os.path.join(project_folder, voc_name + ".xcf")

    try:
        pdb.gimp_xcf_save(0, image, image.active_layer, xcf_path, xcf_path)
        _log("CallMark: auto-saved XCF: %s" % xcf_path)
    except Exception as e:
        _log("CallMark: XCF save failed: %s" % str(e))
        gimp.message("Spectrace: Could not auto-save XCF:\n%s" % str(e))


def _load_callmark_segment_into_gimp(index):
    """
    Generate or reload vocalization segment at index.
    Closes old image, loads new one, runs setup annotation.
    """
    global _CALLMARK_SESSION
    session = _CALLMARK_SESSION

    # Check for existing XCF (previously annotated)
    voc_name = "v%03d" % index
    voc = session["vocalizations"][index]
    individual = voc.get("individual", "unknown")
    clip_basename = os.path.splitext(os.path.basename(session["wav_path"]))[0]
    voc_dir = os.path.join(
        session["project_dir"], clip_basename, individual, voc_name
    )
    xcf_path = os.path.join(voc_dir, voc_name + ".xcf")

    img = None

    if os.path.isfile(xcf_path):
        # Re-open previously annotated XCF
        _log("CallMark: reopening XCF: %s" % xcf_path)
        try:
            img = pdb.gimp_file_load(xcf_path, xcf_path)
            session["project_folders"][index] = voc_dir
        except Exception as e:
            _log("CallMark: XCF load failed: %s, regenerating" % str(e))
            img = None

    if img is None:
        # Generate fresh spectrogram
        result = _generate_callmark_segment(index)
        if result is None:
            gimp.message("Spectrace: Failed to generate spectrogram for vocalization %d." % (index + 1))
            return
        session["project_folders"][index] = result["project_folder"]
        img = _load_png_into_gimp(result["spectrogram_path"])

    # Close old image
    old_image = session.get("current_image")
    if old_image is not None:
        try:
            for display in gimp.display_list():
                try:
                    if pdb.gimp_display_get_window_position(display) is not None:
                        pass
                except Exception:
                    pass
            # Delete all displays of the old image
            display_list = list(gimp.display_list())
            for d in display_list:
                try:
                    if d.image.ID == old_image.ID:
                        pdb.gimp_display_delete(d)
                except Exception:
                    pass
            pdb.gimp_image_delete(old_image)
        except Exception as e:
            _log("CallMark: old image cleanup error: %s" % str(e))

    # Show new image
    pdb.gimp_display_new(img)
    pdb.gimp_displays_flush()

    session["current_image"] = img
    session["current_index"] = index

    # Persist updated index
    _save_callmark_session()

    # Show status
    total = len(session["vocalizations"])
    gimp.message("CallMark: Vocalization %d/%d - %s, %s, Age %d dph" % (
        index + 1, total,
        voc.get("individual", "?"),
        voc.get("category", "?"),
        voc.get("age", 0),
    ))


def spectrace_next_vocalization(image, drawable):
    """Navigate to the next vocalization in the CallMark session."""
    global _CALLMARK_SESSION
    session = _CALLMARK_SESSION

    # Restore session from disk if not active in memory
    if not session["active"]:
        _load_callmark_session()
        session = _CALLMARK_SESSION

    if not session["active"]:
        gimp.message("Spectrace: No active CallMark session.\n\n"
                     "Open a WAV file and select a CallMark Excel export first.")
        return

    current = session["current_index"]
    total = len(session["vocalizations"])

    if current >= total - 1:
        gimp.message("Spectrace: Already at last vocalization (%d/%d)." % (current + 1, total))
        return

    # Auto-save current
    _auto_save_xcf(image)

    # Load next
    _load_callmark_segment_into_gimp(current + 1)


def spectrace_prev_vocalization(image, drawable):
    """Navigate to the previous vocalization in the CallMark session."""
    global _CALLMARK_SESSION
    session = _CALLMARK_SESSION

    # Restore session from disk if not active in memory
    if not session["active"]:
        _load_callmark_session()
        session = _CALLMARK_SESSION

    if not session["active"]:
        gimp.message("Spectrace: No active CallMark session.\n\n"
                     "Open a WAV file and select a CallMark Excel export first.")
        return

    current = session["current_index"]

    if current <= 0:
        gimp.message("Spectrace: Already at first vocalization (1/%d)." % len(session["vocalizations"]))
        return

    # Auto-save current
    _auto_save_xcf(image)

    # Load previous
    _load_callmark_segment_into_gimp(current - 1)


# ============================================================
# WAV FILE LOAD HANDLER
# ============================================================

def _find_existing_spectrogram(project_dir, clip_basename):
    """
    Search project_dir for existing projects matching this clip.
    Returns the spectrogram PNG path from the latest project, or None.

    Projects follow the naming convention: <clip_basename>_<index>/
    Each contains: <clip_basename>_<index>_spectrogram.png
    """
    if not os.path.isdir(project_dir):
        return None

    best_idx = -1
    best_png = None

    try:
        for entry in os.listdir(project_dir):
            entry_path = os.path.join(project_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if not entry.startswith(clip_basename + "_"):
                continue
            suffix = entry[len(clip_basename) + 1:]
            try:
                idx = int(suffix)
            except ValueError:
                continue
            # Check for spectrogram PNG
            png_name = "%s_%d_spectrogram.png" % (clip_basename, idx)
            png_path = os.path.join(entry_path, png_name)
            if os.path.isfile(png_path) and idx > best_idx:
                best_idx = idx
                best_png = png_path
    except OSError:
        return None

    return best_png


def load_wav(filename, raw_filename):
    """
    GIMP file load handler for WAV files.

    When a user selects a .wav file in File > Open, this handler:
    1. Checks for an existing spectrogram in the projects directory
    2. If none found, calls spectrace_wav_bridge.py to generate one
    3. Loads the PNG into GIMP as a new image
    """
    _log("=== load_wav called: %s ===" % filename)

    try:
        pdb.gimp_progress_init("Loading spectrogram...", None)
    except Exception:
        pass

    config = load_config()
    spectrace_root = find_spectrace_root()

    # Determine output directory
    project_dir = config.get("default_project_dir", "projects")
    if not os.path.isabs(project_dir):
        project_dir = os.path.join(spectrace_root, project_dir)

    # Derive clip basename from the WAV filename
    clip_basename = os.path.splitext(os.path.basename(filename))[0]

    # === CallMark import: show file picker before generating spectrogram ===
    callmark_xlsx = _show_callmark_picker()
    if callmark_xlsx is not None:
        _log("CallMark Excel selected: %s" % callmark_xlsx)
        python3 = find_python3()
        callmark_data = _run_bridge(
            spectrace_root, python3,
            ["--mode", "parse-callmark", "--callmark-excel", callmark_xlsx]
        )
        if callmark_data is not None:
            callmark_data["excel_path"] = callmark_xlsx
            img = _load_wav_callmark_mode(filename, clip_basename, project_dir, callmark_data)
            if img is not None:
                return img
            _log("CallMark mode returned None, falling back to normal mode")
        else:
            _log("CallMark parse failed, falling back to normal mode")
            gimp.message("Spectrace: Could not parse CallMark file.\nFalling back to normal mode.")
    else:
        _log("No CallMark file selected, using normal mode")

    # Check for existing spectrogram first
    png_path = _find_existing_spectrogram(project_dir, clip_basename)

    if png_path:
        _log("Found existing spectrogram: %s" % png_path)
    else:
        # No existing project — generate a new spectrogram
        _log("No existing spectrogram for '%s', generating..." % clip_basename)

        python3 = find_python3()
        bridge_script = os.path.join(spectrace_root, "spectrace_wav_bridge.py")
        if not os.path.isfile(bridge_script):
            gimp.message(
                "Spectrace: Cannot find spectrace_wav_bridge.py\n\n"
                "Configure spectrace_root in:\n  %s\n\n"
                "Or install spectrace to ~/spectrace/" % CONFIG_PATH
            )
            raise RuntimeError("spectrace_wav_bridge.py not found at: %s" % bridge_script)

        nfft = config.get("default_nfft", 2048)
        grayscale = config.get("default_grayscale", True)

        cmd = [
            python3, bridge_script,
            "--wav", filename,
            "--output-dir", project_dir,
            "--nfft", str(nfft),
        ]
        if grayscale:
            cmd.append("--grayscale")

        _log("Running: %s" % " ".join(cmd))

        try:
            pdb.gimp_progress_update(0.1)
        except Exception:
            pass

        # Build a clean environment for Python 3 subprocess.
        # GIMP sets PYTHONHOME/PYTHONPATH to its bundled Python 2.7, which
        # fatally breaks any external Python 3 interpreter.
        clean_env = dict(os.environ)
        for key in ["PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                    "PYTHONSTARTUP", "PYTHONCASEOK"]:
            clean_env.pop(key, None)
        conda_bin = os.path.dirname(python3)
        if conda_bin not in clean_env.get("PATH", ""):
            clean_env["PATH"] = conda_bin + ":" + clean_env.get("PATH", "")

        # Call the bridge script
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=spectrace_root,
                env=clean_env,
            )
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.strip() if stderr else "Unknown error"
                _log("Bridge script failed (rc=%d): %s" % (proc.returncode, error_msg))
                gimp.message(
                    "Spectrace: Spectrogram generation failed.\n\n"
                    "Error: %s\n\n"
                    "Check that the spectrace conda environment is set up:\n"
                    "  conda activate spectrace" % error_msg
                )
                raise RuntimeError("Bridge script failed: %s" % error_msg)

            result = json.loads(stdout.strip())
            png_path = result["spectrogram_path"]
            _log("Generated spectrogram: %s" % png_path)

        except (ValueError, KeyError) as e:
            _log("JSON parse error: %s / stdout: %s" % (str(e), stdout))
            gimp.message("Spectrace: Could not parse bridge script output.\n%s" % str(e))
            raise RuntimeError("Bridge script output parse error")

    try:
        pdb.gimp_progress_update(0.8)
    except Exception:
        pass

    # Load the generated PNG into a new GIMP image.
    # We avoid pdb.gimp_file_load() here because calling one file load
    # handler from inside another causes reentrancy failures in GIMP 2.10.
    # Instead, create a blank image and load the PNG as a layer.
    _log("Loading PNG into GIMP: %s (exists=%s)" % (png_path, os.path.isfile(png_path)))
    try:
        # Read PNG dimensions from file header
        import struct
        with open(png_path, "rb") as f:
            f.read(16)  # 8-byte PNG sig + 4-byte IHDR len + 4-byte IHDR tag
            w = struct.unpack(">I", f.read(4))[0]
            h = struct.unpack(">I", f.read(4))[0]
        _log("PNG dimensions: %dx%d" % (w, h))

        img = gimp.Image(w, h, RGB)
        layer = pdb.gimp_file_load_layer(img, png_path)
        _log("Layer loaded: %s" % str(layer))
        pdb.gimp_image_insert_layer(img, layer, None, 0)
        img.flatten()
        pdb.gimp_image_set_filename(img, png_path)
    except Exception as e:
        _log("PNG load failed: %s" % str(e))
        import traceback
        _log(traceback.format_exc())
        gimp.message("Spectrace: Failed to load spectrogram PNG:\n%s" % str(e))
        raise

    try:
        pdb.gimp_progress_end()
    except Exception:
        pass

    _log("WAV loaded successfully as spectrogram: %s" % png_path)
    return img


def register_load_handlers():
    """Called during GIMP's query phase to register WAV as a loadable format."""
    gimp.register_load_handler("file-wav-spectrogram-load", "wav,WAV", "")


# ============================================================
# PLUGIN ENTRY POINTS
# ============================================================

def spectrace_setup(image, drawable, template_xcf=""):
    """
    Main entry point: set up annotation layers and start the background monitor.

    If template_xcf is provided, layers are extracted dynamically from the
    template XCF file. Otherwise falls back to the built-in orca template.
    """
    _log("=== spectrace_setup called (v%s, dynamic template) ===" % PLUGIN_VERSION)

    try:
        pdb.gimp_progress_init("Setting up Spectrace annotation...", None)
    except Exception:
        pass

    # Resolve template
    use_root = ROOT_GROUP_NAME
    use_structure = LAYER_STRUCTURE
    use_sections = LAYER_SECTIONS
    use_colors = LAYER_COLORS

    if template_xcf and os.path.isfile(template_xcf):
        _log("Loading template XCF: %s" % template_xcf)
        try:
            template_img = pdb.gimp_file_load(template_xcf, template_xcf)

            root_name, dyn_structure, dyn_sections, layer_names = \
                extract_template_structure(template_img)

            pdb.gimp_image_delete(template_img)

            if root_name is not None and dyn_structure:
                use_root = root_name
                use_structure = dyn_structure
                use_sections = dyn_sections
                use_colors = generate_layer_colors(layer_names)
                _log("Template loaded: root=%s, %d layers, %d sections" % (
                    root_name, len(layer_names), len(dyn_sections)))
            else:
                _log("Template had no layer groups, using default orca template")
                gimp.message(
                    "Spectrace: No layer groups found in template.\n"
                    "Using default orca template."
                )
        except Exception as e:
            _log("Template load failed: %s" % str(e))
            gimp.message(
                "Spectrace: Could not load template XCF:\n%s\n\n"
                "Using default orca template." % str(e)
            )
    else:
        if template_xcf:
            _log("Template file not found: %s" % template_xcf)

    # Check for existing annotation group
    root_group = find_existing_root_group(image, use_root)
    if root_group is None:
        root_group = create_template_layers(image, use_root, use_structure)
        gimp.message("Spectrace: Created annotation layer structure.")
    else:
        gimp.message("Spectrace: Using existing annotation layers.")

    # Build layer index
    layer_map = build_layer_index(image, root_group)

    if not layer_map:
        gimp.message(
            "Spectrace: No layers found in '%s' group. "
            "Something went wrong during layer creation." % use_root
        )
        return

    # Enforce tool settings
    enforce_pencil_settings()

    # Set initial foreground color
    first_layer_name = None
    if use_sections and use_sections[0][1]:
        first_layer_name = use_sections[0][1][0]
    elif layer_map:
        first_layer_name = list(layer_map.keys())[0]

    if first_layer_name:
        initial_color = use_colors.get(first_layer_name, (255, 0, 0))
        _log("Setting initial color for %s: %s" % (first_layer_name, str(initial_color)))
        set_foreground_color(*initial_color)

    # Verify color
    try:
        current = gimp.get_foreground()
        _log("Initial fg readback: %s" % str(current))
    except Exception as e:
        _log("Initial fg readback failed: %s" % str(e))

    # macOS: switch to pencil tool via keystroke
    if sys.platform == "darwin":
        try:
            subprocess.Popen([
                'osascript', '-e',
                'tell application "System Events" to keystroke "2"'
            ])
        except Exception as e:
            _log("osascript failed: %s" % str(e))

    try:
        pdb.gimp_progress_end()
    except Exception:
        pass

    # Start background monitor with dynamic colors
    _log("Starting background monitor")
    try:
        monitor = SpectraceBackgroundMonitor(image, layer_map, use_colors)
        _log("Monitor created, entering gtk.main()")
    except Exception as e:
        _log("Monitor creation FAILED: %s" % str(e))
        import traceback
        _log(traceback.format_exc())
        return

    gtk.main()
    _log("gtk.main() returned, plugin done")


def spectrace_reset_tools(image, drawable):
    """Quick reset of tool settings without opening the panel."""
    enforce_pencil_settings()
    gimp.message("Spectrace: Tool settings reset (Pencil, 1px, hardness 100).")


# ============================================================
# REGISTRATION
# ============================================================

# --- WAV File Load Handler ---
register(
    "file-wav-spectrogram-load",
    "Load WAV file as spectrogram",
    "Generates a spectrogram from a WAV audio file using the spectrace "
    "Python toolkit (librosa) and loads the result as a GIMP image.",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "WAV Spectrogram",
    None,
    [
        (PF_STRING, "filename", "The filename to load", None),
        (PF_STRING, "raw-filename", "The raw filename to load", None),
    ],
    [(PF_IMAGE, "image", "Output image")],
    load_wav,
    on_query=register_load_handlers,
    menu="<Load>",
)

# --- Setup Annotation (with dynamic template support) ---
register(
    "python-fu-spectrace-setup",
    "Spectrace: Setup annotation layers from template",
    "Creates the annotation layer structure on the current image from a "
    "template XCF file (or the built-in orca template if none selected) "
    "and starts the background color monitor.",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "Setup Annotation...",
    "*",
    [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
        (PF_FILE, "template-xcf", "Template XCF file (empty = default orca)", ""),
    ],
    [],
    spectrace_setup,
    menu="<Image>/Filters/Spectrace",
)

# --- Reset Tools ---
register(
    "python-fu-spectrace-reset-tools",
    "Spectrace: Reset tool settings",
    "Re-applies the correct pencil/eraser settings for spectrace "
    "annotation (Pencil, 1px, hardness 100, dynamics off).",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "Reset Tool Settings",
    "*",
    [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
    ],
    [],
    spectrace_reset_tools,
    menu="<Image>/Filters/Spectrace",
)

# --- Next Vocalization (CallMark) ---
register(
    "python-fu-spectrace-next-voc",
    "Spectrace: Next Vocalization",
    "Navigate to the next vocalization segment in the CallMark session. "
    "Auto-saves the current XCF before advancing.",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "Next Vocalization",
    "*",
    [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
    ],
    [],
    spectrace_next_vocalization,
    menu="<Image>/Filters/Spectrace",
)

# --- Previous Vocalization (CallMark) ---
register(
    "python-fu-spectrace-prev-voc",
    "Spectrace: Previous Vocalization",
    "Navigate to the previous vocalization segment in the CallMark session. "
    "Auto-saves the current XCF before going back.",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "Previous Vocalization",
    "*",
    [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
    ],
    [],
    spectrace_prev_vocalization,
    menu="<Image>/Filters/Spectrace",
)

main()
