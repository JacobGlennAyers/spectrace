#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrace Annotator - GIMP 2.10 Plugin

Constrains GIMP's interface for bioacoustics spectrogram annotation.
Automates template layer setup and provides a simplified control panel
so annotators only need to: pick a layer, draw, and save.

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

# Debug log — check /tmp/spectrace_debug.log if colors misbehave
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
# CONSTANTS
# ============================================================

PLUGIN_VERSION = "1.0.0"
ROOT_GROUP_NAME = "OrcinusOrca_FrequencyContours"

# Layer structure: (name, parent_group_or_None, "group"|"layer")
# Order = XCF stack order (first item = topmost in layers panel).
# Must match orca_template.xcf exactly for downstream pipeline compatibility.
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

# UI-friendly layer list organized by section for radio buttons.
# Values are path-qualified names matching extract_layers_from_xcf() output.
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

# Color mapping: distinct color per layer (RGB tuples).
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
# LAYER CREATION
# ============================================================

def find_existing_root_group(image):
    """Return the existing annotation root group, or None."""
    for layer in image.layers:
        if layer.name == ROOT_GROUP_NAME:
            return layer
    return None


def create_template_layers(image):
    """
    Programmatically create the full annotation layer hierarchy.
    Returns the root layer group.
    """
    width = image.width
    height = image.height

    pdb.gimp_image_undo_group_start(image)

    try:
        # Create root group
        root_group = pdb.gimp_layer_group_new(image)
        pdb.gimp_layer_set_name(root_group, ROOT_GROUP_NAME)
        pdb.gimp_image_insert_layer(image, root_group, None, 0)

        # Track subgroups and their insertion positions
        groups = {}
        position_in_root = 0
        position_in_group = {}

        for name, parent_name, item_type in LAYER_STRUCTURE:
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

    # Last resort: use whatever is currently selected
    return None


def set_foreground_color(r, g, b):
    """Set foreground color using gimpcolor.RGB (best wire serialization)."""
    _log("set_foreground_color(%d, %d, %d)" % (r, g, b))

    # Method 1: gimpcolor.RGB with pdb.gimp_context_set_foreground
    try:
        color = gimpcolor.RGB(r / 255.0, g / 255.0, b / 255.0)
        pdb.gimp_context_set_foreground(color)
        _log("  method 1 (gimpcolor.RGB float) succeeded")
        return
    except Exception as e:
        _log("  method 1 failed: %s" % str(e))

    # Method 2: gimpcolor.RGB with integer 0-255 range
    try:
        color = gimpcolor.RGB(r, g, b)
        pdb.gimp_context_set_foreground(color)
        _log("  method 2 (gimpcolor.RGB int) succeeded")
        return
    except Exception as e:
        _log("  method 2 failed: %s" % str(e))

    # Method 3: gimp.set_foreground with three ints
    try:
        gimp.set_foreground(r, g, b)
        _log("  method 3 (gimp.set_foreground) succeeded")
        return
    except Exception as e:
        _log("  method 3 failed: %s" % str(e))

    # Method 4: tuple form
    try:
        pdb.gimp_context_set_foreground((r, g, b))
        _log("  method 4 (tuple) succeeded")
        return
    except Exception as e:
        _log("  method 4 failed: %s" % str(e))

    _log("  ALL METHODS FAILED")


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
    """Hidden background monitor — polls GIMP's active layer and auto-switches color."""

    def __init__(self, image, layer_map):
        self.image = image
        self.layer_map = layer_map
        self.current_layer_name = None

        # Poll GIMP's active layer every 200ms
        self._timer_id = gobject.timeout_add(200, self._poll_active_layer)

    def _poll_active_layer(self):
        """Check GIMP's active layer and switch foreground color if it changed."""
        try:
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

    def _switch_to_layer(self, lname):
        """Apply a layer switch: set active layer, color, and pencil settings."""
        self.current_layer_name = lname
        _log("POLL: switching to %s" % lname)

        # Enforce pencil settings
        enforce_pencil_settings()

        # Set this layer's foreground color
        color = LAYER_COLORS.get(lname, (255, 255, 255))
        r, g, b = color
        fg = gimpcolor.RGB(r / 255.0, g / 255.0, b / 255.0)
        pdb.gimp_context_set_foreground(fg)
        pdb.gimp_displays_flush()

    def _resolve_layer_path(self, layer):
        """Get path-qualified name for a GIMP layer (e.g. 'Heterodynes/5')."""
        parent = pdb.gimp_item_get_parent(layer)
        if parent is not None and parent.name != ROOT_GROUP_NAME:
            return parent.name + "/" + layer.name
        return layer.name

    def stop(self):
        """Stop the polling timer."""
        if self._timer_id is not None:
            gobject.source_remove(self._timer_id)
            self._timer_id = None


# ============================================================
# PLUGIN ENTRY POINTS
# ============================================================

def spectrace_setup(image, drawable):
    """Main entry point: set up layers and open the control panel."""
    _log("=== spectrace_setup called (v2 polling) ===")
    try:
        pdb.gimp_progress_init("Setting up Spectrace annotation...", None)
    except Exception:
        pass

    # Check for existing layer group
    root_group = find_existing_root_group(image)
    if root_group is None:
        root_group = create_template_layers(image)
        gimp.message("Spectrace: Created annotation layer structure.")
    else:
        gimp.message("Spectrace: Using existing annotation layers.")

    # Build layer index
    layer_map = build_layer_index(image, root_group)

    if not layer_map:
        gimp.message(
            "Spectrace: No layers found in '%s' group. "
            "Something went wrong during layer creation." % ROOT_GROUP_NAME
        )
        return

    # Enforce tool settings
    enforce_pencil_settings()

    # Set initial foreground color to f0_LFC (red)
    first_layer_name = LAYER_SECTIONS[0][1][0]  # "f0_LFC"
    initial_color = LAYER_COLORS.get(first_layer_name, (255, 0, 0))
    _log("Setting initial color for %s: %s" % (first_layer_name, str(initial_color)))
    set_foreground_color(*initial_color)

    # Verify it took effect
    try:
        current = gimp.get_foreground()
        _log("Initial fg readback: %s" % str(current))
    except Exception as e:
        _log("Initial fg readback failed: %s" % str(e))

    _log("STEP A: about to import subprocess")
    import subprocess
    _log("STEP B: subprocess imported")

    if sys.platform == "darwin":
        _log("STEP C: macOS detected, trying osascript")
        try:
            subprocess.Popen([
                'osascript', '-e',
                'tell application "System Events" to keystroke "2"'
            ])
            _log("STEP D: osascript launched")
        except Exception as e:
            _log("STEP D: osascript failed: %s" % str(e))

    _log("STEP E: about to call progress_end")
    try:
        pdb.gimp_progress_end()
        _log("STEP F: progress_end OK")
    except Exception as e:
        _log("STEP F: progress_end skipped: %s" % str(e))

    _log("STEP G: starting background monitor")
    try:
        monitor = SpectraceBackgroundMonitor(image, layer_map)
        _log("STEP H: monitor created, entering gtk.main()")
    except Exception as e:
        _log("STEP H: monitor creation FAILED: %s" % str(e))
        import traceback
        _log(traceback.format_exc())
        return

    # gtk.main() keeps the PDB wire alive so timer callbacks can
    # make PDB calls. Runs until GIMP closes or image is closed.
    gtk.main()
    _log("STEP I: gtk.main() returned, plugin done")


def spectrace_reset_tools(image, drawable):
    """Quick reset of tool settings without opening the panel."""
    enforce_pencil_settings()
    gimp.message("Spectrace: Tool settings reset (Pencil, 1px, hardness 100).")


# ============================================================
# REGISTRATION
# ============================================================

register(
    "python-fu-spectrace-setup",
    "Spectrace: Setup annotation layers and open control panel",
    "Creates the annotation layer structure on the current image and opens "
    "a simplified annotation control panel with layer switching, tool "
    "enforcement, and save functionality.",
    "Spectrace Contributors",
    "MIT License",
    "2025",
    "Setup Annotation...",
    "*",
    [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
    ],
    [],
    spectrace_setup,
    menu="<Image>/Filters/Spectrace",
)

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

main()
