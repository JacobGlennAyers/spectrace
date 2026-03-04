# Spectrace GIMP Plugin — Installation Guide

## Prerequisites

- **GIMP 2.10.x** (NOT 3.0 or later)
- Verify your version: `Help > About` in GIMP should show `2.10.xx`

## Step 1: Find Your GIMP Plug-ins Directory

| Platform | Path |
|----------|------|
| **Linux** | `~/.config/GIMP/2.10/plug-ins/` |
| **Linux (Flatpak)** | `~/.var/app/org.gimp.GIMP/config/GIMP/2.10/plug-ins/` |
| **macOS** | `~/Library/Application Support/GIMP/2.10/plug-ins/` |
| **Windows** | `%APPDATA%\GIMP\2.10\plug-ins\` |

You can also check inside GIMP: `Edit > Preferences > Folders > Plug-ins`

## Step 2: Install the Plugin

1. Copy `spectrace_annotator.py` into the plug-ins directory from Step 1.

2. **Linux/macOS only** — make it executable:
   ```bash
   chmod +x ~/.config/GIMP/2.10/plug-ins/spectrace_annotator.py
   ```

## Step 3 (Optional): Install Simplified Shortcuts

This replaces GIMP's default keyboard shortcuts with a minimal set
(undo, redo, save, zoom only), preventing accidental tool switches.

1. **Back up** your current `menurc`:
   ```bash
   cp ~/.config/GIMP/2.10/menurc ~/.config/GIMP/2.10/menurc.backup
   ```

2. Copy the provided `config/menurc` to your GIMP config directory:
   ```bash
   cp config/menurc ~/.config/GIMP/2.10/menurc
   ```

   Adjust the path for your platform (see table in Step 1, without `/plug-ins/`).

## Step 4: Restart GIMP

Close and reopen GIMP. The plugin appears under:

- `Filters > Spectrace > Setup Annotation...`
- `Filters > Spectrace > Reset Tool Settings`

## Usage

1. Open a spectrogram PNG in GIMP (`File > Open`)
2. Go to `Filters > Spectrace > Setup Annotation...`
3. The plugin will:
   - Create all 26 annotation layers in the correct hierarchy automatically
   - Set the Pencil tool with correct settings (1px, hardness 100)
   - Open the Spectrace control panel
4. Use the control panel to:
   - **Switch layers** — click radio buttons (one at a time)
   - **Switch tools** — Pencil or Eraser buttons
   - **Reset tools** — if settings were accidentally changed
   - **Save** — saves the XCF file
5. Draw on the spectrogram. Use `Ctrl+Z` to undo.
6. When done, close the panel window and save (`Ctrl+S` or `File > Save As`).

## Known Limitations

1. **Tool options panel may show stale values.** The plugin sets tool parameters
   via GIMP's API, but the tool options panel display may not update visually.
   The tool *behaves* correctly. Click "Reset Tools" if unsure.

2. **GIMP menus are still accessible.** The plugin constrains the workflow but
   doesn't remove GIMP's native menus. Installing the stripped `menurc` removes
   most keyboard shortcuts to reduce accidental actions.

3. **GIMP 2.10 only.** This plugin uses Python 2 + PyGTK which is specific to
   GIMP 2.10. It will not work on GIMP 3.0+.

4. **Brush names vary by installation.** The plugin tries several common brush
   names. If no hard brush is found, it uses the currently selected brush and
   prints a warning.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Plugin not in Filters menu | Verify the `.py` file is in the plug-ins directory and is executable (`chmod +x`) |
| `No module named gtk` | Your GIMP installation may be missing Python support. Use the official GIMP 2.10 build. |
| Tool options look wrong | Click "Reset Tools" in the Spectrace panel |
| Layers already exist warning | The plugin detected an existing `OrcinusOrca_FrequencyContours` group and will use it |
| Save button does nothing | Use `File > Save As` first to set a filename, then use the panel's Save button |

## Uninstalling

1. Delete `spectrace_annotator.py` from the plug-ins directory
2. Restore your original `menurc` from backup:
   ```bash
   cp ~/.config/GIMP/2.10/menurc.backup ~/.config/GIMP/2.10/menurc
   ```
3. Restart GIMP
