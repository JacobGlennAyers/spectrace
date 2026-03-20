# Spectrace GIMP Plugin -- Installation Guide

This guide covers installing the Spectrace plugin into GIMP 2.10. For the full project setup (cloning the repo, creating the conda environment), see the main [README](../README.md).

## Prerequisites

- **GIMP 2.10.x** (NOT 3.0 or later)
  - Verify your version: `Help > About` in GIMP should show `2.10.xx`
  - **You must launch GIMP at least once** before running the installer — this creates the configuration directory

- **Spectrace conda environment** (required for WAV file loading)
  - Create it with: `conda env create -f environment.yml && conda activate spectrace`

## Automated Install (Recommended)

Run the cross-platform installer from the repository root:

```bash
python gimp_plugin/install.py              # install
python gimp_plugin/install.py --uninstall  # uninstall
```

The script auto-detects your OS and GIMP config directory. If GIMP is in a non-standard location, set the `GIMP_DIR` environment variable.

The installer performs six steps:

1. **Copies the plugin** (`spectrace_annotator.py`) into GIMP's plug-ins directory
2. **Installs `gimprc`** — hides the menubar and rulers for a cleaner annotation workspace
3. **Installs `menurc`** — strips keyboard shortcuts to prevent accidental operations (only undo, redo, save, and zoom remain)
4. **Installs `toolrc`** — limits the toolbox to Pencil and Eraser only
5. **Installs `sessionrc`** — sets a minimal dock layout (Tool Options + Layers only, no brushes/patterns/fonts panels)
6. **Creates `~/.spectrace/config.json`** — stores the path to your spectrace conda environment and project root

All original GIMP config files are backed up as `*.original` on first install and restored when you uninstall.

> **Note:** The Linux installer auto-detects standard (`~/.config/GIMP/2.10`) and Flatpak (`~/.var/app/org.gimp.GIMP/config/GIMP/2.10`) installations.

<!-- (PICTURE RECOMMENDED: Terminal screenshot showing successful installer output) -->

## Manual Install (Alternative)

If you prefer not to use the install script:

1. **Copy `spectrace_annotator.py`** into the GIMP plug-ins directory:

   | Platform | Plug-ins Directory |
   |----------|-------------------|
   | **macOS** | `~/Library/Application Support/GIMP/2.10/plug-ins/` |
   | **Linux** | `~/.config/GIMP/2.10/plug-ins/` |
   | **Linux (Flatpak)** | `~/.var/app/org.gimp.GIMP/config/GIMP/2.10/plug-ins/` |
   | **Windows** | `%APPDATA%\GIMP\2.10\plug-ins\` |

   You can also check inside GIMP: `Edit > Preferences > Folders > Plug-ins`

2. **Linux/macOS only** — make it executable:
   ```bash
   chmod +x <plug-ins-directory>/spectrace_annotator.py
   ```

3. **(Optional)** Copy config files from `gimp_plugin/config/` to your GIMP config directory:
   - `gimprc` — hides menubar, rulers, toolbox extras
   - `menurc` — strips keyboard shortcuts
   - `toolrc` — pencil + eraser only
   - `sessionrc` — minimal dock layout

4. **Create the spectrace config** at `~/.spectrace/config.json`:
   ```json
   {
     "spectrace_root": "/path/to/your/spectrace",
     "python3_path": "/path/to/conda/envs/spectrace/bin/python",
     "default_nfft": 2048,
     "default_grayscale": true,
     "default_project_dir": "projects"
   }
   ```
   To find your Python path: `conda activate spectrace && which python` (macOS/Linux) or `where python` (Windows).

## Restart GIMP

Close and reopen GIMP. The plugin appears under:

- **`Filters > Spectrace > Setup Annotation...`** — creates annotation layers and starts the background monitor
- **`Filters > Spectrace > Reset Tool Settings`** — re-applies correct pencil/eraser settings

Since the installer strips GIMP's menubar, access menus by **right-clicking the canvas**.

<!-- (PICTURE RECOMMENDED: Screenshot of GIMP after restart showing the stripped-down interface, and right-click menu with Filters > Spectrace visible) -->

## Usage

1. **Open a WAV file** — `File > Open`, select a `.wav` file. The plugin automatically generates a spectrogram and loads it.
2. **Set up annotation** — right-click canvas > `Filters > Spectrace > Setup Annotation...`
   - Leave the template field blank for the default orca template, or browse to a custom `.xcf` template
3. The plugin creates all annotation layers, configures tools, and starts the background monitor
4. **Select a layer** in the Layers panel and draw. The foreground color switches automatically per layer.
5. **Switch tools** between Pencil (locked to 1px) and Eraser (adjustable size)
6. **Save** with `Ctrl+S` or `File > Save As`

<!-- (PICTURE RECOMMENDED: Annotated screenshot showing the complete annotation workflow — layers panel with one selected, spectrogram with drawn contours, and the tool options) -->

## Known Limitations

1. **Tool options panel may show stale values.** The plugin sets tool parameters via GIMP's API, but the panel display may not update visually. The tool *behaves* correctly. Click `Reset Tool Settings` if unsure.

2. **GIMP menus are still accessible.** The plugin constrains the workflow but doesn't remove GIMP's native menus entirely. The stripped `menurc` removes most keyboard shortcuts to reduce accidental actions.

3. **GIMP 2.10 only.** This plugin uses Python 2 + PyGTK (bundled with GIMP 2.10). It will not work on GIMP 3.0+.

4. **Brush names vary by installation.** The plugin tries several common brush names (`1. Pixel`, `2. Hardness 100`, etc.). If no hard brush is found, it uses the currently selected brush and prints a warning.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Plugin not in Filters menu | Verify the `.py` file is in the plug-ins directory, is executable (`chmod +x`), and you restarted GIMP |
| `No module named gtk` | Your GIMP build may be missing Python support. Use the official GIMP 2.10 build. |
| WAV file won't open | Check `~/.spectrace/config.json` — `python3_path` must point to the spectrace conda Python. Test: `conda activate spectrace && python spectrace_wav_bridge.py --wav <file> --output-dir ./projects --nfft 2048 --grayscale` |
| Tool options look wrong | Click `Filters > Spectrace > Reset Tool Settings` |
| Layers already exist warning | The plugin detected an existing annotation group and will reuse it |
| Save button does nothing | Use `File > Save As` first to set a filename, then `Ctrl+S` for subsequent saves |
| Debug information | Check `/tmp/spectrace_debug.log` for detailed plugin logs |

## Uninstalling

Run the uninstaller: `python gimp_plugin/install.py --uninstall`. This:

1. Removes `spectrace_annotator.py` from the plug-ins directory
2. Restores all original config files (`gimprc`, `menurc`, `toolrc`, `sessionrc`) from `*.original` backups
3. Restart GIMP to complete the uninstall

To also remove the spectrace configuration:
```bash
rm -rf ~/.spectrace  # macOS/Linux
```
