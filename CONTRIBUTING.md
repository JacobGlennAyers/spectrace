# Contributing to Spectrace

## Development Environment Setup

1. Clone the repository and create the conda environment:

```bash
git clone <repo-url> && cd spectrace
conda env create -f environment.yml
conda activate spectrace
```

2. Install GIMP **2.10** (required — GIMP 3.x is not supported) from [gimp.org](https://www.gimp.org/downloads/). Launch GIMP once so it creates its config directory, then close it.

3. Install the plugin for development:

```bash
python gimp_plugin/install.py
```

This auto-detects your platform, copies the plugin into GIMP's plug-ins directory, installs a locked-down UI config, and writes a spectrace config pointing to your conda Python (`~/.spectrace/config.json`). On Linux it auto-detects standard and Flatpak installations.

To uninstall: `python gimp_plugin/install.py --uninstall`

After installation, restart GIMP completely.

## GIMP Version Constraint

Spectrace targets **GIMP 2.10** exclusively. The plugin uses Script-Fu and Python-Fu APIs specific to 2.10. Do not use GIMP 3.x APIs (`Gimp.` namespace, GObject introspection) — they are incompatible.

## Running Tests

There is no automated test suite yet. To verify changes:

1. Run the install script for your platform to deploy your updated plugin.
2. Open GIMP, open a `.wav` file (File > Open) to confirm spectrogram rendering.
3. Run **Filters > Spectrace > Setup Annotation** and verify layers, colors, and tool enforcement work correctly.
4. Switch between pencil and eraser — settings (size, opacity, dynamics, force) should auto-enforce.

## Pull Request Guidelines

- Branch from `main` and keep PRs focused on a single change.
- Include a clear description of what changed and why.
- If your change touches the GIMP plugin, describe how you tested it manually (GIMP version, what you verified).
- Do not commit IDE config, `.pyc` files, or project output directories.

## Code Style

- Python code follows PEP 8 with a relaxed line-length limit (120 characters).
- Use descriptive variable names. Prefix internal helpers with `_`.
- GIMP plugin code must remain compatible with GIMP 2.10's bundled Python 2.7 interpreter — avoid Python 3-only syntax in `spectrace_annotator.py`.
- Pipeline / analysis code (outside `gimp_plugin/`) uses Python 3.11+.
