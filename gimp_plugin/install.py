#!/usr/bin/env python3
"""
Spectrace GIMP Plugin Installer / Uninstaller

Cross-platform script that replaces the per-platform shell scripts.
Auto-detects OS, finds the GIMP 2.10 config directory, and installs
(or uninstalls) the plugin + locked-down UI configs.

Usage:
    python gimp_plugin/install.py              # install
    python gimp_plugin/install.py --uninstall  # uninstall
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRACE_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG_FILES = ["gimprc", "menurc", "toolrc", "sessionrc"]


# ── GIMP directory detection ────────────────────────────────────────

def find_gimp_dir():
    """Return the GIMP 2.10 user config directory for this platform."""
    system = platform.system()

    if system == "Darwin":
        path = os.path.expanduser("~/Library/Application Support/GIMP/2.10")
        if os.path.isdir(path):
            return path
        return None

    if system == "Linux":
        standard = os.path.expanduser("~/.config/GIMP/2.10")
        flatpak = os.path.expanduser(
            "~/.var/app/org.gimp.GIMP/config/GIMP/2.10"
        )
        if os.path.isdir(flatpak) and not os.path.isdir(standard):
            print("Detected Flatpak GIMP installation.")
            return flatpak
        if os.path.isdir(flatpak) and os.path.isdir(standard):
            print("Found both standard and Flatpak GIMP configs.")
            print("  Standard: %s" % standard)
            print("  Flatpak:  %s" % flatpak)
            print("Using standard. Set GIMP_DIR env var to override.")
        if os.path.isdir(standard):
            return standard
        return None

    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            path = os.path.join(appdata, "GIMP", "2.10")
            if os.path.isdir(path):
                return path
        return None

    return None


# ── Conda environment detection ─────────────────────────────────────

def find_conda_python():
    """Find the Python interpreter in the spectrace conda environment."""
    home = os.path.expanduser("~")
    is_windows = platform.system() == "Windows"
    exe = "python.exe" if is_windows else "python"
    subdir = "Scripts" if is_windows else "bin"

    # Check common conda base directories
    if is_windows:
        bases = [
            os.path.join(home, "miniconda3"),
            os.path.join(home, "anaconda3"),
            os.path.join(home, "miniforge3"),
            os.path.join(home, "mambaforge"),
            "C:\\miniconda3",
            "C:\\anaconda3",
            "C:\\ProgramData\\miniconda3",
            "C:\\ProgramData\\anaconda3",
        ]
    else:
        bases = [
            os.path.join(home, "miniconda3"),
            os.path.join(home, "anaconda3"),
            os.path.join(home, "miniforge3"),
            os.path.join(home, "mambaforge"),
            "/opt/miniconda3",
            "/opt/anaconda3",
            "/opt/miniforge3",
            "/usr/local/miniconda3",
        ]

    for base in bases:
        candidate = os.path.join(base, "envs", "spectrace", subdir, exe)
        if os.path.isfile(candidate):
            return candidate

    # Try conda run
    conda_cmd = "conda"
    try:
        which_cmd = "where" if is_windows else "which"
        result = subprocess.run(
            [conda_cmd, "run", "-n", "spectrace", which_cmd, "python"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            path = result.stdout.strip().splitlines()[0].strip()
            if os.path.isfile(path):
                return path
    except Exception:
        pass

    return ""


# ── Install ─────────────────────────────────────────────────────────

def install(gimp_dir):
    print("=== Spectrace GIMP Plugin Installer ===")
    print()
    print("GIMP config:    %s" % gimp_dir)
    print("Spectrace root: %s" % SPECTRACE_ROOT)
    print()

    # Back up existing configs (only on first install)
    for f in CONFIG_FILES:
        src = os.path.join(gimp_dir, f)
        backup = src + ".original"
        if os.path.isfile(src) and not os.path.isfile(backup):
            shutil.copy2(src, backup)
            print("  Backed up %s -> %s.original" % (f, f))
    print()

    total_steps = 6

    # 1. Install plugin
    print("[1/%d] Installing plugin..." % total_steps)
    plugins_dir = os.path.join(gimp_dir, "plug-ins")
    os.makedirs(plugins_dir, exist_ok=True)
    plugin_src = os.path.join(SCRIPT_DIR, "spectrace_annotator.py")
    plugin_dst = os.path.join(plugins_dir, "spectrace_annotator.py")
    shutil.copy2(plugin_src, plugin_dst)
    if platform.system() != "Windows":
        os.chmod(plugin_dst, 0o755)
    print("  -> Done")

    # 2-5. Install config files
    config_names = {
        "gimprc": ("gimprc (hides menubar, rulers)", 2),
        "menurc": ("menurc (strips keyboard shortcuts)", 3),
        "toolrc": ("toolrc (pencil + eraser only)", 4),
        "sessionrc": ("sessionrc (minimal dock layout)", 5),
    }
    config_dir = os.path.join(SCRIPT_DIR, "config")
    for fname, (desc, step) in config_names.items():
        print("[%d/%d] Installing %s..." % (step, total_steps, desc))
        src = os.path.join(config_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(gimp_dir, fname))
            print("  -> Done")
        else:
            print("  -> Skipped (not found)")

    # 6. Create spectrace configuration
    print("[6/%d] Creating spectrace configuration..." % total_steps)
    python3_path = find_conda_python()

    if not python3_path:
        print("  WARNING: Could not find spectrace conda environment.")
        print("  WAV file opening will not work until you set python3_path in")
        print("  the config file (see below).")
        print()
        print("  To fix: conda activate spectrace")
        if platform.system() == "Windows":
            print("          where python")
        else:
            print("          which python")
        print("  Then edit the config file with that path.")
        python3_path = "python3"

    config_dir_path = os.path.join(os.path.expanduser("~"), ".spectrace")
    config_file = os.path.join(config_dir_path, "config.json")
    os.makedirs(config_dir_path, exist_ok=True)

    # Merge with existing config if present
    config = {}
    if os.path.isfile(config_file):
        print("  Config already exists, updating spectrace_root and python3_path...")
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except Exception:
            config = {}

    config["spectrace_root"] = SPECTRACE_ROOT
    config["python3_path"] = python3_path
    config.setdefault("default_nfft", 2048)
    config.setdefault("default_grayscale", True)
    config.setdefault("default_project_dir", "projects")

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print("  -> Config written to: %s" % config_file)
    print("     spectrace_root: %s" % SPECTRACE_ROOT)
    print("     python3_path:   %s" % python3_path)

    print()
    print("=== Installed! Close GIMP completely and reopen. ===")
    print()
    print("After restart you will see:")
    print("  - No menubar (right-click canvas for menus)")
    print("  - Only Pencil and Eraser in the toolbox")
    print("  - Only Tool Options (left) and Layers (right)")
    print("  - No brushes, patterns, fonts, or channels docks")
    print()
    print("New features:")
    print("  - File > Open > select a .wav file -> opens as spectrogram")
    print("  - Filters > Spectrace > Setup Annotation -> pick a template .xcf")
    print()
    print("Usage: right-click canvas > Filters > Spectrace > Setup Annotation...")
    print()
    print("To uninstall: python %s --uninstall" % os.path.abspath(__file__))


# ── Uninstall ───────────────────────────────────────────────────────

def uninstall(gimp_dir):
    print("=== Spectrace GIMP Plugin Uninstaller ===")
    print()

    # Remove plugin
    plugin = os.path.join(gimp_dir, "plug-ins", "spectrace_annotator.py")
    if os.path.isfile(plugin):
        os.remove(plugin)
        print("[1/5] Removed plugin")
    else:
        print("[1/5] Plugin not found (skipped)")

    # Restore config files
    step = 2
    for f in CONFIG_FILES:
        src = os.path.join(gimp_dir, f)
        backup = src + ".original"
        if os.path.isfile(backup):
            shutil.move(backup, src)
            print("[%d/5] Restored original %s" % (step, f))
        elif os.path.isfile(src):
            os.remove(src)
            print("[%d/5] Removed spectrace %s (GIMP will use defaults)" % (step, f))
        else:
            print("[%d/5] No %s found (skipped)" % (step, f))
        step += 1

    print()
    print("=== Done! Restart GIMP to complete uninstallation. ===")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spectrace GIMP Plugin Installer"
    )
    parser.add_argument(
        "--uninstall", action="store_true",
        help="Uninstall the plugin and restore original GIMP config",
    )
    args = parser.parse_args()

    # Allow GIMP_DIR env var override
    gimp_dir = os.environ.get("GIMP_DIR") or find_gimp_dir()

    if not gimp_dir:
        print("ERROR: GIMP 2.10 config directory not found.")
        print("Make sure GIMP 2.10 has been launched at least once.")
        print()
        print("If your GIMP config is in a non-standard location, set:")
        print("  GIMP_DIR=/path/to/GIMP/2.10 python %s" % __file__)
        sys.exit(1)

    if args.uninstall:
        uninstall(gimp_dir)
    else:
        install(gimp_dir)


if __name__ == "__main__":
    main()
