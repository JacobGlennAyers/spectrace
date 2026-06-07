#!/bin/bash
# Spectrace GIMP Plugin Installer (macOS)
# Installs the plugin, configures paths, and locks down the GIMP UI for annotation.
# No admin/sudo required.

set -e

GIMP_DIR="$HOME/Library/Application Support/GIMP/2.10"

if [ ! -d "$GIMP_DIR" ]; then
    echo "ERROR: GIMP 2.10 config directory not found at:"
    echo "  $GIMP_DIR"
    echo "Make sure GIMP 2.10 has been launched at least once."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPECTRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Spectrace GIMP Plugin Installer ==="
echo ""
echo "Spectrace root: $SPECTRACE_ROOT"
echo ""

# Back up ALL config files we'll replace (only on first install)
for f in gimprc menurc toolrc sessionrc; do
    if [ -f "$GIMP_DIR/$f" ] && [ ! -f "$GIMP_DIR/$f.original" ]; then
        cp "$GIMP_DIR/$f" "$GIMP_DIR/$f.original"
        echo "  Backed up $f -> $f.original"
    fi
done
echo ""

# 1. Install plugin
echo "[1/6] Installing plugin..."
mkdir -p "$GIMP_DIR/plug-ins"
cp "$SCRIPT_DIR/spectrace_annotator.py" "$GIMP_DIR/plug-ins/"
chmod +x "$GIMP_DIR/plug-ins/spectrace_annotator.py"
echo "  -> Done"

# 2. Install gimprc (hides menubar, rulers, toolbox extras)
echo "[2/6] Installing gimprc (hides menubar, rulers)..."
if [ -f "$SCRIPT_DIR/config/gimprc" ]; then
    cp "$SCRIPT_DIR/config/gimprc" "$GIMP_DIR/"
    echo "  -> Done"
else
    echo "  -> Skipped (no gimprc found)"
fi

# 3. Install stripped shortcuts
echo "[3/6] Installing menurc (strips keyboard shortcuts)..."
if [ -f "$SCRIPT_DIR/config/menurc" ]; then
    cp "$SCRIPT_DIR/config/menurc" "$GIMP_DIR/"
    echo "  -> Done"
else
    echo "  -> Skipped (no menurc found)"
fi

# 4. Install stripped toolbox (pencil + eraser only)
echo "[4/6] Installing toolrc (pencil + eraser only)..."
if [ -f "$SCRIPT_DIR/config/toolrc" ]; then
    cp "$SCRIPT_DIR/config/toolrc" "$GIMP_DIR/"
    echo "  -> Done"
else
    echo "  -> Skipped (no toolrc found)"
fi

# 5. Install minimal session layout (no brushes/patterns/fonts docks)
echo "[5/6] Installing sessionrc (minimal dock layout)..."
if [ -f "$SCRIPT_DIR/config/sessionrc" ]; then
    cp "$SCRIPT_DIR/config/sessionrc" "$GIMP_DIR/"
    echo "  -> Done"
else
    echo "  -> Skipped (no sessionrc found)"
fi

# 6. Create spectrace configuration
echo "[6/6] Creating spectrace configuration..."
SPECTRACE_CONFIG_DIR="$HOME/.spectrace"
SPECTRACE_CONFIG="$SPECTRACE_CONFIG_DIR/config.json"
mkdir -p "$SPECTRACE_CONFIG_DIR"

# Auto-detect conda Python 3
PYTHON3_PATH=""
for CONDA_BASE in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                   "/opt/miniconda3" "/opt/anaconda3" "/opt/miniforge3" "/usr/local/miniconda3"; do
    CANDIDATE="$CONDA_BASE/envs/spectrace/bin/python"
    if [ -x "$CANDIDATE" ]; then
        PYTHON3_PATH="$CANDIDATE"
        break
    fi
done

if [ -z "$PYTHON3_PATH" ]; then
    # Try conda run
    if command -v conda &> /dev/null; then
        PYTHON3_PATH=$(conda run -n spectrace which python 2>/dev/null || echo "")
    fi
fi

if [ -z "$PYTHON3_PATH" ]; then
    echo "  WARNING: Could not find spectrace conda environment."
    echo "  WAV file opening will not work until you set python3_path in:"
    echo "    $SPECTRACE_CONFIG"
    echo ""
    echo "  To fix: conda activate spectrace && which python"
    echo "  Then edit the config file with that path."
    PYTHON3_PATH="python3"
fi

# Write config (preserve existing values if config already exists)
if [ -f "$SPECTRACE_CONFIG" ]; then
    echo "  Config already exists, updating spectrace_root and python3_path..."
    # Use Python to merge configs (available on macOS)
    python3 -c "
import json, sys
try:
    with open('$SPECTRACE_CONFIG', 'r') as f:
        config = json.load(f)
except:
    config = {}
config['spectrace_root'] = '$SPECTRACE_ROOT'
config['python3_path'] = '$PYTHON3_PATH'
config.setdefault('default_nfft', 2048)
config.setdefault('default_grayscale', True)
config.setdefault('default_project_dir', 'projects')
with open('$SPECTRACE_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
" 2>/dev/null || {
    # Fallback: write fresh config
    cat > "$SPECTRACE_CONFIG" << EOFCONFIG
{
  "spectrace_root": "$SPECTRACE_ROOT",
  "python3_path": "$PYTHON3_PATH",
  "default_nfft": 2048,
  "default_grayscale": true,
  "default_project_dir": "projects"
}
EOFCONFIG
}
else
    cat > "$SPECTRACE_CONFIG" << EOFCONFIG
{
  "spectrace_root": "$SPECTRACE_ROOT",
  "python3_path": "$PYTHON3_PATH",
  "default_nfft": 2048,
  "default_grayscale": true,
  "default_project_dir": "projects"
}
EOFCONFIG
fi

echo "  -> Config written to: $SPECTRACE_CONFIG"
echo "     spectrace_root: $SPECTRACE_ROOT"
echo "     python3_path:   $PYTHON3_PATH"

echo ""
echo "=== Installed! Close GIMP completely and reopen. ==="
echo ""
echo "After restart you will see:"
echo "  - No menubar (right-click canvas for menus)"
echo "  - Only Pencil and Eraser in the toolbox"
echo "  - Only Tool Options (left) and Layers (right)"
echo "  - No brushes, patterns, fonts, or channels docks"
echo ""
echo "New features:"
echo "  - File > Open > select a .wav file -> opens as spectrogram"
echo "  - Filters > Spectrace > Setup Annotation -> pick a template .xcf"
echo ""
echo "Usage: right-click canvas > Filters > Spectrace > Setup Annotation..."
echo ""
echo "To restore original GIMP: bash $SCRIPT_DIR/uninstall_mac.sh"
