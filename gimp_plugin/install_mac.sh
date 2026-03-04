#!/bin/bash
# Spectrace GIMP Plugin Installer (macOS)
# Installs the plugin and locks down the GIMP UI for annotation only.
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

echo "=== Spectrace GIMP Plugin Installer ==="
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
echo "[1/5] Installing plugin..."
mkdir -p "$GIMP_DIR/plug-ins"
cp "$SCRIPT_DIR/spectrace_annotator.py" "$GIMP_DIR/plug-ins/"
chmod +x "$GIMP_DIR/plug-ins/spectrace_annotator.py"
echo "  -> Done"

# 2. Install gimprc (hides menubar, rulers, toolbox extras)
echo "[2/5] Installing gimprc (hides menubar, rulers)..."
cp "$SCRIPT_DIR/config/gimprc" "$GIMP_DIR/"
echo "  -> Done"

# 3. Install stripped shortcuts
echo "[3/5] Installing menurc (strips keyboard shortcuts)..."
cp "$SCRIPT_DIR/config/menurc" "$GIMP_DIR/"
echo "  -> Done"

# 4. Install stripped toolbox (pencil + eraser only)
echo "[4/5] Installing toolrc (pencil + eraser only)..."
cp "$SCRIPT_DIR/config/toolrc" "$GIMP_DIR/"
echo "  -> Done"

# 5. Install minimal session layout (no brushes/patterns/fonts docks)
echo "[5/5] Installing sessionrc (minimal dock layout)..."
cp "$SCRIPT_DIR/config/sessionrc" "$GIMP_DIR/"
echo "  -> Done"

echo ""
echo "=== Installed! Close GIMP completely and reopen. ==="
echo ""
echo "After restart you will see:"
echo "  - No menubar (right-click canvas for menus)"
echo "  - Only Pencil and Eraser in the toolbox"
echo "  - Only Tool Options (left) and Layers (right)"
echo "  - No brushes, patterns, fonts, or channels docks"
echo ""
echo "Usage: right-click canvas > Filters > Spectrace > Setup Annotation..."
echo ""
echo "To restore original GIMP: bash $SCRIPT_DIR/uninstall_mac.sh"
