#!/bin/bash
# Spectrace GIMP Plugin Uninstaller (Linux)
# Removes plugin and restores all original configs. No sudo required.

# Detect GIMP config directory (standard or Flatpak)
GIMP_DIR="$HOME/.config/GIMP/2.10"
FLATPAK_DIR="$HOME/.var/app/org.gimp.GIMP/config/GIMP/2.10"

if [ -d "$FLATPAK_DIR" ] && [ ! -d "$GIMP_DIR" ]; then
    GIMP_DIR="$FLATPAK_DIR"
    echo "Detected Flatpak GIMP installation."
fi

echo "=== Spectrace GIMP Plugin Uninstaller (Linux) ==="
echo ""

# Remove plugin
if [ -f "$GIMP_DIR/plug-ins/spectrace_annotator.py" ]; then
    rm "$GIMP_DIR/plug-ins/spectrace_annotator.py"
    echo "[1/5] Removed plugin"
else
    echo "[1/5] Plugin not found (skipped)"
fi

# Restore each config file from .original backup
n=2
for f in gimprc menurc toolrc sessionrc; do
    if [ -f "$GIMP_DIR/$f.original" ]; then
        mv "$GIMP_DIR/$f.original" "$GIMP_DIR/$f"
        echo "[$n/5] Restored original $f"
    elif [ -f "$GIMP_DIR/$f" ]; then
        rm "$GIMP_DIR/$f"
        echo "[$n/5] Removed spectrace $f (GIMP will use defaults)"
    else
        echo "[$n/5] No $f found (skipped)"
    fi
    n=$((n + 1))
done

echo ""
echo "=== Done! Restart GIMP to complete uninstallation. ==="
