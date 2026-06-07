#!/bin/bash
# Spectrace GIMP Plugin Uninstaller (macOS)
# Removes plugin and restores all original configs. No admin/sudo required.

GIMP_DIR="$HOME/Library/Application Support/GIMP/2.10"

echo "=== Spectrace GIMP Plugin Uninstaller ==="
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
