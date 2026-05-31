#!/usr/bin/env bash
#
# install-gimp-linux.sh
# Installs GIMP 2.10.38 (Python 2 / Python-Fu capable) as a desktop app on
# Debian/Ubuntu so it appears in the application menu. Required before
# installing the Spectrace plugin on Linux.
#
# Usage:
#   ./install-gimp-linux.sh                 # downloads the AppImage
#   ./install-gimp-linux.sh /path/to.AppImage   # uses a local copy instead
#
set -euo pipefail

# ---- configuration (point APPIMAGE_URL at your own release mirror) --------
APPIMAGE_URL="https://github.com/TasMania17/Gimp-Appimages-Made-From-Debs/releases/download/Gimp-v2.10.38/gimp-2-10-38-overlay-py2-mm-v3.AppImage"
INSTALL_DIR="$HOME/Applications/gimp-2.10.38"
APPIMAGE_NAME="GIMP-2.10.38.AppImage"
DESKTOP_NAME="GIMP 2.10.38"
DESKTOP_FILE="$HOME/.local/share/applications/gimp-2.10.38.desktop"
# ---------------------------------------------------------------------------

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$1"; }

# 1. libfuse2 sanity check (needed on Debian 13 / Ubuntu 24.04+)
if ! ldconfig -p | grep -q 'libfuse.so.2'; then
    warn "libfuse2 not found. If the AppImage refuses to start, install it:"
    warn "    sudo apt install libfuse2t64      # or: sudo apt install libfuse2"
fi

# 2. Obtain the AppImage (local argument wins over download)
mkdir -p "$INSTALL_DIR"
DEST="$INSTALL_DIR/$APPIMAGE_NAME"
if [ -n "${1:-}" ] && [ -f "${1:-}" ]; then
    say "Using local AppImage: $1"
    cp "$1" "$DEST"
elif [ -f "$DEST" ]; then
    say "AppImage already installed at $DEST (skipping download)"
else
    say "Downloading GIMP 2.10.38 AppImage..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DEST" "$APPIMAGE_URL"
    else
        curl -L -o "$DEST" "$APPIMAGE_URL"
    fi
fi
chmod +x "$DEST"

# 3. Launcher: strips conda/venv off PATH so GIMP uses its bundled Python 2
#    (Python-Fu plugins fail under a Python 3 from an active conda env).
LAUNCHER="$INSTALL_DIR/gimp-launcher.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
HERE="\$(dirname "\$(readlink -f "\$0")")"
CLEAN_PATH=\$(printf '%s' "\$PATH" | tr ':' '\n' | grep -vE 'conda|anaconda|/envs/' | paste -sd:)
exec env PATH="/usr/bin:/bin:\$CLEAN_PATH" "\$HERE/$APPIMAGE_NAME" "\$@"
EOF
chmod +x "$LAUNCHER"

# 4. Icon: reuse GIMP's own if present anywhere, else fall back to theme name
ICON_LINE="gimp"
ICON_SRC="$(find /usr -name 'gimp.png' 2>/dev/null | sort | tail -1 || true)"
if [ -n "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$INSTALL_DIR/gimp.png"
    ICON_LINE="$INSTALL_DIR/gimp.png"
fi

# 5. Desktop entry
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$DESKTOP_NAME
GenericName=Image Editor
Comment=GIMP 2.10.38 with Python-Fu (used for Spectrace annotation)
Exec=$LAUNCHER %F
Icon=$ICON_LINE
Terminal=false
Categories=Graphics;2DGraphics;RasterGraphics;Science;
MimeType=image/png;audio/x-wav;
StartupNotify=true
StartupWMClass=gimp
EOF

# 6. Refresh the application menu
update-desktop-database "$(dirname "$DESKTOP_FILE")" 2>/dev/null || true

say "Done."
echo "  AppImage : $DEST"
echo "  Launcher : $LAUNCHER"
echo "  Search your applications menu for: $DESKTOP_NAME"
