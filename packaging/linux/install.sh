#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
IMDER_BIN="$INSTALL_DIR/imder"
DESKTOP_FILE="$HOME/.local/share/applications/imder.desktop"

if [[ ! -f "$IMDER_BIN" ]]; then
    echo "ERROR: imder binary not found at $IMDER_BIN"
    exit 1
fi

if [[ ! -x "$IMDER_BIN" ]]; then
    echo "Making imder executable..."
    chmod +x "$IMDER_BIN"
fi

echo "Creating desktop shortcut..."
mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Name=IMDER
Comment=Interactive image blender that creates smooth animations
Exec=${IMDER_BIN}
Terminal=false
Type=Application
Categories=Graphics;
StartupNotify=true
DESKTOP

chmod 644 "$DESKTOP_FILE"

if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
fi

SHELL_NAME=""
SHELL_RC=""

if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$(basename "${SHELL:-}")" == "zsh" ]]; then
    SHELL_NAME="zsh"
    SHELL_RC="$HOME/.zshrc"
elif [[ -n "${BASH_VERSION:-}" ]] || [[ "$(basename "${SHELL:-}")" == "bash" ]]; then
    SHELL_NAME="bash"
    SHELL_RC="$HOME/.bashrc"
fi

if [[ -z "$SHELL_RC" ]]; then
    if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_NAME="zsh"
        SHELL_RC="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then
        SHELL_NAME="bash"
        SHELL_RC="$HOME/.bashrc"
    else
        SHELL_NAME="bash"
        SHELL_RC="$HOME/.bashrc"
    fi
fi

ALIAS_MARKER="# >>> IMDER alias >>>"
ALIAS_LINE="alias imder='${IMDER_BIN}'"

if [[ -f "$SHELL_RC" ]]; then
    if grep -qF "$ALIAS_MARKER" "$SHELL_RC"; then
        echo "Shell alias already exists in $SHELL_RC (skipping)"
    else
        echo "Adding alias to $SHELL_RC ($SHELL_NAME)..."
        echo "" >> "$SHELL_RC"
        echo "$ALIAS_MARKER" >> "$SHELL_RC"
        echo "$ALIAS_LINE" >> "$SHELL_RC"
        echo "# <<< IMDER alias <<<" >> "$SHELL_RC"
    fi
else
    echo "Creating $SHELL_RC with alias ($SHELL_NAME)..."
    echo "$ALIAS_MARKER" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    echo "# <<< IMDER alias <<<" >> "$SHELL_RC"
fi

OTHER_RC=""
if [[ "$SHELL_RC" == "$HOME/.bashrc" ]] && [[ -f "$HOME/.zshrc" ]]; then
    OTHER_RC="$HOME/.zshrc"
elif [[ "$SHELL_RC" == "$HOME/.zshrc" ]] && [[ -f "$HOME/.bashrc" ]]; then
    OTHER_RC="$HOME/.bashrc"
fi

if [[ -n "$OTHER_RC" ]]; then
    if grep -qF "$ALIAS_MARKER" "$OTHER_RC"; then
        echo "Alias already exists in $OTHER_RC (skipping)"
    else
        echo "Also adding alias to $OTHER_RC..."
        echo "" >> "$OTHER_RC"
        echo "$ALIAS_MARKER" >> "$OTHER_RC"
        echo "$ALIAS_LINE" >> "$OTHER_RC"
        echo "# <<< IMDER alias <<<" >> "$OTHER_RC"
    fi
fi

echo ""
echo "============================================================"
echo " IMDER installed successfully!"
echo "============================================================"
echo ""
echo " Desktop shortcut:  $DESKTOP_FILE"
echo "                     IMDER now appears in your app menu"
echo "                     under Graphics."
echo ""
echo " Shell alias:       $ALIAS_LINE"
echo "                     Open a new terminal or run:"
echo "                       source $SHELL_RC"
echo "                     Then use: imder cli"
echo ""
echo "============================================================"
