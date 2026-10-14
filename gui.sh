#!/usr/bin/env bash
# Spin up the task-board GUI and open it in a browser window; Ctrl-C / kill closes everything.
#   ./gui.sh [HTTP_PORT] [TCP_PORT]
set -euo pipefail
cd "$(dirname "$0")"

HTTP=${1:-8000}
TCP=${2:-5555}
URL="http://localhost:$HTTP"
PY=.venv/bin/python
PROFILE="taskboard-gui-$$"   # dedicated browser profile: lets us close exactly our window
SERVER=""; BROWSER=""; WIN_PROFILE=""

cleanup() {
  trap - EXIT INT TERM
  echo; echo "Shutting down…"
  [[ -n $SERVER ]] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null || true
  [[ -n $BROWSER ]] && kill "$BROWSER" 2>/dev/null || true
  if [[ -n $WIN_PROFILE ]]; then   # Windows browser launched from WSL
    powershell.exe -NoProfile -Command \
      "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' or Name='msedge.exe'\" |
       Where-Object { \$_.CommandLine -like '*$PROFILE*' } |
       ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" \
      >/dev/null 2>&1 || true
  fi
  [[ -d /tmp/$PROFILE ]] && rm -rf "/tmp/$PROFILE"
}
trap cleanup EXIT INT TERM

[[ -x $PY ]] || { echo "Missing $PY - create the venv first (see README)"; exit 1; }
# WSL: render on the GPU through Mesa's D3D12 driver (otherwise EGL falls back to CPU llvmpipe).
# Prefer an NVIDIA adapter: the Intel iGPU D3D12 path corrupts moving meshes.
if [[ -e /dev/dxg ]]; then
  export GALLIUM_DRIVER=${GALLIUM_DRIVER:-d3d12}
  nvidia-smi -L >/dev/null 2>&1 && export MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}
fi
MUJOCO_GL=${MUJOCO_GL:-egl} "$PY" -m gym_taskboard.gui --http "$HTTP" --port "$TCP" &
SERVER=$!

echo -n "Starting simulator"
until curl -fs "$URL" >/dev/null 2>&1; do
  kill -0 "$SERVER" 2>/dev/null || { echo; echo "Server failed to start"; exit 1; }
  echo -n "."; sleep 0.5
done
echo " ready → $URL  (programmatic port $TCP)"

# Open an app-style window: Linux Chromium/Chrome, else Windows Chrome/Edge from WSL, else default browser.
if B=$(command -v google-chrome || command -v chromium || command -v chromium-browser); then
  "$B" --app="$URL" --user-data-dir="/tmp/$PROFILE" --no-first-run >/dev/null 2>&1 &
  BROWSER=$!
elif grep -qi microsoft /proc/version 2>/dev/null; then
  for W in "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
           "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"; do
    if [[ -x $W ]]; then
      WIN_PROFILE=1
      "$W" --app="$URL" --user-data-dir="C:\\Temp\\$PROFILE" --no-first-run >/dev/null 2>&1 &
      break
    fi
  done
  [[ -n $WIN_PROFILE ]] || cmd.exe /c start "$URL" >/dev/null 2>&1
else
  xdg-open "$URL" >/dev/null 2>&1 || echo "Open $URL in your browser"
fi

echo "Press Ctrl-C to stop."
wait "$SERVER"
