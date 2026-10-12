#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
app_dir=$(cd -- "$script_dir/.." && pwd)
venv=${HISTOLOOP_VENV:-"$app_dir/.venv-portable"}
python=${HISTOLOOP_SETUP_PYTHON:-python3}
command -v "$python" >/dev/null || { echo "Python not found: $python" >&2; exit 2; }
"$python" -c 'import sys, platform; assert sys.version_info[:2] == (3,10) and sys.platform == "linux" and platform.machine() == "x86_64", "The locked runtime requires Python 3.10 on Linux x86_64"'
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip==25.2
"$venv/bin/python" -m pip install -r "$app_dir/requirements.lock"
echo "Installed CPU review runtime in $venv. No frontend build is needed."
