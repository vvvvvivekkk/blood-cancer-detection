#!/usr/bin/env bash
# ============================================================
#  Blood Cancer Detection - first-time setup (Mac / Linux)
#  Installs dependencies, unpacks the trained models, starts
#  the app. Run this ONCE. After that, use ./run.sh.
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo "==============================================="
echo "  Blood Cancer Detection - Setup"
echo "==============================================="
echo

# --- find a Python 3 ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[X] Python was not found."
  echo
  echo "    Install Python 3.12 from https://www.python.org/downloads/"
  echo "    then run this setup again."
  echo
  exit 1
fi

echo "[1/4] Installing dependencies (this can take 5-10 minutes)..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install -r requirements.txt

echo
echo "[2/4] Unpacking the trained models..."
"$PY" run.py --fetch-models

echo
echo "[3/4] Verifying the models..."
"$PY" run.py --check

echo
echo "[4/4] Starting the app at http://localhost:8000"
echo "      Log in with  admin / admin"
echo "      (Press Ctrl+C to stop the app.)"
echo
"$PY" app.py
