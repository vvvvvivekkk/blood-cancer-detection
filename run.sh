#!/usr/bin/env bash
# Blood Cancer Detection - start the app (Mac / Linux). Run ./setup.sh first.
cd "$(dirname "$0")"
PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -z "$PY" ] && { echo "[X] Python not found. Run ./setup.sh first."; exit 1; }
echo "Starting the app at http://localhost:8000  (log in: admin / admin)"
echo "Press Ctrl+C to stop it."
"$PY" app.py
