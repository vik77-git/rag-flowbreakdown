#!/usr/bin/env bash
set -euo pipefail
[ -f .env ] || cp .env.example .env
python -m pip install --no-cache-dir -r requirements.txt
exec uvicorn app.main:app --reload --app-dir backend --port 8000
