#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating virtual environment"
python -m venv venv

echo "==> Activating virtual environment"
source venv/bin/activate

echo "==> Installing dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "==> Copying .env.example to .env"
cp -n .env.example .env || true

echo "==> Running system checks"
python manage.py check

echo "==> Applying migrations"
python manage.py migrate

echo "==> Collecting static files"
python manage.py collectstatic --noinput

echo "==> Done. Start the server with: python manage.py runserver"
