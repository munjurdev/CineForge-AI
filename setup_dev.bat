@echo off
echo ==> Creating virtual environment
python -m venv venv
echo ==> Activating virtual environment
call venv\Scripts\activate.bat
echo ==> Installing dependencies
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo ==> Copying .env.example to .env
copy /Y .env.example .env >nul
echo ==> Running system checks
python manage.py check
echo ==> Applying migrations
python manage.py migrate
echo ==> Collecting static files
python manage.py collectstatic --noinput
echo ==> Done. Start the server with: python manage.py runserver
