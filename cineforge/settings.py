"""
Django settings for CineForge AI project.

CineForge AI — Local AI Video Generation Platform.

See README.md for setup, hardware requirements, and model selection guidance.
"""
from pathlib import Path

from decouple import config  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _path_setting(name: str, default: Path) -> Path:
    """Read a path from the environment, resolving relative paths against BASE_DIR."""
    value = Path(config(name, default=str(default)))
    return value if value.is_absolute() else BASE_DIR / value

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
SECRET_KEY = config(
    "DJANGO_SECRET_KEY",
    default="django-insecure-change-me-in-production-cineforge",
)
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config(
    "ALLOWED_HOSTS",
    default="localhost,127.0.0.1",
    cast=lambda v: [s.strip() for s in v.split(",") if s.strip()],
)

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Project apps
    "core",
    "generator",
    "library",
    "films",
    "system_monitor",
    # Background tasks
    "django_celery_results",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "cineforge.urls"

# Use Django's default test runner.
TEST_RUNNER = "django.test.runner.DiscoverRunner"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "core.context_processors.app_settings",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Database — SQLite for local development and testing (PostgreSQL compatible)
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": config(
            "DB_ENGINE",
            default="django.db.backends.sqlite3",
        ),
        "NAME": str(_path_setting("DB_NAME", BASE_DIR / "db.sqlite3")),
        # PostgreSQL compat options (ignored by SQLite)
        "USER": config("DB_USER", default=""),
        "PASSWORD": config("DB_PASSWORD", default=""),
        "HOST": config("DB_HOST", default=""),
        "PORT": config("DB_PORT", default=""),
        "CONN_MAX_AGE": config("DB_CONN_MAX_AGE", default=600, cast=int),
    }
}

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = config("TIME_ZONE", default="UTC")
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files (CSS, JavaScript, Images)
# ---------------------------------------------------------------------------
STATIC_URL = config("STATIC_URL", default="/static/")
STATIC_ROOT = _path_setting("STATIC_ROOT", BASE_DIR / "staticfiles")
STATICFILES_DIRS = [BASE_DIR / "static"]

# ---------------------------------------------------------------------------
# Media / uploads (videos, images, models)
# ---------------------------------------------------------------------------
MEDIA_URL = config("MEDIA_URL", default="/media/")
MEDIA_ROOT = _path_setting("MEDIA_ROOT", BASE_DIR / "media")

# Subfolders
MEDIA_UPLOADS_DIR = MEDIA_ROOT / "uploads"
MEDIA_TEMP_DIR = MEDIA_ROOT / "temp"
MEDIA_OUTPUTS_DIR = MEDIA_ROOT / "outputs"

# Ensure media subfolders exist at startup (idempotent)
for _d in (MEDIA_UPLOADS_DIR, MEDIA_TEMP_DIR, MEDIA_OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File upload limits
# ---------------------------------------------------------------------------
DATA_UPLOAD_MAX_MEMORY_SIZE = config(
    "DATA_UPLOAD_MAX_MEMORY_SIZE", default=50 * 1024 * 1024, cast=int
)
FILE_UPLOAD_MAX_MEMORY_SIZE = config(
    "FILE_UPLOAD_MAX_MEMORY_SIZE", default=50 * 1024 * 1024, cast=int
)
MAX_UPLOAD_SIZE_MB = config("MAX_UPLOAD_SIZE_MB", default=50, cast=int)
ALLOWED_IMAGE_EXTENSIONS = config(
    "ALLOWED_IMAGE_EXTENSIONS", default=".jpg,.jpeg,.png,.webp"
).split(",")

# ---------------------------------------------------------------------------
# Default primary key field type
# ---------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {name} {message}",
            "style": "{",
        },
        "simple": {
            "format": "[{levelname}] {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": _LOG_DIR / "cineforge.log",
            "formatter": "verbose",
        },
        "celery": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _LOG_DIR / "celery.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "cineforge": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "cineforge.generator": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "cineforge.films": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "celery": {
            "handlers": ["celery"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
# Broker: Redis by default, falling back to memory backend only when Redis
# is not available. The memory broker is NOT suitable for production or
# multiple workers; it exists so the app can start and run tests without Redis.
CELERY_BROKER_URL = config(
    "CELERY_BROKER_URL",
    default="redis://127.0.0.1:6379/0",
)
CELERY_RESULT_BACKEND = config(
    "CELERY_RESULT_BACKEND",
    default="redis://127.0.0.1:6379/1",
)
CELERY_CACHE_BACKEND = config(
    "CELERY_CACHE_BACKEND",
    default="redis://127.0.0.1:6379/2",
)

# Task execution defaults
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_TIME_LIMIT = config("CELERY_TASK_TIME_LIMIT", default=60 * 30, cast=int)  # 30 min
CELERY_TASK_SOFT_TIME_LIMIT = config(
    "CELERY_TASK_SOFT_TIME_LIMIT", default=60 * 25, cast=int
)
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_WORKER_CONCURRENCY = config(
    "CELERY_WORKER_CONCURRENCY", default=1, cast=int
)

if config("CELERY_TASK_ALWAYS_EAGER", default=False, cast=bool):
    CELERY_TASK_ALWAYS_EAGER = True
    CELERY_TASK_EAGER_PROPAGATES = True

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
INFERENCE_DEFAULT_ADAPTER = config(
    "INFERENCE_DEFAULT_ADAPTER", default="comfyui"
).lower()
COMFYUI_BASE_URL = config("COMFYUI_BASE_URL", default="http://127.0.0.1:8188")
COMFYUI_API_TIMEOUT = config("COMFYUI_API_TIMEOUT", default=600, cast=int)
# Exported ComfyUI API workflow. Relative paths resolve against BASE_DIR.
# When empty, the ComfyUI adapter refuses to run instead of submitting a
# placeholder graph that cannot produce a video.
COMFYUI_WORKFLOW_PATH = config("COMFYUI_WORKFLOW_PATH", default="")
if COMFYUI_WORKFLOW_PATH:
    _workflow_path = Path(COMFYUI_WORKFLOW_PATH)
    if not _workflow_path.is_absolute():
        _workflow_path = BASE_DIR / _workflow_path
    COMFYUI_WORKFLOW_PATH = str(_workflow_path)
GENERATION_JOB_TIMEOUT_SECONDS = config(
    "GENERATION_JOB_TIMEOUT_SECONDS", default=60 * 30, cast=int
)

# ---------------------------------------------------------------------------
# Film / FFmpeg defaults
# ---------------------------------------------------------------------------
FFMPEG_BIN = config("FFMPEG_BIN", default="ffmpeg")
# A value that contains a path separator is resolved against BASE_DIR so the
# bundled binary works regardless of the working directory; a bare name like
# "ffmpeg" keeps searching PATH.
if "/" in FFMPEG_BIN or "\\" in FFMPEG_BIN:
    _ffmpeg_path = Path(FFMPEG_BIN)
    FFMPEG_BIN = str(
        _ffmpeg_path if _ffmpeg_path.is_absolute() else BASE_DIR / _ffmpeg_path
    )
FFMPEG_CONCAT_TRANSCODING = config(
    "FFMPEG_CONCAT_TRANSCODING", default="libx264", cast=str
)
FILM_TRANSITION_SECONDS = config("FILM_TRANSITION_SECONDS", default=1.0, cast=float)
# Quality of the re-encoded film output. Lower CRF = better quality.
FILM_VIDEO_CRF = config("FILM_VIDEO_CRF", default=18, cast=int)
# Bitrate used for the voiceover/audio track when a film is re-encoded.
FILM_AUDIO_BITRATE = config("FILM_AUDIO_BITRATE", default="192k")

# ---------------------------------------------------------------------------
# Automatic AI voiceover
# ---------------------------------------------------------------------------
# The video models only generate silent frames, so a spoken track is generated
# from the prompt with edge-tts (free, no API key) and muxed with FFmpeg.
# Set VOICEOVER_ENABLED=false to keep clips silent.
VOICEOVER_ENABLED = config("VOICEOVER_ENABLED", default=True, cast=bool)
# Bengali voices: bn-BD-NabanitaNeural (female), bn-BD-PradeepNeural (male),
# bn-IN-TanishaaNeural (female), bn-IN-BashkarNeural (male).
VOICEOVER_VOICE = config("VOICEOVER_VOICE", default="bn-BD-NabanitaNeural")
VOICEOVER_RATE = config("VOICEOVER_RATE", default="+0%")
VOICEOVER_VOLUME = config("VOICEOVER_VOLUME", default="+0%")
VOICEOVER_AUDIO_BITRATE = config("VOICEOVER_AUDIO_BITRATE", default="192k")
VOICEOVER_MAX_CHARS = config("VOICEOVER_MAX_CHARS", default=240, cast=int)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
ENABLE_SYSTEM_MONITOR = config("ENABLE_SYSTEM_MONITOR", default=True, cast=bool)
