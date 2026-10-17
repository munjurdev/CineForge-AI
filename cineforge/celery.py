"""
Celery application for CineForge AI background video generation jobs.

Usage:
    # Start worker (in a separate terminal)
    celery -A cineforge worker --loglevel=info

    # Start beat scheduler if using periodic tasks
    celery -A cineforge beat --loglevel=info

For development without Redis, the app degrades gracefully:
    - Use the ``--pool=solo`` option for single-process development.
    - With ``CELERY_TASK_ALWAYS_EAGER=True``, tasks run synchronously in-process.
"""
from __future__ import annotations

import logging
import os

from celery import Celery
from celery.signals import worker_process_init

logger = logging.getLogger("cineforge.celery")

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
app = Celery("cineforge")

# Read config from Django settings with CELERY_ prefix
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks in installed apps
app.autodiscover_tasks()


@worker_process_init.connect
def configure_logging_process(**kwargs: object) -> None:
    """Re-configure logging for each Celery worker process."""
    try:
        import django

        django.setup()
    except Exception as exc:  # pragma: no cover - worker startup
        logger.warning("Could not setup Django in worker process: %s", exc)


# ---------------------------------------------------------------------------
# Health check task (used by diagnostics and readiness probes)
# ---------------------------------------------------------------------------
@app.task(bind=True, name="cineforge.tasks.ping")
def ping(self) -> dict[str, str]:
    """Return worker identity for diagnostics."""
    return {
        "worker": getattr(self.request, "hostname", None) or "unknown",
        "pid": str(os.getpid()),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Debug task (keep for development introspection)
# ---------------------------------------------------------------------------
@app.task(bind=True, ignore_result=True)
def debug_task(self) -> None:
    """Print the request for debugging."""
    logger.info("Celery debug request: %s", self.request)
