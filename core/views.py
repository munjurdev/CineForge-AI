"""
Core app views — landing page, dashboard, and shared helpers.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.shortcuts import render


def home(request):
    """Landing page: intro, quick links, and a summary of what is configured."""
    return render(request, "core/home.html", {"page": "home"})


def dashboard(request):
    """Main dashboard: system readiness, counts, and recent activity."""
    from generator.models import GeneratedVideo, GenerationJob
    from generator.services import get_engine_status_summary
    from films.models import FilmProject
    from system_monitor.services import get_system_summary

    system = get_system_summary()
    engine = get_engine_status_summary()

    jobs = GenerationJob.objects.order_by("-created")[:5]
    job_count = GenerationJob.objects.count()
    success_count = GenerationJob.objects.filter(
        status=GenerationJob.Status.SUCCESS
    ).count()
    stats = {
        "jobs": job_count,
        "videos": GeneratedVideo.objects.count(),
        "films": FilmProject.objects.count(),
        "success_rate": round(success_count / job_count * 100) if job_count else 0,
    }

    return render(
        request,
        "core/dashboard.html",
        {
            "page": "dashboard",
            "system": system,
            "engine": engine,
            "stats": stats,
            "recent_jobs": jobs,
        },
    )
