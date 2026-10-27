"""
URL configuration for the generator app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "generator"

urlpatterns = [
    path("generate/", views.generate_video, name="generate_video"),
    path("prompt-assistant/", views.enhance_prompt, name="enhance_prompt"),
    path("image-to-video/", views.generate_image_to_video, name="image_to_video"),
    path("jobs/<uuid:pk>/status/", views.job_status, name="job_status"),
    path("jobs/<uuid:pk>/", views.job_detail, name="job_detail"),
    path("jobs/<uuid:pk>/cancel/", views.cancel_job, name="job_cancel"),
    path("jobs/<uuid:pk>/retry/", views.retry_job, name="job_retry"),
    path("jobs/<uuid:pk>/delete/", views.delete_job, name="job_delete"),
    path("jobs/", views.job_list, name="job_list"),
]
