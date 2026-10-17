"""
Films app views — short-film studio.

CineForge AI — Local AI Video Generation Platform.

This module keeps the film assembly logic intentionally small and testable
while delegating actual FFmpeg invocation to the service layer.
"""
from __future__ import annotations

import logging
from pathlib import Path

from django.contrib import messages
from django.http import FileResponse, HttpResponseNotFound
from django.shortcuts import get_object_or_404, redirect, render

from .forms import CharacterReferenceForm, FilmProjectForm, FilmSceneForm
from .models import CharacterReference, FilmProject, FilmScene
from .services import (
    assemble_film,
    generate_scene_clip,
    move_scene,
    next_scene_order,
)

logger = logging.getLogger("cineforge.films")


def film_list(request):
    """List film projects."""
    projects = FilmProject.objects.prefetch_related("scenes").order_by("-updated")
    return render(request, "films/film_list.html", {
        "page": "film_list",
        "projects": projects,
    })


def film_new(request):
    """Create a new film project."""
    if request.method == "POST":
        form = FilmProjectForm(request.POST)
        if form.is_valid():
            project = form.save()
            messages.success(request, "Film project created.")
            return redirect("films:film_detail", pk=project.pk)
    else:
        form = FilmProjectForm()
    return render(request, "films/film_form.html", {
        "page": "film_form",
        "form": form,
        "project": None,
        "add_scene": False,
    })


def film_detail(request, pk):
    """Show a film project and its scenes."""
    project = get_object_or_404(FilmProject, pk=pk)
    return render(request, "films/film_detail.html", {
        "page": "film_detail",
        "project": project,
        "add_scene": False,
    })


def film_edit(request, pk):
    """Edit a film project and its scenes."""
    project = get_object_or_404(FilmProject, pk=pk)
    if request.method == "POST":
        form = FilmProjectForm(request.POST, instance=project)
        if form.is_valid():
            project = form.save()
            messages.success(request, "Film project updated.")
            return redirect("films:film_detail", pk=project.pk)
    else:
        form = FilmProjectForm(instance=project)
    return render(request, "films/film_form.html", {
        "page": "film_form",
        "form": form,
        "scene_form": FilmSceneForm(prefix="scene"),
        "project": project,
        "add_scene": True,
    })


def film_assemble(request, pk):
    """Trigger film assembly."""
    project = get_object_or_404(FilmProject, pk=pk)
    if request.method == "POST":
        result = assemble_film(project)
        if result["success"]:
            messages.success(request, result["message"] or "Film assembled.")
        else:
            messages.error(request, result["message"] or "Assembly failed.")
    return redirect("films:film_detail", pk=project.pk)


def film_download(request, pk):
    """Serve the assembled movie as a file download."""
    project = get_object_or_404(FilmProject, pk=pk)
    if not project.assembled_video:
        return HttpResponseNotFound("No assembled movie is available.")

    path = Path(project.assembled_video.path)
    if not path.exists():
        return HttpResponseNotFound("Assembled movie file not found on disk.")

    try:
        response = FileResponse(
            path.open("rb"),
            as_attachment=True,
            filename=path.name,
        )
        response["Content-Type"] = "video/mp4"
        return response
    except OSError as exc:
        logger.warning("Could not stream assembled film %s: %s", path, exc)
        messages.error(request, "Could not serve the assembled movie.")
        return redirect("films:film_detail", pk=project.pk)


def scene_add(request, pk):
    """Add a scene to a film project."""
    project = get_object_or_404(FilmProject, pk=pk)
    if request.method == "POST":
        form = FilmSceneForm(request.POST, prefix="scene")
        if form.is_valid():
            scene = form.save(commit=False)
            scene.project = project
            scene.order = next_scene_order(project)
            scene.save()
            messages.success(request, "Scene added.")
            return redirect("films:film_detail", pk=project.pk)
    else:
        form = FilmSceneForm()
    return redirect("films:film_detail", pk=project.pk)


def scene_generate(request, pk, scene_pk):
    """Create and queue a generation job for a single film scene."""
    project = get_object_or_404(FilmProject, pk=pk)
    scene = get_object_or_404(FilmScene, pk=scene_pk, project=project)
    if request.method == "POST":
        queued, message = generate_scene_clip(scene)
        if queued:
            messages.success(request, message or "Scene clip queued.")
        else:
            messages.warning(request, message or "Could not queue the scene clip.")
    return redirect("films:film_detail", pk=project.pk)


def scene_move(request, project_pk, scene_pk, direction):
    """Move a scene one position up or down in the film."""
    project = get_object_or_404(FilmProject, pk=project_pk)
    scene = get_object_or_404(FilmScene, pk=scene_pk, project=project)
    if request.method == "POST":
        if move_scene(scene, direction):
            messages.success(request, "Scene moved.")
        else:
            messages.warning(request, "That scene cannot move any further.")
    return redirect("films:film_detail", pk=project.pk)


def character_list(request):
    """List reusable character profiles and create new ones."""
    if request.method == "POST":
        form = CharacterReferenceForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Character saved.")
            return redirect("films:character_list")
    else:
        form = CharacterReferenceForm()
    return render(request, "films/character_list.html", {
        "page": "character_list",
        "characters": CharacterReference.objects.all(),
        "form": form,
    })


def scene_delete(request, project_pk, scene_pk):
    """Delete a scene from a film project."""
    project = get_object_or_404(FilmProject, pk=project_pk)
    scene = get_object_or_404(FilmScene, pk=scene_pk, project=project)
    scene.delete()
    messages.success(request, "Scene removed.")
    return redirect("films:film_detail", pk=project.pk)
