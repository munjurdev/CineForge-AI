"""
Forms for the films app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django import forms

from .models import CharacterReference, FilmProject, FilmScene


class FilmProjectForm(forms.ModelForm):
    """Form for creating/editing a film project."""

    class Meta:
        model = FilmProject
        fields = ["title", "description", "project_note"]
        widgets = {
            "title": forms.TextInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Film title",
            }),
            "description": forms.Textarea(attrs={
                "rows": 3,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Synopsis (optional)",
            }),
            "project_note": forms.Textarea(attrs={
                "rows": 4,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Notes about character consistency, continuity, and style",
            }),
        }


class FilmSceneForm(forms.ModelForm):
    """Form for creating/editing a film scene."""

    class Meta:
        model = FilmScene
        fields = ["title", "prompt", "negative_prompt", "character_reference", "duration_seconds", "transition_in", "transition_out"]
        widgets = {
            "title": forms.TextInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Scene title (optional)",
            }),
            "prompt": forms.Textarea(attrs={
                "rows": 5,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Describe the scene...",
            }),
            "negative_prompt": forms.Textarea(attrs={
                "rows": 2,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
            }),
            "character_reference": forms.Select(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
            }),
            "duration_seconds": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0.5,
                "max": 30,
                "step": 0.5,
            }),
            "transition_in": forms.Select(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
            }),
            "transition_out": forms.Select(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
            }),
        }


class CharacterReferenceForm(forms.ModelForm):
    """Form for reusable character profiles used across film scenes."""

    class Meta:
        model = CharacterReference
        fields = ["name", "description", "notes", "reference_image"]
        widgets = {
            "name": forms.TextInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Character name",
            }),
            "description": forms.Textarea(attrs={
                "rows": 4,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Physical and stylistic description used in prompts",
            }),
            "notes": forms.Textarea(attrs={
                "rows": 3,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Continuity notes: clothing, props, scars, etc.",
            }),
            "reference_image": forms.ClearableFileInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-100",
            }),
        }

    def clean_reference_image(self):
        from django.conf import settings
        from django.core.files.uploadedfile import UploadedFile

        image = self.cleaned_data.get("reference_image")
        if not image or not isinstance(image, UploadedFile):
            return image

        max_bytes = getattr(settings, "MAX_UPLOAD_SIZE_MB", 50) * 1024 * 1024
        if image.size and image.size > max_bytes:
            raise forms.ValidationError(
                f"Image must be smaller than {max_bytes // (1024 * 1024)} MB."
            )

        allowed = getattr(
            settings, "ALLOWED_IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png", ".webp"]
        )
        extension = f".{image.name.rsplit('.', 1)[-1].lower()}" if "." in image.name else ""
        if extension not in {item.lower() for item in allowed}:
            raise forms.ValidationError(
                f"Unsupported image type. Allowed: {', '.join(allowed)}."
            )
        return image
