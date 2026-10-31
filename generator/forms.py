"""
CineForge AI - Local AI Video Generation Platform.
"""
from __future__ import annotations

from django import forms
from django.core.validators import MaxValueValidator, MinValueValidator

from .models import GenerationJob
from .providers import engine_choices, get_provider


_FIELD_LABELS = {
    "prompt": "Prompt",
    "negative_prompt": "Negative prompt",
    "engine": "Engine / model",
    "model_name": "Model name",
    "duration_seconds": "Duration (seconds)",
    "width": "Width (px)",
    "height": "Height (px)",
    "fps": "FPS",
    "aspect_ratio": "Aspect ratio",
    "seed": "Seed",
    "guidance_scale": "Guidance scale",
    "steps": "Steps",
}

# Shared widget styling so every control on the dark UI looks the same.
_SELECT_CLASS = (
    "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 "
    "text-neutral-100 focus:border-purple-500 focus:outline-none"
)

# The model stores aspect_ratio as free text, so the Select widget would render
# with no options at all. These are the ratios the presets use plus the usual
# extras; the blank entry means "use the engine preset".
ASPECT_RATIO_CHOICES = (
    ("", "Auto (engine preset)"),
    ("16:9", "16:9 — landscape"),
    ("9:16", "9:16 — portrait"),
    ("1:1", "1:1 — square"),
    ("4:3", "4:3 — standard"),
    ("3:4", "3:4 — vertical"),
    ("21:9", "21:9 — ultrawide"),
)


def _set_field_range(field: forms.Field, minimum: float, maximum: float) -> None:
    """Apply numeric bounds to a form field, preserving its widget."""
    field.min_value = minimum
    field.max_value = maximum
    field.validators.extend(
        [MinValueValidator(minimum), MaxValueValidator(maximum)]
    )


class EngineChoiceField(forms.ChoiceField):
    """Engine select whose values come from the provider registry.

    ``GenerationJob.engine`` historically stored uppercase values (``TEST``,
    ``COMFYUI``) while provider specs register lowercase keys (``test``,
    ``veo``), so a submitted value is resolved to its registered key before
    validation. Legacy data therefore keeps working as providers are added.
    """

    def to_python(self, value):
        if value in self.empty_values:
            return ""
        value = str(value).strip()
        spec = get_provider(value)
        return spec.key if spec else value.lower()


def _apply_engine_field(form: forms.Form) -> None:
    """Populate the engine dropdown straight from the provider registry.

    The registry — not the model — owns the list of backends, so adding or
    removing a video API needs neither a migration nor a change here. Cloud
    providers without credentials are listed but labelled with the variable
    they are missing, so the reason is visible before a job is queued.
    """
    if "engine" not in form.fields:
        return
    form.fields["engine"] = EngineChoiceField(
        label=_FIELD_LABELS.get("engine", "Engine / model"),
        choices=[("", "Auto (default engine)")] + engine_choices(),
        required=False,
        widget=forms.Select(attrs={"class": _SELECT_CLASS}),
    )


class GenerationJobForm(forms.ModelForm):
    """Text to video form.

    Generation settings are optional: omitted values are filled from the
    engine preset in the service layer, so the form stays usable for users who
    only want to type a prompt.
    """

    OPTIONAL_FIELDS = (
        "engine",
        "duration_seconds",
        "width",
        "height",
        "fps",
        "guidance_scale",
        "steps",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self.OPTIONAL_FIELDS:
            if name in self.fields:
                self.fields[name].required = False
        for name, label in _FIELD_LABELS.items():
            if name in self.fields:
                self.fields[name].label = label
        _apply_engine_field(self)
        if "aspect_ratio" in self.fields:
            self.fields["aspect_ratio"].widget.choices = ASPECT_RATIO_CHOICES
        _set_field_range(self.fields["duration_seconds"], 0.5, 30)
        _set_field_range(self.fields["width"], 64, 2048)
        _set_field_range(self.fields["height"], 64, 2048)
        _set_field_range(self.fields["fps"], 1, 60)
        _set_field_range(self.fields["guidance_scale"], 0, 20)
        _set_field_range(self.fields["steps"], 1, 500)

    class Meta:
        model = GenerationJob
        fields = [
            "prompt",
            "negative_prompt",
            "engine",
            "model_name",
            "duration_seconds",
            "width",
            "height",
            "fps",
            "aspect_ratio",
            "seed",
            "guidance_scale",
            "steps",
        ]
        widgets = {
            "prompt": forms.Textarea(attrs={
                "rows": 6,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Describe the video you want to generate...",
            }),
            "negative_prompt": forms.Textarea(attrs={
                "rows": 3,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Optional: what to avoid (blurry, distorted, extra limbs, text, watermark...)",
            }),
            "model_name": forms.TextInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "e.g. Wan 2.1 I2V",
            }),
            "duration_seconds": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0.5, "max": 30, "step": 0.5,
            }),
            "width": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 64, "step": 8,
            }),
            "height": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 64, "step": 8,
            }),
            "fps": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 1, "max": 60,
            }),
            "aspect_ratio": forms.Select(attrs={
                "class": _SELECT_CLASS,
            }),
            "engine": forms.Select(attrs={
                "class": _SELECT_CLASS,
            }),
            "seed": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0, "max": 4294967295, "placeholder": "Leave blank for random",
            }),
            "guidance_scale": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0, "max": 20, "step": 0.1, "placeholder": "Optional",
            }),
            "steps": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 1, "max": 500, "placeholder": "Optional",
            }),
        }

    def clean_duration_seconds(self) -> float | None:
        value = self.cleaned_data.get("duration_seconds")
        if value is not None and value <= 0:
            raise forms.ValidationError("Duration must be greater than zero.")
        return value

    def clean_width(self) -> int | None:
        return self._clean_dimension("width")

    def clean_height(self) -> int | None:
        return self._clean_dimension("height")

    def _clean_dimension(self, field: str) -> int | None:
        value = self.cleaned_data.get(field)
        if value is None:
            return getattr(self.instance, field) or None
        if value < 64:
            raise forms.ValidationError(f"{field} must be at least 64 pixels.")
        if value > 2048:
            raise forms.ValidationError(
                f"{field} must be at most 2048 pixels for local generation."
            )
        return value

    def clean(self) -> dict:
        cleaned = super().clean()
        width = cleaned.get("width") or self.instance.width
        height = cleaned.get("height") or self.instance.height
        if width and height:
            if width % 8 or height % 8:
                self.add_warning(
                    "resolution_align",
                    "Width and height should ideally be multiples of 8 for most video models.",
                )
        return cleaned

    def add_warning(self, key: str, message: str) -> None:
        if not hasattr(self, '_warnings'):
            self._warnings: list = []
        self._warnings.append((key, message))

    @property
    def warnings(self):
        return getattr(self, "_warnings", [])

    def clean_aspect_ratio(self) -> str:
        value = self.cleaned_data.get("aspect_ratio")
        if value:
            value = value.strip()
        return value


class ImageToVideoForm(forms.ModelForm):
    """Image to video form."""

    OPTIONAL_FIELDS = (
        "engine",
        "duration_seconds",
        "width",
        "height",
        "fps",
        "guidance_scale",
    )

    reference_image = forms.ImageField(
        label="Reference image",
        help_text="Upload an image to animate.",
        widget=forms.ClearableFileInput(attrs={
            "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-purple-600 hover:file:bg-purple-500 file:text-white",
        }),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self.OPTIONAL_FIELDS:
            if name in self.fields:
                self.fields[name].required = False
        for name, label in _FIELD_LABELS.items():
            if name in self.fields:
                self.fields[name].label = label
        _apply_engine_field(self)
        _set_field_range(self.fields["duration_seconds"], 0.5, 30)
        _set_field_range(self.fields["width"], 64, 2048)
        _set_field_range(self.fields["height"], 64, 2048)
        _set_field_range(self.fields["fps"], 1, 60)
        _set_field_range(self.fields["guidance_scale"], 0, 20)

    class Meta:
        model = GenerationJob
        fields = [
            "prompt",
            "negative_prompt",
            "engine",
            "model_name",
            "duration_seconds",
            "width",
            "height",
            "fps",
            "seed",
            "guidance_scale",
        ]
        widgets = {
            "prompt": forms.Textarea(attrs={
                "rows": 5,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "Describe the motion you want the image to take...",
            }),
            "negative_prompt": forms.Textarea(attrs={
                "rows": 2,
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
            }),
            "model_name": forms.TextInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-purple-500 focus:outline-none",
                "placeholder": "e.g. Wan 2.1 I2V",
            }),
            "duration_seconds": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0.5, "max": 30, "step": 0.5,
            }),
            "width": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 64, "step": 8,
            }),
            "height": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 64, "step": 8,
            }),
            "fps": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 1, "max": 60,
            }),
            "engine": forms.Select(attrs={
                "class": _SELECT_CLASS,
            }),
            "seed": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0, "max": 4294967295, "placeholder": "Leave blank for random",
            }),
            "guidance_scale": forms.NumberInput(attrs={
                "class": "w-full rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-purple-500 focus:outline-none",
                "min": 0, "max": 20, "step": 0.1, "placeholder": "Optional",
            }),
        }

    def clean_duration_seconds(self) -> float | None:
        value = self.cleaned_data.get("duration_seconds")
        if value is not None and value <= 0:
            raise forms.ValidationError("Duration must be greater than zero.")
        return value

    def clean_reference_image(self):
        from django.core.exceptions import ValidationError
        from django.core.files.uploadedfile import UploadedFile
        from django.conf import settings
        image = self.cleaned_data.get("reference_image")
        if not image:
            return image
        if not isinstance(image, UploadedFile):
            return image
        max_bytes = getattr(settings, "MAX_UPLOAD_SIZE_MB", 50) * 1024 * 1024
        if image.size and image.size > max_bytes:
            raise ValidationError(
                f"Image must be smaller than {max_bytes // (1024*1024)} MB."
            )
        allowed = getattr(
            settings, "ALLOWED_IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png", ".webp"]
        )
        ext = image.name.rsplit(".", 1)[-1].lower() if "." in image.name else ""
        if f".{ext}" not in {a.lower() for a in allowed}:
            raise ValidationError(
                f"Unsupported image type. Allowed: {', '.join(allowed)}."
            )
        return image
