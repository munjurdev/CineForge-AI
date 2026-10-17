# CineForge AI — Local AI Video Generation Platform

CineForge AI is a professional, high-quality local AI video generation platform built with **Python and Django**. Enter a natural-language prompt and generate cinematic videos locally using compatible open-source AI video models — no paid APIs or per-generation cloud credits required.

The platform prioritises visual quality, prompt adherence, character consistency, ease of use, and efficient GPU utilisation. It is a functional application, not just a UI prototype.

## Table of Contents

- [Architecture](#architecture)
- [AI model strategy](#ai-model-strategy)
- [Hardware requirements](#hardware-requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Usage](#usage)
- [Testing](#testing)
- [Security and reliability](#security-and-reliability)
- [Limitations and acceptance criteria](#limitations-and-acceptance-criteria)
- [Development](#development)

## Architecture

CineForge AI uses a modular architecture so the AI engine can be replaced without rewriting the Django application.

- **Backend:** Python, Django
- **Frontend:** Django templates, Tailwind CSS, JavaScript
- **Database:** SQLite for local development (PostgreSQL-compatible settings included)
- **AI inference:** ComfyUI API or a compatible local inference backend, accessed through a modular adapter layer
- **Background jobs:** Celery + Redis (with graceful degradation to in-process execution for development)
- **Video processing:** FFmpeg / ffprobe
- **Real-time progress:** polling initially, with optional Django Channels / WebSockets in the future
- **Storage:** local filesystem
- **Configuration:** environment variables and a documented `.env.example`

### Django apps

- `core` — landing page and dashboard
- `generator` — generation jobs, settings, and engine adapters
- `library` — output files and video metadata
- `films` — scenes, character profiles, and movie assembly
- `system_monitor` — hardware and backend diagnostics

### Suggested models

- `GenerationJob`
- `GeneratedVideo`
- `FilmProject`
- `FilmScene`
- `CharacterReference`

Long-running inference is kept out of ordinary HTTP request handlers. The application provides a background execution mechanism, job status tracking, retries, and structured logs.

## AI model strategy

Do not build a new video model from scratch. Integrate an existing, maintained, open-source or locally runnable video-generation model through a modular adapter.

Evaluate options such as:

- Wan family of video models
- LTX-Video
- Other compatible models appropriate to the available hardware

Before selecting a model:

1. Check the current official documentation and license.
2. Check Windows support, VRAM requirements, dependencies, and model size.
3. Choose a model that can realistically run on the detected hardware.
4. If no suitable GPU is available, the application explains the limitations and offers an optional cloud backend separately.
5. Never download large model weights automatically without informing the user about download size and storage requirements.

Local generation does not require a paid API key. Hardware, electricity, storage, and generation time still have real costs.

The adapter layer lives in `generator/adapters.py` with concrete implementations in:

- `generator/adapters_comfyui.py` — ComfyUI adapter
- `generator/adapters_demo.py` — CPU-only demo/test adapter for CI and hardware-less environments

## Hardware requirements

This project is designed to run on a local machine. A GPU is strongly recommended for real AI video generation, but the application does not assume one is available.

Recommended for real generation:

- Windows 10/11 or Linux
- A GPU with enough VRAM for the chosen model and workflow
- FFmpeg installed and on PATH
- Enough free disk space for model weights, outputs, and temp files

For development and testing without a GPU, the included test adapter generates simple animated clips with FFmpeg so the rest of the application (jobs, library, film assembly, downloads, preview) can be exercised end to end.

The `system_monitor:diagnostics` page detects available RAM, GPU, VRAM, CUDA availability, disk space, and installed inference backend where possible, and shows whether the selected model is likely to fit the available hardware.

## Quick start

1. Clone or copy this project to a local folder.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Copy `.env.example` to `.env` and adjust the values.
5. Build the stylesheet (`npm install && npm run build:css`).
6. Run Django checks and migrations.
7. Start the development server.

Example on Windows with PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python manage.py check
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` in your browser.

If you have Celery and Redis available, you can also start a worker:

```powershell
celery -A cineforge worker --loglevel=info
```

For development without Redis, set `CELERY_TASK_ALWAYS_EAGER=True` in `.env` so tasks run synchronously in-process.

## Configuration

Copy `.env.example` to `.env` and review the values. Important settings include:

- `DJANGO_SECRET_KEY` — change this in production
- `DEBUG` — keep `True` only for local development
- `ALLOWED_HOSTS` — comma-separated list of allowed hosts
- `INFERENCE_DEFAULT_ADAPTER` — `comfyui`, `wan`, `ltx`, or `test`
- `COMFYUI_BASE_URL` — URL of the local ComfyUI instance, if used
- `FFMPEG_BIN` — path to ffmpeg if not on PATH
- `FILM_TRANSITION_SECONDS` — crossfade duration used when assembling a film
- `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` — Redis or other broker/backend (`redis` is installed from `requirements.txt` so Celery can resolve the broker transport even in eager mode)
- `MEDIA_ROOT` and `STATIC_ROOT` — storage paths (relative paths resolve against the project root)

All secrets are kept out of source code. The application reads configuration from environment variables via `python-decouple`.

## Project layout

```
manage.py            Django management entry point
requirements.txt     Python dependencies
package.json         Tailwind CSS build scripts
.env.example         Environment variable template
tailwind/
    input.css        Tailwind source (built into static/css/tailwind.css)
cineforge/
    settings.py      Project settings
    urls.py          Root URL configuration
    wsgi.py          WSGI entry point
    asgi.py          ASGI entry point
    celery.py        Celery application
core/
    views.py
    urls.py
    context_processors.py
    templates/core/
generator/
    models.py
    forms.py
    views.py
    urls.py
    services.py
    tasks.py
    prompt_assistant.py
    adapters.py
    adapters_comfyui.py
    adapters_demo.py
    admin.py
    apps.py
    migrations/
    tests/
        test_forms.py
        test_models.py
        test_services.py
        test_adapters.py
        test_tasks.py
library/
    views.py
    urls.py
    apps.py
    tests/
        test_views.py
films/
    models.py
    forms.py
    views.py
    urls.py
    services.py
    admin.py
    apps.py
    migrations/
    tests/
        test_models.py
        test_services.py
        test_views.py
system_monitor/
    views.py
    urls.py
    services.py
    admin.py
    apps.py
    tests/
        test_services.py
templates/
    base.html
media/
    uploads/
    temp/
    outputs/
static/
    css/
    images/
logs/
```

Runtime artifacts are not committed: `db.sqlite3`, `.env`, `staticfiles/`,
`media/*`, `logs/*`, and `venv/` are listed in `.gitignore`.

## Usage

### Text-to-video

1. Go to **Generate video**.
2. Enter a prompt. Prompt examples are available on the page.
3. Optionally add a negative prompt.
4. Configure duration, resolution, FPS, aspect ratio, and seed, subject to model capabilities.
5. Validate settings against the installed model before starting.
6. Generate, cancel, retry, and view generation status.

### Image-to-video

1. Go to **Image to video**.
2. Upload a reference image.
3. Enter a motion prompt.
4. Configure generation settings.
5. Start the job and monitor its status.

### AI prompt assistant

Use the prompt assistant on the generation pages to expand a short idea into a detailed cinematic prompt. Prompts are structured around subject, action, environment, lighting, camera movement, composition, and visual style. The assistant works offline using rule-based templates (`generator/prompt_assistant.py`) and is labelled as rule-based in the UI. It never claims a prompt has been enhanced by an AI model because no language model is called.

### Generation history

History is stored in the database and persists across application restarts. You can preview, download, retry, and delete generated videos from the history page.

### Video library

The library shows a gallery of generated videos with thumbnails, prompt summaries, duration, resolution, and generation date. MP4 download is available when the output format is supported. A disk-usage indicator is shown where available.

### Short-film studio

1. Create a film project.
2. Add scenes, each with its own prompt and generation status.
3. Press **Generate clip** on a scene to queue it through the configured adapter. The resulting job is linked to the scene and the scene's status follows the job.
4. Use reusable character descriptions and reference images (see **Characters**) where the model supports them. Character text is appended to the scene prompt.
5. Reorder scenes with the up/down buttons.
6. Assemble the clips with FFmpeg once at least two scenes have rendered clips.
7. The original clips are preserved and a separate final movie is created.

Scene transitions are applied during assembly: **Cut** joins clips with the fast stream-copy path, while **Fade**, **Dissolve**, **Wipe left**, and **Wipe up** use FFmpeg `xfade` and re-encode the movie (video only). Transition length is controlled by `FILM_TRANSITION_SECONDS` and is clamped to half of the shorter adjacent clip. If clip durations cannot be probed, assembly falls back to hard cuts and says so in the result message. Soundtrack and title-card support are not implemented yet. Character consistency and scene continuity are model-dependent, not guaranteed by this application.

### System diagnostics

The diagnostics page detects available RAM, GPU, VRAM, CUDA availability, disk space, and installed inference backend where possible, and shows whether the selected model is likely to fit the available hardware. It also includes a model compatibility and setup page.

Do not promise successful generation before verifying that a compatible model and workflow are installed.

## Frontend assets

The UI uses Tailwind CSS. The source lives in `tailwind/input.css` and the
compiled stylesheet is served from `static/css/tailwind.css`.

```powershell
npm install
npm run build:css    # one-off minified build
npm run watch:css    # rebuild while developing
```

Run the CSS build after adding or changing template classes.

## Testing

Run the test suite with:

```powershell
python manage.py test
```

The included tests cover business logic and error cases that do not require a GPU. Hardware-dependent features that cannot be tested are explicitly documented below.

## Security and reliability

- The application uses Django CSRF protection and secure file validation.
- The local inference API is not exposed to the public internet by default.
- File deletion is restricted to paths under `MEDIA_ROOT`.
- FFmpeg is invoked with list-form arguments to avoid shell command injection.
- Secrets are kept out of source code.
- Structured logging is configured in `settings.py`.
- Jobs are made recoverable after an application restart where feasible.
- The application avoids running multiple large generations concurrently unless the hardware supports it.

## Limitations and acceptance criteria

The project is complete when:

- The Django application starts on Windows using documented commands.
- The dashboard and generation forms work.
- A user can submit a prompt to an installed, compatible local model.
- The application records the job and displays its real status.
- A successfully generated video can be previewed and downloaded.
- History persists across application restarts.
- Errors such as missing models, failed inference, and insufficient VRAM are handled clearly.
- Multiple generated clips can be assembled into a final short movie.
- Automated tests pass for the components that can be tested without a GPU.
- Any hardware-dependent features that cannot be tested are explicitly documented.

### What was actually tested

On this machine:

- Django system checks pass (`python manage.py check`).
- Migrations are generated for every app with models and apply cleanly.
- The full test suite runs and passes (130 tests; the real-FFmpeg integration
  test is skipped when ffmpeg/ffprobe are absent) for the components that do not
  require a GPU, including the rule-based prompt assistant, scene generation
  bookkeeping, scene reordering, character profiles, and crossfade
  filter-graph/command construction.
- End-to-end prompt-to-video through the real form with the CPU test adapter and
  a real FFmpeg 6.0 static build: job reached SUCCESS, produced a decodable
  128×96 H.264 MP4 (~1s), created a library entry with probed metadata, and the
  download endpoint returned the file.
- End-to-end film workflow: two scenes generated real clips, hard-cut assembly
  produced a decodable 2.0s movie, and selecting the Fade transition produced a
  decodable 1.54s crossfaded movie (1s + 1s with a 0.5s overlap).
- Older-FFmpeg fallback: with FFmpeg 4.1 (no `xfade` filter) assembly degraded
  to hard cuts and reported that FFmpeg 4.3+ is required, instead of failing.
- Every page renders without errors (home, dashboard, both generation forms,
  job list, library, films, character list, diagnostics, model setup, settings).
- The project directory structure, settings, and app configs are consistent.
- The Celery app configures without errors when `django-celery-results` is installed.
- The root URL configuration, app URL configurations, and views are wired correctly.
- The Tailwind stylesheet builds from `tailwind/input.css`.

### What could not be tested here

- Real AI video generation with a local model (no compatible GPU/VRAM available on this machine).
- The ComfyUI adapter against a running ComfyUI instance (no local ComfyUI instance available on this machine).
- Celery worker behaviour with a real broker (Redis not available on this machine), although eager mode was exercised end to end.
- Long-running job progress polling under real inference load.
- A real FFmpeg crossfade render: the integration test is skipped when
  ffmpeg/ffprobe are not installed (they were unavailable in the verification
  environment). Unit tests mock FFmpeg and assert the generated filter graph,
  offsets, clamping, fallback, and command arguments.

These items are documented as hardware- or environment-dependent and are not faked or skipped.

## Development

To work on this project:

1. Keep the architecture modular. The AI engine should be replaceable without rewriting the Django application.
2. Add new adapters to `generator/adapters.py` and the corresponding module.
3. Keep long-running inference out of HTTP handlers.
4. Validate settings against the installed model before starting a job.
5. Display understandable errors when VRAM or dependencies are insufficient.
6. Do not fake video generation with static images or placeholder MP4 files.
7. Do not claim unlimited speed, perfect prompt accuracy, or guaranteed cinematic quality.
8. Do not assume that a powerful GPU is available.
9. Do not invent successful test results.
10. Do not overwrite user files or download massive model files without consent.

## License

This project is intended for local, personal, and experimental use. Always verify the license of any third-party model, workflow, or backend you install, and comply with its terms.
