# Video generation providers

CineForge runs video generation through a **provider registry**: every backend
— cloud API or local — is one module that declares what it is and how to call
it. The engine dropdown, the dashboard status card, the admin select and the
Celery worker all read the same registry, so a provider is added or removed in
exactly one place.

## Built-in providers

| Key | Provider | Credentials | Notes |
| --- | --- | --- | --- |
| `comfyui` | ComfyUI (local) | `COMFYUI_BASE_URL`, `COMFYUI_WORKFLOW_PATH` | Free; your own GPU or the Colab T4 workflow |
| `veo` | Google Veo (Gemini API) | `VEO_API_KEY`, `VEO_MODEL` | Best quality, generates its own audio |
| `runway` | Runway Gen-4 | `RUNWAY_API_KEY` | Camera control, image-to-video |
| `luma` | Luma Ray 2 | `LUMA_API_KEY` | Natural motion; text-to-video only |
| `kling` | Kling AI | `KLING_ACCESS_KEY`, `KLING_SECRET_KEY` | Cheapest per clip; JWT auth |
| `replicate` | Replicate | `REPLICATE_API_TOKEN`, `REPLICATE_MODEL` | One token, hundreds of open models |
| `huggingface` | Hugging Face | `HF_TOKEN`, `HF_MODEL` | Open models, one synchronous request |
| `wan`, `ltx` | — | — | Historical engine values; not implemented (they say so in the dropdown) |

Providers with missing credentials are still listed, labelled with the
variable they need, so the reason is visible before a job is queued.

```bash
# Show/hide providers without deleting code:
VIDEO_PROVIDERS_ENABLED=          # blank = every registered provider
VIDEO_PROVIDERS_DISABLED=luma,replicate
```

## How a request flows

```
form  →  GenerationJob(engine=<provider key>)  →  Celery run_generation_task
      →  get_adapter(engine)                   →  adapter.validate(job)
      →  adapter.generate(job)                 →  media/outputs/<job id>/
```

`CloudVideoAdapter` (in `generator/providers/base.py`) owns everything shared:

* capability validation (reference image, clip length, ignored tuning knobs),
* the submit → poll → download loop with cancellation and timeouts,
* progress reporting (elapsed-time estimate where the API exposes no %),
* HTTP error wording (`401` → "check the API key", `429` → "rate limit"),
* writing the finished file into the job's output directory.

A provider only implements the parts that differ:

```python
class MyAdapter(CloudVideoAdapter):
    name = "myprovider"

    def headers(self): ...       # auth
    def build_payload(self, job): ...
    def submit(self, client, job): ...   # -> remote job id
    def poll(self, client, remote_id): ...  # -> PollResult(state, video_url)
```

Synchronous providers (one request that returns the video) set
`synchronous = True` and implement `run_sync()` instead — see
`huggingface.py`.

## Adding a provider in one file

1. Create `generator/providers/myprovider.py`:

```python
from dataclasses import replace
import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_ASPECT, CAP_I2V, ProviderSpec, read_env

SPEC = ProviderSpec(
    key="myprovider",            # becomes job.engine and the adapter name
    label="My Provider",
    order=45,                    # position in the dropdown
    env_keys=("MYPROVIDER_API_KEY",),   # required, first one is the API key
    optional_env=("MYPROVIDER_API_BASE",),
    capabilities=frozenset({CAP_I2V, CAP_ASPECT}),
    max_duration=10.0,           # seconds; jobs longer than this are rejected
    default_model="fast-1",
    website="https://example.com",
    docs_url="https://docs.example.com",
    pricing="$0.10 per clip",
    estimated_seconds=60.0,      # used to animate progress while polling
)

class MyProviderAdapter(CloudVideoAdapter):
    name = "myprovider"

    def headers(self):
        return {"Authorization": f"Bearer {read_env('MYPROVIDER_API_KEY')}"}

    def submit(self, client, job):
        response = client.post(
            f"{read_env('MYPROVIDER_API_BASE', 'https://api.example.com')}/jobs",
            json={"prompt": job.prompt},
        )
        response.raise_for_status()
        return response.json()["id"]

    def poll(self, client, remote_id):
        data = client.get(f"https://api.example.com/jobs/{remote_id}").json()
        if data["state"] == "done":
            return PollResult(state=PollState.SUCCEEDED, video_url=data["url"])
        return PollResult(state=PollState.RUNNING, message="Rendering…")

register_provider(replace(SPEC, adapter_cls=MyProviderAdapter))
```

2. Add `"generator.providers.myprovider"` to `PROVIDER_MODULES` in
   `generator/providers/registry.py`.

3. Document the keys in `.env.example`.

That is everything — no migration, no form change, no template edit. To remove
a provider, delete the file and its entry in `PROVIDER_MODULES` (or just list
the key in `VIDEO_PROVIDERS_DISABLED`).

## Capabilities

Declare only what the API really supports; the shared validator turns that
into user-facing behaviour:

| Capability | Effect when absent |
| --- | --- |
| `i2v` | Jobs with a reference image are rejected with an explanation |
| `seed`, `steps`, `guidance`, `negative` | Setting them produces a "will be ignored" warning |
| `aspect_ratio`, `resolution` | A warning that the provider picks its own framing |

## Notes and gotchas

* **Veo model ids move fast.** Google retired the Veo 3.1 *preview* models in
  October 2026, so `VEO_MODEL` is deliberately required rather than defaulted.
* **Luma is text-to-video only** here: its image keyframes must be public CDN
  URLs, and CineForge stores reference images privately.
* **Kling signs a JWT** (HS256, 30 min) from your access/secret key pair
  instead of using a bearer token.
* **Hugging Face answers synchronously**, so there is nothing to poll; the
  default request timeout is `VIDEO_PROVIDER_SYNC_TIMEOUT`.
* **Progress bars** on cloud providers advance against `estimated_seconds`
  because most APIs report only queued/running/done.
