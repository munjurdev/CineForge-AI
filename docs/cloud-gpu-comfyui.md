# Real AI video generation with a cloud GPU

This machine (Intel HD 530, 8 GB RAM) cannot run video models locally. This guide
connects CineForge AI to ComfyUI running on a rented GPU instead. Your PC keeps
running the Django site; only inference happens in the cloud.

Tested code path: `generator/adapters_comfyui.py` (reachability check, workflow
token substitution, `/prompt`, `/history`, `/view` download). The cloud steps
below follow the official ComfyUI and RunPod documentation, but they have not
been executed on this machine.

## 1. Rent a GPU

1. Create an account at RunPod (or Vast.ai) and add credit.
2. Deploy a **GPU Pod** with the official **ComfyUI** template.
3. Pick a GPU by model size:

   | Model | Suggested GPU | VRAM |
   |---|---|---|
   | Wan 2.2 **5B TI2V** | RTX 3090 / 4090 / L4 | 16–24 GB |
   | Wan 2.2 **14B T2V** (fp8) | RTX 4090 / A5000 / L40S | 24 GB+ |

   Cost is roughly $0.2–0.6 per GPU-hour depending on the card. Stop the pod when
   you are done — storage and GPU time are billed separately.
4. Expose **HTTP port 8188** on the pod (the template does this by default).
   RunPod shows a URL like `https://<pod-id>-8188.proxy.runpod.net`.
5. Open that URL in a browser and confirm the ComfyUI canvas loads.

## 2. Install a video workflow in ComfyUI

1. In ComfyUI: **Workflow → Browse Templates → Video → Wan 2.2**
   (Text-to-Video, or Image-to-Video for the image workflow).
2. Let ComfyUI Manager download the required models (several GB) — this happens
   on the cloud pod's disk, not on your PC.
3. For the shipped `wan22_t2v_api.json` workflow also install the two
   **lightx2v 4-step** LoRAs into `models/loras/`
   (`wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` and
   `..._low_noise.safetensors`). They cut sampling from 20 steps to 4; without
   them ComfyUI rejects the job with a `lora not found` error.
4. Run the template once in the ComfyUI UI with a short prompt (e.g. 2 seconds,
   480p) and confirm it produces an MP4. Do not continue until this works.

## 3. Export and tokenise the workflow

1. In ComfyUI: **Workflow → Export (API)** / **Save (API Format)**.
2. Save the JSON on your PC as `comfyui/workflows/wan22_t2v_api.json`.
   The repository already ships a Wan 2.2 T2V workflow at that path — if your
   pod uses the models listed above you can skip this export entirely and just
   point `COMFYUI_WORKFLOW_PATH` at the shipped file.
3. Open it and replace the values CineForge should control with tokens:

   | Token | Where to put it |
   |---|---|
   | `{{PROMPT}}` | positive prompt text field |
   | `{{NEGATIVE_PROMPT}}` | negative prompt text field |
   | `{{WIDTH}}`, `{{HEIGHT}}` | latent/video size fields |
   | `{{FPS}}` | frame-rate field, if the workflow has one |
   | `{{FRAMES}}` | frame-count/length field (duration × FPS, rounded to 4n+1) |
   | `{{SEED}}` | sampler seed |
   | `{{MODEL_NAME}}` | optional model label |

   Tokens are placed inside the existing JSON strings/numbers, e.g.

   ```json
   "inputs": { "text": "{{PROMPT}}", "noise_seed": "{{SEED}}" }
   ```

   Keep the quotes: the file stays valid JSON while you edit it, and the
   quotes around numeric tokens are removed when the job runs.

   Keep the surrounding quotes exactly as the exported file had them.

4. Validate before generating:

   ```powershell
   python manage.py comfyui_check
   ```

   It reports whether the workflow JSON parses and whether ComfyUI is reachable.

## 4. Point the site at the pod

In `.env` (restart the server afterwards):

```env
INFERENCE_DEFAULT_ADAPTER=comfyui
COMFYUI_BASE_URL=https://<pod-id>-8188.proxy.runpod.net
COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_t2v_api.json
COMFYUI_API_TIMEOUT=1800
GENERATION_JOB_TIMEOUT_SECONDS=1800
```

Keep `CELERY_TASK_ALWAYS_EAGER=True` for single-user use; the request that starts
a generation will wait for the render, which is fine for short clips but means the
browser tab must stay open. For long renders run a Celery worker with Redis and
set `CELERY_TASK_ALWAYS_EAGER=False`.

## 5. Generate

1. Open `http://127.0.0.1:8000/generate/generate/`.
2. Enter a prompt, keep the duration short (2–4 s) for the first run.
3. Submit, keep the job page open (it polls), then download the MP4 from the job
   page or the library.

Film assembly works on the downloaded clips exactly as before.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| “No ComfyUI workflow is configured” | `COMFYUI_WORKFLOW_PATH` is empty or the file is missing |
| “Cannot connect to ComfyUI” | Pod stopped, wrong URL, or port 8188 not exposed |
| HTTP 502/504 from the proxy URL | Pod is still starting or the port is not ready |
| Workflow rejected (`node_errors`) | A model used by the workflow is not installed on the pod, or a token was placed in the wrong field |
| Render times out | Raise `COMFYUI_API_TIMEOUT` / `GENERATION_JOB_TIMEOUT_SECONDS`, or shorten duration/resolution |
| Generation fails after a pod restart | Pod URLs change; update `COMFYUI_BASE_URL` |

Cloud GPU usage costs real money. Stop the pod when you are not generating.
