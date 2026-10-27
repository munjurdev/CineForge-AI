# Free AI video generation with Google Colab (no GPU, $0)

This machine has no CUDA GPU, and paid GPU pods cost money. This walkthrough
gives CineForge a **free** ComfyUI backend on Google Colab's free T4 GPU, so
real AI video generation works without buying or renting anything.

- Notebook: [`tools/colab/comfyui_free_gpu.ipynb`](../tools/colab/comfyui_free_gpu.ipynb)
- Workflow: [`comfyui/workflows/wan22_5b_t2v_api.json`](../comfyui/workflows/wan22_5b_t2v_api.json)

## Why the 5B model

The default shipped workflow targets Wan 2.2 **A14B** (two 14B experts), which
needs a real 24 GB GPU. Colab's free tier gives a **T4 (16 GB)**, which is a
poor fit for 14B but comfortable for Wan 2.2 **TI2V-5B** — ComfyUI documents
that it works well on 8 GB VRAM with native offloading. So the free route uses:

| File | Goes in | Size |
|---|---|---|
| `wan2.2_ti2v_5B_fp16.safetensors` | `models/diffusion_models/` | 9.31 GB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` | 6.27 GB |
| `wan2.2_vae.safetensors` | `models/vae/` | 1.31 GB |

Total ~18 GB, downloaded by the notebook (resumable, skipped when already
present).

## 1. Run the notebook

1. Open [Colab](https://colab.research.google.com/) with a free Google account.
2. **Upload** `tools/colab/comfyui_free_gpu.ipynb` (File → Upload notebook), or
   create a blank notebook and paste the cells from it.
3. If Colab gives you a CPU runtime: Runtime → Change runtime type → **T4 GPU**.
4. Run the cells in order:
   - **1** GPU check (`nvidia-smi` must show a T4)
   - **2** installs ComfyUI
   - **3** downloads the three model files (~18 GB, first run only)
   - **4** starts ComfyUI and a **cloudflared** tunnel, then prints a
     `https://….trycloudflare.com` URL
   - **5** smoke-tests the exact workflow CineForge will submit — it must print
     `queued prompt_id: …`, not `REJECTED`

Leave the Colab tab open — the runtime (and the tunnel) die when it closes.

## 2. Point CineForge at it

Edit `.env` and restart the server:

```env
COMFYUI_BASE_URL=https://<the-url-printed-by-cell-4>.trycloudflare.com
COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_5b_t2v_api.json
COMFYUI_API_TIMEOUT=1800
GENERATION_JOB_TIMEOUT_SECONDS=1800
```

Then validate. CineForge checks ComfyUI's `/system_stats` API route:

```powershell
python manage.py comfyui_check
```

It must print **Workflow OK** *and* reach ComfyUI. If it says "Cannot reach
ComfyUI", the runtime disconnected — re-run notebook cells 1-4 and paste the
**new** URL (it changes every run).

## 3. Generate

1. Open `http://127.0.0.1:8000/generate/generate/`.
2. Enter a prompt. Keep the first runs small: **832x480**, 2-3 s, 24 fps.
   The bundled 5B workflow uses its documented **20 steps / CFG 5** baseline;
   higher step counts take longer and do not guarantee a better result.
3. Submit and keep the tab open — CineForge polls until the MP4 arrives, then it
   lands in the library as usual.

Rules for this workflow: **width and height must be divisible by 16** (832x480
is fine), duration ≤ 30 s, and longer clips cost more T4 time.

## Limits and safety

| | |
|---|---|
| Cost | $0 — no card, no credits, nothing is billed |
| Quota | Google enforces free-tier GPU hours per day; when you hit it the runtime stops |
| Session | Runtime disconnects when idle or over quota; the tunnel URL then dies |
| Models | Live on the runtime disk — re-download after a disconnect unless you keep them on Google Drive |
| Speed | T4 + 5B is minutes per short clip, not seconds |

**The tunnel URL is public and unauthenticated.** Anyone who has it can queue
jobs on your free GPU. It is a random URL you never share, but treat it as
secret: don't post it, and stop the runtime when you are done
(Runtime → Disconnect and delete runtime).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot reach ComfyUI` | Runtime disconnected or cell 4 not re-run — get the new URL |
| `REJECTED … node_errors` in cell 5 | Model file missing or misnamed — re-run cell 3 and compare the printed file list |
| `lora not found` | You are pointing at the 14B workflow — use `wan22_5b_t2v_api.json`, or install the files the error names |
| 502/504 from the tunnel | Tunnel still starting — wait a few seconds and re-run `comfyui_check` |
| CUDA out of memory | Lower resolution (e.g. 640x368) or shorten the clip |
| Render times out | `COMFYUI_API_TIMEOUT` / `GENERATION_JOB_TIMEOUT_SECONDS` are 1800 s by default; raise them or shorten the clip |
| Generation worked yesterday, not today | Colab sessions are ephemeral — re-run cells 1-4, new URL |

## Other routes

- Paid cloud GPU (RunPod + the 14B workflow): [`docs/cloud-gpu-comfyui.md`](cloud-gpu-comfyui.md)
- No GPU at all, want a non-AI clip to exercise the pipeline: set
  `INFERENCE_DEFAULT_ADAPTER=test` in `.env` (CPU demo via FFmpeg)