# ComfyUI API workflows

## Shipped workflow: `wan22_t2v_api.json` (Wan 2.2 14B, powerful GPU)

A ready-made **Wan 2.2 text-to-video** API workflow lives in this folder and is
the value of `COMFYUI_WORKFLOW_PATH` in `.env`. It runs only when your ComfyUI
instance has these exact files installed:

| ComfyUI folder | File |
|---|---|
| `models/diffusion_models/` | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` |
| `models/diffusion_models/` | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` |
| `models/text_encoders/` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| `models/vae/` | `wan_2.1_vae.safetensors` |
| `models/loras/` | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` |
| `models/loras/` | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` |

The two LoRA files are the **lightx2v 4-step distillation LoRAs** (ComfyUI
Manager → search `lightx2v 4steps`; they are also fetched automatically by the
cloud template in `docs/cloud-gpu-comfyui.md`). They are what makes the render
fast:

| | Without LoRA | With LoRA (shipped default) |
|---|---|---|
| Sampling steps | 20 (10 high + 10 low) | **4 (2 high + 2 low)** |
| CFG | 3.5 (two forward passes per step) | **1.0** (distilled — one pass per step) |
| Model passes while sampling | ~40 | **4** |

Two-stage sampling runs euler/simple with shift 8 either way, and the result is
saved as MP4 through the `SaveVideo` node. `{{FRAMES}}` (duration × FPS,
rounded up to 4n+1) fills the latent `length`, so the duration set on the
generation form really controls the clip length.

Prefer the slower, higher-quality path? Delete nodes `105` and `106` (the two
`LoraLoaderModelOnly` nodes), point the `model` input of samplers `96`/`95`
back at `93`/`94`, and restore `steps: 20`, `cfg: 3.5`, `start_at_step` 0/10.

## Shipped workflow: `wan22_5b_t2v_api.json` (Wan 2.2 5B, free/weak GPUs)

A tokenised conversion of ComfyUI's official **Wan 2.2 TI2V-5B** template — for
free tiers (Colab T4) and any GPU where the 14B pair does not fit. ComfyUI
documents 5B as comfortable on 8 GB VRAM.

| ComfyUI folder | File |
|---|---|
| `models/diffusion_models/` | `wan2.2_ti2v_5B_fp16.safetensors` (9.31 GB) |
| `models/text_encoders/` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (6.27 GB) |
| `models/vae/` | `wan2.2_vae.safetensors` (1.31 GB) |

Points of difference from the 14B workflow: a single diffusion model (no
high/low-noise pair, no LoRA), the `Wan22ImageToVideoLatent` node (5B's VAE is
compressed 16x, so size inputs must be **divisible by 16**), and the official
`uni_pc` / cfg 5 / 20-step sampling. Select it with
`COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_5b_t2v_api.json`.
See [docs/colab-free-gpu.md](../../docs/colab-free-gpu.md) for the free Colab
setup that runs it.

## Your own workflow

Put workflows exported from ComfyUI with **Save (API Format)** in this folder and
point `COMFYUI_WORKFLOW_PATH` at the file, for example:

```env
COMFYUI_WORKFLOW_PATH=comfyui/workflows/my_workflow_api.json
```

Replace the prompt, negative prompt, seed, size, FPS, and duration values inside
the exported JSON with these tokens before using it:

`{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{FPS}}`,
`{{FRAMES}}`, `{{DURATION}}`, `{{SEED}}`, `{{MODEL_NAME}}`.

Keep the surrounding quotes on every token — the file then stays valid JSON,
so editors and linters parse it without errors. Quotes around numeric tokens
are stripped while rendering (`"length": "{{FRAMES}}"` → `"length": 49`), so
ComfyUI still receives a JSON number. The bare form (`"length": {{FRAMES}}`)
also works but makes the file invalid JSON on disk.

See `docs/cloud-gpu-comfyui.md` for the full setup walkthrough and
`python manage.py comfyui_check` to validate the file and the server URL.
