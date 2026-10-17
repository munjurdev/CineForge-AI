# ComfyUI API workflows

Put workflows exported from ComfyUI with **Save (API Format)** in this folder and
point `COMFYUI_WORKFLOW_PATH` at the file, for example:

```env
COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_t2v_api.json
```

Replace the prompt, negative prompt, seed, size, FPS, and duration values inside
the exported JSON with these tokens before using it:

`{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{FPS}}`,
`{{DURATION}}`, `{{SEED}}`, `{{MODEL_NAME}}`.

See `docs/cloud-gpu-comfyui.md` for the full setup walkthrough and
`python manage.py comfyui_check` to validate the file and the server URL.
