"""Build tools/colab/comfyui_free_gpu.ipynb.

Run from the project root:

    python tools/colab/build_notebook.py

The smoke-test cell embeds a sample-value copy of
comfyui/workflows/wan22_5b_t2v_api.json so the notebook is self-contained.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = ROOT / "comfyui" / "workflows" / "wan22_5b_t2v_api.json"
OUT = Path(__file__).resolve().parent / "comfyui_free_gpu.ipynb"


def build() -> None:
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    # Sample values instead of {{TOKENS}} — this is what CineForge will send.
    wf["6"]["inputs"]["text"] = "A red fox running through deep snow, cinematic, slow motion"
    wf["7"]["inputs"]["text"] = "blurry, watermark, text, low quality"
    wf["55"]["inputs"].update({"width": 832, "height": 480, "length": 49})
    wf["3"]["inputs"]["seed"] = 42
    wf["57"]["inputs"]["fps"] = 24
    wf_json = json.dumps(wf, indent=2)

    cells = [
        ("markdown", MD_INTRO),
        ("code", CELL_GPU),
        ("code", CELL_INSTALL),
        ("code", CELL_MODELS),
        ("code", CELL_TUNNEL),
        ("code", CELL_SMOKE.format(workflow=wf_json)),
        ("markdown", MD_END),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "execution_count": None if kind == "code" else None,
                "outputs": [] if kind == "code" else None,
                "source": source,
            }
            for kind, source in cells
        ],
    }
    # nbformat: markdown cells have no execution_count/outputs
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            cell.pop("execution_count", None)
            cell.pop("outputs", None)

    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


MD_INTRO = """# CineForge AI — free GPU (Google Colab) + ComfyUI

This notebook gives CineForge a **free** ComfyUI backend, so it can generate real
AI videos with no local GPU and no paid GPU pod. Cost: $0.

Cells, in order:
1. Checks the free GPU (T4, 16 GB)
2. Installs ComfyUI
3. Downloads the Wan 2.2 **5B** model set (~18 GB, resumable and re-run safe)
4. Starts ComfyUI + a free **cloudflared** tunnel (no account, no API key) and prints the public URL
5. Smoke-tests the exact workflow CineForge submits

Then paste the printed URL into CineForge's `.env` — full steps in
`docs/colab-free-gpu.md`.

**Free-tier limits:** Google enforces quotas and session length; the runtime
disconnects when you hit them, which also kills the tunnel URL. Model files live
on the runtime disk, so after a disconnect re-run cell 3 (it skips anything it
already has) and paste the **new** URL into `.env`.
"""

CELL_GPU = """# 1) Free GPU check — we need a T4 (16 GB). If you got a CPU runtime,
#    menu: Runtime -> Change runtime type -> T4 GPU.
!nvidia-smi"""

CELL_INSTALL = """# 2) Install ComfyUI (torch is already present on Colab)
import os
import pathlib

if not pathlib.Path("/content/ComfyUI").exists():
    !git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI

os.chdir("/content/ComfyUI")
!pip install -q -r requirements.txt
print("ComfyUI ready at /content/ComfyUI")"""

CELL_MODELS = """# 3) Download the Wan 2.2 5B model set (~18 GB).
#    Resumable and idempotent: files that are already complete are skipped,
#    partial ones continue with `curl -C -`. Safe to re-run every session.
import pathlib
import subprocess

FILES = {
    "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors": (
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        9999658848,
    ),
    "vae/wan2.2_vae.safetensors": (
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        1409400960,
    ),
    "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors": (
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        6735906897,
    ),
}

root = pathlib.Path("/content/ComfyUI/models")
for rel, (url, size) in FILES.items():
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        print(f"have  {rel}")
        continue
    print(f"get   {rel}  ({size / 1e9:.2f} GB)")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
         "-o", str(dest), url],
        check=True,
    )
    got = dest.stat().st_size
    print(f"      {got / 1e9:.2f} GB" + ("" if got == size else "  !! size mismatch"))

print("models:", [p.name for p in root.rglob("*.safetensors")])"""

CELL_TUNNEL = """# 4) Start ComfyUI + a free cloudflared tunnel, then print the public URL
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

cloudflared = pathlib.Path("/usr/local/bin/cloudflared")
if not cloudflared.exists():
    subprocess.run(
        ["curl", "-L", "-o", str(cloudflared),
         "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
        check=True,
    )
    subprocess.run(["chmod", "+x", str(cloudflared)], check=True)

# clean up anything from a previous run of this cell
subprocess.run(["pkill", "-f", "main.py"], check=False)
subprocess.run(["pkill", "-f", "cloudflared"], check=False)
time.sleep(2)

subprocess.Popen(
    [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
    cwd="/content/ComfyUI",
    stdout=open("/content/comfyui.log", "a"),
    stderr=subprocess.STDOUT,
)
tunnel_log = pathlib.Path("/content/tunnel.log")
tunnel_log.write_text("")
subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8188"],
    stdout=tunnel_log.open("a"),
    stderr=subprocess.STDOUT,
)


def wait_for(url, tries=300, label="ComfyUI"):
    # First ComfyUI start on Colab can take well over two minutes (imports +
    # model scan), so poll patiently and say what is happening.
    for elapsed in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print(f"  {label} is up (waited {elapsed + 1}s)")
                    return True
        except Exception:
            pass
        if (elapsed + 1) % 15 == 0:
            print(f"  ...still waiting for {label} ({elapsed + 1}s)")
        time.sleep(1)
    return False


started = wait_for("http://127.0.0.1:8188/system_stats")
if not started:
    print("\\nComfyUI did NOT answer within 5 minutes. It may still be starting,")
    print("or it crashed — check the log below, then re-run this cell.")
    print("Last lines of /content/comfyui.log:")
    print(pathlib.Path("/content/comfyui.log").read_text(errors="ignore")[-3000:])

public = None
for _ in range(90):
    text = pathlib.Path("/content/tunnel.log").read_text(errors="ignore")
    match = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", text)
    if match:
        public = match.group(0)
        break
    time.sleep(1)

print("=" * 68)
print("Public ComfyUI URL:", public)
print("=" * 68)
if public:
    tunnel_ready = wait_for(f"{public}/system_stats", tries=60, label="Public tunnel")
    if not tunnel_ready:
        print("The tunnel URL is not returning ComfyUI yet. Check /content/tunnel.log:")
        print(tunnel_log.read_text(errors="ignore")[-2000:])
print("Paste into CineForge .env, then restart the CineForge server:")
print(f"  COMFYUI_BASE_URL={public}")
print("  COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_5b_t2v_api.json")
print("  COMFYUI_API_TIMEOUT=1800")"""

CELL_SMOKE = """# 5) Smoke test — POST the same workflow CineForge submits (sample values).
#    Keep this dict in sync with comfyui/workflows/wan22_5b_t2v_api.json.
#    If ComfyUI accepts it, a render starts; watch /content/comfyui.log.
import json
import urllib.error
import urllib.request

WORKFLOW = {workflow}

request = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({{"prompt": WORKFLOW, "client_id": "cineforge-smoke"}}).encode(),
    headers={{"Content-Type": "application/json"}},
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    print("queued prompt_id:", payload.get("prompt_id"))
    print("render running — follow it with:  tail -f /content/comfyui.log")
except urllib.error.HTTPError as error:
    print("REJECTED", error.code)
    print(error.read().decode(errors="ignore")[:3000])
except Exception as error:
    print("could not reach http://127.0.0.1:8188 —", error)
    print("Re-run cell 4 (it restarts ComfyUI and the tunnel).")
    print(pathlib.Path("/content/comfyui.log").read_text(errors="ignore")[-2000:])"""

MD_END = """## Connect CineForge

1. Copy the `COMFYUI_BASE_URL` printed above into your CineForge `.env`.
2. Also set:
   - `COMFYUI_WORKFLOW_PATH=comfyui/workflows/wan22_5b_t2v_api.json`
   - `COMFYUI_API_TIMEOUT=1800`
3. Restart the CineForge server and run `python manage.py comfyui_check` — it must
   print **Workflow OK** and report ComfyUI as reachable.
4. Generate from `/generate/generate/`. Width and height must be divisible by
   **16** (the default 832x480 is fine); duration is capped at 30 s.

## When it stops

- Leave the Colab tab open. Free-tier quota or idle limits disconnect the runtime
  and kill the tunnel URL; re-run cells 1-4, paste the **new** URL into `.env`
  and restart the CineForge server.
- To avoid re-downloading ~18 GB every session, keep a copy of
  `ComfyUI/models` on Google Drive and copy it back in cell 3 (optional).
- Done? Runtime -> Disconnect and delete runtime. Nothing is ever billed.
"""

if __name__ == "__main__":
    build()