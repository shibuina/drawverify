#!/usr/bin/env python
"""Local web demo of the DrawVerify validators: draw a circle and an arrow on a LIBERO frame,
pick a caption, and see what each trained model says.

  demo/run.sh                 # creates the venv once, then serves http://127.0.0.1:8765
  demo/run.sh --selftest      # load the models, score one drawn sketch, print the timings

Strokes are drawn by `libero_sketch.render.render_human` (the look the models were trained on)
and scored by `training/validator.score` without generating a rationale (the paper's direct
readout). Adapters live in output/adapters/<name>/ (adapter_config.json + adapter_model.safetensors,
copied from the training runs); frames come from output/extracted (stage 01).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # the 8-bit copy leaves 0.6 GB on a 4 GB card
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "training")]
from libero_sketch import render as R  # noqa: E402
import validator as V  # noqa: E402

BASE = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTERS = REPO / "output" / "adapters"
# (adapter dir, name on the page, one-line description); the first one is the paper's final model
MODELS = [("2b-full4-ans", "DrawVerify (answer only)", "trained on generated sketches, answer only"),
          ("2b-full4", "Explain, then answer", "trained on generated sketches with a rationale target"),
          ("2b-full4-hum-s0", "Explain, then answer + 144 human sketches", "the row above, fine-tuned on human sketches")]
LOCK = threading.Lock()   # one sketch on the GPU at a time


def scenes() -> list[dict]:
    """Four first frames per suite, spread over its tasks."""
    out = []
    for suite in sorted((REPO / "output" / "extracted").iterdir()):
        tasks = sorted(p for p in suite.iterdir() if p.is_dir())
        for task in tasks[::max(1, len(tasks) // 4)][:4]:
            frame = task / "demo_0" / "frames" / "t0_rgb.png"
            if frame.exists():
                out.append({"path": frame, "task": task.name.split("_", 1)[1].replace("_", " ")})
    return out


def load(bits: int | None):
    """bits: 0 = the training precision (bf16, about 5 GB of GPU memory), 8 or 4 = a bitsandbytes copy
    (3.1 / 2.6 GB), None = pick by the free GPU memory. A model split between GPU and CPU works but took
    76 s per sketch on a 4 GB laptop, so a small card gets a quantized copy instead. Against the paper's
    bf16 runs on human test sketches, the 8-bit copy gives the same verdict on 98 / 95 / 100 % (n = 60,
    the three MODELS in order) and the 4-bit copy on 97 / 87 / 95 % (n = 150)."""
    import torch
    from peft import PeftModel
    if bits is None:
        free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
        bits = 0 if free >= 6 or not torch.cuda.is_available() else 8 if free >= 3.3 else 4
    processor = V.load_processor(BASE)
    if bits:
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
             if bits == 4 else BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForImageTextToText.from_pretrained(BASE, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0})
    else:
        model = V.load_model(BASE)
    for i, (name, _, _) in enumerate(MODELS):
        path = str(ADAPTERS / name)
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    return model.eval(), processor, bits


def stroke(value, required: bool):
    """A pointer path from the page -> clipped [[x, y], ...] in 0..1, or None."""
    if value is None and not required:
        return None
    if not isinstance(value, list) or not 2 <= len(value) <= 4000:
        raise ValueError("a stroke needs 2 to 4000 points")
    pts = np.asarray(value, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("stroke points must be [x, y] numbers")
    return np.clip(pts, 0.0, 1.0)


def check(model, processor, frame: Path, circle, arrow, caption: str) -> dict:
    caption = " ".join(str(caption).split())
    if not 1 <= len(caption) <= 80:
        raise ValueError("the caption needs 1 to 80 characters")
    rgb = np.asarray(Image.open(frame).convert("RGB"))
    img = Image.fromarray(R.render_human(rgb, stroke(circle, True), stroke(arrow, False)))
    results, t0 = [], time.time()
    with LOCK:
        for name, label, note in MODELS:
            model.set_adapter(name)
            p = V.score(model, processor, [{"image": img, "caption": caption}], max_new_tokens=0)[0]["p_wrong"]
            results.append({"label": label, "note": note, "p_wrong": p})
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "caption": caption, "results": results, "seconds": round(time.time() - t0, 2)}


def serve(model, processor, bits: int, port: int) -> None:
    sc, page = scenes(), Path(__file__).parent / "index.html"
    info = {"scenes": [{"id": i, "task": s["task"]} for i, s in enumerate(sc)], "bits": bits}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def json(self, code: int, obj) -> None:
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                return self.send(200, page.read_bytes(), "text/html; charset=utf-8")   # read per request: page edits need no restart
            if self.path == "/api/scenes":
                return self.json(200, info)
            if self.path.startswith("/scene/"):
                i = self.path[7:].split(".")[0]
                if i.isdigit() and int(i) < len(sc):       # an index, never a path from the client
                    return self.send(200, sc[int(i)]["path"].read_bytes(), "image/png")
            self.json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/check":
                return self.json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n < 2_000_000:
                    raise ValueError("request too large")
                d = json.loads(self.rfile.read(n))
                i = d.get("scene")
                if not isinstance(i, int) or not 0 <= i < len(sc):
                    raise ValueError("unknown scene")
                self.json(200, check(model, processor, sc[i]["path"], d.get("circle"), d.get("arrow"), d.get("caption", "")))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self.json(400, {"error": str(e)})

        def log_message(self, *a):   # keep the terminal for the timings
            pass

    print(f"DrawVerify demo: http://127.0.0.1:{port}   ({len(sc)} scenes, {len(MODELS)} models; Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def selftest(model, processor) -> None:
    """One drawn sketch through the whole path: payload -> render -> every model."""
    t = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([0.5 + 0.08 * np.cos(t), 0.4 + 0.08 * np.sin(t)], 1).tolist()
    arrow = np.stack([np.linspace(0.58, 0.8, 30), np.linspace(0.4, 0.3, 30)], 1).tolist()
    try:
        stroke([[0, 0]], True)
        raise SystemExit("a one-point stroke was accepted")
    except ValueError:
        pass
    frame = scenes()[0]["path"]
    for cap, arr in (("put this there", arrow), ("pick up this", None)):
        out = check(model, processor, frame, circle, arr, cap)
        assert all(0.0 <= r["p_wrong"] <= 1.0 for r in out["results"]) and out["image"].startswith("data:image/png")
        print(f'"{cap}", arrow={arr is not None}: {out["seconds"]} s  ' + "  ".join(f'{r["label"]}: {r["p_wrong"]:.3f}' for r in out["results"]))
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bits", type=int, choices=(0, 4, 8), help="0 = training precision, 4 / 8 = quantized; default: by free GPU memory")
    a = ap.parse_args()
    missing = [n for n, _, _ in MODELS if not (ADAPTERS / n / "adapter_model.safetensors").exists()]
    if missing:
        raise SystemExit(f"missing adapters under {ADAPTERS}: {missing}")
    t0 = time.time()
    model, processor, bits = load(a.bits)
    print(f"models loaded in {time.time() - t0:.0f} s ({f'{bits}-bit copy' if bits else 'training precision'})", flush=True)
    selftest(model, processor) if a.selftest else serve(model, processor, bits, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
