#!/usr/bin/env python3
"""A blind labelling UI for the rendered human sketches.

Shows one sketch and its deictic caption and asks the question the validator is
asked — is this pair coherent and unambiguous — not the question the geometric
label answers, which is whether the drawing matches the demonstration.

Deliberately withheld while labelling, because either would anchor the answer:
  * the geometric verdict (`sketch_ok`)
  * the instruction the participant was actually shown
  * what any model predicted

Labels append to labels.jsonl, so the session resumes and nothing is overwritten.
Order is shuffled with a fixed seed, so a resumed run continues the same sequence.

    python labeling/label_server.py --port 8791
"""

from __future__ import annotations

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from urllib.parse import urlparse

def _human_eval_root() -> Path:
    """The human-evaluation working directory: holds out/ and the label files.

    $HUMAN_EVAL_ROOT wins. Otherwise the `human_eval/` folder next to this repo (the lab
    layout is <x>/DrawVLA/{data_gen,human_eval}), then one inside the repo.
    """
    env = os.environ.get("HUMAN_EVAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]          # this file lives in <repo>/labeling/
    for cand in (repo.parent / "human_eval", repo / "human_eval"):
        if cand.is_dir():
            return cand
    return repo.parent / "human_eval"


ROOT = _human_eval_root()
SET = ROOT / "out" / "human-eval"          # overridden by --set
LABELS = ROOT / "out" / "labels.jsonl"     # overridden by --labels

CRITERION = (
    "Answer <b>right</b> only when the circle, arrow, image and caption are mutually "
    "consistent and leave exactly one actionable interpretation. Answer <b>wrong</b> "
    "when any required cue is missing, malformed, inconsistent, detached, ambiguous, "
    "or not visibly groundable. Use only visible evidence."
)

PAGE = """<!doctype html><meta charset=utf-8><title>Sketch labelling</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 Inter,system-ui,sans-serif;color:#17201e;background:#f8f6f1;
     display:grid;place-items:center;min-height:100vh}
.wrap{width:min(1000px,calc(100% - 32px));display:grid;grid-template-columns:1fr 340px;gap:24px;align-items:start}
.card{background:#fff;border:1px solid #e2e6e1;border-radius:18px;padding:20px;
      box-shadow:0 18px 50px rgba(31,46,40,.08)}
img{width:100%;image-rendering:auto;border-radius:12px;display:block;background:#222}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.02em}
.cap{font-size:26px;font-weight:750;letter-spacing:-.02em;margin:6px 0 12px}
.cur{font-size:13px;font-weight:700;padding:7px 11px;border-radius:9px;margin-bottom:14px;
     display:flex;justify-content:space-between;align-items:center}
.cur.right{background:#e6f3ec;color:#0f5c38}.cur.wrong{background:#fbe9e7;color:#8f2c1e}
.cur.unsure{background:#eceee9;color:#5d6763}.cur.none{background:#f4f5f2;color:#8a938f}
.crit{font-size:12.5px;color:#5d6763;background:#f2f5f2;border-left:3px solid #147a4a;
      padding:10px 12px;border-radius:0 8px 8px 0;margin-bottom:16px}
button{width:100%;min-height:44px;margin-bottom:8px;border:0;border-radius:11px;
       font:inherit;font-weight:730;cursor:pointer}
.r{background:#147a4a;color:#fff}.w{background:#c0392b;color:#fff}
.u{background:#eceee9}.k{background:#17201e;color:#fff}
.nav{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.nav button{background:#f0f2ee;min-height:36px;font-size:13px;font-weight:650}
.sel{outline:3px solid #17201e;outline-offset:2px}
kbd{background:#f3f4f1;color:#17201e;border:1px solid #d7dbd6;border-bottom-width:2px;
    border-radius:5px;padding:1px 5px;font-size:11px}
.meta{color:#66706d;font-size:13px;margin-top:12px}
.bar{height:6px;background:#e3e7e2;border-radius:99px;overflow:hidden;margin-top:8px}
.bar span{display:block;height:100%;background:#147a4a;width:0}
</style>
<div class=wrap>
  <div class=card><img id=img alt="human sketch"></div>
  <div class=card>
    <h1>Is this a valid prompt?</h1>
    <div class=cap id=cap>...</div>
    <div class="cur none" id=cur><span id=curtext>not labelled yet</span><span id=pos></span></div>
    <div class=crit>__CRITERION__</div>
    <button class=r id=bR onclick="send('right')">Right <kbd>R</kbd></button>
    <button class=w id=bW onclick="send('wrong')">Wrong <kbd>W</kbd></button>
    <button class=u id=bU onclick="send('unsure')">Unsure <kbd>U</kbd></button>
    <button class=k onclick="step(1)">Keep &amp; next <kbd>space</kbd></button>
    <div class=nav>
      <button onclick="step(-1)">&larr; Back <kbd>B</kbd></button>
      <button onclick="nextUnlabelled()">Next unlabelled <kbd>N</kbd></button>
    </div>
    <div class=meta id=meta></div>
    <div class=bar><span id=barfill></span></div>
  </div>
</div>
<script>
let i=0, total=0, shown=0, cur=null;
async function show(idx){
  const r = await (await fetch('/api/item?i='+idx)).json();
  i=r.index; total=r.total; cur=r.response_id; shown=performance.now();
  document.getElementById('img').src = r.image + '?v=' + cur;
  document.getElementById('cap').textContent = '\u201c' + r.caption + '\u201d';
  const box=document.getElementById('cur');
  box.className = 'cur ' + (r.current_label || 'none');
  document.getElementById('curtext').textContent =
      r.current_label ? ('current label: ' + r.current_label) : 'not labelled yet';
  document.getElementById('pos').textContent = (r.index+1) + ' / ' + r.total;
  for (const [k,b] of [['right','bR'],['wrong','bW'],['unsure','bU']])
     document.getElementById(b).classList.toggle('sel', r.current_label===k);
  document.getElementById('meta').textContent = r.labelled + ' of ' + r.total + ' labelled';
  document.getElementById('barfill').style.width = (100*r.labelled/r.total) + '%';
}
async function send(label){
  if(cur===null) return;
  await fetch('/api/label',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({response_id:cur,label:label,ms:Math.round(performance.now()-shown)})});
  step(1);
}
function step(d){ show(Math.max(0, Math.min(total-1, i+d))); }
async function nextUnlabelled(){
  const r = await (await fetch('/api/next_unlabelled?from='+(i+1))).json();
  if(r.index===null){ alert('everything is labelled'); return; }
  show(r.index);
}
addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='r')send('right'); else if(k==='w')send('wrong'); else if(k==='u')send('unsure');
  else if(k===' '||k==='enter'){e.preventDefault();step(1);}
  else if(k==='b'||k==='arrowleft')step(-1);
  else if(k==='arrowright')step(1);
  else if(k==='n')nextUnlabelled();
});
show(0);
</script>
"""


def load_items(seed: int) -> list[dict]:
    rows = [json.loads(l) for l in (SET / "human_sketches.jsonl").open(encoding="utf-8")]
    items = [{"response_id": r["response_id"], "caption": r["caption"],
              "image": "/images/" + Path(r["image"]).name} for r in rows]
    random.Random(seed).shuffle(items)
    return items


def labelled_ids() -> dict[str, str]:
    if not LABELS.exists():
        return {}
    out = {}
    for line in LABELS.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["response_id"]] = d["label"]
    return out


def make_handler(items: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            body = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE.replace("__CRITERION__", CRITERION), "text/html; charset=utf-8")
            if path.startswith("/images/"):
                name = Path(path).name
                target = SET / "images" / name
                if not target.is_file() or target.parent != (SET / "images"):
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, target.read_bytes(), "image/png")
            if path in ("/api/item", "/api/next"):
                from urllib.parse import parse_qs
                done = labelled_ids()
                query = parse_qs(urlparse(self.path).query)
                if path == "/api/next":          # kept for the old client
                    remaining = [i for i in items if i["response_id"] not in done]
                    if not remaining:
                        return self._send(200, json.dumps({"done": True, "labelled": len(done)}))
                    index = items.index(remaining[0])
                else:
                    try:
                        index = int(query.get("i", ["0"])[0])
                    except ValueError:
                        index = 0
                index = max(0, min(len(items) - 1, index))
                item = dict(items[index])
                item.update({"index": index, "total": len(items), "labelled": len(done),
                             "current_label": done.get(item["response_id"]), "done": False})
                return self._send(200, json.dumps(item))
            if path == "/api/next_unlabelled":
                from urllib.parse import parse_qs
                done = labelled_ids()
                start_at = int(parse_qs(urlparse(self.path).query).get("from", ["0"])[0] or 0)
                order = list(range(start_at, len(items))) + list(range(0, start_at))
                found = next((j for j in order if items[j]["response_id"] not in done), None)
                return self._send(200, json.dumps({"index": found}))
            if path == "/api/stats":
                done = labelled_ids()
                counts = {}
                for v in done.values():
                    counts[v] = counts.get(v, 0) + 1
                return self._send(200, json.dumps({"labelled": len(done), "total": len(items),
                                                   "counts": counts}))
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if path == "/api/label":
                d = json.loads(raw)
                if d.get("label") not in ("right", "wrong", "unsure"):
                    return self._send(400, json.dumps({"error": "bad label"}))
                LABELS.parent.mkdir(parents=True, exist_ok=True)
                with LABELS.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"response_id": d["response_id"], "label": d["label"],
                                         "ms": int(d.get("ms") or 0),
                                         "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                return self._send(200, json.dumps({"ok": True}))
            if path == "/api/undo":
                if LABELS.exists():
                    lines = [l for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]
                    LABELS.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""), encoding="utf-8")
                return self._send(200, json.dumps({"ok": True}))
            return self._send(404, b"not found", "text/plain")
    return Handler


def main() -> int:
    global SET, LABELS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--seed", type=int, default=7, help="shuffle seed; keep it to resume the same order")
    ap.add_argument("--set", type=Path, default=SET, dest="dataset",
                    help="rendered set to label (dir with human_sketches.jsonl + images/)")
    ap.add_argument("--labels", type=Path, default=LABELS,
                    help="labels file; use a fresh one per dataset")
    args = ap.parse_args()
    SET, LABELS = args.dataset, args.labels
    items = load_items(args.seed)
    done = labelled_ids()
    print(f"set: {SET}")
    print(f"labels: {LABELS}")
    print(f"{len(items)} sketches, {len(done)} already labelled -> http://localhost:{args.port}")
    print("keys: R right  W wrong  U unsure  space keep & next  B back  N next unlabelled")
    print("re-labelling just appends; the latest label for a sketch wins")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(items)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
