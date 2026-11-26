"""Unified LLM client: Gemini or OpenAI, JSON responses, on-disk cache, retries.

Keys: GEMINI_API_KEY (or GOOGLE_API_KEY) / OPENAI_API_KEY, from the environment or a
`data_gen/.env` file (KEY=value lines, git-ignored). provider="none" disables calls so
the pipeline runs template-only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}


def load_dotenv(path: Path | None = None) -> None:
    p = path or Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLMClient:
    def __init__(self, provider: str, model: str | None = None,
                 cache_dir: str | Path = "output/llm_cache", max_retries: int = 4):
        load_dotenv()
        self.provider = provider.lower()
        if self.provider not in ("gemini", "openai", "none"):
            raise ValueError(f"provider must be gemini|openai|none, got {provider}")
        self.model = model or DEFAULT_MODELS.get(self.provider, "")
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    # ------------------------------------------------------------- public --

    def complete_json(self, prompt: str, image_png: bytes | None = None) -> dict | None:
        """Return parsed JSON dict, or None on unrecoverable failure / provider none."""
        if not self.enabled:
            return None
        key = self._cache_key(prompt, image_png)
        cached = self._cache_get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                text = self._call(prompt, image_png)
                obj = _extract_json(text)
                self._cache_put(key, obj, prompt)
                self.stats["calls"] += 1
                return obj
            except Exception as e:  # rate limits, transient net, malformed JSON
                if attempt == self.max_retries - 1:
                    self.stats["failures"] += 1
                    print(f"[llm] giving up after {self.max_retries} tries: {e}")
                    return None
                time.sleep(delay)
                delay *= 2

    # ---------------------------------------------------------- providers --

    def _call(self, prompt: str, image_png: bytes | None) -> str:
        if self.provider == "gemini":
            return self._call_gemini(prompt, image_png)
        return self._call_openai(prompt, image_png)

    def _call_gemini(self, prompt: str, image_png: bytes | None) -> str:
        from google import genai
        from google.genai import types
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise SystemExit("GEMINI_API_KEY / GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        contents: list = [prompt]
        if image_png is not None:
            contents.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
        resp = self._client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        return resp.text

    def _call_openai(self, prompt: str, image_png: bytes | None) -> str:
        from openai import OpenAI
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY not set")
            self._client = OpenAI()
        content: list | str = prompt
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"})
        return resp.choices[0].message.content

    # -------------------------------------------------------------- cache --

    def _cache_key(self, prompt: str, image_png: bytes | None) -> str:
        h = hashlib.sha256()
        h.update(self.provider.encode())
        h.update(self.model.encode())
        h.update(prompt.encode())
        if image_png:
            h.update(image_png)
        return h.hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())["response"]
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, obj: dict, prompt: str) -> None:
        p = self.cache_dir / f"{key}.json"
        p.write_text(json.dumps(
            {"provider": self.provider, "model": self.model,
             "prompt_head": prompt[:200], "response": obj}, indent=1))


def _extract_json(text: str) -> dict:
    """Parse JSON, tolerating markdown fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:120]!r}")
    return json.loads(t[start:end + 1])
