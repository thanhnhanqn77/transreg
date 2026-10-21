"""Standalone Ollama Client with API Key rotation and Thinking Policy enforcement."""
import json
import os
from pathlib import Path
import re
import socket
import time
from urllib import error, request

KEY_PATTERN = r"[a-f0-9]{32}\.[A-Za-z0-9_\-]+"


class AllKeysUnavailable(RuntimeError):
    pass


class OllamaClient:
    """Ollama API Client using standard Python urllib (no external requests dependency)."""
    def __init__(self, keys, model, host, timeout=300, max_tokens=8192,
                 retries=3, sleep=time.sleep, opener=request.urlopen):
        self.keys = list(dict.fromkeys(keys))
        if not self.keys:
            raise ValueError("No valid Ollama API keys found.")
        self.index = 0
        self.model, self.host = model, host.rstrip("/")
        self.timeout, self.max_tokens = timeout, max_tokens
        self.retries, self.sleep, self.opener = retries, sleep, opener
        self.url = self.host if self.host.endswith("/api/chat") else self.host + "/api/chat"

    def rotate(self, reason, attempt):
        old = self.index
        self.index = (self.index + 1) % len(self.keys)
        print(f"Key #{old + 1}: {reason}; switching to #{self.index + 1}/{len(self.keys)}.", flush=True)
        if attempt % len(self.keys) == 0:
            cycle = attempt // len(self.keys)
            self.sleep(min(5.0 * cycle, 30.0))
        else:
            self.sleep(1.0)

    def generate(self, prompt):
        data = self.generate_response(prompt)
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama returned empty final content.")
        if data.get("done_reason") == "length":
            if not re.search(r"\{\s*(true|false|unknown|uncertain)\s*\}", text, re.I):
                raise RuntimeError("Output truncated; increase --max_tokens.")
        return text

    def generate_response(self, prompt, max_tokens=None, think=False, system=None, response_format=None):
        tokens = self.max_tokens if max_tokens is None else max_tokens
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0,
                "num_predict": tokens,
                "num_ctx": max(self.max_tokens * 2, 32768)
            }
        }
        if think is not None:
            body['think'] = think
        if response_format is not None:
            body['format'] = response_format
        payload = json.dumps(body).encode()
        max_attempts = max(len(self.keys) * max(self.retries, 15), 60)
        
        for attempt in range(1, max_attempts + 1):
            req = request.Request(
                self.url,
                data=payload,
                headers={
                    "Authorization": "Bearer " + self.keys[self.index],
                    "Content-Type": "application/json"
                }
            )
            try:
                with self.opener(req, timeout=self.timeout) as response:
                    data = json.load(response)
                if data.get("error"):
                    self.rotate("API error in response", attempt)
                    continue
                return data
            except error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (401, 402, 403, 408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"Ollama HTTP {status}; check model/endpoint.") from None
                self.rotate(f"HTTP {status}", attempt)
            except (error.URLError, TimeoutError, socket.timeout):
                if attempt == max_attempts:
                    raise RuntimeError("Ollama network/timeout failure after retries.") from None
                self.sleep(1.0)
        raise AllKeysUnavailable(f"All {len(self.keys)} keys unavailable after {max_attempts} attempts.")


def read_keys(key_or_path=None):
    """Read API keys from direct token string, environment variable, or optional keys file."""
    tokens = []
    
    # 1. Environment variables (primary headless method)
    env = os.getenv("OLLAMA_API_KEYS", "") or os.getenv("OLLAMA_API_KEY", "")
    if env:
        tokens.extend(re.findall(KEY_PATTERN, env))

    # 2. Direct string or path passed in
    if key_or_path:
        if isinstance(key_or_path, (list, tuple)):
            for item in key_or_path:
                tokens.extend(re.findall(KEY_PATTERN, str(item)))
        elif isinstance(key_or_path, str):
            # Check if direct key string
            matched = re.findall(KEY_PATTERN, key_or_path)
            if matched:
                tokens.extend(matched)
            elif Path(key_or_path).is_file():
                content = Path(key_or_path).read_text(encoding="utf-8-sig")
                content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))
                tokens.extend(re.findall(KEY_PATTERN, content))

    # 3. Fallback: local keys.txt if present
    if not tokens and Path("keys.txt").is_file():
        content = Path("keys.txt").read_text(encoding="utf-8-sig")
        content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))
        tokens.extend(re.findall(KEY_PATTERN, content))

    keys = list(dict.fromkeys(tokens))
    if not keys:
        raise ValueError(
            "No Ollama tokens found via direct argument, OLLAMA_API_KEY environment variable, or keys.txt; "
            "expected 32 hex chars + '.' + token."
        )
    return keys

