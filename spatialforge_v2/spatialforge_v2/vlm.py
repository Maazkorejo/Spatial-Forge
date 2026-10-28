"""
Thin VLM adapters. Each takes (system, user, images_png_bytes, schema, temperature) and returns raw text.
Model names and free-tier availability change often: verify the model id with your provider.
"""
from __future__ import annotations

import base64
from typing import Optional, Protocol

import requests


class VLM(Protocol):
    def complete(self, system: str, user: str, images: list[bytes],
                 schema: Optional[dict] = None, temperature: float = 0.2) -> str: ...


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class OllamaVLM:
    """Local fallback. `format=<json schema>` makes Ollama enforce the schema while decoding."""

    def __init__(self, model: str = "llama3.2-vision", host: str = "http://localhost:11434", timeout: int = 180):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def complete(self, system, user, images, schema=None, temperature=0.2):
        body = {
            "model": self.model, "stream": False,
            "format": schema if schema else "json",
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [_b64(i) for i in images]},
            ],
        }
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatVLM:
    """Groq / OpenRouter / vLLM / any OpenAI-compatible chat endpoint with image support.
    Set strict_schema=True only if the model+provider support response_format=json_schema."""

    def __init__(self, base_url: str, api_key: str, model: str, strict_schema: bool = False, timeout: int = 120):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.strict_schema, self.timeout = strict_schema, timeout

    def complete(self, system, user, images, schema=None, temperature=0.2):
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(i)}} for i in images]
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        if schema and self.strict_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "assembly", "schema": schema, "strict": True}}
        else:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class MockVLM:
    """Returns canned outputs in order; for tests without a model."""

    def __init__(self, outputs: list[str]):
        self.outputs, self.i = outputs, 0

    def complete(self, system, user, images, schema=None, temperature=0.2):
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return out
