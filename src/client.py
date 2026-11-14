"""Raw-text completion client and JSONL logger (plan section 3.7).

Only /v1/completions with a prompt string: no chat endpoint, no server-side parsing,
no retries. Every sampling setting is sent on every request.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .models import ModelConfig


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    request: dict  # body as sent, minus the prompt (logged separately)
    latency_s: float


def sampling_for(cfg: ModelConfig, seed: int) -> dict:
    return {
        "temperature": 0,  # greedy; see plan decision 7
        "seed": seed,
        "max_tokens": cfg.max_tokens,
        "stop": list(cfg.stop),
        "skip_special_tokens": False,
        # The rendered prompt already holds any BOS the template wants; vLLM would add another.
        "add_special_tokens": False,
    }


class CompletionClient:
    def __init__(self, endpoint: str, server_model: str, sampling: dict, timeout_s: float = 900):
        self.url = endpoint.rstrip("/") + "/v1/completions"
        self.server_model = server_model
        self.sampling = sampling
        self.http = httpx.Client(timeout=timeout_s)

    def __call__(self, prompt: str) -> Completion:
        body = {"model": self.server_model, "prompt": prompt, **self.sampling}
        start = time.monotonic()
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") not in ("stop", "length"):
            # e.g. "tool_calls": the server parsed the output itself, which we rule out.
            raise RuntimeError(f"unexpected finish_reason {choice.get('finish_reason')!r}: server-side processing?")
        request = {k: v for k, v in body.items() if k != "prompt"}
        return Completion(choice["text"], choice.get("finish_reason"), request, time.monotonic() - start)


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
