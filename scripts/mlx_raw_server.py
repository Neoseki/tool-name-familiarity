"""Minimal raw-completion server for local testing on Apple Silicon. Stand-in for vLLM.

mlx_lm.server strips tool-call and reasoning text out of /v1/completions output
(mlx-lm 0.32.2), which is hidden processing we rule out (plan 3.7). This server does
only: tokenize the prompt as given, greedy-decode, stop on EOS / stop strings / max_tokens,
return the full decoded text.

uv run --no-project --with mlx-lm==0.32.2 python scripts/mlx_raw_server.py \
    --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()

    model, tokenizer = load(snapshot_download(a.model, revision=a.revision))

    def complete(body: dict) -> dict:
        if body.get("temperature", 0) != 0:
            raise ValueError("this server only does greedy decoding (temperature=0)")
        add_special = body.get("add_special_tokens", True)
        prompt_ids = tokenizer.encode(body["prompt"], add_special_tokens=add_special)
        stops = body.get("stop") or []
        text, finish, n = "", "length", 0
        for r in stream_generate(model, tokenizer, prompt_ids, max_tokens=body["max_tokens"]):
            text += r.text
            n = r.generation_tokens
            hit = [s for s in stops if s in text]
            if hit:
                text = text[: min(text.index(s) for s in hit)]
                finish = "stop"
                break
            if r.finish_reason:
                finish = r.finish_reason
                break
        return {
            "object": "text_completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": n},
        }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send(200, {"status": "ok"} if self.path == "/health" else {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._send(404, {"error": "only /v1/completions"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            start = time.monotonic()
            try:
                out = complete(body)
            except Exception as e:  # report, never retry
                return self._send(400, {"error": str(e)})
            print(f"completion: {out['usage']} {out['choices'][0]['finish_reason']} {time.monotonic() - start:.1f}s", flush=True)
            self._send(200, out)

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"serving {a.model}@{a.revision} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
