"""Run one model x one condition over a list of task ids.

uv run python -m src.run --model qwen3-local --condition original \
    --ids tasks/dev_ids.json --endpoint http://127.0.0.1:8080 --server-model Qwen/Qwen3-1.7B
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from . import bfcl_io
from .client import CompletionClient, JsonlLog, sampling_for
from .loop import run_entry
from .mapping import NameMap
from .models import MODELS
from .render import load_tokenizer
from .score import score_entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--condition", required=True, choices=["original"])  # twin: step 3
    p.add_argument("--ids", required=True, type=Path, help="JSON list of task ids")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--server-model", required=True, help="model name the server expects")
    p.add_argument("--server", required=True, help="server name and version, logged, e.g. vllm==0.x or mlx-lm==0.32.2")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/raw"))
    a = p.parse_args()

    cfg = MODELS[a.model]
    ids = json.loads(a.ids.read_text())
    entries = bfcl_io.load_entries(ids)
    gts = bfcl_io.load_ground_truth(ids)
    tokenizer = load_tokenizer(cfg)
    sampling = sampling_for(cfg, a.seed)
    client = CompletionClient(a.endpoint, a.server_model, sampling)
    run_name = f"{cfg.slug}_{a.condition}_{a.run_id}"
    log = JsonlLog(a.out / f"{run_name}.jsonl")

    base = {
        "run": run_name, "model": cfg.slug, "hf_repo": cfg.hf_repo, "model_revision": cfg.revision,
        "condition": a.condition, "server": a.server, "bfcl_eval": importlib.metadata.version("bfcl-eval"),
        "transformers": importlib.metadata.version("transformers"), "host": platform.platform(),
    }
    passed = 0
    for tid in ids:
        entry = entries[tid]
        names = NameMap.identity(bfcl_io.original_tools(entry))
        bfcl_io.reset_state(run_name, entry)
        r = run_entry(entry, cfg, tokenizer, client, names, run_name, log.write, base)
        verdict = score_entry(entry, r.decoded, gts[tid], run_name, r.force_quit)
        passed += bool(verdict["valid"])
        log.write({**base, "type": "entry", "task_id": tid, "valid": verdict["valid"], "checker": verdict,
                   "decoded": r.decoded, "steps": r.steps, "rejected": r.rejected, "intrusions": r.intrusions,
                   "force_quit": r.force_quit})
        print(f"{tid}: {'PASS' if verdict['valid'] else 'FAIL ' + verdict.get('error_type', '')} "
              f"(steps={r.steps}, rejected={r.rejected})")
    log.close()
    print(f"{passed}/{len(ids)} passed -> {a.out / (run_name + '.jsonl')}")


if __name__ == "__main__":
    main()
