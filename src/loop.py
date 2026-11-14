"""Multi-turn loop mirroring BFCL's inference_multi_turn_FC (bfcl-eval 2026.3.23).

Per turn: add the user message(s), then step until the model replies without a call,
its output fails to parse (BFCL ends the turn on a decode error), or the step counter
exceeds MAXIMUM_STEP_LIMIT (counter resets each turn; exceeding it force-quits the entry).
"""

from dataclasses import dataclass, field
from typing import Callable

from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT

from . import bfcl_io
from .client import Completion
from .mapping import NameMap, param_names, resolve
from .models import ModelConfig
from .parsers.base import OK
from .render import render

Generate = Callable[[str], Completion]


@dataclass
class EntryRun:
    decoded: list[list[list[str]]] = field(default_factory=list)  # turn -> step -> accepted original call strings
    force_quit: bool = False
    steps: int = 0
    intrusions: int = 0  # original function names (H3)
    arg_intrusions: int = 0  # original argument names
    rejected: int = 0


def run_entry(
    entry: dict,
    cfg: ModelConfig,
    tokenizer,
    generate: Generate,
    names: NameMap,
    run_name: str,
    log: Callable[[dict], None],
    base_record: dict,
) -> EntryRun:
    tools_original = bfcl_io.original_tools(entry)
    tools_model = names.model_tools(tools_original)
    shown, originals = param_names(tools_model), param_names(tools_original)
    messages: list[dict] = []
    out = EntryRun()

    for turn, user_msgs in enumerate(entry["question"]):
        assert user_msgs, f"{entry['id']} turn {turn} has no user message (miss_func is out of scope)"
        messages.extend({**m, "content": names.user_text(m["content"])} for m in user_msgs)
        turn_decoded: list[list[str]] = []
        step = 0
        while True:
            prompt = render(tokenizer, messages, tools_model, cfg)
            completion = generate(prompt)
            parsed = cfg.parser.parse(completion.text)
            record = {
                **base_record,
                "type": "step",
                "task_id": entry["id"],
                "turn": turn,
                "step": step,
                "prompt": prompt,
                "request": completion.request,
                "latency_s": completion.latency_s,
                "raw_output": completion.text,
                "finish_reason": completion.finish_reason,
                "parse_status": parsed.status,
                "parse_error": parsed.error,
                "parsed_calls": [{"name": c.name, "arguments": c.arguments} for c in parsed.calls],
            }
            out.steps += 1

            messages.append(cfg.parser.assistant_message(parsed))
            if parsed.status != OK:
                log({**record, "turn_end": parsed.status})
                break

            resolved = [resolve(c, names, shown, originals) for c in parsed.calls]
            accepted = [r.call_str for r in resolved if r.accepted]
            results = iter(bfcl_io.execute(accepted, entry, run_name) if accepted else [])
            outputs = [names.tool_output(next(results)) if r.accepted else r.error for r in resolved]
            if accepted:
                turn_decoded.append(accepted)
            out.rejected += sum(not r.accepted for r in resolved)
            out.intrusions += sum(r.intrusion == "function" for r in resolved)
            out.arg_intrusions += sum(r.intrusion == "argument" for r in resolved)

            messages.extend({"role": "tool", "content": o} for o in outputs)

            step += 1
            force = step > MAXIMUM_STEP_LIMIT
            log({
                **record,
                "mapped_calls": accepted,
                "rejected_calls": [{"name": r.call.name, "error": r.error, "intrusion": r.intrusion} for r in resolved if not r.accepted],
                "tool_outputs": outputs,
                "turn_end": "force_quit" if force else None,
            })
            if force:
                out.force_quit = True
                break

        out.decoded.append(turn_decoded)
        if out.force_quit:
            break
    return out
