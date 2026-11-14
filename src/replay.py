"""Replay BFCL ground truth through our full pipeline as if a model had produced it.

Used by the end-to-end test now and by the twin replay gate (plan 8.2) in step 3.
"""

import ast
import importlib
import inspect
import json

from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING

from .client import Completion
from .parsers.base import ToolCall


def _signatures(entry: dict) -> dict[str, inspect.Signature]:
    sigs = {}
    for cls_name in entry["involved_classes"]:
        cls = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name]), cls_name)
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                sigs[name] = inspect.signature(fn)
    return sigs


def gt_calls(entry: dict, ground_truth: list[list[str]]) -> list[list[ToolCall]]:
    """Ground-truth call strings -> structured calls, positional args named via the backend signature."""
    sigs = _signatures(entry)
    turns = []
    for turn in ground_truth:
        calls = []
        for s in turn:
            node = ast.parse(s, mode="eval").body
            name = node.func.id
            params = [p for p in sigs[name].parameters if p != "self"]
            args = {params[i]: ast.literal_eval(a) for i, a in enumerate(node.args)}
            args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
            calls.append(ToolCall(name, args))
        turns.append(calls)
    return turns


def qwen3_text(calls: list[ToolCall]) -> str:
    return "\n".join(
        f"<tool_call>\n{json.dumps({'name': c.name, 'arguments': c.arguments})}\n</tool_call>" for c in calls
    )


class ScriptedModel:
    """Emits each turn's calls in one step, then a plain reply that ends the turn."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    @classmethod
    def from_turns(cls, turns: list[list[ToolCall]], render_calls=qwen3_text) -> "ScriptedModel":
        outputs = []
        for calls in turns:
            if calls:
                outputs.append(render_calls(calls))
            outputs.append("Done.")
        return cls(outputs)

    def __call__(self, prompt: str) -> Completion:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("scripted model ran out of outputs: loop took more steps than scripted")
        return Completion(self.outputs.pop(0), "stop", {"scripted": True}, 0.0)
