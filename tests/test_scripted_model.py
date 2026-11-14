"""End to end without an LLM: ground truth rendered as Qwen3 output, through render ->
parse -> name check -> BFCL backend -> checker. Must pass on every dev task."""

import pytest

from conftest import DEV_IDS
from src import bfcl_io
from src.loop import run_entry
from src.mapping import NameMap
from src.replay import ScriptedModel, gt_calls
from src.score import score_entry


def replay(entry, gt, cfg, tok, run):
    model = ScriptedModel.from_turns(gt_calls(entry, gt))
    logs = []
    bfcl_io.reset_state(run, entry)
    r = run_entry(entry, cfg, tok, model, NameMap.identity(bfcl_io.original_tools(entry)), run, logs.append, {})
    verdict = score_entry(entry, r.decoded, gt, run, r.force_quit)
    return r, verdict, logs, model


@pytest.mark.parametrize("tid", DEV_IDS)
def test_replay_passes(tid, entries, gts, qwen_cfg, qwen_tok):
    r, verdict, logs, model = replay(entries[tid], gts[tid], qwen_cfg, qwen_tok, "replay")
    assert verdict == {"valid": True}, verdict
    assert not model.outputs and r.rejected == 0 and not r.force_quit
    assert len(r.decoded) == len(gts[tid])
    for rec in logs:  # traceability gate: every step has prompt, raw output, parse status
        assert rec["prompt"] and rec["raw_output"] and rec["parse_status"]


def test_rerun_is_identical_after_reset(entries, gts, qwen_cfg, qwen_tok):
    tid = "multi_turn_base_0"
    first = replay(entries[tid], gts[tid], qwen_cfg, qwen_tok, "rerun")[2]
    second = replay(entries[tid], gts[tid], qwen_cfg, qwen_tok, "rerun")[2]
    assert [x["tool_outputs"] for x in first if "tool_outputs" in x] == [x["tool_outputs"] for x in second if "tool_outputs" in x]


def test_step_limit_force_quits(entries, qwen_cfg, qwen_tok):
    entry = entries["multi_turn_base_1"]
    call = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    model = ScriptedModel([call] * 50)
    bfcl_io.reset_state("limit", entry)
    r = run_entry(entry, qwen_cfg, qwen_tok, model, NameMap.identity(bfcl_io.original_tools(entry)), "limit", lambda _: None, {})
    assert r.force_quit and len(r.decoded) == 1 and r.steps == 21  # BFCL: quits when count > 20
    assert score_entry(entry, r.decoded, [[]] * 4, "limit", r.force_quit)["error_type"] == "multi_turn:force_terminated"


@pytest.mark.parametrize("tid", DEV_IDS)
def test_replay_missing_a_call_fails(tid, entries, gts, qwen_cfg, qwen_tok):
    # Negative control: drop the last ground-truth call of the last non-empty turn.
    entry, gt = entries[tid], gts[tid]
    turns = gt_calls(entry, gt)
    last = max(i for i, t in enumerate(turns) if t)
    turns[last] = turns[last][:-1]
    model = ScriptedModel.from_turns(turns)
    bfcl_io.reset_state("neg", entry)
    r = run_entry(entry, qwen_cfg, qwen_tok, model, NameMap.identity(bfcl_io.original_tools(entry)), "neg", lambda _: None, {})
    assert not score_entry(entry, r.decoded, gt, "neg", r.force_quit)["valid"]
