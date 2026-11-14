"""Parsers against each model's own chat template.

The template renders an assistant message with known calls; the text it adds after the
generation prompt is what a model emitting that message would produce. Cut at the first
stop string (as the server does) and it must parse back to the same calls. Then the
full ground-truth replay runs in each model's format.
"""

import pytest

from conftest import DEV_IDS
from src import bfcl_io
from src.loop import run_entry
from src.mapping import NameMap
from src.models import MODELS
from src.parsers.base import NO_CALL, OK, ParseResult, ToolCall
from src.render import load_tokenizer, render
from src.replay import ScriptedModel, gt_calls
from src.score import score_entry

SLUGS = ["qwen3-local", "olmo3-7b", "llmjp-4.1-8b"]
TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
         for n in ("cd", "ls", "echo")]
USER = [{"role": "user", "content": "go"}]


@pytest.fixture(scope="module", params=SLUGS)
def model(request):
    cfg = MODELS[request.param]
    return cfg, load_tokenizer(cfg)


def as_generated(cfg, tok, assistant: dict, tools=TOOLS) -> str:
    prefix = render(tok, USER, tools, cfg)
    full = tok.apply_chat_template(USER + [assistant], tools=tools, add_generation_prompt=False, tokenize=False, **cfg.template_kwargs)
    assert full.startswith(prefix), "template renders history differently from the generation prompt"
    text = full[len(prefix):]
    cuts = [text.index(s) for s in cfg.stop if s in text]
    return text[: min(cuts)] if cuts else text


def calls_msg(cfg, calls):
    """The assistant message the parser itself would put into history for these calls."""
    return cfg.parser.assistant_message(ParseResult(OK, calls=calls))


CASES = [
    [ToolCall("cd", {"folder": "document"})],
    [ToolCall("cd", {"folder": "a b"}), ToolCall("ls", {"a": True})],
    [ToolCall("echo", {"content": "line1\nline2 \"quoted\" <tag> & 'x'", "file_name": "n.txt", "n": -3, "x": 1.5,
                       "l": [1, None, False], "d": {"k": "v"}})],
]


@pytest.mark.parametrize("calls", CASES, ids=["one", "parallel", "awkward-values"])
def test_round_trip(model, calls):
    cfg, tok = model
    msg = calls_msg(cfg, calls)
    parsed = cfg.parser.parse(as_generated(cfg, tok, msg))
    assert parsed.status == OK, (parsed.error, as_generated(cfg, tok, msg))
    assert parsed.calls == calls


def test_final_answer_has_no_call(model):
    cfg, tok = model
    parsed = cfg.parser.parse(as_generated(cfg, tok, {"role": "assistant", "content": "All done."}))
    assert parsed.status == NO_CALL and parsed.content == "All done."


@pytest.mark.parametrize("tid", DEV_IDS)
def test_replay_in_model_format(model, tid, entries, gts):
    cfg, tok = model
    entry, gt = entries[tid], gts[tid]
    tools = bfcl_io.original_tools(entry)

    def emit(calls):
        msg = calls_msg(cfg, calls)
        return as_generated(cfg, tok, msg, tools)

    def done():
        return as_generated(cfg, tok, {"role": "assistant", "content": "Done."}, tools)

    outputs = []
    for calls in gt_calls(entry, gt):
        if calls:
            outputs.append(emit(calls))
        outputs.append(done())
    run = f"tmpl_{cfg.slug}"
    bfcl_io.reset_state(run, entry)
    r = run_entry(entry, cfg, tok, ScriptedModel(outputs), NameMap.identity(tools), run, lambda _: None, {})
    assert score_entry(entry, r.decoded, gt, run, r.force_quit) == {"valid": True}
