from src import bfcl_io
from src.loop import run_entry
from src.mapping import NameMap, not_found, resolve, to_call_str
from src.parsers.base import ToolCall
from src.replay import ScriptedModel, qwen3_text

TOOLS = [{"type": "function", "function": {"name": n}} for n in ("ls", "cd")]
ORIGINAL = {"ls", "cd"}


def test_identity_accepts_and_formats_like_bfcl():
    r = resolve(ToolCall("cd", {"folder": "a'b"}), NameMap.identity(TOOLS), ORIGINAL)
    assert r.accepted and r.call_str == "cd(folder=\"a'b\")" == to_call_str("cd", {"folder": "a'b"})


def test_unknown_name_rejected_with_generic_error():
    r = resolve(ToolCall("rm_everything", {}), NameMap.identity(TOOLS), ORIGINAL)
    assert not r.accepted and r.error == not_found("rm_everything") and not r.intrusion


def test_original_name_in_twin_is_an_intrusion():
    twin = NameMap({"list_entries": "ls", "change_folder": "cd"})
    r = resolve(ToolCall("ls", {}), twin, ORIGINAL)
    assert not r.accepted and r.intrusion and r.error == not_found("ls")
    assert resolve(ToolCall("list_entries", {"a": True}), twin, ORIGINAL).call_str == "ls(a=True)"


def test_argument_name_injection_rejected():
    r = resolve(ToolCall("ls", {"a=__import__('os'),b": 1}), NameMap.identity(TOOLS), ORIGINAL)
    assert not r.accepted


def test_rejected_call_never_executes_and_is_logged(entries, qwen_cfg, qwen_tok):
    entry = entries["multi_turn_base_1"]
    bad = qwen3_text([ToolCall("rm_everything", {}), ToolCall("mkdir", {"dir_name": "zz_probe"})])
    model = ScriptedModel([bad] + ["Done."] * len(entry["question"]))
    logs = []
    run = "rejtest"
    bfcl_io.reset_state(run, entry)
    r = run_entry(entry, qwen_cfg, qwen_tok, model, NameMap.identity(bfcl_io.original_tools(entry)), run, logs.append, {})
    step0 = logs[0]
    assert step0["rejected_calls"] == [{"name": "rm_everything", "error": not_found("rm_everything"), "intrusion": False}]
    assert step0["mapped_calls"] == ["mkdir(dir_name='zz_probe')"]
    assert step0["tool_outputs"][0] == not_found("rm_everything")  # results stay in call order
    assert r.decoded[0] == [["mkdir(dir_name='zz_probe')"]] and r.rejected == 1
    # the error text reaches the model in the next prompt
    assert not_found("rm_everything") in model.prompts[1]
    bfcl_io.reset_state(run, entry)
