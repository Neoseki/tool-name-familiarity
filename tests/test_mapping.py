from src import bfcl_io
from src.loop import run_entry
from src.mapping import NameMap, no_argument, not_found, param_names, resolve, to_call_str
from src.parsers.base import ToolCall
from src.replay import ScriptedModel, qwen3_text


def tool(name, *params):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {p: {} for p in params}}}}


ORIGINAL_TOOLS = [tool("ls", "a"), tool("cd", "folder"), tool("cat", "file_name")]
ORIG = param_names(ORIGINAL_TOOLS)
IDENT = NameMap.identity(ORIGINAL_TOOLS)

# A twin as step 3 will build it: functions and parameters renamed.
TWIN = NameMap(
    {"list_entries": "ls", "change_folder": "cd", "show_file": "cat"},
    params={"ls": {"show_hidden": "a"}, "cd": {"target": "folder"}, "cat": {"entry_name": "file_name"}},
)
TWIN_SHOWN = param_names([tool("list_entries", "show_hidden"), tool("change_folder", "target"), tool("show_file", "entry_name")])


def test_identity_accepts_and_formats_like_bfcl():
    r = resolve(ToolCall("cd", {"folder": "a'b"}), IDENT, ORIG, ORIG)
    assert r.accepted and r.call_str == "cd(folder=\"a'b\")" == to_call_str("cd", {"folder": "a'b"})


def test_unknown_function_rejected_with_generic_error():
    r = resolve(ToolCall("rm_everything", {}), IDENT, ORIG, ORIG)
    assert not r.accepted and r.error == not_found("rm_everything") and r.intrusion == ""


def test_unknown_argument_rejected_in_original_condition():
    r = resolve(ToolCall("cd", {"folder": "x", "force": True}), IDENT, ORIG, ORIG)
    assert not r.accepted and r.error == no_argument("cd", "force") and r.intrusion == ""


def test_argument_name_injection_rejected():
    r = resolve(ToolCall("ls", {"a=__import__('os'),b": 1}), IDENT, ORIG, ORIG)
    assert not r.accepted and r.error.startswith("Error: function 'ls' has no argument")


def test_twin_translates_function_and_arguments():
    r = resolve(ToolCall("list_entries", {"show_hidden": True}), TWIN, TWIN_SHOWN, ORIG)
    assert r.accepted and r.call_str == "ls(a=True)"


def test_original_function_name_in_twin_is_an_intrusion():
    r = resolve(ToolCall("ls", {"a": True}), TWIN, TWIN_SHOWN, ORIG)
    assert not r.accepted and r.intrusion == "function" and r.error == not_found("ls")


def test_original_argument_name_in_twin_is_an_intrusion():
    # The gap: show_file(file_name=...) would succeed on the backend if passed through.
    r = resolve(ToolCall("show_file", {"file_name": "notes.txt"}), TWIN, TWIN_SHOWN, ORIG)
    assert not r.accepted and r.intrusion == "argument" and r.error == no_argument("show_file", "file_name")


def test_same_error_text_in_both_conditions():
    # A made-up argument gets the same message shape whichever condition it is in.
    a = resolve(ToolCall("cd", {"bogus": 1}), IDENT, ORIG, ORIG)
    b = resolve(ToolCall("change_folder", {"bogus": 1}), TWIN, TWIN_SHOWN, ORIG)
    assert a.error == no_argument("cd", "bogus") and b.error == no_argument("change_folder", "bogus")
    assert a.intrusion == b.intrusion == ""


def test_rejected_calls_never_execute_and_are_logged(entries, qwen_cfg, qwen_tok):
    entry = entries["multi_turn_base_1"]
    bad = qwen3_text([
        ToolCall("rm_everything", {}),
        ToolCall("mkdir", {"dir_name": "zz_probe", "parents": True}),  # unknown argument
        ToolCall("mkdir", {"dir_name": "zz_ok"}),
    ])
    model = ScriptedModel([bad] + ["Done."] * len(entry["question"]))
    logs = []
    run = "rejtest"
    bfcl_io.reset_state(run, entry)
    r = run_entry(entry, qwen_cfg, qwen_tok, model, NameMap.identity(bfcl_io.original_tools(entry)), run, logs.append, {})
    step0 = logs[0]
    assert [c["error"] for c in step0["rejected_calls"]] == [not_found("rm_everything"), no_argument("mkdir", "parents")]
    assert step0["mapped_calls"] == ["mkdir(dir_name='zz_ok')"]  # only the valid call reaches BFCL
    assert step0["tool_outputs"][:2] == [not_found("rm_everything"), no_argument("mkdir", "parents")]  # call order kept
    assert r.decoded[0] == [["mkdir(dir_name='zz_ok')"]] and r.rejected == 2
    assert no_argument("mkdir", "parents") in model.prompts[1]  # the model sees the error
    bfcl_io.reset_state(run, entry)


def test_every_ground_truth_call_fits_the_shown_schema():
    # The argument check must never reject a correct call: all 1142 calls in all 200 tasks.
    from bfcl_eval.utils import load_file
    from src.replay import gt_calls

    ids = [e["id"] for e in load_file(bfcl_io.TASK_FILE)]
    entries, gts = bfcl_io.load_entries(ids), bfcl_io.load_ground_truth(ids)
    for tid in ids:
        tools = bfcl_io.original_tools(entries[tid])
        names, shown = NameMap.identity(tools), param_names(tools)
        for turn in gt_calls(entries[tid], gts[tid]):
            for c in turn:
                assert resolve(c, names, shown, shown).accepted, (tid, c)
