from src.parsers.base import NO_CALL, OK, PARSE_ERROR
from src.parsers.qwen3 import Qwen3Parser

P = Qwen3Parser()


def test_single_call():
    r = P.parse('<tool_call>\n{"name": "ls", "arguments": {"a": true}}\n</tool_call>')
    assert r.status == OK and r.calls[0].name == "ls" and r.calls[0].arguments == {"a": True}


def test_multiple_calls_with_text_keep_order():
    raw = 'Sure.\n<tool_call>\n{"name": "cd", "arguments": {"folder": "x"}}\n</tool_call>\n<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    r = P.parse(raw)
    assert r.status == OK and [c.name for c in r.calls] == ["cd", "ls"] and r.content == "Sure."


def test_no_call():
    r = P.parse("All done.")
    assert r.status == NO_CALL and r.content == "All done."


def test_malformed_json_is_not_repaired():
    r = P.parse('<tool_call>\n{"name": "ls", "arguments": {"a": true}\n</tool_call>')
    assert r.status == PARSE_ERROR and not r.calls


def test_unclosed_tag():
    assert P.parse('<tool_call>\n{"name": "ls", "arguments": {}}').status == PARSE_ERROR


def test_arguments_must_be_object():
    assert P.parse('<tool_call>\n{"name": "ls", "arguments": "{}"}\n</tool_call>').status == PARSE_ERROR


def test_unknown_name_still_parses():
    # Rejection is the mapping layer's job, not the parser's.
    r = P.parse('<tool_call>\n{"name": "list_entries", "arguments": {}}\n</tool_call>')
    assert r.status == OK and r.calls[0].name == "list_entries"


def test_think_block_split_out():
    r = P.parse("<think>\nhmm\n</think>\n\nOK")
    assert r.status == NO_CALL and r.reasoning == "hmm" and r.content == "OK"
