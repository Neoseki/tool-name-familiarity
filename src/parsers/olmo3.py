"""OLMo 3 Instruct format, read from its chat template (rev 6e5971d9):

    <function_calls>name(key=<json value>, ...)\nname2(...)</function_calls>

Values are JSON (the template renders them with `tojson`), so true/false/null appear
as bare names. That is the format, not a repair. Positional arguments, expressions or
any other statement are parse errors.
"""

import ast
import re

from .base import NO_CALL, OK, PARSE_ERROR, ParseResult, Parser, ToolCall

BLOCK = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
JSON_NAMES = {"true": True, "false": False, "null": None}


def _value(node: ast.expr):
    if isinstance(node, ast.Name) and node.id in JSON_NAMES:
        return JSON_NAMES[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -_value(node.operand)
    if isinstance(node, ast.List):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys):
        return {k.value: _value(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"not a JSON value: {ast.unparse(node)}")


def _calls(body: str) -> list[ToolCall]:
    calls = []
    for stmt in ast.parse(body.strip()).body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)):
            raise ValueError(f"not a function call: {ast.unparse(stmt)}")
        call = stmt.value
        if call.args or any(kw.arg is None for kw in call.keywords):
            raise ValueError(f"positional or ** arguments: {ast.unparse(call)}")
        calls.append(ToolCall(call.func.id, {kw.arg: _value(kw.value) for kw in call.keywords}))
    return calls


class Olmo3Parser(Parser):
    def parse(self, raw: str) -> ParseResult:
        bodies = BLOCK.findall(raw)
        content = BLOCK.sub("", raw).strip()
        if raw.count("<function_calls>") != len(bodies) or raw.count("</function_calls>") != len(bodies):
            return ParseResult(PARSE_ERROR, content=raw.strip(), error="unbalanced <function_calls> tags")
        if not bodies:
            return ParseResult(NO_CALL, content=content)
        try:
            calls = [c for b in bodies for c in _calls(b)]
        except (SyntaxError, ValueError) as e:
            return ParseResult(PARSE_ERROR, content=raw.strip(), error=str(e))
        if not calls:
            return ParseResult(PARSE_ERROR, content=raw.strip(), error="empty <function_calls> block")
        return ParseResult(OK, calls=calls, content=content)
