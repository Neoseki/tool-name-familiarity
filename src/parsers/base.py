from dataclasses import dataclass, field

OK = "ok"
NO_CALL = "no_call"
PARSE_ERROR = "parse_error"


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ParseResult:
    status: str  # OK | NO_CALL | PARSE_ERROR
    calls: list[ToolCall] = field(default_factory=list)
    content: str = ""  # visible text outside tool calls and reasoning; on PARSE_ERROR, the unparsed text
    reasoning: str = ""
    error: str = ""


class Parser:
    def parse(self, raw: str) -> ParseResult:
        raise NotImplementedError

    def assistant_message(self, parsed: ParseResult) -> dict:
        """How this step goes back into the chat history, in the shape the model's template expects."""
        msg = {"role": "assistant", "content": parsed.content}
        if parsed.calls:
            msg["tool_calls"] = [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in parsed.calls
            ]
        return msg


def call_from_json(obj) -> ToolCall:
    """Validate a decoded {"name": ..., "arguments": {...}} object. No repair."""
    if not isinstance(obj, dict):
        raise ValueError(f"tool call is not an object: {obj!r}")
    name, args = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"tool call has no string name: {obj!r}")
    if not isinstance(args, dict):
        raise ValueError(f"tool call arguments are not an object: {obj!r}")
    return ToolCall(name, args)
