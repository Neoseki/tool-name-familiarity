"""LLM-jp-4.1 thinking (Harmony-style), read from its chat template (rev 7c19bc90).

Generation starts after `<|start|>assistant`. Output is a sequence of messages:

    <|channel|>analysis<|message|>...<|end|>
    <|start|>assistant to=functions.ls<|channel|>commentary <|constrain|>json<|message|>{"a": true}<|call|>
    <|channel|>final<|message|>...<|return|>

Parallel calls are several `to=functions.X` messages, all but the last ending in <|end|>.
The recipient may sit before or after the channel tag. Terminators may be missing when
the server strips the stop token; a body cut short by max_tokens fails JSON parsing.
"""

import json
import re

from .base import NO_CALL, OK, PARSE_ERROR, ParseResult, Parser, ToolCall

TERMINATORS = re.compile(r"<\|(?:end|call|return)\|>")
RECIPIENT = re.compile(r"to=([^\s<]+)")
CHANNEL = re.compile(r"<\|channel\|>\s*(\w+)")


class LlmJpParser(Parser):
    def parse(self, raw: str) -> ParseResult:
        reasoning, content, calls = [], [], []
        try:
            for seg in raw.split("<|start|>"):
                if not seg.strip():
                    continue
                if "<|message|>" not in seg:
                    raise ValueError(f"message without <|message|>: {seg[:80]!r}")
                header, body = seg.split("<|message|>", 1)
                body = TERMINATORS.split(body, 1)[0]
                recipient = RECIPIENT.search(header)
                channel = CHANNEL.search(header)
                if recipient:
                    target = recipient.group(1)
                    if not target.startswith("functions."):
                        raise ValueError(f"unknown recipient {target!r}")
                    args = json.loads(body)
                    if not isinstance(args, dict):
                        raise ValueError(f"arguments are not an object: {body[:80]!r}")
                    calls.append(ToolCall(target.removeprefix("functions."), args))
                elif channel and channel.group(1) == "analysis":
                    reasoning.append(body.strip())
                elif channel and channel.group(1) in ("final", "commentary"):
                    content.append(body.strip())
                else:
                    raise ValueError(f"message without channel or recipient: {header!r}")
        except ValueError as e:
            visible = TERMINATORS.sub("", re.sub(r"<\|[a-z]+\|>", " ", raw)).strip()
            return ParseResult(PARSE_ERROR, content=visible, reasoning="\n".join(reasoning), error=str(e))
        status = OK if calls else NO_CALL
        return ParseResult(status, calls=calls, content="\n".join(content), reasoning="\n".join(reasoning))

    def assistant_message(self, parsed: ParseResult) -> dict:
        # The template refuses content + thinking together with tool calls, and drops
        # earlier-turn analysis anyway. Keep reasoning in "thinking", text in "content".
        if parsed.calls:
            msg = super().assistant_message(ParseResult(parsed.status, parsed.calls))
            del msg["content"]
            if parsed.reasoning:
                msg["thinking"] = parsed.reasoning
            return msg
        return {"role": "assistant", "content": parsed.content, "thinking": parsed.reasoning}
