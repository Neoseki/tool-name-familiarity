"""Qwen3 format: zero or more <tool_call>\n{json}\n</tool_call> blocks after optional text."""

import json
import re

from .base import NO_CALL, OK, PARSE_ERROR, ParseResult, Parser, call_from_json

THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class Qwen3Parser(Parser):
    def parse(self, raw: str) -> ParseResult:
        reasoning = "\n".join(m.strip() for m in THINK.findall(raw))
        text = THINK.sub("", raw)
        bodies = BLOCK.findall(text)
        content = BLOCK.sub("", text).strip()

        if text.count("<tool_call>") != len(bodies) or text.count("</tool_call>") != len(bodies):
            return ParseResult(PARSE_ERROR, content=text.strip(), reasoning=reasoning, error="unbalanced <tool_call> tags")
        if not bodies:
            return ParseResult(NO_CALL, content=content, reasoning=reasoning)
        try:
            calls = [call_from_json(json.loads(b)) for b in bodies]
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            return ParseResult(PARSE_ERROR, content=text.strip(), reasoning=reasoning, error=str(e))
        return ParseResult(OK, calls=calls, content=content, reasoning=reasoning)
