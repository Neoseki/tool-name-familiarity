"""Per-model settings. Identical across conditions; only the endpoint differs Mac vs pod.

Revisions are pinned before the real run (plan section 8.1). None = not pinned yet.
"""

from dataclasses import dataclass, field

from .parsers.base import Parser
from .parsers.llmjp import LlmJpParser
from .parsers.olmo3 import Olmo3Parser
from .parsers.qwen3 import Qwen3Parser


@dataclass(frozen=True)
class ModelConfig:
    slug: str
    hf_repo: str
    revision: str | None
    parser: Parser
    template_kwargs: dict = field(default_factory=dict)
    stop: tuple[str, ...] = ()
    max_tokens: int = 2048


MODELS = {
    "qwen3-local": ModelConfig(
        "qwen3-local", "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", Qwen3Parser(),
        template_kwargs={"enable_thinking": False}, stop=("<|im_end|>",),
    ),
    "qwen3-8b": ModelConfig(
        "qwen3-8b", "Qwen/Qwen3-8B", None, Qwen3Parser(),
        template_kwargs={"enable_thinking": False}, stop=("<|im_end|>",),
    ),
    # Template default: function-calling system prompt; tool results in an "environment" role.
    "olmo3-7b": ModelConfig(
        "olmo3-7b", "allenai/Olmo-3-7B-Instruct", None, Olmo3Parser(),
        stop=("<|im_end|>", "<|endoftext|>"),
    ),
    # The template writes today's date into the system prompt unless given one: fixed here so
    # every run, in both conditions, sees the same prompt. Reasoning effort as llm-jp used for BFCL.
    "llmjp-4.1-8b": ModelConfig(
        "llmjp-4.1-8b", "llm-jp/llm-jp-4.1-8b-thinking", None, LlmJpParser(),
        template_kwargs={"reasoning_effort": "medium", "conversation_start_date": "2026-11-14"},
        stop=("<|call|>", "<|return|>"), max_tokens=8192,
    ),
}
