"""Model-facing names <-> BFCL's original names, and unknown-name rejection.

The model only ever sees model-facing names. In the original condition the map is the
identity; in the twin condition (step 3) it renames functions, parameters and entities.
Every call is checked here before anything reaches a BFCL backend.
"""

import keyword
from dataclasses import dataclass, field

from .parsers.base import ToolCall


def not_found(name: str) -> str:
    """Same generic error in both conditions (plan section 3.7)."""
    return f"Error: function '{name}' not found"


@dataclass
class NameMap:
    functions: dict[str, str]  # model-facing -> original
    params: dict[str, dict[str, str]] = field(default_factory=dict)  # original fn -> {model param -> original param}
    entities: dict[str, str] = field(default_factory=dict)  # model-facing -> original

    @classmethod
    def identity(cls, original_tools: list[dict]) -> "NameMap":
        return cls({t["function"]["name"]: t["function"]["name"] for t in original_tools})

    @property
    def is_identity(self) -> bool:
        return all(k == v for k, v in self.functions.items()) and not self.params and not self.entities

    def model_tools(self, original_tools: list[dict]) -> list[dict]:
        if not self.is_identity:
            raise NotImplementedError("twin tool rendering is step 3")
        return original_tools

    def user_text(self, text: str) -> str:
        if self.entities:
            raise NotImplementedError("entity mapping is step 3")
        return text

    def tool_output(self, text: str) -> str:
        if self.entities:
            raise NotImplementedError("entity mapping is step 3")
        return text


@dataclass
class Resolved:
    call: ToolCall  # as the model emitted it
    accepted: bool
    call_str: str = ""  # original-name BFCL call string, if accepted
    error: str = ""  # message returned to the model, if rejected
    intrusion: bool = False  # rejected name is an original name absent from the model's tool list


def resolve(call: ToolCall, names: NameMap, original_names: set[str]) -> Resolved:
    original = names.functions.get(call.name)
    if original is None:
        intrusion = call.name in original_names and not names.is_identity
        return Resolved(call, False, error=not_found(call.name), intrusion=intrusion)

    pmap = names.params.get(original, {})
    args = {}
    for k, v in call.arguments.items():
        k = pmap.get(k, k)
        # Argument names end up inside a string BFCL eval()s: only plain identifiers get through.
        if not k.isidentifier() or keyword.iskeyword(k):
            return Resolved(call, False, error=f"Error: invalid argument name {k!r}")
        args[k] = v
    return Resolved(call, True, call_str=to_call_str(original, args))


def to_call_str(name: str, args: dict) -> str:
    """Same formatting as BFCL's convert_to_function_call: keyword args, repr values."""
    return f"{name}({','.join(f'{k}={v!r}' for k, v in args.items())})"
