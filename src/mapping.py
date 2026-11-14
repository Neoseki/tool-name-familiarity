"""Model-facing names <-> BFCL's original names, and unknown-name rejection.

The model only ever sees model-facing names. In the original condition the map is the
identity; in the twin condition (step 3) it renames functions, parameters and entities.
Every call is checked here before anything reaches a BFCL backend.
"""

from dataclasses import dataclass, field

from .parsers.base import ToolCall


def not_found(name: str) -> str:
    """Same generic errors in both conditions."""
    return f"Error: function '{name}' not found"


def no_argument(name: str, arg: str) -> str:
    return f"Error: function '{name}' has no argument '{arg}'"


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
    # "function" / "argument": the rejected name is an original name the model was not shown
    intrusion: str = ""


def param_names(tools: list[dict]) -> dict[str, set[str]]:
    """Function name -> its parameter names, from an OpenAI-style tool list."""
    return {t["function"]["name"]: set(t["function"]["parameters"].get("properties", {})) for t in tools}


def resolve(call: ToolCall, names: NameMap, shown: dict[str, set[str]], originals: dict[str, set[str]]) -> Resolved:
    """Check a call against the schema the model was shown, then translate it to original names.

    shown: model-facing function -> parameter names, as in the tool list the model saw.
    originals: original function -> parameter names, as the BFCL backend knows them.
    Unknown function or argument names are rejected the same way in both conditions; the
    backend would accept original names in the twin condition, so they must never reach it.
    """
    original = names.functions.get(call.name)
    if original is None or call.name not in shown:
        intrusion = "function" if call.name in originals else ""
        return Resolved(call, False, error=not_found(call.name), intrusion=intrusion)

    for k in call.arguments:
        if k not in shown[call.name]:
            intrusion = "argument" if k in originals[original] else ""
            return Resolved(call, False, error=no_argument(call.name, k), intrusion=intrusion)

    pmap = names.params.get(original, {})
    args = {pmap.get(k, k): v for k, v in call.arguments.items()}
    return Resolved(call, True, call_str=to_call_str(original, args))


def to_call_str(name: str, args: dict) -> str:
    """Same formatting as BFCL's convert_to_function_call: keyword args, repr values."""
    return f"{name}({','.join(f'{k}={v!r}' for k, v in args.items())})"
