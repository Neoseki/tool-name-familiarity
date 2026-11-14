"""Thin access layer over the pinned bfcl-eval package.

We use BFCL's task data, backend classes and checker only. Prompt formatting and
output decoding are ours (README, Pipeline).
"""

import copy
import re

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import load_file, populate_test_cases_with_predefined_functions

CATEGORY = "multi_turn_base"
TASK_FILE = PROMPT_PATH / f"BFCL_v4_{CATEGORY}.json"
GT_FILE = POSSIBLE_ANSWER_PATH / f"BFCL_v4_{CATEGORY}.json"


def load_entries(ids: list[str]) -> dict[str, dict]:
    """Task entries by id, with BFCL's function docs attached under "function"."""
    wanted = set(ids)
    entries = [e for e in load_file(TASK_FILE) if e["id"] in wanted]
    missing = wanted - {e["id"] for e in entries}
    if missing:
        raise KeyError(f"unknown task ids: {sorted(missing)}")
    populate_test_cases_with_predefined_functions(entries)
    return {e["id"]: e for e in entries}


def load_ground_truth(ids: list[str]) -> dict[str, list[list[str]]]:
    wanted = set(ids)
    return {g["id"]: g["ground_truth"] for g in load_file(GT_FILE) if g["id"] in wanted}


def original_tools(entry: dict) -> list[dict]:
    """OpenAI-style tool list for the entry, as passed to HF chat templates.

    BFCL's own converter does the type mapping (dict -> object, float -> number, ...).
    The non-standard "response" schema is dropped; this is the same in both conditions.
    """
    tools = convert_to_tool(copy.deepcopy(entry["function"]), GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)
    for t in tools:
        t.pop("response", None)
    return [{"type": "function", "function": t} for t in tools]


def execute(call_strs: list[str], entry: dict, run_name: str) -> list[str]:
    """Run call strings (original names) against the entry's live backend instances."""
    results, _ = multi_turn_utils.execute_multi_turn_func_call(
        call_strs,
        entry["initial_config"],
        entry["involved_classes"],
        run_name,
        entry["id"],
        long_context=False,
        is_evaL_run=False,
    )
    return results


def reset_state(run_name: str, entry: dict) -> None:
    """Drop the backend instances BFCL cached in module globals for this entry.

    BFCL never clears them, so without this a rerun under the same name would continue
    from the old state. Covers the inference instances and the checker's two copies.
    """
    store = vars(multi_turn_utils)
    for name in (run_name, f"{run_name}_eval", f"{run_name}_ground_truth_eval"):
        for cls in entry["involved_classes"]:
            key = re.sub(r"[-./:]", "_", f"{name}_{entry['id']}_{cls}_instance")
            store.pop(key, None)
