from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

from . import bfcl_io


def score_entry(entry: dict, decoded: list[list[list[str]]], ground_truth: list[list[str]], run_name: str, force_quit: bool) -> dict:
    """BFCL's verdict for one entry, including the eval runner's turn-count check.

    Clears the cached backend state for this entry afterwards (inference and checker copies).
    """
    try:
        if force_quit or len(decoded) != len(ground_truth):
            return {"valid": False, "error_type": "multi_turn:force_terminated",
                    "error_message": f"{len(decoded)} model turns vs {len(ground_truth)} ground-truth turns"}
        return multi_turn_checker(decoded, ground_truth, entry, bfcl_io.CATEGORY, run_name)
    finally:
        bfcl_io.reset_state(run_name, entry)
