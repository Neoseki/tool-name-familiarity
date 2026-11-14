"""BFCL's own checker accepts its ground truth."""

from conftest import DEV_IDS
from src.score import score_entry


def test_ground_truth_passes_checker(entries, gts):
    for tid in DEV_IDS[:2]:
        decoded = [[turn] if turn else [] for turn in gts[tid]]
        verdict = score_entry(entries[tid], decoded, gts[tid], "gtcheck", force_quit=False)
        assert verdict == {"valid": True}, (tid, verdict)


def test_turn_count_mismatch_fails(entries, gts):
    tid = DEV_IDS[0]
    decoded = [[turn] for turn in gts[tid][:-1]]
    verdict = score_entry(entries[tid], decoded, gts[tid], "gtcheck", force_quit=False)
    assert verdict["error_type"] == "multi_turn:force_terminated"
