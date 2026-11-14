"""Draw the 20 pilot tasks from BFCL multi_turn_base (plan section 3.2). Deterministic.

Every task has exactly one main backend (50 tasks each); the other classes only ever
appear as a second backend. Strata = main backend: 8 file-system tasks (most familiar
names), 4 from each of the other three.

No outcome-based exclusions: tasks every model fails count as "both fail" and drop out
of McNemar on their own, while excluding original-condition failures would only remove
tasks that could show c (fail original, pass twin), biasing toward H1.

The rest of each stratum, in seeded order, is the reserve list. A reserve replaces a
selected task only for a pipeline reason found before any model run (e.g. it fails the
twin replay gate), taking the next unused reserve from the same stratum. Record it.

uv run python scripts/select_tasks.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_eval.utils import load_file  # noqa: E402

from src import bfcl_io  # noqa: E402

SEED = 20261114
QUOTA = {"GorillaFileSystem": 8, "VehicleControlAPI": 4, "TradingBot": 4, "TravelAPI": 4}
OUT = Path(__file__).resolve().parent.parent / "tasks"


def num(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def main() -> None:
    entries = load_file(bfcl_io.TASK_FILE)
    strata = {main: [] for main in QUOTA}
    for e in entries:
        (main,) = set(QUOTA) & set(e["involved_classes"])
        strata[main].append(e["id"])

    rng = random.Random(SEED)
    selected, reserves = [], {}
    for main in QUOTA:  # fixed order, so the draw is reproducible
        order = sorted(strata[main], key=num)
        rng.shuffle(order)
        selected += order[: QUOTA[main]]
        reserves[main] = order[QUOTA[main]:]

    by_id = {e["id"]: e for e in entries}
    selected.sort(key=num)
    (OUT / "selected_ids.json").write_text(json.dumps(selected, indent=1) + "\n")
    (OUT / "selection.json").write_text(json.dumps({
        "seed": SEED,
        "source": f"bfcl-eval {bfcl_io.TASK_FILE.name}",
        "quota": QUOTA,
        "selected": [{"id": t, "involved_classes": by_id[t]["involved_classes"], "turns": len(by_id[t]["question"])}
                     for t in selected],
        "reserves_in_order": reserves,
        "replacements": [],
    }, indent=1) + "\n")
    print("\n".join(f"{t}  {by_id[t]['involved_classes']}" for t in selected))


if __name__ == "__main__":
    main()
