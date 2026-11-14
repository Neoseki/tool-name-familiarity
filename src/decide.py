"""The pre-registered decision rule (README, Pre-registration), as code.

Pass only the models that passed every validity gate (README), including Floor.
"""

from dataclasses import dataclass
from math import comb


@dataclass
class ModelResult:
    b: int  # pass original, fail twin
    c: int  # fail original, pass twin
    intrusions: int  # twin-condition calls to original function names
    intrusion_tasks: int  # distinct tasks with at least one intrusion
    original_nonexistent: int  # original-condition calls to function names not in the tool list

    @property
    def net(self) -> int:
        return self.b - self.c

    @property
    def p(self) -> float:
        """Exact one-sided McNemar: P(X >= b), X ~ Binomial(b + c, 1/2)."""
        n = self.b + self.c
        return sum(comb(n, k) for k in range(self.b, n + 1)) / 2**n if n else 1.0

    @property
    def h3(self) -> bool:
        return self.intrusions >= 3 and self.intrusion_tasks >= 2 and self.intrusions > self.original_nonexistent


ROWS = ["strong", "suggestive", "reversed", "behaviour_without_score_change", "null", "inconclusive"]


def decide(models: dict[str, ModelResult]) -> str:
    """First matching row, top to bottom."""
    ms = list(models.values())
    nets = [m.net for m in ms]
    if any(m.p < 0.05 and any(o is not m and o.net >= 3 for o in ms) for m in ms):
        return "strong"
    if sum(n >= 3 for n in nets) >= 2:
        return "suggestive"
    if sum(n <= -3 for n in nets) >= 2:
        return "reversed"
    if ms and all(n <= 2 for n in nets) and any(m.h3 for m in ms):
        return "behaviour_without_score_change"
    if ms and all(abs(n) <= 1 for n in nets) and not any(m.h3 for m in ms):
        return "null"
    return "inconclusive"
