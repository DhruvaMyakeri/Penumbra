"""Multiple-comparison correction for a scenario suite.

A garage run tests dozens of situations against the same control. At alpha = 0.025 per
test, twenty null situations produce a false discovery about half the time. Reporting
raw per-test p-values from a suite that size is how a testing tool starts inventing
vulnerabilities.

Two corrections, reported side by side because they answer different questions:

**Benjamini-Hochberg (FDR).** Controls the expected *proportion* of reported
vulnerabilities that are false. This is the right default for a discovery tool: the
output is a shortlist to investigate, and tolerating a known fraction of false leads
buys much more power than demanding near-certainty on every one.

**Bonferroni (FWER).** Controls the probability of *any* false discovery. Appropriate
when a single claim will be acted on directly. Much stricter, and reported so a reader
can apply the harder bar without recomputing anything.

Both are computed over every scenario that produced a test - a render the gate
rejected never ran a test and is not part of the family.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Corrected:
    """One test's standing after correction."""

    name: str
    p_value: float
    rank: int
    bh_threshold: float
    passes_fdr: bool
    passes_bonferroni: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "p_value": round(self.p_value, 5),
            "rank": self.rank,
            "bh_threshold": round(self.bh_threshold, 5),
            "passes_fdr": self.passes_fdr,
            "passes_bonferroni": self.passes_bonferroni,
        }


@dataclass
class SuiteCorrection:
    n_tests: int
    alpha: float
    results: list[Corrected]
    bonferroni_alpha: float

    @property
    def discoveries(self) -> list[str]:
        return [r.name for r in self.results if r.passes_fdr]

    @property
    def strong_discoveries(self) -> list[str]:
        return [r.name for r in self.results if r.passes_bonferroni]

    def to_dict(self) -> dict:
        return {
            "n_tests": self.n_tests,
            "alpha": self.alpha,
            "bonferroni_alpha": round(self.bonferroni_alpha, 6),
            "discoveries_fdr": self.discoveries,
            "discoveries_bonferroni": self.strong_discoveries,
            "per_test": [r.to_dict() for r in self.results],
        }


def correct(p_values: dict[str, float], *, alpha: float = 0.025) -> SuiteCorrection:
    """Benjamini-Hochberg plus Bonferroni over a suite of tests.

    `p_values` maps scenario name to its p-value. Only scenarios that actually ran a
    test belong here; gate-rejected renders are not part of the family.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return SuiteCorrection(0, alpha, [], alpha)

    # BH: find the largest rank k where p(k) <= k/n * alpha, then everything at or
    # below that rank is a discovery. Walking backwards is what makes the procedure
    # a step-up rather than a per-test comparison.
    cutoff_rank = 0
    for rank, (_, p) in enumerate(items, start=1):
        if p <= rank / n * alpha:
            cutoff_rank = rank

    bonf = alpha / n
    results = [
        Corrected(
            name=name,
            p_value=p,
            rank=rank,
            bh_threshold=rank / n * alpha,
            passes_fdr=rank <= cutoff_rank,
            passes_bonferroni=p <= bonf,
        )
        for rank, (name, p) in enumerate(items, start=1)
    ]
    return SuiteCorrection(n_tests=n, alpha=alpha, results=results, bonferroni_alpha=bonf)
