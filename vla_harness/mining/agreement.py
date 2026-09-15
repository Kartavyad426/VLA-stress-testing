"""Taxonomy validation -- supplementing kappa, not raising it (F6a).

The survey's answer to "how does anyone validate a taxonomy is CORRECT" is
negative: nobody does. Four substitutes exist and only one can falsify a
taxonomy. So we report several weak instruments honestly rather than one strong
one we do not have.

**kappa is the wrong instrument at any threshold.** It measures whether raters
agree, not whether the partition carves the space. `{failure on a Tuesday,
failure not on a Tuesday}` scores kappa = 1.0. That is construct validity, and
raising 0.5 -> 0.7 addresses the wrong axis. kappa is additionally depressed by
exactly the class imbalance failure taxonomies always have (the kappa paradox),
which is why percentage agreement and Gwet's AC1 belong beside it.
"""
from __future__ import annotations

from collections import Counter


def percent_agreement(a, b) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0


def cohens_kappa(a, b) -> float:
    n = len(a)
    if not n:
        return 0.0
    po = percent_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def gwet_ac1(a, b) -> float:
    """Gwet's AC1 -- robust to the prevalence problem that depresses kappa.

    Chance agreement is estimated from the OVERALL marginal, not from each
    rater's own distribution, so a dominant category does not drive the
    correction toward 1 and the statistic toward 0.
    """
    n = len(a)
    if not n:
        return 0.0
    po = percent_agreement(a, b)
    cats = set(a) | set(b)
    q = len(cats)
    if q < 2:
        return 1.0
    pi = {k: (a.count(k) + b.count(k)) / (2 * n) for k in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def rater_report(a, b, labels=None) -> dict:
    """All three statistics plus the imbalance that makes kappa misleading."""
    n = len(a)
    dist = Counter(a)
    top = dist.most_common(1)[0][1] / n if n else 0
    return {"n": n, "percent_agreement": round(percent_agreement(a, b), 4),
            "cohens_kappa": round(cohens_kappa(a, b), 4),
            "gwet_ac1": round(gwet_ac1(a, b), 4),
            "n_categories": len(set(a) | set(b)),
            "majority_class_share": round(top, 3),
            "note": ("kappa measures rater consistency, NOT whether the "
                     "taxonomy is correct. High majority-class share depresses "
                     "kappa relative to AC1 (kappa paradox) -- read all three.")}


def name_based_reassignment(assignments, induced) -> dict:
    """The partial construct check (survey §8.5).

    A held-out judge sees ONLY cluster names and descriptions -- never the
    induced grouping -- and assigns held-out episodes. Agreement with the
    induced partition tests whether the cluster has a **communicable common
    property**.

    Unlike kappa on a hand-designed taxonomy, this tests the PARTITION rather
    than the raters. It does **not** establish the property is the RIGHT one.
    Report it as partial; nothing we found does better.
    """
    keys = [k for k in assignments if k in induced]
    a = [assignments[k] for k in keys]
    b = [induced[k] for k in keys]
    r = rater_report(a, b)
    r.update({"instrument": "name-based re-assignment",
              "tests": "whether clusters have a communicable common property",
              "does_not_test": "whether that property is the correct one",
              "status": "PARTIAL -- report as such"})
    return r
