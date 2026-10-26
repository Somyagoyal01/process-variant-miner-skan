"""Directly-follows graph, variants and a decision rule from a normalised log."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def traces(log: pd.DataFrame) -> pd.Series:
    return log.sort_values(["case_id", "ts"]).groupby("case_id")["activity"].apply(tuple)


def dfg(tr: pd.Series) -> Counter:
    c = Counter()
    for t in tr:
        c.update(zip(t, t[1:]))
    return c


def discover_edges(tr: pd.Series, min_frac: float = 0.02) -> set[tuple[str, str]]:
    """Keep directly-follows edges seen in at least min_frac of cases (drops rare noise)."""
    n = len(tr)
    return {e for e, k in dfg(tr).items() if k / n >= min_frac}


def fitness(tr: pd.Series, edges: set) -> float:
    """Share of cases whose every step-to-step move is allowed by the model."""
    ok = sum(all(e in edges for e in zip(t, t[1:])) for t in tr)
    return ok / len(tr)


def variants(tr: pd.Series) -> Counter:
    return Counter(tr)


def learn_rule(cases: pd.DataFrame, tr: pd.Series, activity: str, features=("amount", "vendor_new"),
               max_depth: int = 2):
    """Which case attributes decide whether `activity` happens? Small tree -> readable rule."""
    y = tr.reindex(cases["case_id"]).map(lambda t: int(activity in t)).to_numpy()
    X = cases[list(features)].to_numpy()
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, random_state=0).fit(X, y)
    return tree, export_text(tree, feature_names=list(features), decimals=0)
