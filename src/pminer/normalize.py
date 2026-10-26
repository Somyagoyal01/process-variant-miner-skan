"""Turn raw, team-specific UI/event labels into canonical activities.

No labels are used here. Three signals:
  1. text similarity of the labels (TF-IDF over words + character n-grams),
  2. where in a case the label usually happens (mean relative position),
  3. cannot-link: two labels that often appear in the SAME case are probably
     different steps, even if their text is similar ("PR created" vs "PO created").
Noise actions (opening mail, refreshing) are found first: they can happen at
any point of a case, so their position in the case is spread out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def collapse_repeats(ev: pd.DataFrame) -> pd.DataFrame:
    """Drop a row when it repeats the previous raw label in the same case (double clicks)."""
    ev = ev.sort_values(["case_id", "ts"]).reset_index(drop=True)
    prev = ev.groupby("case_id")["raw_label"].shift()
    return ev[ev["raw_label"] != prev].reset_index(drop=True)


def position_spread(ev: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each label's relative position in its case (0 = first, 1 = last)."""
    ev = ev.copy()
    ev["pos"] = ev.groupby("case_id").cumcount()
    ev["len"] = ev.groupby("case_id")["raw_label"].transform("size")
    ev["rel"] = ev["pos"] / (ev["len"] - 1).clip(lower=1)
    return ev.groupby("raw_label")["rel"].agg(["mean", "std"]).fillna(0)


def find_noise(ev: pd.DataFrame, threshold: float = 0.2) -> set[str]:
    """Real steps happen at a typical point in a case; UI noise can happen anywhere.

    A label whose relative position has a standard deviation above `threshold`
    (uniform noise gives ~0.29) is treated as noise. Rare-but-real steps such as
    rework still sit at a stable point, so they survive.
    """
    spread = position_spread(ev)
    return set(spread[spread["std"] > threshold].index)


class LabelNormalizer:
    def __init__(self, distance_threshold: float = 0.6, w_text: float = 0.4,
                 w_pos: float = 0.6, cannot_link: float = 0.05):
        self.distance_threshold = distance_threshold
        self.w_text, self.w_pos, self.cannot_link = w_text, w_pos, cannot_link
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)

    def _vec(self, labels):
        return hstack([self.word.transform(labels), self.char.transform(labels)]).tocsr()

    def fit(self, ev: pd.DataFrame, noise: set[str]):
        ev = ev[~ev["raw_label"].isin(noise)].copy()
        ev["pos"] = ev.groupby("case_id").cumcount()
        ev["len"] = ev.groupby("case_id")["raw_label"].transform("size")
        ev["rel"] = ev["pos"] / (ev["len"] - 1).clip(lower=1)
        stats = ev.groupby("raw_label").agg(rel=("rel", "mean"), n_cases=("case_id", "nunique"))
        labels = stats.index.tolist()
        self.word.fit(labels)
        self.char.fit(labels)
        X = self._vec(labels)
        text_d = 1 - cosine_similarity(X)
        pos = stats["rel"].to_numpy()
        pos_d = np.minimum(1.0, np.abs(pos[:, None] - pos[None, :]) * 5)
        D = self.w_text * text_d + self.w_pos * pos_d
        # cannot-link from co-occurrence in the same case
        sets = ev.groupby("case_id")["raw_label"].apply(set)
        idx = {l: i for i, l in enumerate(labels)}
        co = np.zeros((len(labels), len(labels)))
        for s in sets:
            ids = [idx[l] for l in s]
            for a in ids:
                co[a, ids] += 1
        n = stats["n_cases"].to_numpy()
        co_rate = co / np.minimum(n[:, None], n[None, :])
        np.fill_diagonal(co_rate, 0)
        D[co_rate > self.cannot_link] = 1.0
        np.fill_diagonal(D, 0)
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=self.distance_threshold).fit(D)
        # name each cluster after its most frequent member label
        freq = ev["raw_label"].value_counts()
        self.label_to_cluster, self.cluster_names = {}, {}
        for lab, c in zip(labels, cl.labels_):
            self.label_to_cluster[lab] = int(c)
        for c in set(cl.labels_):
            members = [l for l, k in self.label_to_cluster.items() if k == c]
            self.cluster_names[int(c)] = max(members, key=lambda m: freq[m])
        self.labels_, self.X_ = labels, X
        self.stats_ = stats
        return self

    def transform_log(self, ev: pd.DataFrame, noise: set[str]) -> pd.DataFrame:
        ev = ev[~ev["raw_label"].isin(noise)].copy()
        ev["activity"] = ev["raw_label"].map(lambda l: self.cluster_names[self.label_to_cluster[l]])
        prev = ev.groupby("case_id")["activity"].shift()
        return ev[ev["activity"] != prev].reset_index(drop=True)  # same step logged twice

    def assign_new(self, new_labels: list[str]) -> list[str]:
        """Map unseen label text to the closest known cluster (text only, no context)."""
        sims = cosine_similarity(self._vec(new_labels), self.X_)
        best = sims.argmax(axis=1)
        return [self.cluster_names[self.label_to_cluster[self.labels_[j]]] for j in best]
