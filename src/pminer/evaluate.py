"""Score discovery against the generator's ground truth and the hand-written held-out sets."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .discover import discover_edges, fitness, learn_rule, traces, variants
from .generate import CANONICAL, TRUE_EDGES, GenConfig, generate
from .normalize import LabelNormalizer, collapse_repeats, find_noise

DATA = Path(__file__).resolve().parents[2] / "data"


def cluster_to_canonical(ev: pd.DataFrame, norm: LabelNormalizer) -> dict[str, str]:
    """Majority vote: which true step do the events in each discovered cluster belong to?"""
    ev = ev[ev["raw_label"].isin(norm.label_to_cluster)]
    names = ev["raw_label"].map(lambda l: norm.cluster_names[norm.label_to_cluster[l]])
    out = {}
    for name, grp in ev.groupby(names)["true_activity"]:
        out[name] = grp.value_counts().idxmax()
    return out


def run_all(cfg: GenConfig = GenConfig(), min_frac: float = 0.02) -> dict:
    ev_raw, cases = generate(cfg)
    ev = collapse_repeats(ev_raw)
    noise = find_noise(ev)
    true_noise = set(ev.loc[ev.true_activity == "NOISE", "raw_label"])
    norm = LabelNormalizer().fit(ev, noise)
    log = norm.transform_log(ev, noise)
    c2c = cluster_to_canonical(ev, norm)

    # 1) label normalisation: share of real events whose cluster maps to the right step
    real = ev[ev.true_activity != "NOISE"]
    real = real[real.raw_label.isin(norm.label_to_cluster)]
    pred = real.raw_label.map(lambda l: c2c[norm.cluster_names[norm.label_to_cluster[l]]])
    label_acc = float((pred == real.true_activity).mean())

    # 2) process model
    tr = traces(log)
    tr_c = tr.map(lambda t: tuple(c2c[a] for a in t))
    edges = discover_edges(tr_c, min_frac)
    tp = len(edges & TRUE_EDGES)
    edge_p, edge_r = tp / max(len(edges), 1), tp / len(TRUE_EDGES)
    fit = fitness(tr_c, edges)
    raw_variants = cases["true_variant"].nunique()
    found_variants = len(variants(tr_c))
    raw_label_variants = ev_raw.groupby("case_id")["raw_label"].apply(tuple).nunique()

    # 3) decision rule for the conditional step, checked on hand-written cases
    tree, rule_text = learn_rule(cases, tr_c, "Manager approval")
    ho_cases = pd.read_csv(DATA / "heldout_cases_handwritten.csv")
    ho_pred = tree.predict(ho_cases[["amount", "vendor_new"]].to_numpy())
    rule_acc_heldout = float((ho_pred == ho_cases["needed_approval"]).mean())

    # 4) unseen wording, written by hand before running the pipeline
    ho = pd.read_csv(DATA / "heldout_labels_handwritten.csv")
    ho_assigned = [c2c[a] for a in norm.assign_new(ho.raw_label.tolist())]
    ho["predicted"] = ho_assigned
    ho_acc = float((ho["predicted"] == ho["canonical"]).mean())

    return {
        "n_cases": cfg.n_cases,
        "n_raw_labels": int(ev_raw.raw_label.nunique()),
        "n_clusters": len(norm.cluster_names),
        "noise_labels_found": sorted(noise),
        "noise_precision": len(noise & true_noise) / max(len(noise), 1),
        "noise_recall": len(noise & true_noise) / max(len(true_noise), 1),
        "label_accuracy": round(label_acc, 4),
        "edge_precision": round(edge_p, 3),
        "edge_recall": round(edge_r, 3),
        "missing_edges": sorted(TRUE_EDGES - edges),
        "extra_edges": sorted(edges - TRUE_EDGES),
        "fitness": round(fit, 4),
        "raw_label_sequences": int(raw_label_variants),
        "discovered_variants": int(found_variants),
        "discovered_variants_no_mapping": int(len(variants(tr))),
        "true_variants": int(raw_variants),
        "rule_text": rule_text,
        "rule_acc_heldout_cases": rule_acc_heldout,
        "heldout_label_accuracy": round(ho_acc, 3),
        "heldout_misses": ho[ho.predicted != ho.canonical][["raw_label", "canonical", "predicted"]].to_dict("records"),
        "unmapped_canonical": sorted(set(CANONICAL) - set(c2c.values())),
        "top_variants": [(" > ".join(v), k) for v, k in variants(tr_c).most_common(5)],
    }
