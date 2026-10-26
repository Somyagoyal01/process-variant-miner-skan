import pandas as pd

from pminer.discover import discover_edges, fitness, traces
from pminer.evaluate import run_all
from pminer.generate import STRESS, TRUE_EDGES, GenConfig, generate, needs_approval
from pminer.normalize import LabelNormalizer, collapse_repeats, find_noise

SMALL = GenConfig(n_cases=400, seed=3)


def test_generator_is_reproducible_and_follows_policy():
    ev1, cs1 = generate(SMALL)
    ev2, _ = generate(SMALL)
    assert ev1.equals(ev2)
    agree = (cs1.had_approval == cs1.apply(lambda r: int(needs_approval(r.amount, r.vendor_new)), axis=1)).mean()
    assert 0.9 < agree < 1.0  # policy mostly followed, with a few exceptions


def test_collapse_repeats_removes_double_clicks():
    ev = pd.DataFrame({"case_id": ["a"] * 4, "ts": [1, 2, 3, 4], "raw_label": ["x", "x", "y", "x"]})
    assert collapse_repeats(ev).raw_label.tolist() == ["x", "y", "x"]


def test_noise_found_without_labels():
    ev, _ = generate(SMALL)
    ev = collapse_repeats(ev)
    assert find_noise(ev) == set(ev.loc[ev.true_activity == "NOISE", "raw_label"])


def test_cannot_link_keeps_similar_text_apart():
    ev, _ = generate(SMALL)
    ev = collapse_repeats(ev)
    norm = LabelNormalizer().fit(ev, find_noise(ev))
    c = norm.label_to_cluster
    assert c["PR created"] != c["PO created"]
    assert c["Goods receipt"] != c["Invoice received"]


def test_fitness_and_edges():
    tr = pd.Series([("a", "b", "c"), ("a", "c")])
    edges = discover_edges(tr, min_frac=0.5)
    assert edges == {("a", "b"), ("b", "c"), ("a", "c")}
    assert fitness(tr, {("a", "b"), ("b", "c")}) == 0.5


def test_end_to_end_normal_and_stress():
    m = run_all(GenConfig(n_cases=600, seed=5))
    assert m["edge_recall"] == 1.0 and m["noise_precision"] == 1.0
    assert m["rule_acc_heldout_cases"] >= 0.8
    s = run_all(STRESS)
    assert s["edge_recall"] == 1.0
    assert s["fitness"] < 1.0  # missing events break replay: this is expected and reported
    assert 0 < s["heldout_label_accuracy"] < 1.0
