"""Synthetic procure-to-pay UI/event logs with the mess real logs have.

Every trace comes from a known ground-truth process so that discovery can be
scored. The mess we add on purpose:
  * the same step is labelled differently by different teams/tools,
  * UI noise (opening mail, refreshing a screen) between real steps,
  * double clicks (the same raw action logged twice in a row),
  * concurrency (goods receipt and invoice can arrive in either order),
  * a rework loop (match fails -> resolve mismatch -> match again),
  * a conditional step whose rule depends on case attributes,
  * a few policy exceptions, so the rule is not perfectly clean.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

CANONICAL = [
    "Create purchase requisition",
    "Manager approval",
    "Create purchase order",
    "Send PO to vendor",
    "Receive goods",
    "Receive invoice",
    "Three-way match",
    "Resolve mismatch",
    "Post invoice",
    "Pay vendor",
]

# How three different teams / systems name each step in the logs.
SURFACE_FORMS = {
    "Create purchase requisition": ["Create purchase requisition", "PR created", "New PR - SAP ME51N", "Requisition submitted"],
    "Manager approval": ["Manager approval", "Approve PR", "PR approved by manager", "Approval - L1"],
    "Create purchase order": ["Create purchase order", "PO created", "Convert PR to PO", "ME21N create PO"],
    "Send PO to vendor": ["Send PO to vendor", "PO emailed to supplier", "Dispatch purchase order", "Transmit PO"],
    "Receive goods": ["Goods receipt", "GRN posted", "Receive goods", "MIGO goods receipt"],
    "Receive invoice": ["Invoice received", "Vendor invoice received", "Scan invoice", "Invoice captured"],
    "Three-way match": ["Three-way match", "3-way match check", "Match PO GRN invoice", "Invoice matching"],
    "Resolve mismatch": ["Resolve price mismatch", "Mismatch resolved", "Fix quantity variance", "Clear match exception"],
    "Post invoice": ["Post invoice", "Invoice posted", "MIRO post invoice", "Book invoice"],
    "Pay vendor": ["Pay vendor", "Payment run", "Payment released", "F110 payment"],
}
NOISE = ["Open Outlook", "Refresh screen", "Copy to clipboard", "Idle 10 min", "Open Teams chat"]

APPROVAL_LIMIT = 50_000

# Ground-truth directly-follows edges of the real process (used only for scoring).
TRUE_EDGES = {
    ("Create purchase requisition", "Manager approval"),
    ("Create purchase requisition", "Create purchase order"),
    ("Manager approval", "Create purchase order"),
    ("Create purchase order", "Send PO to vendor"),
    ("Send PO to vendor", "Receive goods"),
    ("Send PO to vendor", "Receive invoice"),
    ("Receive goods", "Receive invoice"),
    ("Receive invoice", "Receive goods"),
    ("Receive goods", "Three-way match"),
    ("Receive invoice", "Three-way match"),
    ("Three-way match", "Resolve mismatch"),
    ("Resolve mismatch", "Three-way match"),
    ("Three-way match", "Post invoice"),
    ("Post invoice", "Pay vendor"),
}


@dataclass
class GenConfig:
    n_cases: int = 1500
    noise_rate: float = 0.25      # chance of a noise action after a step
    double_click_rate: float = 0.08
    mismatch_rate: float = 0.18
    exception_rate: float = 0.03  # approval policy not followed
    mix_rate: float = 0.15        # chance a step uses another team's label
    missing_rate: float = 0.0     # chance a real step was never logged
    seed: int = 7


# Harder setting: more noise, more mixed naming, missing events, more exceptions.
STRESS = GenConfig(noise_rate=0.6, double_click_rate=0.15, exception_rate=0.08,
                   mix_rate=0.4, missing_rate=0.05, seed=11)


def needs_approval(amount: float, vendor_new: int) -> bool:
    return amount > APPROVAL_LIMIT or vendor_new == 1


def generate(cfg: GenConfig = GenConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, cases). events: case_id, ts, raw_label, true_activity."""
    rng = random.Random(cfg.seed)
    events, cases = [], []
    for i in range(cfg.n_cases):
        cid = f"C{i:05d}"
        team = rng.randrange(4)  # which naming convention this team's tool uses
        amount = round(rng.lognormvariate(10.3, 1.0), 2)
        vendor_new = int(rng.random() < 0.15)
        approval = needs_approval(amount, vendor_new)
        if rng.random() < cfg.exception_rate:
            approval = not approval
        steps = ["Create purchase requisition"]
        if approval:
            steps.append("Manager approval")
        steps += ["Create purchase order", "Send PO to vendor"]
        recv = ["Receive goods", "Receive invoice"]
        rng.shuffle(recv)
        steps += recv + ["Three-way match"]
        while rng.random() < cfg.mismatch_rate and steps.count("Resolve mismatch") < 3:
            steps += ["Resolve mismatch", "Three-way match"]
        steps += ["Post invoice", "Pay vendor"]
        variant = " > ".join(steps)
        t = 0
        for s in steps:
            # mostly the team's own label, sometimes another form (shared tools)
            forms = SURFACE_FORMS[s]
            if rng.random() < cfg.missing_rate and s not in ("Create purchase requisition", "Pay vendor"):
                continue  # the step happened but the log never captured it
            label = forms[team] if rng.random() >= cfg.mix_rate else rng.choice(forms)
            reps = 2 if rng.random() < cfg.double_click_rate else 1
            for _ in range(reps):
                t += rng.randint(1, 600)
                events.append((cid, t, label, s))
            if rng.random() < cfg.noise_rate:
                t += rng.randint(1, 120)
                events.append((cid, t, rng.choice(NOISE), "NOISE"))
        cases.append((cid, amount, vendor_new, int("Manager approval" in steps), variant))
    ev = pd.DataFrame(events, columns=["case_id", "ts", "raw_label", "true_activity"])
    cs = pd.DataFrame(cases, columns=["case_id", "amount", "vendor_new", "had_approval", "true_variant"])
    return ev, cs
