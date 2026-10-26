# process-variant-miner-skan

*Personal project by Somya Goyal, built while applying to Skan AI. Not affiliated with or endorsed by Skan AI; uses synthetic data only.*

Recover the real process from messy UI/event logs, with no labels: normalise team-specific action names into canonical steps, drop UI noise, discover the process graph and its variants, and learn the decision rule behind a conditional step. Everything is scored against ground truth and against **hand-written held-out data the code never saw**.

## Why this matters for Skan AI

- **The problem.** An agent that operates enterprise software needs to know the process (steps, branches, rules, exceptions), but that knowledge is spread across noisy logs where every team names the same step differently.
- **Who feels it.** The team building process representations for downstream agents. The Applied Scientist I posting describes "messy event logs", "process variant detection, noise removal, and trace consolidation", and "conformance checking against ground-truth processes".
- **What this shows.** A small, auditable pipeline for exactly that: label normalisation using text, position and cannot-link signals; noise removal; a directly-follows graph; variants; and a readable decision rule. It also includes the evaluation harness the posting asks for, with ground-truth conformance, a stress setting, and held-out sets written by hand.
- **Expected impact (estimate).** In a P2P-style log with 45 raw labels and 1,485 distinct raw sequences per 1,500 cases, the pipeline collapses them to 10 steps and 16 variants, so an analyst or agent reviews 16 shapes instead of ~1,500. The assumption is that real logs have similar label fragmentation. This is not measured on real customer data.

Source: the Skan AI *Applied Scientist I* job description on skanai.keka.com (Oct 2026). Process- and task-mining tools from other vendors (for example Celonis and UiPath) work on the same kind of problem, which is why I framed the evaluation around conformance and variants.

## Demo

```bash
pip install -r requirements.txt
python run.py          # writes results/metrics.json and results/process_model.md
pytest -q              # 6 tests
```

## Approach

1. **Synthetic log** (`generate.py`): a procure-to-pay process with 10 steps, a conditional manager approval (amount > 50,000 or a new vendor, with 3% policy exceptions), concurrent goods/invoice receipt, and a rework loop. Each step has 4 team-specific names (e.g. `PR created`, `New PR - SAP ME51N`). The generator adds UI noise (`Open Outlook`, `Refresh screen`) and double clicks. A **stress** setting adds more noise and mixed naming, 8% policy exceptions and 5% missing events.
2. **Noise removal** (`normalize.find_noise`): real steps happen at a typical point in a case, while UI noise happens anywhere. A label whose relative position has a standard deviation above 0.2 is treated as noise.
3. **Label normalisation** (`LabelNormalizer`): agglomerative clustering on a distance that mixes TF-IDF text similarity (words + character n-grams) and mean relative position. A **cannot-link** rule applies when two labels often occur in the same case, which keeps `PR created` and `PO created` apart even though their text is nearly the same.
4. **Discovery** (`discover.py`): a directly-follows graph with a frequency floor, fitness (the share of cases that replay on the model), variants, and a depth-2 decision tree for *when* manager approval happens.
5. **Evaluation** (`evaluate.py`): label accuracy, noise precision/recall, edge precision/recall vs the true model, fitness, variant counts, and two hand-written held-out sets:
   - `data/heldout_labels_handwritten.csv`: 20 new phrasings (e.g. "Line manager sign-off", "Bank transfer to supplier") written before the pipeline was run;
   - `data/heldout_cases_handwritten.csv`: 10 cases near the approval boundary.

## Results (from `python run.py`)

| metric | normal log | stress log |
|---|---|---|
| raw labels → clusters (true steps: 10) | 45 → 10 | 45 → 13 (over-split) |
| noise precision / recall | 1.00 / 1.00 | 1.00 / 1.00 |
| label accuracy (events) | 1.00 | 1.00 |
| edge precision / recall vs true model | 1.00 / 1.00 | 0.78 / 1.00 |
| fitness (cases replayable) | 1.00 | 0.93 |
| distinct raw sequences → discovered variants (true) | 1,485 → 16 (16) | 1,500 → 216 (15) |
| approval rule on 10 hand-written cases | 9/10 | 9/10 |
| **unseen wording, hand-written (20 labels)** | **14/20 = 0.70** | 0.70 |

Learned rule (normal log): `amount > ~50,039 OR vendor_new → approval`, which is close to the true 50,000. The miss is a hand-written case at 50,001, right on the boundary.

### What the numbers actually say

- **The synthetic log is easier than real logs.** Steps sit at very stable positions, so position does most of the work, and the perfect scores on the normal log mostly reflect that. Don't read them as accuracy on real data.
- **Stress exposes the real failure modes.** First, *missing events* create false "skip" edges (e.g. `Create purchase order → Receive goods`), which drop edge precision to 0.78 and fitness to 0.93. Second, *over-splitting* (13 clusters for 10 steps) inflates the variant count from 15 to 216 without mapping. Both are the kinds of failure the job description talks about turning into research problems.
- **Text-only mapping of new wording reaches 70%.** The 6 misses need meaning, not character overlap: "Bank transfer to supplier" → *Pay vendor*, and "Supplier bill arrived" → *Receive invoice*. That is where an embedding model or an LLM-as-judge step would come in. I did not run one here (offline only), so I make no claim about how much it would help.

## Limitations / next steps

- Synthetic data, a single process family, and no real UI traces or screen recordings.
- Missing-event repair (e.g. infer `Send PO` when `Create PO → Receive goods`) and a cluster-count criterion that doesn't use labels (currently a fixed threshold, with a sensitivity sweep in mind).
- Swap TF-IDF for sentence embeddings, and add an LLM-as-judge check on cluster names, calibrated against the hand-written set.
- Conformance with a Petri net / inductive miner (e.g. pm4py) instead of a directly-follows graph, to handle concurrency explicitly.

## Files

- `src/pminer/generate.py`: synthetic log + ground truth (true edges, true variants, approval policy)
- `src/pminer/normalize.py`: double-click collapse, noise detection, label clustering with cannot-link
- `src/pminer/discover.py`: directly-follows graph, fitness, variants, decision rule
- `src/pminer/evaluate.py`: all metrics + held-out checks
- `run.py`: runs both settings and writes `results/`
- `tests/test_pipeline.py`: 6 tests
