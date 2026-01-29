# KARY200_OBJECTS OQA Dataset

This package contains a 200 item k-ary OBJECTS attribute table and a set of
posterior-entropy statistics for Optimal Question Asking (OQA) style evaluation.

## Repository layout

```text
200-kary-oqa-dataset/
├─ data/
│  ├─ kary200_Objects.json
│  ├─ items.txt
│  ├─ attributes.txt
│  └─ equivalence_classes.json
├─ plots/
│  ├─ kary200_entropy_summary.csv
│  ├─ kary200_entropy_summary.json
│  ├─ kary200_runs_30targets.csv
│  ├─ kary200_summary_30targets.csv
│  ├─ posterior_entropy_summary.csv
│  ├─ gpt5_entropy.csv
│  ├─ gemini_2_5_pro_entropy.csv
│  ├─ claude_sonnet_4_5_entropy.csv
│  ├─ grok4_entropy.csv
│  ├─ oracle_entropy.csv
│  ├─ kary200_entropy_plot.png
│  └─ make_plot.py
├─ prompts/
│  ├─ prompt_plain.txt
│  ├─ prompt_strict_json.txt
│  └─ prompt_system.txt
├─ oracle/
│  └─ kary200_exact_oracle_dp.py
├─ metadata.json
└─ README.md
```

## Data files

- `data/kary200_Objects.json` is the canonical attribute table mapping each
  object ID (hex strings `"0000"` through `"00c7"`) to a dictionary of
  categorical attributes (`"color"`, `"shape"`, `"material"`, `"size"`,
  `"pattern"`, `"origin"`, `"use_case"`, `"energy"`).
- `data/items.txt` lists the object IDs, one per line, in the same order
  used when sampling hidden targets.
- `data/attributes.txt` lists the attribute names, one per line.
- `data/equivalence_classes.json` contains the same objects grouped by
  unique attribute vectors. In this dataset all 200 objects form their own
  equivalence class (no duplicates).

## Oracle dynamic program

The folder `oracle/` contains `kary200_exact_oracle_dp.py`, a self-contained
Python 3 module that implements the exact information-theoretic oracle for
multi-way (k-ary) attribute queries. Given an `items` dictionary loaded
from `data/kary200_Objects.json`, the class `KaryOracleDP`:

1. Builds the optimal decision tree that minimizes the expected number of
   attribute questions needed to isolate the target object.
2. Provides a convenience method `simulate_target(target_id)` that returns
   the entropy trajectory (in bits) as the oracle asks optimal questions.

If you sample 30 random target IDs and average the trajectories from
`simulate_target`, you recover the pink “Oracle (Optimal)” curve in
`plots/kary200_entropy_plot.png`.

## LLM evaluation protocol

The plots and summary tables assume the following closed-world guessing
game:

1. A hidden target is drawn uniformly from the 200 objects.
2. At each step a model asks a question about a single attribute
   (for example, the object's color or origin).
3. The environment answers truthfully with the attribute's value.
4. The model implicitly updates its candidate set and continues querying
   until only one object remains.

Posterior entropies are reported as `log2(# of remaining options)` and
are averaged across 30 random targets per model. Entropy at step 1
matches the prior `log2(200)` for every model, including the oracle.

## Plot bundle usage

- The figure `plots/kary200_entropy_plot.png` visualizes mean entropy
  trajectories and one-standard-deviation error bars for five agents:
  GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, and the Oracle.
- Per-model summaries live in the `*_entropy.csv` files listed above;
  `kary200_entropy_summary.csv` aggregates them into a single table, and
  `posterior_entropy_summary.csv` collects just the means by step.
- To regenerate the figure, run:

```bash
cd 200-kary-oqa-dataset/plots
python make_plot.py
```

All files are plain text (JSON or CSV) and require only standard Python
scientific libraries (NumPy, pandas, Matplotlib) to manipulate.

## Technical note

For the API experiments, the APIs were accessed on 16 November 2025.
