# 25_CARS OQA Dataset

This package contains a 25-item CARS attribute table and ready-to-use prompts for running Optimal Question Asking (OQA) API experiments. It also includes an entropy plot bundle for quick visualization and benchmarking.

## Repository layout

```
25-cars-oqa-dataset/
├─ data/
│  ├─ 25_cars.json
│  ├─ items.txt
│  ├─ attributes.txt
│  └─ equivalence_classes.json
├─ plots/
│  ├─ 25_cars_entropy_summary.csv
│  ├─ 25_cars_entropy_summary.json
│  ├─ 25_cars_entropy_seeds.csv
│  ├─ 25_cars_entropy_plot.png
│  └─ make_plot.py
├─ prompts/
│  ├─ prompt_plain.txt
│  ├─ prompt_strict_json.txt
│  └─ prompt_system.txt
├─ metadata.json
└─ README.md
```

## Contents
- `data/25_cars.json` — canonical boolean attribute table (25 cars × attributes).
- `data/items.txt` — list of car names.
- `data/attributes.txt` — list of attribute names.
- `data/equivalence_classes.json` — items grouped by identical attribute vectors.
- `metadata.json` — dataset metadata.
- `prompts/prompt_plain.txt` — plain-text game prompt.
- `prompts/prompt_strict_json.txt` — strict JSON-output prompt for programmatic evaluation.
- `prompts/prompt_system.txt` — short system role to stabilize outputs.
- `plots/25_cars_entropy_summary.csv` — tidy table with columns: model, step, entropy_bits_mean, entropy_bits_std, entropy_bits_lo, entropy_bits_hi.
- `plots/25_cars_entropy_summary.json` — JSON version of the same table plus a small metadata block.
- `plots/25_cars_entropy_seeds.csv` — per-turn samples for variation analysis (4 models × 8 steps × 10 seeds = 320 rows).
- `plots/25_cars_entropy_plot.png` — line plot with error bars.
- `plots/make_plot.py` — script that recreates the figure from the CSV.

## Game protocol
- Hidden target is sampled uniformly from `data/items.txt`.
- The agent asks yes/no questions about attributes. Answers are truthful and noise-free.
- The dialog ends when a single candidate remains, or when a single equivalence class remains.
- External tools are disallowed for the agent.

## Dataset stats
- Items: 25
- Attributes: 20
- Duplicates present: False
- Equivalence classes: 25

## Quickstart (pseudo)
1. Send `prompts/prompt_system.txt` as the system message.
2. Send `prompts/prompt_plain.txt` or `prompts/prompt_strict_json.txt` as the user message.
3. Loop:
   - Read the model's `question` (or parse the JSON).
   - Answer "Yes" or "No" from `data/25_cars.json` for the hidden target.
   - Feed the answer back to the model.
   - Stop when only one candidate (or one equivalence class) remains.

## Plot bundle usage
- The figure `plots/25_cars_entropy_plot.png` visualizes entropy in bits across steps for four series: GPT 4.1, Gemini 2.0 Flash, Claude Haiku 4.5, and Oracle.
- The CSV in `plots/25_cars_entropy_summary.csv` drives the plot.
- To regenerate the figure:

```bash
cd plots
python make_plot.py
```