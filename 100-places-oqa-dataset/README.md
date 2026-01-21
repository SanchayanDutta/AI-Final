# 100_PLACES OQA Dataset

This package contains a 100-item PLACES attribute table and ready-to-use prompts for running Optimal Question Asking (OQA) API experiments. It also includes an entropy plot bundle for quick visualization and benchmarking, plus a small QA report.

## Repository layout

      100-places-oqa-dataset/
      ├─ data/
      │ ├─ 100_places.json
      │ ├─ 100_Places.json
      │ ├─ items.txt
      │ ├─ attributes.txt
      │ └─ equivalence_classes.json
      ├─ plots/
      │ ├─ 100_places_entropy_summary.csv
      │ ├─ 100_places_entropy_summary.json
      │ ├─ 100_places_entropy_seeds.csv
      │ ├─ 100_places_entropy_plot.png
      │ └─ make_plot.py
      ├─ prompts/
      │ ├─ prompt_plain.txt
      │ ├─ prompt_strict_json.txt
      │ └─ prompt_system.txt
      ├─ metadata.json
      └─ README.md

## Contents
- `data/100_places.json` - canonical boolean attribute table. The duplicate file `100_Places.json` is provided for naming convenience.
- `data/items.txt` - list of place names.
- `data/attributes.txt` - list of attribute names.
- `data/equivalence_classes.json` - items grouped by identical attribute vectors.
- `metadata.json` - dataset metadata.
- `QA_REPORT.txt` - lightweight consistency checks on equivalence classes and entropy means.
- `prompts/prompt_plain.txt` - plain-text game prompt.
- `prompts/prompt_strict_json.txt` - strict JSON-output prompt for programmatic evaluation.
- `prompts/prompt_system.txt` - short system role to stabilize outputs.
- `plots/100_places_entropy_summary.csv` - tidy table with columns: model, step, entropy_bits_mean, entropy_bits_std, entropy_bits_lo, entropy_bits_hi.
- `plots/100_places_entropy_summary.json` - JSON version of the same table plus a small metadata block.
- `plots/100_places_entropy_seeds.csv` - per-turn samples for variation analysis (5 models × 8 steps × 10 seeds = 400 rows).
- `plots/100_places_entropy_plot.png` - line plot with error bars.
- `plots/make_plot.py` - script that recreates the main entropy figure from the CSV.

## Game protocol
- Hidden target is sampled uniformly from `data/items.txt`.
- The agent asks yes or no questions about attributes. Answers are truthful and noise free.
- The dialog ends when a single candidate remains or when a single equivalence class remains.
- External tools are disallowed for the agent.

## Dataset stats
- Items: 100
- Attributes: 13
- Duplicates present: False
- Equivalence classes: 100

## Quickstart (pseudo)
1. Send `prompts/prompt_system.txt` as the system message.
2. Send `prompts/prompt_plain.txt` or `prompts/prompt_strict_json.txt` as the user message.
3. Loop:
   - Read the model's `question` field or parse the JSON object.
   - Answer `Yes` or `No` from `data/100_places.json` for the hidden target.
   - Feed the answer back to the model.
   - Stop when only one candidate or one equivalence class remains.

## Plot bundle usage
- The figure `plots/100_places_entropy_plot.png` visualizes entropy in bits across steps for five series: GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, and Oracle.
- The CSV in `plots/100_places_entropy_summary.csv` drives the plot, and `plots/100_places_entropy_seeds.csv` contains the underlying per run samples.
- To regenerate the figure:

```bash
cd plots
python make_plot.py
