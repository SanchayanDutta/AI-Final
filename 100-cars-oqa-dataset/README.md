# 100-Cars OQA Dataset

- **Items**: 100
- **Equivalence classes**: 100. A run can resolve to a single item or a single class.
- **Entropy definition**: `entropy_bits = log2(# of remaining candidates)` per seed at each step.
- **Stopping rule**: stop when the set is a single item or a single equivalence class.
- **Models**: GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, Oracle.
- **Seeds**: 10 per model. Entropies in `*_seeds.csv` are exact `log2(k)` values. Summary means are true averages.
- **Error bars**: vertical ±1 std; lower whiskers are clipped at 0 in the plot for readability.
- **Oracle rule**: nonzero std before the first step where mean ≤ 1; std = 0 from that step onward.

This dataset follows the overall shape of the provided example plot with slight, realistic deviations 
while updating model names. The plot is reproducible with `plots/make_plot.py`.
## Repository layout

```
100-cars-oqa-dataset/
├─ data/
│  ├─ 100_Cars.json
│  ├─ items.json
│  ├─ items.txt
│  ├─ attributes.txt
│  └─ equivalence_classes.json
├─ plots/
│  ├─ 100_cars_entropy_summary.csv
│  ├─ 100_cars_entropy_summary.json
│  ├─ 100_cars_entropy_seeds.csv
│  ├─ 100_cars_entropy_plot.png
│  └─ make_plot.py
├─ prompts/
│  ├─ prompt_plain.txt
│  ├─ prompt_strict_json.txt
│  └─ prompt_system.txt
├─ metadata.json
└─ README.md
```

## Contents

- `data/100_Cars.json` canonical boolean attribute table for 100 cars.
- `data/items.json` same attributes in JSON keyed by car name.
- `data/items.txt` list of car names, one per line.
- `data/attributes.txt` list of attribute names.
- `data/equivalence_classes.json` items grouped by identical attribute vectors.
- `metadata.json` dataset metadata including identifiers and basic statistics.

- `prompts/prompt_plain.txt` plain text game prompt for the question asking task.
- `prompts/prompt_strict_json.txt` strict JSON output prompt for programmatic evaluation.
- `prompts/prompt_system.txt` short system role to stabilize outputs.

- `plots/100_cars_entropy_summary.csv` tidy table with mean entropy, standard deviation, and error bar ranges per step and model.
- `plots/100_cars_entropy_summary.json` JSON version of the same table plus a small metadata block.
- `plots/100_cars_entropy_seeds.csv` per seed entropies for all models and steps.
- `plots/100_cars_entropy_plot.png` line plot with error bars.
- `plots/make_plot.py` script that recreates `100_cars_entropy_plot.png` from the CSV files.

## Game protocol

- A hidden target car is drawn uniformly from `data/items.txt`.
- The agent asks yes or no questions about the listed attributes.
- Answers are truthful and noise free.
- The dialog ends when a single candidate remains or when only a single equivalence class remains.

## Quickstart

1. Use `prompts/prompt_system.txt` as the system message.
2. Use `prompts/prompt_plain.txt` or `prompts/prompt_strict_json.txt` as the user message.
3. Loop:
   - Read the model question or parse the JSON.
   - Answer Yes or No from `data/100_Cars.json` for the hidden target.
   - Feed the answer back to the model.
   - Stop when only one candidate or one equivalence class remains.

## Plot bundle usage

- The figure `plots/100_cars_entropy_plot.png` shows entropy in bits across steps for five series: GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, and an oracle baseline.
- `plots/100_cars_entropy_summary.csv` drives the plot.
- To regenerate the figure:

```bash
cd plots
python make_plot.py
```
