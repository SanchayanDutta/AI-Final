# 300_KARY OQA Dataset

This package contains a 300 item k-ary attribute table and ready to use prompts for running Optimal Question Asking (OQA) API experiments. It also includes an entropy plot bundle for quick visualization and benchmarking.

## Repository layout

```
300-kary-oqa-dataset/
├─ data/
│  ├─ kary300_Items.json
│  ├─ kary300_Items.json
│  ├─ items.txt
│  ├─ attributes.txt
│  └─ equivalence_classes.json
├─ plots/
│  ├─ kary300_entropy_summary.csv
│  ├─ kary300_entropy_summary.json
│  ├─ kary300_entropy_plot_30targets.png
│  └─ make_plot.py
├─ prompts/
│  ├─ prompt_plain.txt
│  ├─ prompt_strict_json.txt
│  └─ prompt_system.txt
├─ metadata.json
└─ README.md
```

## Contents
- `data/kary300_Items.json` - canonical k-ary attribute table (duplicate file `kary300_Items.json` is provided for naming convenience).
- `data/items.txt` - list of item identifiers.
- `data/attributes.txt` - list of attribute names.
- `data/equivalence_classes.json` - items grouped by identical attribute vectors.
- `metadata.json` - dataset metadata.
- `prompts/prompt_plain.txt` - plain text game prompt for the k-ary setting.
- `prompts/prompt_strict_json.txt` - strict JSON output prompt for programmatic evaluation.
- `prompts/prompt_system.txt` - short system role to stabilize outputs.
- `plots/kary300_entropy_summary.csv` - tidy table with columns: model, step, entropy_bits_mean, entropy_bits_std, entropy_bits_lo, entropy_bits_hi.
- `plots/kary300_entropy_summary.json` - JSON version of the same table.
- `plots/kary300_entropy_plot_30targets.png` - line plot with error bars for all models.
- `plots/make_plot.py` - script that recreates the figure from the summary CSV.
- `plots/gpt5_summary.csv` and friends - original per model summaries preserved from the source package.

## Game protocol
- Hidden target is sampled uniformly from `data/items.txt`.
- The agent asks k-ary attribute questions such as "What is the color" or "What is the energy source". Answers are truthful and noise free.
- The dialog ends when a single candidate remains, or when a single equivalence class remains.
- External tools are disallowed for the agent.

## Dataset stats
- Items: 300
- Attributes: 8
- Duplicates present: False
- Equivalence classes: 300

## Quickstart (pseudo)
1. Send `prompts/prompt_system.txt` as the system message.
2. Send `prompts/prompt_plain.txt` or `prompts/prompt_strict_json.txt` as the user message.
3. Loop:
   - Read the model's `question` (or parse the JSON).
   - Answer with the appropriate attribute value from `data/kary300_Items.json` for the hidden target.
   - Feed the answer back to the model.
   - Stop when only one candidate (or one equivalence class) remains.

## Plot bundle usage
- The figure `plots/kary300_entropy_plot_30targets.png` visualizes entropy in bits across steps for five series: GPT 5, Gemini 2.5 Pro, Claude Sonnet 4.5, Grok 4, and Oracle.
- The CSV in `plots/kary300_entropy_summary.csv` drives the plot.
- To regenerate the figure:

```bash
cd plots
python make_plot.py
```
