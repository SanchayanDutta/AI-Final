# K-ary 100 Objects OQA Dataset (30 targets)

This bundle packages a small Optimal Question Asking (OQA) experiment on a 100‑object
k‑ary attribute table together with oracle and model entropy trajectories.

The goal is to study how quickly different agents reduce uncertainty about a hidden
object when they ask multi‑way (k‑ary) questions about its attributes.

## Contents

```
100-kary-oqa-dataset/
├─ metadata.json
├─ README.md
├─ data/
│  └─ oqa_kary100_dataset.json
├─ oracle_dp/
│  └─ oracle_kary100_dp.py
├─ plots/
│  ├─ kary100_entropy_seeds.csv
│  ├─ kary100_entropy_summary.csv
│  ├─ kary100_entropy_summary.json
│  ├─ kary100_entropy_plot.png
│  ├─ oqa_kary100_entropy_plot_30targets_crossing.png
│  └─ make_plot.py
└─ prompts/
   └─ prompt_template_generic.txt
```

### Attribute table

`data/oqa_kary100_dataset.json` stores the 100 objects and their discrete attributes.
Keys are string identifiers (e.g., `"0000"`) and each value is a dictionary with
entries such as `color`, `shape`, `material`, `size`, `pattern`, `origin`,
`use_case`, and `energy`. Each object has a unique attribute vector. fileciteturn0file2

### Oracle dynamic program

`oracle_dp/oracle_kary100_dp.py` implements the exact information‑theoretic oracle
used for the pink “Oracle (Optimal)” curve:

* It takes the attribute table as a Python dictionary (you can load the JSON and pass
  the resulting dict).
* It builds an optimal k‑ary decision tree by dynamic programming, assuming a uniform
  prior over objects and noiseless answers.
* For any chosen target object, it returns the posterior entropy trajectory
  `H_t = log2(# remaining candidates after t questions)` under the optimal policy.
* In the released summary and plot we use a 1-based step index where step 1 is the
  prior entropy `H_0 = log2(100)` (before any question), and steps 2–10 correspond
  to `H_1, ..., H_9`.
* Averaging these trajectories over 30 targets reproduces the oracle line in the main
  plot when shifted in this way.

### Entropy trajectories and summary statistics

`plots/kary100_entropy_seeds.csv` contains per‑run entropy trajectories for five
agents:

* GPT 5  
* Gemini 2.5 Pro  
* Claude Sonnet 4.5  
* Grok 4  
* Oracle (Optimal)

Each row records the model, dialog step (1–10), a run index `seed` (1–30), and the
entropy in bits at that step. Step 1 always equals the prior `log2(100)` for every
run and agent; steps 2–10 store `log2(n)` where `n` is the number of remaining
candidate objects after each question in that run, with values clipped at zero.

`plots/kary100_entropy_summary.csv` aggregates these trajectories, reporting the mean
entropy, standard deviation, and a simple ±1σ interval for each model and step.
The same information is mirrored in JSON form in
`plots/kary100_entropy_summary.json`.

### Plot generation

`plots/make_plot.py` reads `kary100_entropy_summary.csv` and regenerates
`kary100_entropy_plot.png`, which shows mean entropy with error bars across steps
for all agents. GPT 5 is shown in Prussian blue, Gemini 2.5 Pro in orange,
Claude Sonnet 4.5 in green, Grok 4 in red, and the oracle in violet.

To regenerate the figure:

```bash
cd plots
python make_plot.py
```

This will overwrite `kary100_entropy_plot.png` with the plot corresponding to the
current summary file.
