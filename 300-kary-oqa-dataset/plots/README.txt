K-ary 300 Objects: Posterior Entropy Across Steps

This folder contains posterior entropy statistics for the 300-way OQA setting, aggregated over 30 target objects for five models:
- GPT 5
- Gemini 2.5 Pro
- Claude Sonnet 4.5
- Grok 4
- Oracle (Optimal)

Files
- kary300_runs_30targets.csv          Per-target entropy trajectories (model, run_id, step)
- kary300_summary_30targets.csv       Mean and standard deviation of entropy (in bits) per model and step
- kary300_entropy_summary.csv         Mean, std, and mean ± std per model and step
- kary300_entropy_summary.json        JSON version of the same summary table
- gpt5_summary.csv                    Per-model summary in the compact format
- gemini_2_5_pro_summary.csv          Per-model summary in the compact format
- claude_sonnet_4_5_summary.csv       Per-model summary in the compact format
- grok4_summary.csv                   Per-model summary in the compact format
- oracle_summary.csv                  Per-model summary in the compact format
- kary300_entropy_plot.png            Line plot with error bars (30 targets)
- kary300_entropy_plot_30targets.png  Same plot, kept for backward compatibility
- make_plot.py                        Script that recreates the figure from the summary CSV

Usage
- Run the script to regenerate the figure:
  cd plots
  python make_plot.py
