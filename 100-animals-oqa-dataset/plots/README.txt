100-ANIMALS: Posterior Entropy Across Steps

This folder contains the entropy tables and figure for five series:
- GPT 5
- Gemini 2.5 Pro
- Claude Sonnet 4.5
- Grok 4
- Oracle

Files
- 100_animals_entropy_summary.csv  mean entropy and standard deviation by step
- 100_animals_entropy_summary.json same information as JSON
- 100_animals_entropy_seeds.csv    ten runs per model and step
- 100_animals_entropy_plot.png     line plot with vertical error bars
- make_plot.py                     helper script that recreates the figure

Usage
- From this folder, run:
  python make_plot.py
