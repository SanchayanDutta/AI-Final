CARS-100: Posterior Entropy Across Steps

This folder contains a compact dataset and a plot for five series:
- GPT 5
- Gemini 2.5 Pro
- Claude Sonnet 4.5
- Grok 4
- Oracle

Files
- 100_cars_entropy_summary.csv           Tidy table of mean entropy and standard deviation per step
- 100_cars_entropy_summary.json          Same as JSON with a small metadata block
- 100_cars_entropy_seeds.csv             Ten samples per step and model to illustrate variation
- 100_cars_entropy_plot.png              Line plot with visible error bars
- make_plot.py                           Script that recreates the figure from the CSV

Usage
- Run the script to regenerate the figure:
  cd plots
  python make_plot.py
