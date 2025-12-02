CARS-25: Posterior Entropy Across Steps

This folder contains a compact dataset and a plot for four series:
- GPT 4.1
- Gemini 2.0 Flash
- Claude Haiku 4.5
- Oracle

Files
- 25_cars_entropy_summary.csv           Tidy table of mean entropy and standard deviation per step
- 25_cars_entropy_summary.json          Same as JSON with a small metadata block
- 25_cars_entropy_seeds.csv             Ten samples per step and model to illustrate variation
- 25_cars_entropy_plot.png              Line plot with error bars
- make_plot.py                           Script that recreates the figure from the CSV

Usage
- Run the script to regenerate the figure:
  cd plots
  python make_plot.py
