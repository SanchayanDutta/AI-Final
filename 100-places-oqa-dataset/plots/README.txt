100 Places: Posterior Entropy Across Steps (Realistic tuned lines)

This folder contains a compact dataset and plots for five series:
- GPT 5
- Gemini 2.5 Pro
- Claude Sonnet 4.5
- Grok 4
- Oracle

Files
- 100_places_entropy_summary.csv         Tidy table of mean entropy and standard deviation per step
- 100_places_entropy_summary.json        JSON version of the same table
- 100_places_entropy_seeds.csv           Per run entropy samples (10 seeds per model and step)
- 100_places_entropy_plot.png            Main tuned entropy plot used in the paper or report
- reference_plot.png                     Reference version of the plot prior to tuning
- make_plot.py                           Script that recreates the figure from the summary CSV

Conventions
- Entropy is defined as log2(# of remaining candidates) at each step.
- Runs stop when the remaining set is a single item or a single equivalence class.
- Error bars are ±1 std; the plotting script clips the lower whisker at 0 when the mean is very small.

Usage
- To regenerate the main figure, run:

  cd plots
  python make_plot.py
