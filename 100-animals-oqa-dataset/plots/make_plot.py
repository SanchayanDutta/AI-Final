import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLOR_MAP = {
    "Oracle": "purple",
    "GPT 5": "blue",
    "Gemini 2.5 Pro": "orange",
    "Claude Sonnet 4.5": "green",
    "Grok 4": "red",
}
LEGEND_ORDER = ["Oracle", "GPT 5", "Gemini 2.5 Pro", "Claude Sonnet 4.5", "Grok 4"]

df = pd.read_csv("100_animals_entropy_summary.csv")

present = list(df["model"].unique())
ordered = [m for m in LEGEND_ORDER if m in present] + [m for m in present if m not in LEGEND_ORDER]

plt.figure(figsize=(8, 5))
for model_name in ordered:
    g = df[df["model"] == model_name].sort_values("step").copy()
    x = g["step"].to_numpy()
    y = g["entropy_bits_mean"].to_numpy()
    std = g["entropy_bits_std"].to_numpy()
    lower = np.minimum(std, y)
    upper = std
    yerr = np.vstack([lower, upper])
    color = COLOR_MAP.get(model_name, None)
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3, label=model_name, color=color)

plt.xlabel("Step")
plt.ylabel("Entropy (bits)")
plt.title("100 Animals Dataset: Entropy (bits) Across Steps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("100_animals_entropy_plot.png", dpi=160)
plt.close()
