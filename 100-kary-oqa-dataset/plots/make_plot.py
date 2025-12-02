import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Order of models in the legend
order = ["GPT 5", "Gemini 2.5 Pro", "Claude Sonnet 4.5", "Grok 4", "Oracle (Optimal)"]

# Explicit color mapping
color_map = {
    "GPT 5": "#003153",  # Prussian Blue
    "Gemini 2.5 Pro": "tab:orange",
    "Claude Sonnet 4.5": "tab:green",
    "Grok 4": "tab:red",
    "Oracle (Optimal)": "violet",
}

df = pd.read_csv("kary100_entropy_summary.csv")

plt.figure(figsize=(9, 5))
for model in order:
    g = df[df["model"] == model].sort_values("step")
    if g.empty:
        continue
    x = g["step"].to_numpy()
    y = g["entropy_bits_mean"].to_numpy()
    std = g["entropy_bits_std"].to_numpy()

    # Asymmetric error bars, but never below zero entropy
    lower = np.minimum(std, y)
    upper = std
    yerr = np.vstack([lower, upper])

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="-o",
        capsize=3,
        label=model,
        color=color_map.get(model, None),
    )

plt.title("K-ary 100 Objects Dataset: Entropy (in bits) Across Steps")
plt.xlabel("Step")
plt.ylabel("Entropy = log2(# of Remaining Options)")
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("kary100_entropy_plot.png", dpi=160)
plt.close()