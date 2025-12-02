# Auto-generated: fixed legend order and asymmetric error bars clipping at 0 if needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("25_places_entropy_summary.csv")
legend_order = ["GPT 4.1", "Gemini 2.0 Flash", "Claude Haiku 4.5", "Oracle"]
present = list(df["model"].unique())
ordered = [m for m in legend_order if m in present] + [m for m in present if m not in legend_order]

needs_asym_clip = (df["entropy_bits_mean"] - df["entropy_bits_std"]).min() < -1e-9

plt.figure(figsize=(8, 5))
for model_name in ordered:
    g = df[df["model"] == model_name].sort_values("step").copy()
    x = g["step"].to_numpy()
    y = g["entropy_bits_mean"].to_numpy()
    std = g["entropy_bits_std"].to_numpy()
    if needs_asym_clip:
        lower = np.minimum(std, y)
        upper = std
        yerr = np.vstack([lower, upper])
    else:
        yerr = std
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3, label=model_name)

plt.xlabel("Step")
plt.ylabel("Entropy = log2(# of Remaining Options)")
plt.title("Average entropy by step (25 places OQA)")
plt.legend()
plt.tight_layout()
plt.savefig("25_places_entropy_plot.png", dpi=160)
plt.close()
