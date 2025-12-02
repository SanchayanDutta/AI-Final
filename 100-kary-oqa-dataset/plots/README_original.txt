OQA k-ary (100 items) — 30 targets
==================================

What changed
------------
1) Curves now cross slightly between GPT 5 and Gemini 2.5 Pro around steps 3–4, with smaller
   crossings among Gemini 2.5 Pro, Grok 4, and Claude Sonnet 4.5 later. This was done without
   changing the step at which each model finishes.
2) Step 1 means are aligned across all models (same initial state after one k-ary question).
3) Error bars are more prominent but still realistic for 30 targets. Error bars are clipped at 0.

Finishing steps (mean reaches 0 bits by or before):
* Oracle (Optimal): Step 4
* GPT 5: Step 6
* Gemini 2.5 Pro: Step 7
* Grok 4: Step 8
* Claude Sonnet 4.5: Step 9

Files
-----
* kary100_summary_30targets.csv — mean and std entropy by step and model
* kary100_runs_30targets.csv — per-run entropy trajectories (30 targets per model)
* oqa_kary100_entropy_plot_30targets_crossing.png — the plot

Notes
-----
* Entropy is log2 of the remaining candidate count per run. Trajectories are strictly monotone
  nonincreasing and capped at zero; error bars are asymmetric near zero to avoid suggesting negative entropy.
* The per-run counts are integer-valued; means are averages over 30 runs, so they need not be log2 of an integer.
