"""Figure 1 of the Track 1 report: leaderboard bridge from our first upload to the final submission.

Each bar is labelled with the absolute leaderboard score of that upload (4 decimals, as Codabench reports); the
green length is the gain over the previous row. Submission ids and scores: models/fusion_*.json (candidate_of).
"""
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = [("First upload (921278)", 1.4591),
         ("v14: library + 8 frozen heads", 1.5808),
         ("+ fine-tuned AST (aFG)", 1.5868),
         ("+ frozen BEATs (bH)", 1.58896),
         ("+ fine-tuned BEATs, Dasheng", 1.59382),
         ("boundary calibration", 1.59722),
         ("med 13, thr 0.34", 1.59992),
         ("med 15 (P2)", 1.6000),
         ("other-branch gap 2.0 (Q1)", 1.6002),
         ("+ bFS:4 (final, 994586)", 1.600895)]
RANK1 = 1.617289


def r4(v):
    return str(Decimal(str(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


fig, ax = plt.subplots(figsize=(3.5, 2.1), dpi=300)
prev = 0.0
for i, (name, v) in enumerate(STEPS):
    base = i == 0
    lo, h = (1.40, v - 1.40) if base else (prev, v - prev)
    last = i == len(STEPS) - 1
    ax.barh(i, h, left=lo, height=0.6, color="#9e9e9e" if base else "#4caf50", edgecolor="none")
    ax.text(v + 0.002, i, r4(v), va="center", fontsize=6.5, fontweight="bold" if last else "normal")
    if not last:
        ax.plot([v, v], [i + 0.3, i + 0.7], color="#757575", lw=0.5, ls=":")
    prev = v
n = len(STEPS)
ax.axvline(RANK1, color="#1565c0", lw=0.7, ls="--")
ax.text(RANK1 - 0.001, -0.75, f"rank 1: {RANK1:.4f}", color="#1565c0", fontsize=6, ha="right")
ax.set_yticks(range(n), [s for s, _ in STEPS], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(1.40, 1.64)
ax.set_xlabel("Leaderboard combined score (F1 + Dice)", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_bridge.pdf"))
print("ok")
