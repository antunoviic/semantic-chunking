"""Forest plot of the 30 paired comparisons in eval_results/final_table_v4_nofilter.json.
Output: thesis/overleaf/figures/forest_v4.pdf (and a PNG preview next to it)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = json.load(open("eval_results/final_table_v4_nofilter.json"))["tests"]
DOCS = [("nasa", "NASA"), ("rfc9110", "RFC 9110"), ("wells", "Wells")]
BASES = [("fixed_matched", "Fixed-size"), ("recursive_matched", "Recursive (default)"),
         ("recursive_sent_matched", "Recursive (sentence ends)"),
         ("semantic_matched", "Semantic (tuned)"), ("packing_matched", "Packing (no model)")]
COL = {"nasa": "#1b6ca8", "rfc9110": "#c0392b", "wells": "#2e8b57"}
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.9), sharey=True)
rows = [(d, b) for d, _ in DOCS for b, _ in BASES]
ypos = {r: -(i + i // len(BASES) * 0.8) for i, r in enumerate(rows)}
for ax, k in zip(axes, (1, 3)):
    ax.axvline(0, color="0.4", lw=0.8)
    for t in T:
        if t["k"] != k:
            continue
        y = ypos[(t["doc"], t["base"])]
        ax.plot([t["lo"], t["hi"]], [y, y], color=COL[t["doc"]], lw=1.4)
        ax.plot(t["d"], y, "o", ms=4.5, color=COL[t["doc"]],
                mfc=COL[t["doc"]] if t["reject"] else "white", mew=1.2)
    ax.set_title(f"Hit@{k}")
    ax.set_xlabel("LLM $-$ baseline (pp)")
    ax.set_xlim(-12, 22)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_yticks([ypos[r] for r in rows])
axes[0].set_yticklabels([dict(BASES)[b] for _, b in rows])
for d, name in DOCS:
    ys = [ypos[(d, b)] for b, _ in BASES]
    axes[1].text(1.03, sum(ys) / len(ys), name, rotation=270, va="center", ha="left",
                 color=COL[d], fontweight="bold", transform=axes[1].get_yaxis_transform(), clip_on=False)
fig.tight_layout(rect=(0, 0, 0.96, 1))
Path("eval_results/analysis").mkdir(parents=True, exist_ok=True)
fig.savefig("eval_results/analysis/forest_v4.pdf")
fig.savefig("eval_results/analysis/forest_v4_preview.png", dpi=160)
print("written")
