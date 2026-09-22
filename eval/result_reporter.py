from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ResultReporter:
    """SRP: prints the results table and saves JSON + Markdown + chart files."""

    def __init__(self, results_dir: Path = Path("./eval_results")) -> None:
        self._dir = results_dir

    @staticmethod
    def _sorted(results: list[dict]) -> list[dict]:
        return sorted(results, key=lambda r: r["mrr"], reverse=True)

    def print_table(self, results: list[dict], k: int) -> None:
        """One row per strategy, ranked by MRR.

        The name column is sized from the data. It used to be a fixed 15
        characters while names run to 33, so every long name pushed its own
        row right and the header stopped lining up with the numbers.
        """
        hr_key = f"hit_rate@{k}"
        ranked = self._sorted(results)
        name_w = max([len(r["strategy"]) for r in ranked] + [len("Strategy")])
        rank_w = len(str(len(ranked)))

        cols = (f"{'':>{rank_w}}  {'Strategy':<{name_w}}  {'Chunks':>7} {'Avg Len':>7} "
                f"{'Hit@1':>7} {f'Hit@{k}':>7} {'MRR':>6} {'Avg Dist':>8} {'Ctx Chars':>9}")
        rule = "-" * len(cols)
        print(f"\n{'=' * len(cols)}\n{cols}\n{rule}")

        for i, r in enumerate(ranked, 1):
            best = " *" if i == 1 else "  "
            print(
                f"{i:>{rank_w}}. {r['strategy']:<{name_w}}  "
                f"{r['chunk_count']:>7} "
                f"{r['avg_chunk_len']:>7} "
                f"{r.get('hit_rate@1', 0.0):>6.1f}% "
                f"{r[hr_key]:>6.1f}% "
                f"{r['mrr']:>6.3f} "
                f"{r['avg_dist_top1']:>8.3f} "
                f"{r.get('avg_retrieved_chars', 0):>9}"
                f"{best}".rstrip()
            )
        print(rule)
        print("* best MRR   |   Ctx Chars = characters returned per query, "
              "the context cost of the hit rate")
        print("=" * len(cols))

    def save_json(self, results: list[dict], doc_stem: str) -> None:
        self._dir.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._dir / f"{doc_stem}_{ts}.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[results] Saved to {out}")

    def save_markdown(self, results: list[dict], k: int, doc_stem: str) -> None:
        """Markdown table for direct use in the thesis."""
        self._dir.mkdir(exist_ok=True)
        hr_key = f"hit_rate@{k}"
        lines = [
            f"# Chunking Strategy Comparison — {doc_stem}",
            "",
            f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"| Strategy | Chunks | Avg Len | Hit@1 | Hit@{k} | MRR | Avg Dist Top-1 | Ctx Chars/Query |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self._sorted(results):
            lines.append(
                f"| {r['strategy']} | {r['chunk_count']} | {r['avg_chunk_len']} "
                f"| {r.get('hit_rate@1', 0.0):.1f}% | {r[hr_key]:.1f}% | {r['mrr']:.3f} "
                f"| {r['avg_dist_top1']:.3f} | {r.get('avg_retrieved_chars', 0)} |"
            )
        out = self._dir / f"{doc_stem}_results.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[results] Markdown table saved to {out}")

    def save_charts(self, results: list[dict], k: int, doc_stem: str) -> None:
        """Four horizontal bar charts, one per metric.

        Horizontal rather than vertical because the strategy names are long:
        rotated, they overlapped each other and the value labels ran together.
        Each chart is sorted by the value it shows, so the order carries the
        message. Context cost uses a log scale because semantic_lc exceeds the
        other strategies by an order of magnitude and would otherwise squash
        every remaining bar into an invisible sliver.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[charts] matplotlib not installed — skipping")
            return

        self._dir.mkdir(exist_ok=True)
        hr_key = f"hit_rate@{k}"

        fig, axes = plt.subplots(2, 2, figsize=(16, max(8, 0.5 * len(results) + 4)))
        fig.suptitle(f"Chunking Strategy Comparison — {doc_stem}", fontsize=15, y=0.98)

        def barh(ax, key, title, fmt, *, better_low=False, log=False):
            rows = sorted(results, key=lambda r: r.get(key, 0), reverse=better_low)
            names = [r["strategy"] for r in rows]
            values = [r.get(key, 0) for r in rows]
            # Both sort orders put the best value last, i.e. at the bottom:
            # ascending for higher-is-better, descending for lower-is-better.
            best = len(values) - 1
            colors = ["#2e7d32" if i == best else "#90a4ae" for i in range(len(values))]
            bars = ax.barh(names, values, color=colors, height=0.68)
            ax.set_title(title, fontsize=11, pad=8)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            if log:
                ax.set_xscale("log")
                ax.set_xlim(max(1, min(values) * 0.6), max(values) * 3)
            else:
                ax.set_xlim(0, max(values) * 1.22 if max(values) else 1)
            for b, v in zip(bars, values):
                ax.text(b.get_width() * (1.06 if log else 1) + (0 if log else max(values) * 0.015),
                        b.get_y() + b.get_height() / 2, fmt.format(v),
                        va="center", ha="left", fontsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        barh(axes[0][0], hr_key, f"Hit Rate@{k} (%) — higher is better", "{:.1f}")
        barh(axes[0][1], "mrr", "MRR — higher is better", "{:.3f}")
        barh(axes[1][0], "avg_dist_top1", "Avg distance top-1 — lower is better",
             "{:.3f}", better_low=True)
        barh(axes[1][1], "avg_retrieved_chars",
             "Context cost: chars retrieved per query (log scale) — lower is cheaper",
             "{:.0f}", better_low=True, log=True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = self._dir / f"{doc_stem}_comparison.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[charts] Saved to {out}")
