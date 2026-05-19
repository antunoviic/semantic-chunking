from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ResultReporter:
    """SRP: prints the results table and saves JSON + chart files."""

    def __init__(self, results_dir: Path = Path("./eval_results")) -> None:
        self._dir = results_dir

    def print_table(self, results: list[dict], k: int) -> None:
        hr_key = f"hit_rate@{k}"
        header = f"{'Strategy':<15} {'Chunks':>7} {'Avg Len':>8} {f'Hit@{k}':>8} {'MRR':>7} {'Avg Dist':>9}"
        sep = "=" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        for r in results:
            print(
                f"{r['strategy']:<15} "
                f"{r['chunk_count']:>7} "
                f"{r['avg_chunk_len']:>8} "
                f"{r[hr_key]:>7.1f}% "
                f"{r['mrr']:>7.3f} "
                f"{r['avg_dist_top1']:>9.3f}"
            )
        print(sep)

    def save_json(self, results: list[dict], pdf_stem: str) -> None:
        self._dir.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._dir / f"{pdf_stem}_{ts}.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[results] Saved to {out}")

    def save_charts(self, results: list[dict], k: int, pdf_stem: str) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("Agg")
        except ImportError:
            print("[charts] matplotlib not installed — skipping")
            return

        self._dir.mkdir(exist_ok=True)
        hr_key = f"hit_rate@{k}"
        names  = [r["strategy"] for r in results]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Chunking Strategy Comparison — {pdf_stem}", fontsize=13)

        def bar(ax, values, title, ylabel, fmt="{:.1f}"):
            bars = ax.bar(names, values, color=colors[:len(names)])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=20)

        bar(axes[0], [r[hr_key] for r in results],         f"Hit Rate@{k} (%)",                  "%",  "{:.1f}")
        bar(axes[1], [r["mrr"] for r in results],           "MRR (higher = better)",              "",   "{:.3f}")
        bar(axes[2], [r["avg_dist_top1"] for r in results], "Avg Distance Top-1\n(lower = better)", "", "{:.3f}")

        plt.tight_layout()
        out = self._dir / f"{pdf_stem}_comparison.png"
        plt.savefig(out, dpi=150)
        print(f"[charts] Saved to {out}")
