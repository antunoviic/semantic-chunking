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
        hr_key = f"hit_rate@{k}"
        ranked = self._sorted(results)
        header = (
            f"{'Strategy':<16} {'Chunks':>7} {'Avg Len':>8} {'Hit@1':>7} "
            f"{f'Hit@{k}':>7} {'MRR':>7} {'Avg Dist':>9} {'Ctx Chars':>10}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        for i, r in enumerate(ranked):
            marker = "*" if i == 0 else " "
            print(
                f"{r['strategy']:<15}{marker} "
                f"{r['chunk_count']:>7} "
                f"{r['avg_chunk_len']:>8} "
                f"{r.get('hit_rate@1', 0.0):>6.1f}% "
                f"{r[hr_key]:>6.1f}% "
                f"{r['mrr']:>7.3f} "
                f"{r['avg_dist_top1']:>9.3f} "
                f"{r.get('avg_retrieved_chars', 0):>10}"
            )
        print(sep)
        print("* best MRR — Ctx Chars = avg characters retrieved per query (context cost)")

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
        cmap   = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(names))]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Chunking Strategy Comparison — {doc_stem}", fontsize=13)

        def bar(ax, values, title, ylabel, fmt="{:.1f}"):
            bars = ax.bar(names, values, color=colors)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=25)

        bar(axes[0][0], [r[hr_key] for r in results],                    f"Hit Rate@{k} (%)",                       "%", "{:.1f}")
        bar(axes[0][1], [r["mrr"] for r in results],                      "MRR (higher = better)",                   "",  "{:.3f}")
        bar(axes[1][0], [r["avg_dist_top1"] for r in results],            "Avg Distance Top-1\n(lower = better)",    "",  "{:.3f}")
        bar(axes[1][1], [r.get("avg_retrieved_chars", 0) for r in results],
            "Context Cost\n(avg chars retrieved per query, lower = cheaper)", "chars", "{:.0f}")

        plt.tight_layout()
        out = self._dir / f"{doc_stem}_comparison.png"
        plt.savefig(out, dpi=150)
        print(f"[charts] Saved to {out}")
