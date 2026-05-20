"""
End-to-end RAG Evaluator.

For each chunking strategy:
  1. Retrieve top-k chunks from ChromaDB
  2. Generate an answer with the local Qwen model
  3. Score the answer with GPT-4 (or Qwen as fallback)

Metrics per strategy:
  - avg_score   : average GPT score (0-2) across all questions
  - score_pct   : avg_score / 2 * 100  (percentage)
  - per-question detail saved to JSON for manual inspection
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from llm_chunker.interfaces import LLMClient
from llm_chunker.vectorstore import VectorStore


_GENERATE_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say so briefly."
)

_EVALUATE_SYSTEM = (
    "You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system. "
    "You will be given a question, a reference text (ground truth), and a generated answer. "
    "Score the generated answer:\n"
    "  2 = fully correct and complete based on the reference\n"
    "  1 = partially correct or incomplete\n"
    "  0 = incorrect, irrelevant, or question not answered\n"
    "Respond with ONLY the number 0, 1, or 2. Nothing else."
)


class RAGEvaluator:

    def __init__(
        self,
        generator: LLMClient,
        evaluator: LLMClient,
        results_dir: Path = Path("./eval_results"),
    ) -> None:
        self._gen   = generator   # local Qwen — generates answers
        self._eval  = evaluator   # GPT-4 (or Qwen fallback) — scores answers
        self._dir   = results_dir
        self._dir.mkdir(exist_ok=True)

    def evaluate_all(
        self,
        store: VectorStore,
        strategy_collections: dict[str, str],   # name → collection_name
        qa_pairs: list[dict],
        k: int = 3,
        pdf_stem: str = "doc",
    ) -> list[dict]:
        """
        Evaluate every strategy and return summary results.

        strategy_collections: e.g. {"llm": "eval_llm", "fixed_256": "eval_fixed_256", ...}
        """
        all_results = []
        for name, collection in strategy_collections.items():
            print(f"  [rag-eval] {name}...")
            result = self._evaluate_strategy(name, collection, store, qa_pairs, k)
            all_results.append(result)
            print(f"    score: {result['score_pct']:.1f}%  ({result['avg_score']:.2f}/2.00)")

        self._save(all_results, pdf_stem)
        return all_results

    def _evaluate_strategy(
        self,
        name: str,
        collection: str,
        store: VectorStore,
        qa_pairs: list[dict],
        k: int,
    ) -> dict:
        details = []
        for qa in qa_pairs:
            results   = store.query(collection, qa["question"], k=k)
            context   = "\n\n---\n\n".join(r.chunk_text for r in results)
            answer    = self._generate(qa["question"], context)
            score     = self._score(qa["question"], qa["source_text"], answer)
            details.append({
                "question":    qa["question"],
                "source_text": qa["source_text"],
                "context":     context,
                "answer":      answer,
                "score":       score,
            })

        scores    = [d["score"] for d in details]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "strategy":   name,
            "avg_score":  round(avg_score, 3),
            "score_pct":  round(avg_score / 2 * 100, 1),
            "n":          len(scores),
            "details":    details,
        }

    def _generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return self._gen.chat(messages)

    def _score(self, question: str, source_text: str, answer: str) -> int:
        messages = [
            {"role": "system", "content": _EVALUATE_SYSTEM},
            {"role": "user",   "content": (
                f"Question: {question}\n\n"
                f"Reference text: {source_text}\n\n"
                f"Generated answer: {answer}"
            )},
        ]
        raw = self._eval.chat(messages).strip()
        # extract first digit
        for ch in raw:
            if ch in "012":
                return int(ch)
        return 0  # fallback if model doesn't follow instructions

    def _save(self, results: list[dict], pdf_stem: str) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = self._dir / f"{pdf_stem}_rag_eval_{ts}.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rag-eval] Detailed results saved to {json_path}")

        md_path = self._dir / f"{pdf_stem}_rag_eval_{ts}.md"
        md_path.write_text(self._to_markdown(results, pdf_stem), encoding="utf-8")
        print(f"[rag-eval] Review file saved to {md_path}  ← upload this to Claude")

    @staticmethod
    def _to_markdown(results: list[dict], pdf_stem: str) -> str:
        lines = [
            f"# RAG Evaluation — {pdf_stem}",
            "",
            "**Task for Claude:** For each question below, score the generated answer:",
            "- **2** = fully correct and complete based on the reference text",
            "- **1** = partially correct or incomplete",
            "- **0** = incorrect, irrelevant, or not answered",
            "",
            "At the end, provide a summary table: Strategy | Avg Score | Score %",
            "",
            "---",
            "",
        ]
        for strategy_result in results:
            name = strategy_result["strategy"]
            lines.append(f"## Strategy: `{name}`")
            lines.append("")
            for i, d in enumerate(strategy_result["details"], 1):
                lines += [
                    f"### Q{i}: {d['question']}",
                    "",
                    f"**Reference text:** {d['source_text']}",
                    "",
                    f"**Generated answer:** {d['answer']}",
                    "",
                    f"*(Local scorer gave: {d['score']}/2)*",
                    "",
                    "---",
                    "",
                ]
        return "\n".join(lines)

    @staticmethod
    def print_table(results: list[dict]) -> None:
        print()
        print("=" * 45)
        print(f"{'Strategy':<16} {'Score %':>8}  {'Avg (0-2)':>10}  {'N':>4}")
        print("=" * 45)
        for r in sorted(results, key=lambda x: x["score_pct"], reverse=True):
            print(f"{r['strategy']:<16} {r['score_pct']:>7.1f}%  {r['avg_score']:>10.3f}  {r['n']:>4}")
        print("=" * 45)
        print()
