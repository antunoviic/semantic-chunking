# Full Strategy Comparison — nasa, rfc9110, wells



## nasa (n = 292 questions, literal set)

Evaluated: 2026-09-08 00:15

| Strategy | Chunks | Avg Len | Hit@1 | Hit@3 | MRR | Avg Dist Top-1 | Ctx Chars/Query |
|---|---:|---:|---:|---:|---:|---:|---:|
| llm_incremental_parentchild | 3668 | 701 | 76.0% | 89.7% | 0.835 | 0.285 | 2248 |
| llm_incremental_headings | 1047 | 686 | 69.5% | 84.2% | 0.779 | 0.318 | 2178 |
| llm_incremental | 1035 | 703 | 69.2% | 83.6% | 0.779 | 0.320 | 2195 |
| llm_incremental_headings_hybrid | 1036 | 701 | 69.2% | 83.2% | 0.778 | 0.321 | 2201 |
| recursive_matched_701 | 1237 | 691 | 63.4% | 79.5% | 0.721 | 0.333 | 2165 |
| fixed_matched_701 | 1250 | 697 | 60.3% | 76.7% | 0.687 | 0.336 | 2092 |
| recursive | 1880 | 470 | 58.9% | 74.0% | 0.674 | 0.319 | 1443 |
| fixed_512 | 1762 | 508 | 58.6% | 74.7% | 0.671 | 0.322 | 1525 |
| fixed_256 | 3447 | 252 | 42.5% | 51.7% | 0.475 | 0.297 | 756 |
| semantic_lc | 302 | 2693 | 12.0% | 20.5% | 0.193 | 0.448 | 78660 |

---

## rfc9110 (n = 296 questions)

Evaluated: 2026-09-08 00:47

| Strategy | Chunks | Avg Len | Hit@1 | Hit@3 | MRR | Avg Dist Top-1 | Ctx Chars/Query |
|---|---:|---:|---:|---:|---:|---:|---:|
| recursive | 1393 | 361 | 85.1% | 94.6% | 0.900 | 0.271 | 1138 |
| llm_incremental_parentchild | 2360 | 700 | 85.1% | 92.2% | 0.892 | 0.257 | 2444 |
| llm_incremental_headings_hybrid | 637 | 700 | 83.1% | 90.9% | 0.874 | 0.288 | 2159 |
| llm_incremental | 642 | 695 | 83.1% | 90.5% | 0.871 | 0.289 | 2207 |
| llm_incremental_headings | 642 | 695 | 83.1% | 90.5% | 0.871 | 0.289 | 2204 |
| llm_enriched | 642 | 756 | 80.4% | 91.2% | 0.857 | 0.286 | 2642 |
| recursive_matched_700 | 730 | 692 | 79.4% | 90.2% | 0.856 | 0.298 | 2138 |
| llm_window | 735 | 600 | 75.3% | 89.9% | 0.826 | 0.294 | 1842 |
| fixed_matched_700 | 721 | 694 | 68.9% | 85.1% | 0.775 | 0.309 | 2088 |
| fixed_512 | 1014 | 507 | 64.5% | 78.7% | 0.720 | 0.299 | 1523 |
| semantic_lc | 154 | 3222 | 60.1% | 78.7% | 0.703 | 0.359 | 13813 |
| fixed_256 | 1991 | 251 | 51.4% | 59.8% | 0.559 | 0.276 | 755 |

---

## wells (n = 400 questions)

Evaluated: 2026-09-09 12:52

| Strategy | Chunks | Avg Len | Hit@1 | Hit@3 | MRR | Avg Dist Top-1 | Ctx Chars/Query |
|---|---:|---:|---:|---:|---:|---:|---:|
| recursive_matched_703 | 887 | 705 | 87.5% | 95.8% | 0.919 | 0.390 | 2455 |
| llm_incremental_parentchild | 3090 | 703 | 85.5% | 92.2% | 0.889 | 0.349 | 2294 |
| llm_incremental_headings | 871 | 703 | 83.5% | 92.0% | 0.880 | 0.390 | 2322 |
| fixed_matched_703 | 944 | 700 | 80.5% | 91.5% | 0.859 | 0.395 | 2098 |
| recursive | 1732 | 377 | 77.5% | 86.2% | 0.820 | 0.371 | 1258 |
| fixed_512 | 1335 | 509 | 76.5% | 87.8% | 0.818 | 0.387 | 1525 |
| semantic_lc | 229 | 2687 | 60.2% | 78.0% | 0.699 | 0.457 | 11158 |
| fixed_256 | 2611 | 253 | 50.0% | 59.8% | 0.551 | 0.368 | 758 |


---

## Summary — Hit@1 by document

| Document | Structure | llm_incremental (best variant) | Best classical baseline | Verdict |
|---|---|---:|---:|---|
| nasa | implicit | 69.5% (headings) | 63.4% (recursive_matched) | LLM chunking wins |
| rfc9110 | explicit | 83.1% | 85.1% (recursive / parentchild, tied) | roughly neutral |
| wells | none (prose) | 83.5% | 87.5% (recursive_matched) | LLM chunking loses |

Confirms the U-shaped relationship: the advantage of LLM-based chunking scales
with how much *implicit* structure the LLM can exploit that a length-based
splitter cannot see — strongest on nasa, gone on rfc9110 (already
heading-delimited, so a naive splitter benefits just as much), reversed on
wells (no structure to find, so the length-matched baseline wins on pure
retrieval density).
