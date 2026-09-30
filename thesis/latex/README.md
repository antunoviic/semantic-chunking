# LaTeX source of the thesis

Source of *LLM-Based Semantic Chunking for Retrieval-Augmented Generation: A
Comparison with Rule-Based Chunking Strategies* (Ivan Antunovic, bachelor's
thesis, University of Vienna, 2026; supervisor Dr. Marian Lux). The compiled
PDF is one level up, in [`../thesis.pdf`](../thesis.pdf).

The library it describes is the package in this repository; the experiments
behind chapter 4 are documented in
[`eval/EXPERIMENTS.md`](../../eval/EXPERIMENTS.md).

## Building

pdfLaTeX with biber (biblatex, IEEE style):

```
latexmk -pdf main.tex
```

On Overleaf, set the compiler to pdfLaTeX and the main document to `main.tex`.

## Files

```
main.tex                          preamble, title page, abstract, bibliography
chapters/01_introduction.tex      motivation and research question
chapters/02_background.tex        background and related work
chapters/03_implementation.tex    library, developer guide, verification, use of AI tools
chapters/04_evaluation.tex        datasets, design, results, threats to validity
chapters/05_lessons_learned.tex   findings, lessons learned, limitations
figures/                          architecture diagram and forest plot
references.bib                    bibliography
```
