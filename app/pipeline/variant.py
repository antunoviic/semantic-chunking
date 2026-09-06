from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkVariant:
    #names the variants and ablations

    incremental: bool = True
    respect_headings: bool = True
    smart_split: bool = True
    heading_mode: str = "regex"      # "regex" (shipped) | "hybrid" (Regex + LLM-Fallback)

    def _resolve(self, respect_headings: bool | None, smart_split: bool | None,
                heading_mode: str | None) -> tuple[bool, bool, str]:
        return (
            self.respect_headings if respect_headings is None else respect_headings,
            self.smart_split if smart_split is None else smart_split,
            self.heading_mode if heading_mode is None else heading_mode,
        )

    def key(self, respect_headings: bool | None = None,
            smart_split: bool | None = None, heading_mode: str | None = None) -> str:
        if not self.incremental:
            return ""                      # window standard name
        headings, smart, hmode = self._resolve(respect_headings, smart_split, heading_mode)
        base = "incremental_headings" if headings else "incremental"
        if headings and hmode != "regex":
            base += f"_{hmode}"
        return base if smart else base + "_midpoint"

    def label(self, respect_headings: bool | None = None,
              smart_split: bool | None = None, heading_mode: str | None = None) -> str:
        if not self.incremental:
            return "llm_window"
        headings, smart, hmode = self._resolve(respect_headings, smart_split, heading_mode)
        base = "llm_incremental_headings" if headings else "llm_incremental"
        if headings and hmode != "regex":
            base += f"_{hmode}"
        return base if smart else base + "_midpoint"
