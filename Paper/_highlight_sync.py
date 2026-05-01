#!/usr/bin/env python3
"""Mark passages in frontiers_highlighted.tex with \\textcolor{blue}{} when they
differ from Paper/frontiers/frontiers.tex (baseline 'old paper')."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "frontiers" / "frontiers.tex"
NEW = ROOT / "frontiers_clean_without highlights.tex"
HL = ROOT / "frontiers_highlighted.tex"


def extract_body(tex: str) -> str:
    i = tex.find("\\begin{document}")
    j = tex.find("\\end{document}", i)
    if i == -1 or j == -1:
        return tex
    return tex[i : j + len("\\end{document}")]


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_color_braces(s: str) -> str:
    """Remove outer \\textcolor{blue}{...} wrappers (non-nested safe pass)."""
    out = s
    for _ in range(500):
        m = re.search(r"\\textcolor\{blue\}\{", out)
        if not m:
            break
        start = m.start()
        inner_start = m.end()
        depth = 1
        i = inner_start
        while i < len(out) and depth:
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
                if depth == 0:
                    out = out[:start] + out[inner_start:i] + out[i + 1 :]
                    break
            i += 1
        else:
            break
    return out


def paragraph_chunks(body: str) -> list[str]:
    parts = re.split(r"\n\s*\n", body)
    return [p.strip() for p in parts if len(p.strip()) > 120]


def main() -> None:
    old = extract_body(OLD.read_text(encoding="utf-8", errors="replace"))
    new = extract_body(NEW.read_text(encoding="utf-8", errors="replace"))
    old_norm = normalize(old)

    new_chunks = paragraph_chunks(new)
    novel: list[str] = []
    for ch in new_chunks:
        n = normalize(ch)
        if n and n not in old_norm:
            novel.append(ch)

    print(f"Old chars: {len(old_norm)}, novel paragraph-like chunks vs old: {len(novel)}")
    for i, ch in enumerate(novel[:25]):
        print(f"--- {i+1} ({len(ch)} chars) ---")
        print(ch[:220].replace("\n", " ") + ("..." if len(ch) > 220 else ""))

    hl = HL.read_text(encoding="utf-8", errors="replace")
    hl_plain = strip_color_braces(hl)
    # Quick sanity: plain highlighted should match new roughly
    if normalize(extract_body(hl_plain)) != normalize(new):
        print("WARN: stripped highlighted != new clean (structural drift)")

    # Report chunks from new not found in highlighted with blue (heuristic)
    missing_blue = []
    for ch in novel[:80]:
        snippet = ch[:80].replace("\n", " ")
        if snippet in hl and "\\textcolor{blue}{" not in hl[hl.find(snippet) - 40 : hl.find(snippet) + 5]:
            missing_blue.append(snippet[:60])

    print(f"Heuristic lines possibly missing blue highlight: {len(missing_blue)}")


if __name__ == "__main__":
    main()
