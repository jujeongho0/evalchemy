# Adapted from https://artificialanalysis.ai/methodology/intelligence-benchmarking#multiple-choice-extraction-regex

import re


def get_multiple_choice_answer(pred: str):    
    s = pred.strip()

    # 1) Single-letter response -> use directly
    if len(s) == 1 and s.isalpha():
        return s.upper()

    # 2) Primary pattern: **정답** : **X** (with optional markdown formatting)
    primary = r"(?i)[\*\_]{0,2}정답[\*\_]{0,2}\s*:\s*[\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])"
    m = re.findall(primary, s)
    if m:
        return m[-1].upper()

    # 3) Fallback patterns (try in sequence; return last match within the first pattern that matches)
    fallbacks = [
        # LaTeX boxed notation (e.g., \boxed{A} or \boxed{The answer is A})
        r"\\boxed\{[^}]*([A-Z])[^}]*\}",
        # Natural language (e.g., "정답은 B")
        r"정답은\s*([a-zA-Z])",
        # With parenthesis in LaTeX style (e.g., "정답은 \(C\)")
        r"정답은\s*\\\(([a-zA-Z])",
        # Choice format (e.g., "(D) some answer text")
        r"\(([A-Z])\)\s*[^A-Z]*",
        # Explicit statement (e.g., "E가 정답")
        r"([A-Z])\s+가\s+정답",
        # Standalone letter at end of response
        r"([A-Z])\s*$",
        # Letter followed by period (e.g., "F.")
        r"([A-Z])\s*\.",
        # Letter followed by non-word character
        r"([A-Z])\s*[^\w]",
    ]

    for pat in fallbacks:
        m = re.findall(pat, s, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()

    return pred