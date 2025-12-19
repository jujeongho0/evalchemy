"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""

import re


def get_multiple_choice_answer(pred: str):
    # Try to pull out “정답은 X입니다.”, “정답은 {X}입니다.” or “정답은 \boxed{X}입니다.”
    m = re.search(r"(?:Exact\s+)?정답은\s*(?:\\boxed)?\{?([A-Z])\}?입니다.", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: isolate any single capital letter
    candidates = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", pred.upper())
    if candidates:
        return candidates[-1]

    # Final fallback: return the whole trimmed string
    return pred.strip().rstrip(".").rstrip("/")
