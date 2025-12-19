"""
The logic from KMMLU-Pro codebase at https://github.com/LG-AI-EXAONE/KMMLU-Pro:
"""

import re


def get_multiple_choice_answer(pred: str):
    # Try to pull out “정답: X”
    m = re.findall(r"(?i)정답[^A-E]*:[^A-E]*([A-E])", pred)
    extracted_answer = m[-1] if m else ""

    return extracted_answer
