import asyncio
import copy
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(
    api_key="", # TODO: You need to enter your OpenAI API KEY.
    timeout=300.0,
    max_retries=1
)

# Adopted from https://artificialanalysis.ai/methodology/intelligence-benchmarking#quality-checker-prompt
JUDGE_PROMPT = """Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {correct_answer}
    Expression 2: {response}"""


class ExtractedAnswer(BaseModel):
    equivalent: Literal["Yes", "No"]
    strict: Literal[True]  # 100% reliability


async def judge_answer(correct_answer, response, judge):
    prompt = JUDGE_PROMPT.format(correct_answer=correct_answer, response=response)
    try:
        response = await client.beta.chat.completions.parse(
            model=judge,
            max_completion_tokens=4096,  # overkill for judge
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "equivalent": content.equivalent,
        }
    except Exception as e:  # very, very rare
        print("Error:", e)
        return None


async def add_judge_response(question, predictions, judge):
    unique_id = question["id"]
    prediction = copy.deepcopy(predictions[unique_id])  # not in-place
    correct_answer = question["answer"]

    if "judge_response" in prediction:  # already judged
        return unique_id, prediction

    response = prediction["response"]
    content = await judge_answer(correct_answer, response, judge)

    if content is not None:
        prediction["judge_response"] = content  # local in-place
        return unique_id, prediction
    else:
        return None, None


async def judge_all_responses(questions, predictions, num_workers, judge):
    async def bound_func(question):
        async with semaphore:
            content = await add_judge_response(question, predictions, judge)
            return content

    semaphore = asyncio.Semaphore(num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results
