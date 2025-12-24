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

JUDGE_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""


class ExtractedAnswer(BaseModel):
    response: str


async def extract_answer(question, correct_answer, response, judge):
    prompt = JUDGE_PROMPT.format(question=question, official_answer=correct_answer, candidate_answer=response)
    try:
        response = await client.beta.chat.completions.parse(
            model=judge,
            max_completion_tokens=4096,  # overkill for judge
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "response": content.response,
        }
    except Exception as e:  # very, very rare
        print("Error:", e)
        return None


async def add_judge_response(question, predictions, judge):
    unique_id = question["id"]
    prediction = copy.deepcopy(predictions[unique_id])  # not in-place
    question_text = question["question"]
    correct_answer = question["answer"]

    if "judge_response" in prediction:  # already judged
        return unique_id, prediction

    response = prediction["response"]
    content = await extract_answer(question_text, correct_answer, response, judge)

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
