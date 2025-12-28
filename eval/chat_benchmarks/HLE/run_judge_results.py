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

# FIXME: Adopte from https://artificialanalysis.ai/methodology/intelligence-benchmarking#hle-equality-checker-prompt
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


async def extract_answer(question, correct_answer, response, judge):
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
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
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence,
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
