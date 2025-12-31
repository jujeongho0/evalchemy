import logging
import os
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

# Modified version of https://arxiv.org/pdf/2403.06412
PROMPT = """주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.

질문: {problem}
보기: A: {A}, B: {B}, C: {C}, D: {D}

당신의 답변 마지막 줄은 다음과 같은 형식으로 작성해야 합니다: '정답: A/B/C/D' (예: '정답: A')."""

PROMPT_CONTEXT = """주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.

맥락: {context}
질문: {problem}
보기: A: {A}, B: {B}, C: {C}, D: {D}

당신의 답변 마지막 줄은 다음과 같은 형식으로 작성해야 합니다: '정답: A/B/C/D' (예: '정답: A')."""

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for CLIcK benchmark"
    )


class CLIcKBenchmark(BaseBenchmark):

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        # FIXME
        thinking_budget: Optional[int] = None,
        parse_think: Optional[bool] = False,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "EunsuKim/CLIcK"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 1
        # FIXME
        self.thinking_budget = thinking_budget
        self.parse_think = parse_think

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        all_outputs = []
        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                if not example["paragraph"]:
                    messages = [
                        {
                            "role": "user",
                            "content": PROMPT.format(
                                problem=example["question"], A=example["choices"][0], B=example["choices"][1], C=example["choices"][2], D=example["choices"][3]
                            ),
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": PROMPT_CONTEXT.format(
                                context=example["paragraph"], problem=example["question"], A=example["choices"][0], B=example["choices"][1], C=example["choices"][2], D=example["choices"][3]
                            ),
                        },
                    ]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": True,
                            "temperature": 0.7,
                            "max_new_tokens": self.max_new_tokens,
                            "seed": seed,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for CLIcK...")
            outputs = self.compute(model=model, inputs=all_instances, thinking_budget=self.thinking_budget, parse_think=self.parse_think) # FIXME
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]
            example["answer"] = chr(ord("A") + example["choices"].index(example["answer"]))

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([example["answer"] == example["model_answers"][i] for example in examples])

            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        # Calculate overall statistics
        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std = np.std([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        dataset = load_dataset(self.dataset_name, cache_dir=HF_HUB_CACHE)
        questions = [row for row in dataset["train"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions
