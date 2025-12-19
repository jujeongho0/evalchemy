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

PROMPT = """다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 ABCD 중 하나이고, "정답:" 뒤에 와야 합니다. 정답을 고르기 전에 차근차근 생각하고 추론하세요.
 
{problem}

A) {A}
B) {B}
C) {C}
D) {D}"""

PROMPT_FIVE = """다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 ABCDE 중 하나이고, "정답:" 뒤에 와야 합니다. 정답을 고르기 전에 차근차근 생각하고 추론하세요.
 
{problem}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}"""

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for KMMLUPro benchmark"
    )


class KMMLUProBenchmark(BaseBenchmark):

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "LGAI-EXAONE/KMMLU-Pro"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 1 # FIXME

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
                if len(example["options"]) == 4:
                    messages = [
                        {
                            "role": "user",
                            "content": PROMPT.format(
                                problem=example["question"], A=example["options"][0], B=example["options"][1], C=example["options"][2], D=example["options"][3]
                            ),
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": PROMPT_FIVE.format(
                                problem=example["question"], A=example["options"][0], B=example["options"][1], C=example["options"][2], D=example["options"][3], E=example["options"][4]
                            ),
                        },
                    ]

                templated_messages = self._prepare_messages(messages, model)

                # FIXME: Non-thinking
                # templated_messages = templated_messages + "<think>\n\n</think>\n\n"

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
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
            self.logger.info("Generating responses for KMMLUPro...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([chr(ord("A")+int(example["solution"])-1) == example["model_answers"][i] for example in examples])

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
        questions = [row for row in dataset["test"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions
