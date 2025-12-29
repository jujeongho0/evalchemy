import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from run_judge_results import judge_all_responses

from eval.task import BaseBenchmark

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for AALCR benchmark"
    )


class AALCRBenchmark(BaseBenchmark):

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 262144,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        # FIXME
        thinking_budget: Optional[int] = None,
        parse_think: Optional[bool] = False,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 1 # FIXME
        # FIXME
        self.thinking_budget = thinking_budget
        self.parse_think = parse_think

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()

        # Prepare instances for model
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    {
                        "role": "user",
                        "content": example["question"],
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
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for AALCR...")
            outputs = self.compute(model=model, inputs=all_instances, thinking_budget=self.thinking_budget, parse_think=self.parse_think) # FIXME
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            examples_list.append(example)

        return {"examples": examples_list}

    def evaluate_responses(
        self, results: Dict[str, Any], judge: str = "gpt-4o-mini-2024-07-18" # TODO: Qwen/Qwen3-235B-A22B-Instruct-2507
    ) -> Dict[str, float]:

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        questions = []
        for example in examples:
            questions.append({"id": example["question_id"], "question": example["question"], "answer": example["answer"]})
            example["judge_responses"] = []

        all_results = []
        for i in range(self.n_repeat):
            predictions = {example["question_id"]: {"response": example["model_outputs"][i]} for example in examples}

            eval_results = asyncio.run(judge_all_responses(questions, predictions, num_workers=2, judge=judge))

            solved = 0
            for j, (unique_id, predictions) in enumerate(eval_results):
                if unique_id is not None:
                    solved += ("correct" in predictions["judge_response"]["response"].lower() and "incorrect" not in predictions["judge_response"]["response"].lower())
                    examples[j]["judge_responses"].append(predictions["judge_response"])

            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )
        
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

    def load_questions(self) -> Dataset:
        self.logger.info("Loading AALCR questions from source...")
        dataset = load_dataset("ArtificialAnalysis/AA-LCR", cache_dir=HF_HUB_CACHE)
        dataset = [row for row in dataset["test"]]
        if self.debug:
            dataset = dataset[:2]

        questions = []
        for d in dataset:
            document_set_path = os.path.join("eval/chat_benchmarks/AALCR/data", d["document_category"], d["document_set_id"])

            docs = []
            for filename in d["data_source_filenames"].split(";"):
                try:
                    document_path = os.path.join(document_set_path, filename)
                    with open(document_path, encoding="utf-8") as f:
                        docs.append(f.read())
                except:
                    self.logger.info(f"{document_path} file does not exist...") # FIXME: Certain .txt files are not present in the dataset.
            
            documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
            question = f"BEGIN INPUT DOCUMENTS\n\n{documents_text}\n\nEND INPUT DOCUMENTS\n\nAnswer the following question using the input documents provided above.\n\nSTART QUESTION\n\n{d['question']}\n\nEND QUESTION\n"

            questions.append({"question_id": d["question_id"], "question": question, "answer": d["answer"]})
        
        return questions
