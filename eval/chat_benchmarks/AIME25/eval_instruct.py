import re
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

# FIXME: Adopted from https://artificialanalysis.ai/methodology/intelligence-benchmarking#mathematical-questions
PROMPT = """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{problem}

Remember to put your answer inside \\boxed{{}}."""


class AIME25Benchmark(BaseBenchmark):
    """
    AIME25 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/aime25

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIME25/data/aime25.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        # FIXME
        thinking_budget: Optional[int] = None,
        parse_think: Optional[bool] = False,
    ):
        """
        Initialize AIME25 benchmark.

        Args:
            data_file: File containing the AIME25 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 16 # FIXME
        # FIXME
        self.thinking_budget = thinking_budget
        self.parse_think = parse_think

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()
        # Prepare instances for model
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["problem"])},
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

                # Add repetition information to instance metadata
                instance.repeat_idx = i
                instance.metadata = {
                    "problem_id": str(example["id"]) if "id" in example else str(idx),
                    "expected_answer": str(example["answer"]),
                    "reference_solution": str(example["solution"]) if "solution" in example else "",
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for AIME25...")
            outputs = self.compute(model=model, inputs=all_instances, thinking_budget=self.thinking_budget, parse_think=self.parse_think) # FIXME
            all_outputs.append(outputs)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.parse_math_answer(o) for o in outputs]
            example["answer"] = self.postprocess(example["answer"])

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([self.is_equiv(example["model_answers"][i], example["answer"]) for example in examples])
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

    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIME25 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def is_equiv(self, str1, str2):
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = self._strip_string(str1)
            ss1 = self.postprocess(ss1)
            ss2 = self._strip_string(str2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    def postprocess(self, s):
        s = str(s).strip()
        try:
            float_value = float(s)
            return str(int(float_value)) if float_value.is_integer() else str(float_value)
        except Exception:
            return s
        
    def parse_math_answer(self, raw_string):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                answer = s[len(left) : -1]
                if "=" in answer:
                    answer = answer.split("=")[-1].lstrip(" ")
                return answer
            except Exception:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None
            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx is None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        def get_answer_with_dollar_sign(s):
            first_pattern = r"\$(.*)\$"
            last_match = None
            matches = re.findall(first_pattern, s)
            if matches:
                last_match = matches[-1]
                if "=" in last_match:
                    last_match = last_match.split("=")[-1].lstrip(" ")
            return last_match

        def get_answer_without_dollar_sign(s):
            last_match = None
            if "=" in s:
                last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
                if "\\n" in last_match:
                    last_match = last_match.split("\\n")[0]
            else:
                pattern = "(?:\\$)?\\d+(?:\\.\\d+)?(?![\\w\\d])"
                matches = re.findall(pattern, s)
                if matches:
                    last_match = matches[-1]
            return last_match

        if "\\boxed" in raw_string:
            answer = remove_boxed(last_boxed_only_string(raw_string))
        else:
            answer = get_answer_with_dollar_sign(raw_string)
            if not answer:
                answer = get_answer_without_dollar_sign(raw_string)
        return answer

    def _fix_fracs(self, string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _strip_string(self, string):
        # linebreaks
        string = string.replace("\n", "")
        # print(string)

        # remove inverse spaces
        string = string.replace("\\!", "")
        # print(string)

        # replace \\ with \
        string = string.replace("\\\\", "\\")
        # print(string)

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        # print(string)

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        # print(string)

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = self._remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace(r"\%", "")

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string
