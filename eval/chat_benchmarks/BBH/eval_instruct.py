import re
import logging
import os
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.filters.extraction import RegexFilter
from lm_eval.tasks.bbh.cot_zeroshot.utils import ExtendedRegexFilter, MapRegexFilter, NumberParseRegexFilter, WordSortFilter, MultiChoiceRegexFilter

from eval.task import BaseBenchmark

# Adopted from https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bbh/cot_zeroshot
REGEX_BOOLEAN_EXPRESSIONS = "\\b(True|False)\\b"
REGEX_CAUSAL_JUDGEMENT = "\\b(Yes|No|yes|no)\\b"
REGEX_DATE_UNDERSTANDING = "(\\([A-Z]\\))"
REGEX_DISAMBIGUATION_QA = "(\\([A-Z]\\))"
REGEX_DYCK_LANGUAGES = "(?<= )([\" \\[\\(<{}>\\)\\]]+)|([\" \\[\\(<{}>\\)\\]]+)"
REGEX_FORMAL_FALLACIES = "\\b(valid|invalid)\\b"
REGEX_GEOMETRIC_SHAPES = "(\\([A-Z]\\))"
REGEX_HYPERBATON = "(\\([A-Z]\\))"
REGEX_LOGICAL_DEDUCTION_FIVE_OBJECTS = "(\\([A-Z]\\))"
REGEX_LOGICAL_DEDUCTION_SEVEN_OBJECTS = "(\\([A-Z]\\))"
REGEX_LOGICAL_DEDUCTION_THREE_OBJECTS = "(\\([A-Z]\\))"
REGEX_MOVIE_RECOMMENDATION = "(\\([A-Z]\\))"
REGEX_MULTISTEP_ARITHMETIC_TWO = "([-0-9]+)"
REGEX_NAVIGATE = "\\b(Yes|No|yes|no)\\b"
REGEX_OBJECT_COUNTING = "([-0-9]+)"
REGEX_PENGUINS_IN_A_TABLE = "(\\([A-Z]\\))"
REGEX_REASONING_ABOUT_COLORED_OBJECTS = "(\\([A-Z]\\))"
REGEX_RUIN_NAMES = "(\\([A-Z]\\))"
REGEX_SALIENT_TRANSLATION_ERROR_DETECTION = "(\\([A-Z]\\))"
REGEX_SNARKS = "(\\([A-Z]\\))"
REGEX_SPORTS_UNDERSTANDING = {r"\b(no|not plausible)\b": "no", r"\b(yes|plausible)\b": "yes"}
REGEX_TEMPORAL_SEQUENCES = "(\\([A-Z]\\))"
REGEX_TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS = "(\\([A-Z]\\))"
REGEX_TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS = "(\\([A-Z]\\))"
REGEX_TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS = "(\\([A-Z]\\))"
REGEX_WEB_OF_LIES = {r"\b(no|does not tell the truth|is not telling the truth)\b": "no", r"\b(yes|tells the truth|is telling the truth)\b": "yes"}

PROMPT_BOOLEAN_EXPRESSIONS = """Evaluate the result of a random Boolean expression.

Q: {problem}
A: Let's think step by step."""

PROMPT_CAUSAL_JUDGEMENT = """Answer questions about causal attribution.

Q: {problem}
A: Let's think step by step."""

PROMPT_DATE_UNDERSTANDING = """Infer the date from context.

Q: {problem}
A: Let's think step by step."""

PROMPT_DISAMBIGUATION_QA = """Clarify the meaning of sentences with ambiguous pronouns.

Q: {problem}
A: Let's think step by step."""

PROMPT_DYCK_LANGUAGES = """Correctly close a Dyck-n word.

Q: {problem}
A: Let's think step by step."""

PROMPT_FORMAL_FALLACIES = """Distinguish deductively valid arguments from formal fallacies.

Q: {problem}
A: Let's think step by step."""

PROMPT_GEOMETRIC_SHAPES = """Name geometric shapes from their SVG paths.

Q: {problem}
A: Let's think step by step."""

PROMPT_HYPERBATON = """Order adjectives correctly in English sentences.

Q: {problem}
A: Let's think step by step."""

PROMPT_LOGICAL_DEDUCTION_FIVE_OBJECTS = """A logical deduction task which requires deducing the order of a sequence of objects.

Q: {problem}
A: Let's think step by step."""

PROMPT_LOGICAL_DEDUCTION_SEVEN_OBJECTS = """A logical deduction task which requires deducing the order of a sequence of objects.

Q: {problem}
A: Let's think step by step."""

PROMPT_LOGICAL_DEDUCTION_THREE_OBJECTS = """A logical deduction task which requires deducing the order of a sequence of objects.

Q: {problem}
A: Let's think step by step."""

PROMPT_MOVIE_RECOMMENDATION = """Recommend movies similar to the given list of movies.

Q: {problem}
A: Let's think step by step."""

PROMPT_MULTISTEP_ARITHMETIC_TWO = """Solve multi-step arithmetic problems.

Q: {problem}
A: Let's think step by step."""

PROMPT_NAVIGATE = """Given a series of navigation instructions, determine whether one would end up back at the starting point.

Q: {problem}
A: Let's think step by step."""

PROMPT_OBJECT_COUNTING = """Questions that involve enumerating objects and asking the model to count them.

Q: {problem}
A: Let's think step by step."""

PROMPT_PENGUINS_IN_A_TABLE = """Answer questions about a table of penguins and their attributes.

Q: {problem}
A: Let's think step by step."""

PROMPT_REASONING_ABOUT_COLORED_OBJECTS = """Answer extremely simple questions about the colors of objects on a surface.

Q: {problem}
A: Let's think step by step."""

PROMPT_RUIN_NAMES = """Select the humorous edit that 'ruins' the input movie or musical artist name.

Q: {problem}
A: Let's think step by step."""

PROMPT_SALIENT_TRANSLATION_ERROR_DETECTION = """Detect the type of error in an English translation of a German source sentence.

Q: {problem}
A: Let's think step by step."""

PROMPT_SNARKS = """Determine which of two sentences is sarcastic.

According to Cambridge University Dictionary, sarcasm is "the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone's feelings or to criticize something in a humorous way." Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.

Q: {problem}
A: Let's think step by step."""

PROMPT_SPORTS_UNDERSTANDING = """Determine whether an artificially constructed sentence relating to sports is plausible or not.

Q: {problem}
A: Let's think step by step."""

PROMPT_TEMPORAL_SEQUENCES = """Task description: Answer questions about which times certain events could have occurred.

Q: {problem}
A: Let's think step by step."""

PROMPT_TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS = """A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.

Q: {problem}
A: Let's think step by step."""

PROMPT_TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS = """A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.

Q: {problem}
A: Let's think step by step."""

PROMPT_TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS = """A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.

Q: {problem}
A: Let's think step by step."""

PROMPT_WEB_OF_LIES = """Evaluate a random boolean function expressed as a word problem.

Q: {problem}
A: Let's think step by step."""

PROMPT_WORD_SORTING = """Sort a list of words.

Q: {problem}
A: Let's think step by step."""

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for KMMLUPro benchmark"
    )


class BBHBenchmark(BaseBenchmark):

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
        self.dataset_name = "SaylorTwift/bbh"
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
                prompt = eval("PROMPT_" + example["subset"].upper())

                messages = [{"role": "user", "content": prompt.format(problem=example["input"])}]

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
            self.logger.info("Generating responses for BBH...")
            outputs = self.compute(model=model, inputs=all_instances, thinking_budget=self.thinking_budget, parse_think=self.parse_think) # FIXME
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = []
            for o in outputs:
                simple = ["boolean_expressions", "causal_judgement", "dyck_languages", "formal_fallacies", "navigate"]
                multi_choice = [
                    "date_understanding", "disambiguation_qa", "flexible-extract", "geometric_shapes", "hyperbaton", \
                    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation", "penguins_in_a_table", \
                    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks", "temporal_sequences", \
                    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects"
                ]
                number_parse = ["multistep_arithmetic_two", "object_counting"]
                map_ = ["sports_understanding", "web_of_lies"]
                word_sort = ["word_sorting"]

                if example["subset"] in simple:
                    regex_pattern = eval("REGEX_" + example["subset"].upper())
                    filt = RegexFilter(
                        regex_pattern=regex_pattern,
                        group_select=-1,
                    )
                    example["model_answers"].append(filt.apply([[o]], [example])[0][0].lower())

                elif example["subset"] in multi_choice:
                    regex_pattern = eval("REGEX_" + example["subset"].upper())
                    filt = MultiChoiceRegexFilter(
                        regex_pattern=regex_pattern,
                        group_select=-1,
                        ignore_case=True,
                        ignore_punctuation=True,
                    )
                    example["model_answers"].append(filt.apply([[o]], [example])[0][0].lower())

                elif example["subset"] in number_parse:
                    regex_pattern = eval("REGEX_" + example["subset"].upper())
                    filt = NumberParseRegexFilter(
                        regex_pattern=regex_pattern,
                        group_select=-1,
                    )
                    example["model_answers"].append(filt.apply([[o]], [example])[0][0].lower())

                elif example["subset"] in map_:
                    regex_pattern_to_value = eval("REGEX_" + example["subset"].upper())
                    filt = MapRegexFilter(
                        regex_pattern_to_value=regex_pattern_to_value,
                        group_select=-1,
                        ignore_case=True,
                    )
                    example["model_answers"].append(filt.apply([[o]], [example])[0][0].lower())
                
                elif example["subset"] in word_sort:
                    filt = WordSortFilter()
                    example["model_answers"].append(filt.apply([[o]], [example])[0][0].lower())

                else:
                    raise NotImplementedError

            example["target"] = example["target"].lower()

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([example["target"] == example["model_answers"][i] for example in examples])

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
        subsets = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', \
            'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', \
            'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', \
            'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', \
            'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', \
            'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
        ]
        questions = []
        for subset in subsets:
            dataset = load_dataset(self.dataset_name, subset, cache_dir=HF_HUB_CACHE)
            for row in dataset["test"]:
                questions.append({"subset": subset, "input": row["input"], "target": row["target"]})
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions
