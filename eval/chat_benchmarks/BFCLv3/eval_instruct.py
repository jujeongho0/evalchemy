import re
import json
import logging
import os
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .multi_turn_utils import execute_multi_turn_func_call, is_empty_execute_response
from .type_mappings import JAVA_TYPE_CONVERSION, JS_TYPE_CONVERSION
from .java_type_converter import java_type_converter
from .js_type_converter import js_type_converter

PROMPT = """You are a helpful assistant and an expert in function composition. You can answer general questions using your internal knowledge OR invoke functions when necessary. Follow these strict guidelines:

1. FUNCTION CALLS:
- ONLY use functions that are EXPLICITLY listed in the function list below
- If NO functions are listed (empty function list []), respond ONLY with internal knowledge or "I don't have access to [Unavailable service] information"
- If a function is not in the list, respond ONLY with internal knowledge or "I don't have access to [Unavailable service] information"
- If ALL required parameters are present AND the query EXACTLY matches a listed function's purpose: output ONLY the function call(s)
- Use exact format: [func_name1(param1=value1, param2=value2), func_name2(...)]
Examples:
CORRECT: [get_weather(location="Vancouver"), calculate_route(start="Boston", end="New York")] <- Only if get_weather and calculate_route are in function list
INCORRECT: get_weather(location="New York")
INCORRECT: Let me check the weather: [get_weather(location="New York")]
INCORRECT: [get_events(location="Singapore")] <- If function not in list

2. RESPONSE RULES:
- For pure function requests matching a listed function: ONLY output the function call(s)
- For knowledge questions: ONLY output text
- For missing parameters: ONLY request the specific missing parameters
- For unavailable services (not in function list): output ONLY with internal knowledge or "I don't have access to [Unavailable service] information". Do NOT execute a function call.
- If the query asks for information beyond what a listed function provides: output ONLY with internal knowledge about your limitations
- NEVER combine text and function calls in the same response
- NEVER suggest alternative functions when the requested service is unavailable
- NEVER create or invent new functions not listed below

3. STRICT BOUNDARIES:
- ONLY use functions from the list below - no exceptions
- NEVER use a function as an alternative to unavailable information
- NEVER call functions not present in the function list
- NEVER add explanatory text to function calls
- NEVER respond with empty brackets
- Use proper Python/JSON syntax for function calls
- Check the function list carefully before responding

4. TOOL RESPONSE HANDLING:
- When receiving tool responses: provide concise, natural language responses
- Don't repeat tool response verbatim
- Don't add supplementary information

Here is a list of functions in JSON format that you can invoke:
"""

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]

NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for BFCLv3 benchmark"
    )


class BFCLv3Benchmark(BaseBenchmark):

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        # FIXME
        thinking_budget: Optional[int] = None,
        parse_think: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "llamastack/bfcl_v3"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for example in examples:
            example["model_outputs"] = []

        max_turns = 8 # BFCLv3's max turn is 8.
        for mt in range(1, max_turns + 1):
            all_instances, call_indices = [], []
            for idx, example in enumerate(examples):
                turns = json.loads(example["turns"])
                functions = json.loads(example["functions"])

                if mt > len(turns):
                    continue

                messages = []
                for turn in range(1, mt + 1):
                    if turn == 1:
                        messages.append({"role": "user", "content": turns[0][1]["content"].strip()})

                    else:
                        messages.append({"role": "assistant", "content": examples[idx]["model_outputs"][turn - 2]})

                        try:
                            messages.append({"role": "user", "content": turns[turn - 1][0]["content"].strip()})

                        except: # 'multi_turn_miss_func' task
                            user_content = []
                            missed_functions = json.loads(example["missed_functions"])
                            for k, v in missed_functions.items():
                                for v_ in v:
                                    for missed_function in v_:
                                        function_name = missed_function["name"]
                                        functions.append(missed_function)
                                        user_content.append(f"Supply function \"{function_name}\".")
                                    
                            messages.append({"role": "user", "content": "\n".join(user_content)})
                        
                system_content = PROMPT + str(functions)
                messages.insert(0, {"role": "system", "content": system_content})

                templated_messages = tokenizer.apply_chat_template(
                    messages,
                    tools=functions,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "temperature": 0.0,
                            "max_new_tokens": self.max_new_tokens,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
                all_instances.append(instance)
                call_indices.append(idx)

            if not all_instances:
                continue

            # Generate model responses
            self.logger.info(f"Generating responses of {mt} turns for BFCLv3...")
            outputs = self.compute(model=model, inputs=all_instances, thinking_budget=None, parse_think=False) # FIXME: Agent benchmarking requires no thinking.

            for ci, output in zip(call_indices, outputs):
                examples[ci]["model_outputs"].append(output)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example in examples:
            example["model_answers"] = [self.parse_tool_calls(emo) for emo in example["model_outputs"]]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        solved = 0
        for example in examples:
            test_category = example["test_category"]
            func_description = json.loads(example["functions"])
            model_output = example["model_answers"]
            possible_answer = json.loads(example["ground_truth"])
            language = example["language"]

            if "relevance" in test_category or "irrelevance" in test_category:
                contain_func_call = False
                decode_error = None

                try:
                    contain_func_call = True
                    if self.is_empty_output(model_output):
                        contain_func_call = False
                except Exception as e:
                    contain_func_call = False
                    decode_error = str(e)

                if "irrelevance" in test_category:
                    success = not contain_func_call
                else:
                    success = contain_func_call

                if not success:
                    temp = {"valid": success}
                    if "irrelevance" in test_category:
                        temp["error"] = ["Valid syntax. Successfully decode AST when it should not."]
                        temp["error_type"] = "irrelevance_error:decoder_success"
                    else:
                        temp["error"] = [f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"]
                        temp["error_type"] = "relevance_error:decoder_failed"
                    example["eval_checker"] = temp
                else:
                    example["eval_checker"] = {"valid": True}

            else:
                if "multi_turn" in test_category:
                    example["eval_checker"] = self.multi_turn_checker(model_output, possible_answer, example)

                else:
                    if "parallel" in example["id"]:
                        example["eval_checker"] = self.parallel_function_checker_no_order(func_description, model_output[0], possible_answer, language)
                    
                    elif "multiple" in example["id"]:
                        example["eval_checker"] = self.multiple_function_checker(func_description, model_output[0], possible_answer, language)
                    
                    else:
                        if len(example["model_answers"]) != 1:
                            example["eval_checker"] = {
                                "valid": False,
                                "error": ["Wrong number of functions."],
                                "error_type": "simple_function_checker:wrong_count"
                            }

                        else:
                            example["eval_checker"] = self.simple_function_checker(func_description[0], model_output[0][0], possible_answer[0], language)
                            
            if example["eval_checker"]["valid"]:
                solved += 1

        results.update(
            {
                "num_total": num_questions,
                "num_solved": solved,
                "accuracy": solved / num_questions,
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

    def parse_tool_calls(self, text):
        tool_calls = []

        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        matches = pattern.findall(text)

        for match in matches:
            try:
                data = json.loads(match)
            
                name = data["name"]
                arguments = data.get("arguments", {})

                tool_calls.append({name: arguments})

            except:
                continue
        
        if tool_calls:
            return tool_calls
        else:
            return [{}]

    def is_function_calling_format_output(self, decoded_output):
        if type(decoded_output) != list:
            return False
        for item in decoded_output:
            if type(item) != dict:
                return False
            if len(item) != 1:
                return False
            if type(list(item.values())[0]) != dict:
                return False
        return True

    def is_empty_output(self, decoded_output):
        if not self.is_function_calling_format_output(decoded_output):
            return True
        if len(decoded_output) == 0:
            return True
        if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
            return True
        return False

    def _compare_instances(self, model_obect, ground_truth_object):
        assert type(model_obect) == type(
            ground_truth_object
        ), "Objects are not of the same type."
        differences = {}
        valid = True
        for attr_name in vars(ground_truth_object):
            # We don't check for private attributes
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(model_obect, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)

            if model_attr != ground_truth_attr:
                valid = False
                differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

        return valid, differences

    def state_checker(self, model_instances, ground_truth_instances):
        for class_name, ground_truth_instance in ground_truth_instances.items():
            model_instance = model_instances[class_name]
            valid, differences = self._compare_instances(model_instance, ground_truth_instance)

            if not valid:
                model_instance_attributes = {
                    key: value
                    for key, value in vars(model_instance).items()
                    if not key.startswith("_")
                }
                ground_truth_instance_attributes = {
                    key: value
                    for key, value in vars(ground_truth_instance).items()
                    if not key.startswith("_")
                }

                return {
                    "valid": False,
                    "error_message": f"Model instance for {class_name} does not match the state with ground truth instance.",
                    "error_type": "multi_turn:instance_state_mismatch",
                    "details": {
                        "differences": differences,
                        "model_instance_state": model_instance_attributes,
                        "ground_truth_instance_state": ground_truth_instance_attributes,
                    },
                }

        return {"valid": True}

    def _is_subsequence_unordered(self, list1, list2):
        list2_copy = list2[:]
        
        missing_elements = []
        for item in list1:
            try:
                list2_copy.remove(item)
            except ValueError:
                missing_elements.append(item)
        
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements

    def response_checker(self, model_response_list, ground_truth_response_list, turn_index):
        is_subsequence, missing_items = self._is_subsequence_unordered(
            ground_truth_response_list, model_response_list
        )
        if not is_subsequence:
            return {
                "valid": False,
                "error_message": f"Model response execution results so far does not contain all the ground truth response execution results for turn {turn_index}.",
                "error_type": "multi_turn:execution_response_mismatch",
                "details": {
                    "missing_items": missing_items,
                    "model_response (including all previous turns)": model_response_list,
                    "ground_truth_response (only the current turn)": ground_truth_response_list,
                },
            }

        return {"valid": True}

    def multi_turn_checker(self, multi_turn_model_result_list_decoded, multi_turn_ground_truth_list, test_entry):
        initial_config = json.loads(test_entry["initial_config"])
        involved_classes = test_entry["involved_classes"]
        test_entry_id = test_entry["id"]
        test_category = test_entry["test_category"]
        execution_results = []
        all_turn_model_execution_results = []

        for turn_index, single_turn_ground_truth_list in enumerate(multi_turn_ground_truth_list):
            single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]

            single_turn_model_execution_results = []
            single_turn_model_execution_results_uncombined = []
            single_turn_ground_truth_execution_results = []
            model_instances = {}
            single_step_model_execution_results = []
        
            for single_step_model_response in single_turn_model_response_list:
                single_step_model_execution_results, model_instances = (
                    execute_multi_turn_func_call(
                        func_call_list=single_step_model_response,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name="",
                        test_entry_id=test_entry_id,
                        long_context=(
                            "long_context" in test_category or "composite" in test_category
                        ),
                        is_evaL_run=True,
                    )
                )
                single_turn_model_execution_results.extend(single_step_model_execution_results)
                single_turn_model_execution_results_uncombined.append(single_step_model_execution_results)

            single_turn_ground_truth_execution_results, ground_truth_instances = (
                execute_multi_turn_func_call(
                    func_call_list=single_turn_ground_truth_list,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name="ground_truth",
                    test_entry_id=test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=True,
                )
            )

            all_turn_model_execution_results.extend(single_turn_model_execution_results)
            execution_results.append(
                {
                    "model": single_turn_model_execution_results_uncombined,
                    "ground_truth": single_turn_ground_truth_execution_results,
                }
            )

            if len(single_turn_ground_truth_list) > 0:
                if not single_turn_model_response_list or is_empty_execute_response(
                    single_turn_model_response_list
                ):
                    return {
                        "valid": False,
                        "error_message": f"Model response list is empty for turn {turn_index}",
                        "error_type": "multi_turn:empty_turn_model_response",
                        "details": {
                            "execution_result": execution_results,
                        },
                    }

            if not single_turn_ground_truth_list:
                continue

            assert len(model_instances) == len(
                ground_truth_instances
            ), f"Model instances and ground truth instances do not match in length for turn {turn_index}. Model instances: {len(model_instances)}, Ground truth instances: {len(ground_truth_instances)}"
            assert set(model_instances.keys()) == set(ground_truth_instances.keys())

            state_check_result = self.state_checker(model_instances, ground_truth_instances)
            if not state_check_result["valid"]:
                state_check_result["execution_result"] = execution_results
                return state_check_result

            response_check_result = self.response_checker(
                all_turn_model_execution_results,
                single_turn_ground_truth_execution_results,
                turn_index,
            )
            if not response_check_result["valid"]:
                return response_check_result

        return {"valid": True}

    def find_description(self, func_descriptions, name):
        if type(func_descriptions) == list:
            for func_description in func_descriptions:
                if func_description["name"] == name:
                    return func_description
            return None
        else:
            # it is a dict, there is only one function
            return func_descriptions

    def get_possible_answer_type(self, possible_answer):
        for answer in possible_answer:
            if answer != "":  # Optional parameter
                return type(answer)
        return None

    def type_checker(self, param, value, possible_answer, expected_type_description, expected_type_converted, nested_type_converted):
        # NOTE: This type checker only supports nested type checking for one level deep.
        # We didn't implement recursive type checking for nested types, as it's not needed for the current use case and it's very complex.

        result = {
            "valid": True,
            "error": [],
            "is_variable": False,
            "error_type": "type_error:simple",
        }

        is_variable = False
        # check for the case where a variable is used instead of a actual value.
        # use the type in possible_answer as the expected type
        possible_answer_type = self.get_possible_answer_type(possible_answer)
        # if possible_answer only contains optional parameters, we can't determine the type
        if possible_answer_type != None:
            # we are being precise here.
            # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
            if possible_answer_type != expected_type_converted:
                is_variable = True

        # value is the same type as in function description
        if type(value) == expected_type_converted:
            # We don't need to do recursive check for simple types
            if nested_type_converted == None:
                result["is_variable"] = is_variable
                return result
            else:
                for possible_answer_item in possible_answer:
                    flag = True  # Each parameter should match to at least one possible answer type.
                    # Here, we assume that each item should be the same type. We could also relax it.
                    if type(possible_answer_item) == list:
                        for value_item in value:
                            checker_result = self.type_checker(
                                param,
                                value_item,
                                possible_answer_item,
                                str(nested_type_converted),
                                nested_type_converted,
                                None,
                            )
                            if not checker_result["valid"]:
                                flag = False
                                break

                    if flag:
                        return {"valid": True, "error": [], "is_variable": is_variable}

                result["valid"] = False
                result["error"] = [
                    f"Nested type checking failed for parameter {repr(param)}. Expected outer type {expected_type_description} with inner type {str(nested_type_converted)}. Parameter value: {repr(value)}."
                ]
                result["error_type"] = "type_error:nested"

        # value is not as expected, check for the case where a variable is used instead of a actual value
        # use the type in possible_answer as the expected type
        possible_answer_type = self.get_possible_answer_type(possible_answer)
        # if possible_answer only contains optional parameters, we can't determine the type
        if possible_answer_type != None:
            # we are being precise here.
            # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
            if type(value) == possible_answer_type:
                result["is_variable"] = True
                return result

        result["valid"] = False
        result["error"].append(
            f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, got {type(value).__name__}. Parameter value: {repr(value)}."
        )
        result["error_type"] = "type_error:simple"
        return result

    def standardize_string(self, input_string):
        """
        This function standardizes the string by removing all the spaces, ",./-_*^" punctuation, and converting it to lowercase
        It will also convert all the single quotes to double quotes
        This is used to compare the model output with the possible answers
        We don't want to punish model for answer like April 1, 2024 vs April 1,2024, vs April 1 2024
        """
        regex_string = r"[ \,\.\/\-\_\*\^]"
        return re.sub(regex_string, "", input_string).lower().replace("'", '"')

    def dict_checker(self, param, model_output, possible_answers):
        # This function works for simple dictionaries, but not dictionaries with nested dictionaries.
        # The current dataset only contains simple dictionaries, so this is sufficient.

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        for i in range(len(possible_answers)):

            if possible_answers[i] == "":
                continue

            result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

            flag = True

            possible_answer = possible_answers[i]
            # possible_anwer is a single dictionary

            for key, value in model_output.items():
                if key not in possible_answer:
                    result["valid"] = False
                    result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                    result["error_type"] = "value_error:dict_key"
                    flag = False
                    break

                standardize_value = value
                # If the value is a string, we need to standardize it
                if type(value) == str:
                    standardize_value = self.standardize_string(value)

                # We also need to standardize the possible answers if they are string
                standardize_possible_answer = []
                for i in range(len(possible_answer[key])):
                    if type(possible_answer[key][i]) == str:
                        standardize_possible_answer.append(
                            self.standardize_string(possible_answer[key][i])
                        )
                    else:
                        standardize_possible_answer.append(possible_answer[key][i])

                if standardize_value not in standardize_possible_answer:
                    result["valid"] = False
                    result["error"].append(
                        f"Invalid value for parameter {repr(key)}: {repr(value)}. Expected one of {standardize_possible_answer}."
                    )
                    result["error_type"] = "value_error:dict_value"
                    flag = False
                    break

            for key, value in possible_answer.items():
                if key not in model_output and "" not in value:
                    result["valid"] = False
                    result["error"].append(f"Missing dict key parameter: '{key}'.")
                    result["error_type"] = "value_error:dict_key"
                    flag = False
                    break

            if flag:
                return {"valid": True, "error": []}

        return result

    def list_dict_checker(self, param, model_output, possible_answers):
        # This function takes in a list of dictionaries and checks if each dictionary is valid
        # The order of the dictionaries in the list must match the order of the possible answers

        result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

        for answer_index in range(len(possible_answers)):
            flag = True  # True means so far, all dictionaries are valid

            # Only proceed if the number of dictionaries in the list matches the number of dictionaries in the possible answers
            if len(model_output) != len(possible_answers[answer_index]):
                result["valid"] = False
                result["error"] = ["Wrong number of dictionaries in the list."]
                result["error_type"] = "value_error:list_dict_count"
                flag = False
                continue

            for dict_index in range(len(model_output)):
                result = self.dict_checker(
                    param,
                    model_output[dict_index],
                    [possible_answers[answer_index][dict_index]],
                )
                if not result["valid"]:
                    flag = False
                    break
            if flag:
                return {"valid": True, "error": []}

        return result

    def string_checker(self, param, model_output, possible_answer):
        standardize_possible_answer = []
        standardize_model_output = self.standardize_string(model_output)
        for i in range(len(possible_answer)):
            if type(possible_answer[i]) == str:
                standardize_possible_answer.append(self.standardize_string(possible_answer[i]))

        if standardize_model_output not in standardize_possible_answer:
            return {
                "valid": False,
                "error": [
                    f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}. Case insensitive."
                ],
                "error_type": "value_error:string",
            }

        return {"valid": True, "error": []}

    def list_checker(self, param, model_output, possible_answer):
        # Convert the tuple to a list

        standardize_model_output = list(model_output)

        # If the element in the list is a string, we need to standardize it
        for i in range(len(standardize_model_output)):
            if type(standardize_model_output[i]) == str:
                standardize_model_output[i] = self.standardize_string(model_output[i])

        standardize_possible_answer = []
        # We also need to standardize the possible answers
        for i in range(len(possible_answer)):
            standardize_possible_answer.append([])
            for j in range(len(possible_answer[i])):
                if type(possible_answer[i][j]) == str:
                    standardize_possible_answer[i].append(
                        self.standardize_string(possible_answer[i][j])
                    )
                else:
                    standardize_possible_answer[i].append(possible_answer[i][j])

        if standardize_model_output not in standardize_possible_answer:
            return {
                "valid": False,
                "error": [
                    f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
                ],
                "error_type": "value_error:list/tuple",
            }

        return {"valid": True, "error": []}

    def simple_function_checker(self, func_description, model_output, possible_answer, language):
        possible_answer = list(possible_answer.values())[0]
        # Extract function name and parameters details
        func_name = func_description["name"]
        param_details = func_description["parameters"]["properties"]
        required_params = func_description["parameters"]["required"]

        # Initialize a result dictionary
        result = {
            "valid": True,
            "error": [],
            "error_type": "simple_function_checker:unclear",
        }

        # Check if function name matches
        if func_name not in model_output:
            result["valid"] = False
            result["error"].append(
                f"Function name {repr(func_name)} not found in model output."
            )
            result["error_type"] = "simple_function_checker:wrong_func_name"
            return result

        model_params = model_output[func_name]

        # Check for required parameters in model output
        for param in required_params:
            if param not in model_params:
                result["valid"] = False
                result["error"].append(f"Missing required parameter: {repr(param)}.")
                result["error_type"] = "simple_function_checker:missing_required"
                return result

        # Validate types and values for each parameter in model output
        for param, value in model_params.items():
            if param not in param_details or param not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected parameter: {repr(param)}.")
                result["error_type"] = "simple_function_checker:unexpected_param"
                return result

            full_param_details = param_details[param]
            expected_type_description = full_param_details["type"]  # This is a string
            is_variable = False
            nested_type_converted = None

            if language == "Java":
                expected_type_converted = JAVA_TYPE_CONVERSION[expected_type_description]

                if expected_type_description in JAVA_TYPE_CONVERSION:
                    if type(value) != str:
                        result["valid"] = False
                        result["error"].append(
                            f"Incorrect type for parameter {repr(param)}. Expected type String, got {type(value).__name__}. Parameter value: {repr(value)}."
                        )
                        result["error_type"] = "type_error:java"
                        return result

                    if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                        nested_type = param_details[param]["items"]["type"]
                        nested_type_converted = JAVA_TYPE_CONVERSION[nested_type]
                        value = java_type_converter(
                            value, expected_type_description, nested_type
                        )
                    else:
                        value = java_type_converter(value, expected_type_description)

            elif language == "JavaScript":
                expected_type_converted = JS_TYPE_CONVERSION[expected_type_description]

                if expected_type_description in JS_TYPE_CONVERSION:
                    if type(value) != str:
                        result["valid"] = False
                        result["error"].append(
                            f"Incorrect type for parameter {repr(param)}. Expected type String, got {type(value).__name__}. Parameter value: {repr(value)}."
                        )
                        result["error_type"] = "type_error:js"
                        return result

                    if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                        nested_type = param_details[param]["items"]["type"]
                        nested_type_converted = JS_TYPE_CONVERSION[nested_type]
                        value = js_type_converter(value, expected_type_description, nested_type)
                    else:
                        value = js_type_converter(value, expected_type_description)

            elif language == "Python":
                expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
                if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

            else:
                raise ValueError(f"Unsupported language: {language}")

            # We convert all tuple value to list when the expected type is tuple.
            # The conversion is necessary because any tuple in the possible answer would become a list after being processed through json.dump() and json.load().
            # This does introduce some false positive (eg, when the model provides a list value instead of tuple). We hope to find a better solution in the future.
            if expected_type_description == "tuple" and type(value) == tuple:
                value = list(value)

            # Allow python auto conversion from int to float
            if (
                language == "Python"
                and expected_type_description == "float"
                and type(value) == int
            ):
                value = float(value)

            # Type checking
            # In fact, we only check for Python here.
            # Type check for other languages are handled by the type converter, and so their value (after conversion) is always correct.
            type_check_result = self.type_checker(
                param,
                value,
                possible_answer[param],
                expected_type_description,
                expected_type_converted,
                nested_type_converted,
            )
            is_variable = type_check_result["is_variable"]
            if not type_check_result["valid"]:
                return type_check_result

            # It doesn't make sense to special handle dictionaries and list of dictionaries if the value is a variable.
            # We can just treat the variable as a string and use the normal flow.
            if not is_variable:
                # Special handle for dictionaries
                if expected_type_converted == dict:
                    result = self.dict_checker(param, value, possible_answer[param])
                    if not result["valid"]:
                        return result
                    continue

                # Special handle for list of dictionaries
                elif expected_type_converted == list and nested_type_converted == dict:
                    result = self.list_dict_checker(param, value, possible_answer[param])
                    if not result["valid"]:
                        return result
                    continue

                # Special handle for strings
                elif expected_type_converted == str:
                    # We don't check for case sensitivity for string, as long as it's not a variable
                    result = self.string_checker(param, value, possible_answer[param])
                    if not result["valid"]:
                        return result
                    continue

                elif expected_type_converted == list:
                    result = self.list_checker(param, value, possible_answer[param])
                    if not result["valid"]:
                        return result
                    continue

            # Check if the value is within the possible answers
            if value not in possible_answer[param]:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer[param]}."
                )
                result["error_type"] = "value_error:others"
                return result

        # Check for optional parameters not provided but allowed
        for param in possible_answer:
            if param not in model_params and "" not in possible_answer[param]:
                result["valid"] = False
                result["error"].append(
                    f"Optional parameter {repr(param)} not provided and not marked as optional."
                )
                result["error_type"] = "simple_function_checker:missing_optional"
                return result

        return result

    def parallel_function_checker_no_order(self, func_descriptions, model_output, possible_answers, language):
        if len(model_output) != len(possible_answers):
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "parallel_function_checker_no_order:wrong_count",
            }
        
        matched_indices = []

        # We go throught the possible answers one by one, and eliminate the model output that matches the possible answer
        # It must be this way because we need ground truth to fetch the correct function description
        for i in range(len(possible_answers)):
            # possible_answers[i] is a dictionary with only one key
            func_name_expected = list(possible_answers[i].keys())[0]
            func_description = self.find_description(func_descriptions, func_name_expected)

            all_errors = []

            for index in range(len(model_output)):
                if index in matched_indices:
                    continue

                result = self.simple_function_checker(func_description, model_output[index], possible_answers[i], language)

                if result["valid"]:
                    matched_indices.append(index)
                    break
                else:
                    all_errors.append(
                        {
                            f"Model Result Index {index}": {
                                "sub_error": result["error"],
                                "sub_error_type": result["error_type"],
                                "model_output_item": model_output[index],
                                "possible_answer_item": possible_answers[i],
                            }
                        }
                    )

            if not result["valid"]:
                considered_indices = [
                    i for i in range(len(model_output)) if i not in matched_indices
                ]
                all_errors.insert(
                    0,
                    f"Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers.",
                )
                return {
                    "valid": False,
                    "error": all_errors,
                    "error_type": "parallel_function_checker_no_order:cannot_find_match",
                }

        return {"valid": True, "error": []}

    def multiple_function_checker(self, func_descriptions, model_output, possible_answers, language):
        if len(model_output) != len(possible_answers):
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count",
            }

        # possible_answers is a list of only one dictionary with only one key
        func_name_expected = list(possible_answers[0].keys())[0]
        func_description = self.find_description(func_descriptions, func_name_expected)
        return self.simple_function_checker(func_description, model_output[0], possible_answers[0], language)
