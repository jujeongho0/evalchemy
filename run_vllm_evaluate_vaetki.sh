#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1
export HUGGING_FACE_HUB_TOKEN="your_hf_token"
export OPENROUTER_API_KEY="your_openrouter_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="NC-AI-consortium-VAETKI/VAETKI"
OUTPUT_PATH="/path/to/output_path"

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks GPQADiamond \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 24576 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks MMLUPro \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks HLE \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 24576 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KMMLUPro \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks CLIcK \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KoBALT \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks LiveCodeBenchv6_official \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks AIME25 \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks HRM8K \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks IFEval \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks IFBench \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 24576 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks AALCR \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 1024 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks BFCLv3 \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks MATH500 \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \

python3 -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks BBH \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --thinking_budget 16384 \
    --parse_think \
    --verbosity INFO \
