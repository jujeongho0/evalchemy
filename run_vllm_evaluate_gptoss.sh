#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1
export HUGGING_FACE_HUB_TOKEN="your_hf_token"
export OPENROUTER_API_KEY="your_openrouter_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="openai/gpt-oss-120b"
OUTPUT_PATH="/path/to/output_path"

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks MMLUPro \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KMMLUPro \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks CLIcK \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KoBALT \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks LiveCodeBenchv6_official \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks HRM8K \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \
    
python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks IFBench \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks AALCR \
    --batch_size auto \
    --max_tokens 131072 \
    --apply_chat_template \
    --output_path ${OUTPUT_PATH} \
    --parse_think "<|message|>" \
    --verbosity INFO \
