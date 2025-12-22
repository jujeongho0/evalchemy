#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="/path/to/wbl_model"

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks GPQADiamond \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks MMLUPro \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks HLE \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks KMMLUPro \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks LiveCodeBenchv6_official \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --trust_remote_code \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks AIME25 \
    --batch_size auto \
    --max_tokens 8192 \
    --apply_chat_template \
    --output_path results \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},pipeline_parallel_size=8,trust_remote_code=True \
    --tasks AALCR \
    --batch_size auto \
    --max_tokens 262144 \
    --apply_chat_template \
    --output_path results \
