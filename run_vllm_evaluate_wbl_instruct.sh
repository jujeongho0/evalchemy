#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_CACHE="/path/to/.cache/huggingface/hub"

MODEL_PATH="/path/to/wbl_model"

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks GPQADiamond \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 24576 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks MMLUPro \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

# python -m eval.eval \
#     --model vllm \
#     --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
#     --tasks HLE \
#     --batch_size auto \
#     --max_tokens 32768 \
#     --apply_chat_template \
#     --output_path results \
#     --thinking_budget 24576 \
#     --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KMMLUPro \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks CLIcK \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks KoBALT \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks LiveCodeBenchv6_official \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks AIME25 \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks HRM8K \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks IFEval \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks IFBench \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 24576 \
    --parse_think \

# python -m eval.eval \
#     --model vllm \
#     --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
#     --tasks AALCR \
#     --batch_size auto \
#     --max_tokens 32768 \
#     --apply_chat_template \
#     --output_path results \
#     --thinking_budget 16384 \
#     --parse_think \

# python -m eval.eval \
#     --model vllm \
#     --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
#     --tasks BFCLv3 \
#     --batch_size auto \
#     --max_tokens 32768 \
#     --apply_chat_template \
#     --output_path results \
#     --thinking_budget 16384 \
#     --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks MATH500 \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \

python -m eval.eval \
    --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True \
    --tasks BBH \
    --batch_size auto \
    --max_tokens 32768 \
    --apply_chat_template \
    --output_path results \
    --thinking_budget 16384 \
    --parse_think \
