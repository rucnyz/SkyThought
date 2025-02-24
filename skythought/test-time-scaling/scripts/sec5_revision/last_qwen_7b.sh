#!/bin/bash

# Server: vllm serve Qwen/Qwen2.5-Coder-7B-Instruct
MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=8 \
        --selection=oracle \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --num_round ${MAX_ROUND} \
	--context last \
        --api_name openai/Qwen/Qwen2.5-Coder-7B-Instruct \
        --api_base http://localhost:8000/v1 \
        --selection oracle_all_rounds \
        --result_json_path="results/sec5_revision_last_qwen_7b_${difficulty}_max_round_${MAX_ROUND}.json"
done
