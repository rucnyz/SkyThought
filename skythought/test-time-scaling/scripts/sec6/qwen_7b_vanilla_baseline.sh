MAX_ROUND=5
for difficulty in easy medium hard
do
    python evaluate_multiprocess.py \
        --difficulty=${difficulty} \
        --temperature=0.7 \
        --num_threads=16 \
        --n=8 \
        --selection=first\
        --test_generator 4o-mini \
        --lcb_version release_v4 \
        --start_date 2024-08-01 \
        --end_date 2024-12-01 \
        --num_round ${MAX_ROUND} \
        --api_name openai/Qwen/Qwen2.5-Coder-7B-Instruct \
        --api_base http://localhost:8000/v1 \
        --result_json_path="results/sec6_qwen7b_vanilla_baseline_${difficulty}_max_round_${MAX_ROUND}.json" \
        --load_cached_preds \
        --cached_preds_path="results/sec5_revision_vanilla_qwen_7b_${difficulty}_max_round_${MAX_ROUND}.json"
done