```shell
cd ./skythought/train/LLaMA-Factory
pip install -e .
pip install nvitop # nvitop -m
# 32B
CUDA_VISIBLE_DEVICES=0,2,3,4 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/qwen2_full_sft.yaml
# mix data
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29500 \
llamafactory-cli train my_scripts/vd_qwq_sky_qwen2_full_sft.yaml

# 3b
CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29502 \
llamafactory-cli train my_scripts/qwen2_3B_full_sft.yaml
# 7b
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/qwen2_7B_full_sft.yaml
# 7b coder
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29500 \
llamafactory-cli train my_scripts/qwen2_coder_7B_full_sft.yaml

# lora for 32B
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/qwen2_32B_lora_sft.yaml

# repro pipeline
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/qwen2_my_full_sft.yaml

# our dataset 32B
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29500 \
llamafactory-cli train my_scripts/vd_ds_qwen2_full_sft.yaml
# qwq
CUDA_VISIBLE_DEVICES=6,7 MASTER_PORT=29500 \
llamafactory-cli train my_scripts/vd_qwq_qwen2_full_sft.yaml

# our dataset 7B
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/vd_ds_qwen2_7b_full_sft.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29501 \
llamafactory-cli train my_scripts/qwen2_full_simpo.yaml

python gpu_monitor.py --interval 30 "MASTER_PORT=29501 llamafactory-cli train my_scripts/vd_ds_qwen2_full_sft.yaml"
```

```shell
python my_eval.py --model NovaSky-AI/Sky-T1-32B-Preview \
--evals=MATH500,AIME,GPQADiamond \
--tp=4 \
--output_file=results.txt \
--temperatures=0
```

| Metric       | Sky-T1-32B-Preview | my-32B-240 | 32B  | my-7B-720 | 7B    | 3B   | QwQ  | ds-r1 | o1-preview |
|--------------|--------------------|------------|------|-----------|-------|------|------|-------|------------|
| Math500      | 87.6               | 86.4       | 82.2 | 65.2      | 76.4  | 63.8 | 90.8 |       | 81.4       |
| AIME2024     | 46.67              | 33.33      | 23.3 | 13.33     | 10.0  | 6.67 | 40.0 |       | 40.0       |
| GPQA-Diamond | 51.01              | 50.0       | 44.4 | 27.27     | 34.85 | 29.8 | 54.0 |       | 75.2       |