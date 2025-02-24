# Skythought-evals: Data Generation and Evaluation Tools


## Requirements 

Make sure you have installed the `skythought` package as outlined in the [README.md](/README.md#usage).

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```

## Generation and Evaluation

### Benchmark Evaluation

Given below are two examples for evaluation. For a walkthrough on the basics, please refer to the [example](../../examples/evaluate.ipynb). 

```shell
skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task aime  --backend vllm --backend-args tensor_parallel_size=8  --sampling-params temperature=0.6,top_p=0.95 --n 8 --result-dir ./
skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task gpqa_diamond --backend vllm --backend-args tensor_parallel_size=8 --sampling-params temperature=0.6,top_p=0.95 --n 8
```

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 


The results will be saved in a folder in `result-dir`:

```bash
result-dir/
├── Qwen_QwQ-32B-Preview_aime_myHash
│   ├── results.json # contains the full results for the benchmark
│   └── summary.json # contains summary of the run with configuration and metrics
```

### Scaling evaluation with Ray

You can scale evaluations across multiple model replicas (and across multiple nodes) using [ray](https://docs.ray.io) backend:

```shell
skythought evaluate --model Qwen/QwQ-32B-Preview --task aime --backend ray --backend-args tensor_parallel_size=4,num_replicas=4 --result-dir ./
```

By default, we make use of the configuration in [ray_configs/ray_config.yaml](./ray_configs/ray_config.yaml). You can also customize the following parameters for ray: 


### Optimized settings for 32B and 7B models

The following are optimized settings on a 8xH100 or a 8xA100 node. We recommend using `ray` backend for best performance. 

For 32B models, we recommend using the default backend configuration for best performance. 

```shell
skythought evaluate --model Qwen/QwQ-32B-Preview --task aime24 --backend ray --result-dir ./
```

For 7B models, we recommend using `tensor_parallel_size=1` and `num_replicas=8` for best performance. FOr example, the previous command will change to:

```shell
skythought evaluate --model Qwen/Qwen2-7B-Instruct --task math500 --backend ray --backend-args tensor_parallel_size=1,num_replicas=8 --result-dir ./
```

#### Multi-node inference

Note that if you have a ray cluster setup, you can scale the number of replicas as needed with `num_replicas` argument in `backend-args` to make full use of your cluster. Make sure to execute the script on the head node and ensure that `--result-dir` is a valid directory that the head node can write to. 

### Best-of-N Evaluation

You can use the `--n` parameter to specify the number of generations per problem. For `n>1` , we calculate pass

```bash
skythought evaluate --model Qwen/Qwen2-7B-Instruct --task math500 --backend ray --backend-args tensor_parallel_size=1,num_replicas=8 --sampling-params temperature=0.7,max_tokens=4096 --n 64 --result-dir ./
```

### Distill and Reject Sampling
Currently we support distill and reject sampling for NUMINA, APPS, and TACO datasets. For NUMINA, the source can be one from `[amc_aime, math, olympiads]`.

#### Example Usage

```shell
skythought generate --model Qwen/QwQ-32B-Preview --task apps --backend ray --backend-args tensor_parallel_size=8 --sampling-params max_tokens=16384 --result-dir $SKYT_HOME/data
```

### Reproducibility Issues


We've noticed that it can be hard to reproduce results in reasoning benchmarks. Beyond the lack of agreed sampling parameters and metrics in the field at the moment, there can be significant differences in results across different evaluation codebases, and even for the same codebase with a different set of dependencies. In half-precision (bfloat16 or float16), numerical error accumulation will change outputs ever so slightly, which can dramatically alter final performance. There are three factors we've noticed that affect results:

- Long context generations: Errors can accumulate so that the output changes at 1k+ tokens, which compound as you keep generating. Since we typically set max tokens to be 16k or 32k tokens, the final solution will change significantly
- vLLM settings:  With vLLM, we’ve also noticed that at half-precision, different batch sizes can affect downstream evaluation results by a few percentage points. Further, different tensor parallelism settings can also change results in half-precision.
- vLLM version: Different versions of vLLM will use different CUDA-Toolkit or Flash attention versions. Even for the same settings, these differences in the underlying kernels used can change results. 

 We recommend to run all evaluation benchmarks at full precision, i.e float32 to avoid this. By default, we run evaluation in `float32`, which can be customized with the `--backend-args` flag for local inference. In full-precision, evaluation results should be robust to changes in batch size, tensor parallel size, version differences, etc.