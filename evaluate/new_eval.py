import argparse
import json

from datasets import load_dataset as hf_load_dataset
from lighteval.tasks.requests import Doc
import numpy as np
import torch
from lighteval_tasks import expr_gold_metric
from testing_util import extract_answer, math_equal, strip_answer_string
from vllm import LLM, SamplingParams


def load_model(
    model_path: str, tokenizer_path: str | None = None, tp: int = 1, seed: int = 1234
):
    return LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        seed=seed,
    )


class MaxThinkLimiter:
    def __init__(
        self,
        max_think_tokens_soft: int,
        max_think_tokens_hard: int,
        stop_think_token_id: int,
    ):
        self.max_think_tokens_soft = max_think_tokens_soft
        self.max_think_tokens_hard = max_think_tokens_hard
        self.stop_think_token_id = stop_think_token_id

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """
        LogitsProcessor is a function that takes a list of previously generated
        tokens and a tensor of the logits for the next token, and returns a modified
        tensor of logits to sample from.

        Gradually increase the probability of '</think>' token
        """
        curr_len = len(token_ids)
        if curr_len > self.mex_think_tokens_soft:
            # balance between token with max logits and stop_think
            max_logits = logits.max()
            curr_logits = logits[self.stop_think_token_id]
            logits[self.stop_think_token_id] = curr_logits + (
                max_logits - curr_logits
            ) * (
                (curr_len - self.max_think_tokens_soft)
                / (self.max_think_tokens_hard - self.max_think_tokens_soft)
            )

        return logits


class AIME2024:
    def __init__(self):
        self.data_path = "HuggingFaceH4/aime_2024"
        self.data = self.load_data()

    def load_data(self):
        dataset = hf_load_dataset(self.data_path)
        return dataset["train"]

    @staticmethod
    def generate_prompt(problem: str):
        instruction = (
            "Let's think step by step and output the final answer within \\boxed{}."
        )
        return f"{problem} {instruction}"

    def generate_prompts(self):
        return [self.generate_prompt(x["problem"]) for x in self.data]

    def generate_answers(self):
        return [x["answer"] for x in self.data]

    def generate_conversations(self):
        return [
            [
                {
                    "role": "user",
                    "content": self.generate_prompt(x["problem"]),
                }
            ]
            for x in self.data
        ]

    def generate_lighteval_docs(self):
        return [
            Doc(
                task_name="aime24",
                query=self.generate_prompt(x["problem"]),
                choices=[x["answer"]],
                gold_index=0,
            )
            for x in self.data
        ]


def _judge_one_skythought(
    responses: str | list[str], gold: str, use_last_number: bool = True
):
    if isinstance(responses, str):
        responses = [responses]
    gold = strip_answer_string(gold)
    ret = []
    for response in responses:
        response = extract_answer(response, use_last_number=use_last_number)
        response = strip_answer_string(response)
        ret.append(float(math_equal(response, gold)))

    return ret


def _judge_one_lighteval(responses: str | list[str], gold: str, doc: Doc):
    if isinstance(responses, str):
        responses = [responses]
    return [
        expr_gold_metric.sample_level_fn([gold], [response], doc)
        for response in responses
    ]


def judge_skythought(
    responses: list[str] | list[list[str]],
    gold: list[str],
    use_last_number: bool = True,
) -> list[float]:
    assert len(responses) == len(gold)
    results = []
    for response, g in zip(responses, gold):
        results.append(_judge_one_skythought(response, g, use_last_number))

    return results


def judge_lighteval(
    responses: list[str],
    gold: list[str],
    docs: list[Doc],
) -> list[float]:
    results = []
    for response, g, doc in zip(responses, gold, docs):
        results.append(_judge_one_lighteval(response, g, doc))

    return results


def main(
    model_path: str,
    seed: int = 1234,
    tp: int = 1,
    output_json: str = "results.json",
    n_generations: int = 16,
):
    model = load_model(model_path, tp=tp, seed=seed)

    aime_2024 = AIME2024()
    conversations = aime_2024.generate_conversations()
    prompts = aime_2024.generate_prompts()
    golds = aime_2024.generate_answers()
    docs = aime_2024.generate_lighteval_docs()

    sampling_params = SamplingParams(
        n=n_generations,
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        logits_processors=[
            MaxThinkLimiter(
                max_think_tokens_soft=2048,
                max_think_tokens_hard=2256,
                stop_think_token_id=model.get_tokenizer().encode(
                    "</think>", add_special_tokens=False
                )[0],
            )
        ],
    )
    responses = model.chat(conversations, sampling_params=sampling_params)
    raw_responses = [[out.text for out in res.outputs] for res in responses]

    metrics = {
        "lighteval": judge_lighteval(raw_responses, golds, docs),
        "skythought": judge_skythought(raw_responses, golds),
    }

    for metric, results in metrics.items():
        print(f"[*] {metric} - {np.mean(results):.4f}")

    results = []

    for idx, sample in enumerate(aime_2024.data):
        records = []
        for sample_idx in range(n_generations):
            records.append(
                {
                    "response": raw_responses[idx][sample_idx],
                    "metrics": {m: metrics[m][idx][sample_idx] for m in metrics},
                    "token_usage": {
                        "prompt_tokens": len(responses[idx].prompt_token_ids),
                        "completion_tokens": len(
                            responses[idx].outputs[sample_idx].token_ids
                        ),
                    },
                }
            )

        agg_metrics = {}
        for m in metrics:
            agg_metrics[f"{m}_mean"] = sum(metrics[m][idx]) / len(metrics[m][idx])
            agg_metrics[f"{m}_best"] = max(metrics[m][idx])

        results.append(
            {
                "raw_data": sample,
                "prompt": prompts[idx],
                "answer": golds[idx],
                "responses": records,
                "metrics": agg_metrics,
            }
        )

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model on AIME 2024 dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to evaluate.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results.json",
        help="Output JSON file to save results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=16,
        help="Number of generations to sample.",
    )

    args = parser.parse_args()

    main(
        model_path=args.model,
        seed=args.seed,
        tp=args.tp,
        n_generations=args.n_generations,
        output_json=args.output_json,
    )
# python3 evaluate/new_eval.py --model agentica-org/DeepScaleR-1.5B-Preview --output-json X.json --n-generations 1 --seed 1234
