import json
import argparse


def convert_logs_to_clean_json(logs, model):
    logs = json.loads(logs)
    output = []
    for data in logs.values():
        problem = data["problem"]
        solution = data["solution"]
        assert len(data["responses"]) == 1
        # get first value from responses dict
        model_response = list(data["responses"].values())[0]
        metadata = "\n".join(
            [
                f"Model: {model}",
                f"Problem id: {data['id']}",
                f"Answer: {data['answer']}",
                f"Tokens: {data['token_usages']}",
                f"Correctness: {model_response['correctness']}",
            ]
        )
        response = model_response["content"]

        output.append(
            {
                "problem": problem,
                "solution": solution,
                "response": response.split("</think>")[-1].split("<|end_of_thought|>")[-1],
                "model_response": response,
                "metadata": metadata,
            }
        )
    return json.dumps(output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert logs to clean JSON format.")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--in_json", type=str, help="Input JSON file")
    parser.add_argument("--out_json", type=str, help="Output JSON file")
    args = parser.parse_args()

    with open(args.in_json, "r") as file:
        logs = file.read()
    clean_json = convert_logs_to_clean_json(logs, args.model)
    with open(args.out_json, "w") as file:
        file.write(clean_json)
