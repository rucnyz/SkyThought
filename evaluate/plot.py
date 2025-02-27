import glob
import json

import pandas as pd
import seaborn as sns


def structure_results(results, model_name):
    columns = [
        "model",
        "completion_tokens",
        "lighteval_eval",
        "skythought_eval",
        "problem_id",
    ]
    data = []
    for res in results:
        problem_id = res["raw_data"]["id"]
        for record in res["responses"]:
            completion_tokens = record["token_usage"]["completion_tokens"]
            lighteval_eval = record["metrics"]["lighteval"]
            skythought_eval = record["metrics"]["skythought"]
            data.append(
                [
                    model_name,
                    completion_tokens,
                    lighteval_eval,
                    skythought_eval,
                    problem_id,
                ]
            )

    df = pd.DataFrame(data, columns=columns)
    return df


dfs = []
for file in glob.glob("out/new_eval_0225/*.json"):
    with open(file) as f:
        results = json.load(f)
    df = structure_results(results, file.split("/")[-1].split(".")[0])
    dfs.append(df)


final_df = pd.concat(dfs)

ax = sns.histplot(
    data=final_df[final_df["model"] == "DeepScaleR"],
    x="completion_tokens",
    hue="skythought_eval",
    bins=30,
    common_norm=False,
    kde=True,
)
ax.set_title("Token length distribution of DeepScaleR")
