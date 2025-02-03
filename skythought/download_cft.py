from datasets import load_dataset
import json
import os
import argparse


def download_and_save_datasets(output_path, selected_configs=None):
    """
    Download WebInstruct-CFT datasets from Hugging Face and save them in JSON format

    Args:
        output_path: Path to save the output files
        selected_configs: List of configurations to download, if None, download all configs
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # All available configurations
    available_configs = {
        "600k": "WebInstruct-CFT-600K",
        "50k": "WebInstruct-CFT-50K",
        "4k": "WebInstruct-CFT-4K",
    }

    # If no configs specified, use all configs
    configs_to_download = []
    if selected_configs:
        for config in selected_configs:
            if config.lower() in available_configs:
                configs_to_download.append(available_configs[config.lower()])
            else:
                print(f"Warning: Unknown configuration '{config}', will be skipped")
    else:
        configs_to_download = list(available_configs.values())

    if not configs_to_download:
        print("No valid configurations selected")
        return

    try:
        for config in configs_to_download:
            print(f"Downloading {config} dataset...")

            # Load dataset from Hugging Face
            dataset = load_dataset("TIGER-Lab/WebInstruct-CFT", config)

            # Get training data
            train_data = dataset["train"]

            # Convert to list format
            data_list = train_data.to_list()

            # Save as JSON file
            output_file = os.path.join(output_path, f"{config}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)

            print(f"{config} has been saved to {output_file}")

    except Exception as e:
        print(f"Error occurred during download: {str(e)}")


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Download WebInstruct-CFT datasets")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../data/",
        help="Output directory path",
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="*",
        choices=["600k", "50k", "4k"],
        help="Configurations to download, multiple choices allowed. Example: -c 600k 50k",
    )

    args = parser.parse_args()

    print(
        "Selected configurations:",
        args.configs if args.configs else "all configurations",
    )
    print("Output path:", args.output)

    download_and_save_datasets(args.output, args.configs)


if __name__ == "__main__":
    main()
