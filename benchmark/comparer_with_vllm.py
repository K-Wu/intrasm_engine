from datasets import load_dataset, load_metric

# The prompt, i.e., how an input is formed, by the Alpaca team is at https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release
raw_datasets = load_dataset("tatsu-lab/alpaca")
# vLLM uses the ShareGPT dataset at https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
raw_datasets = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")
