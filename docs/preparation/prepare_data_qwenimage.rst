Prepare Data for Post-Training (Qwen-Image)
========================================

Last updated: 02/20/2026.

Before starting the post-training job, we need to prepare the data for
the policy training. The data should be stored in the parquet format.

For OCR task, we can obtain the raw dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr.

We need to prepare the messages and solutions for each sample in the dataset. 
The following code snippet shows how to prepare the data for post-training.

.. code:: python
    import os

    import datasets

    SYSTEM_PROMPT = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )
    NEGATIVE_USER_PROMPT = " "


    def extract_solution(solution_str):
        # The solution is stored in the format: 'The image displays "xxx".'
        return solution_str.split('"')[1]


    def make_map_fn(split):
        def process_fn(example, idx):
            text = example.pop("text")
            solution = extract_solution(text)
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": NEGATIVE_USER_PROMPT},
                ],
                "ability": "ocr",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn


    if __name__ == "__main__":
        data_source = os.path.expanduser("~/dataset/ocr/")

        dataset = datasets.load_dataset(data_source)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

        local_dir = os.path.expanduser("~/data/ocr")

        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
