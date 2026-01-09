# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os

import datasets
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class QwenDataset(Dataset):
    """
    Dataset of text prompts, e.g., for text-guided vision generation task.

    Args:
        data_files (str or list): Path(s) to (parquet, json, or txt) file(s) containing prompts.
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the prompts.
        config (DictConfig): the data config.
        template (str): The template to format the prompt.
        max_samples (int): Maximum number of samples to load. If -1, load all samples.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        template: str = None,
        max_samples: int = -1,
        **kwargs,
    ):
        if not isinstance(data_files, list):
            data_files = [data_files]
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/qwen"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.data_source = config.get("data_source")
        self.serialize_dataset = False
        self.use_shm = config.get("use_shm", False)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)

        # NOTE: hard code for Qwen-Image
        self.tokenizer_max_length = 1024
        DEFAULT_TEMPLATE = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode = template or DEFAULT_TEMPLATE
        self.prompt_template_encode_start_idx = 34

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        if data_files[0].endswith(".txt"):
            return
        from verl.utils.fs import copy_to_local

        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        # read files
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".txt"):
                dataframe = datasets.load_dataset("text", data_files=parquet_file)["train"]
                dataframe = dataframe.rename_column("text", self.prompt_key)
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        # sample max_samples
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        # apply chat template
        self.prompts = [self.prompt_template_encode.format(e) for e in self.dataframe[self.prompt_key]]

        # check truncation
        if self.truncation == "error":
            for prompt in self.prompts:
                if len(prompt) > self.max_prompt_length:
                    raise RuntimeError(f"Prompt length {len(prompt)} is longer than {self.max_prompt_length}.")

        # filter out too long prompts
        self.prompts = self.maybe_filter_out_long_prompts(self.prompts)

        # tokenize prompts
        txt_tokens = self.tokenizer(
            self.prompts,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = txt_tokens.input_ids
        self.attention_masks = txt_tokens.attention_mask

    def maybe_filter_out_long_prompts(self, prompts: list):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            prompts = [x for x in prompts if len(x) <= self.max_prompt_length]
        return prompts

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    @staticmethod
    def get_ground_truth(prompt: str, data_source: str):
        if data_source == "ocr":
            ground_truth = prompt.split('"')[1]
            return ground_truth
        elif data_source == "prompt":
            return prompt
        else:
            return None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row_dict: dict = self.dataframe[idx]

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        # add model tensor inputs
        row_dict["input_ids"] = self.input_ids[idx]
        row_dict["attention_mask"] = self.attention_masks[idx]

        # add reward related non-tensor inputs
        row_dict["raw_prompt"] = self.prompts[idx]
        row_dict["reward_model"] = {}
        row_dict["data_source"] = self.data_source
        ground_truth = self.get_ground_truth(row_dict["raw_prompt"], row_dict["data_source"])
        if ground_truth is not None:
            row_dict["reward_model"]["ground_truth"] = ground_truth

        return row_dict

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[QwenDataset]: list of QwenDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in QwenDataset\n"
                "reason: _read_files_and_tokenize() not called or data file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("QwenDataset dataframe is None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")
        if total_samples % num_splits != 0:
            raise ValueError(f"Cannot split dataset size {total_samples} into {num_splits} splits")
        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = [self.dataframe[i] for i in range(start_idx, end_idx)]

            split_dataset = QwenDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits
