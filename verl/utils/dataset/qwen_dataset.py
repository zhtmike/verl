
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import os
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class QwenDataset(Dataset):
    """
    Dataset of text prompts, e.g., for text-guided vision generation task.
    Args:
        data_files (str): Path to the text file containing prompts.
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the prompts.
        config (OmegaConf): the data config.
        template (str): The template to format the prompt.
        max_samples (int): Maximum number of samples to load. If -1, load all samples.
    """
    def __init__(
        self, data_files: str, tokenizer: PreTrainedTokenizer, config: DictConfig, template: str = None, max_samples: int = -1,  **kwargs
    ):
        self.file_path = os.path.join(data_files)
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        with open(self.file_path) as f:
            self.prompts = [line.strip() for line in f.readlines()]

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        if self.truncation == "error":
            for prompt in self.prompts:
                if len(prompt) > self.max_prompt_length:
                    raise RuntimeError(
                        f"Prompt length {len(prompt)} is longer than {self.max_prompt_length}."
                    )

        if self.filter_overlong_prompts:
            self.prompts = [x for x in self.prompts if len(x) <= self.max_prompt_length]

        if self.max_samples > 0 and self.max_samples < len(self.prompts):
            self.prompts = self.prompts[: self.max_samples]

        # NOTE: hard code for Qwen-Image for now
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = template or "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        # tokenize prompts
        txt = [self.prompt_template_encode.format(e) for e in self.prompts]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx, padding=True, truncation=True, return_tensors="pt"
        )
        self.input_ids = txt_tokens.input_ids,
        self.attention_masks = txt_tokens.attention_mask,

        self.data_source = config.data_source
        self.reward_model_style = config.reward_model_style


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
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "reward_model": {"style": self.reward_model_style},
            "data_source": self.data_source,
        }
        ground_truth = self.get_ground_truth(self.prompt[idx], item["data_source"])
        if ground_truth is not None:
            item["reward_model"]["ground_truth"] = ground_truth
        return item
