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
import os

import pytest
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.protocol import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_omni_rollout_spmd import vLLMOmniRollout


@pytest.fixture
def mock_data() -> DataProto:
    model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
    tokenizer_path = os.path.join(model_path, "tokenizer")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # NOTE: hard code for Qwen-Image
    tokenizer_max_length = 1024
    DEFAULT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    template = DEFAULT_TEMPLATE
    drop_idx = 34

    test_prompt = "a photo of a cat"
    test_prompt_2 = "a photo of a dog"
    negative_prompt = ""

    txt = [template.format(e) for e in [test_prompt, test_prompt_2]]
    txt_tokens = tokenizer(
        txt,
        max_length=tokenizer_max_length + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    negative_txt = [template.format(e) for e in [negative_prompt, negative_prompt]]
    negative_txt_tokens = tokenizer(
        negative_txt,
        max_length=tokenizer_max_length + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    data = DataProto.from_single_dict(
        {
            "prompt_ids": txt_tokens.input_ids,
            "prompt_mask": txt_tokens.attention_mask,
            "negative_prompt_ids": negative_txt_tokens.input_ids,
            "negative_prompt_mask": negative_txt_tokens.attention_mask,
        }
    )
    return data


class TestvLLMOmniRollout:
    @classmethod
    def setup_class(cls):
        model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
        tokenizer_path = os.path.join(model_path, "tokenizer")

        diffusion_config = RolloutConfig()
        model_config = HFModelConfig(path=model_path, tokenizer_path=tokenizer_path)

        cls.rollout_engine = vLLMOmniRollout(diffusion_config, model_config, None)
        cls._prefix = "origin_"

    @pytest.mark.skip
    @pytest.mark.asyncio
    async def test_generate_sequences(self, mock_data: DataProto):
        result = await self.rollout_engine.generate_sequences(mock_data)
        expected_batch_keys = ["responses"]
        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch."

        assert result.batch.batch_size[0] == 2, f"Expected batch size 2, got {result.batch.batch_size[0]}."
        images_pil = result.batch["responses"].permute(0, 2, 3, 1).numpy().astype("uint8")

        # TODO: for visualization, drop later
        for i, image in enumerate(images_pil):
            image_path = os.path.join(f"{self._prefix}{i}.jpg")
            Image.fromarray(image).save(image_path)


class TestvLLMOmniRolloutCustomizedPipeline:
    @classmethod
    def setup_class(cls):
        model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
        tokenizer_path = os.path.join(model_path, "tokenizer")

        diffusion_config = RolloutConfig()

        custom_pipeline = "verl.workers.utils.vllm_omni_patch.pipelines.pipeline_qwenimage.QwenImagePipelineWithLogProb"
        model_config = HFModelConfig(path=model_path, tokenizer_path=tokenizer_path, custom_pipeline=custom_pipeline)

        cls.rollout_engine = vLLMOmniRollout(diffusion_config, model_config, None)
        cls._prefix = "custom_"

    @pytest.mark.asyncio
    async def test_generate_sequences(self, mock_data: DataProto):
        result = await self.rollout_engine.generate_sequences(mock_data)
        expected_batch_keys = ["responses"]
        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch."

        assert result.batch.batch_size[0] == 2, f"Expected batch size 2, got {result.batch.batch_size[0]}."
        images_pil = result.batch["responses"].permute(0, 2, 3, 1).numpy().astype("uint8")

        # TODO: for visualization, drop later
        for i, image in enumerate(images_pil):
            image_path = os.path.join(f"{self._prefix}{i}.jpg")
            Image.fromarray(image).save(image_path)
