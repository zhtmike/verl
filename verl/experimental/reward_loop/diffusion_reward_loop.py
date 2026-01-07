# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import base64
import logging
import os
from io import BytesIO

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from PIL import Image
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.utils.ray_utils import get_event_loop

from . import RewardLoopWorker
from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class DiffusionRewardLoopWorker(RewardLoopWorker):
    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        """
        Prepeare discriminative reward model inputs: input prompt and output image.
        """
        assert len(data) == 1, "DiffusionRewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        prompt_str = self.input_tokenizer.decode(data_item.batch["prompts"], skip_special_tokens=True)
        response_image = data_item.batch["responses"]

        image_base64 = await self.pil_image_to_base64(response_image)
        query = self.prepare_query(chat, prompt_str, image_base64)

        chat.append({"role": "assistant", "content": query})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        return rm_prompt

    async def pil_image_to_base64(self, image: Image.Image) -> str:
        # To avoid blocking the event loop, run the conversion in a thread pool
        base64_image = await get_event_loop().run_in_executor(self._executor, self._pil_image_to_base64, image)
        return base64_image

    def _pil_image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_image = f"data:image;base64,{encoded_image_text}"
        return base64_image

    def prepare_query(self, chat, prompt, image_base64: str) -> list:
        query = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
        ]
        return query


class DiffusionRewardLoopManager:
    """
    DiffusionRewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward_model.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                DiffusionRewardLoopWorker.options(
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

    # this func is used to replace the legacy fsdp/megatron RewardModelWorker.compute_rm_score
    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = torch.tensor(scores, dtype=torch.float32)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
