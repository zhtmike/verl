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
import logging
import os

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from .reward_manager import get_reward_manager_cls
from .reward_model import RewardModelManager
from . import RewardLoopWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class DiffusionRewardLoopWorker(RewardLoopWorker):
    #TODO: (susan) compute_score_disrm

    def _init_reward_fn(self):
        self.reward_fn = get_custom_reward_fn(self.config)

        # Load reward loop manager class
        # Support both registry and importlib loading methods
        reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")

        if reward_loop_source == "register":
            # Load from registry (default behavior)
            reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
        elif reward_loop_source == "importlib":
            # Load from external module using importlib
            from verl.utils.import_utils import load_extern_object

            reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path", None)
            reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name", None)

            assert reward_loop_module_path is not None, (
                "reward_loop_module_path must be set when reward_loop_source='importlib'"
            )
            assert reward_loop_class_name is not None, (
                "reward_loop_class_name must be set when reward_loop_source='importlib'"
            )

            reward_manager_cls = load_extern_object(
                module_path=reward_loop_module_path, object_name=reward_loop_class_name
            )
        else:
            raise ValueError(f"Unknown reward_loop_source: {reward_loop_source}. Must be 'register' or 'importlib'")

        self.reward_loop = reward_manager_cls(
            self.config, None, self.reward_fn, self.reward_router_address,
        )

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
        rm_scores = torch.tensor(
            scores, dtype=torch.float32
        )
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
