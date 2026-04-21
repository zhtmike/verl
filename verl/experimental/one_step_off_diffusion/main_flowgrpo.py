# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""
Diffusion-only one-step-off-policy entrypoint for FlowGRPO training.
"""

import asyncio
import os
import socket

import hydra
import ray

from verl.experimental.one_step_off_diffusion.ray_diffusion_trainer import OneStepOffRayFlowGRPOTrainer
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler


@ray.remote(num_cpus=10, max_concurrency=100)  # please make sure main_task is not scheduled on head
class OneStepDiffusionTaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        if os.path.exists(os.path.join(local_path, "processor")):
            processor_path = os.path.join(local_path, "processor")
        else:
            processor_path = local_path
        processor = hf_processor(processor_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, max_samples=config.data.get("val_max_samples", -1)
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the FlowGRPO trainer.
        trainer = OneStepOffRayFlowGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="one_step_off_flowgrpo_trainer", version_base=None)
def main(config):
    from time import time

    from verl.trainer.main_flowgrpo import run_flowgrpo

    start_time = time()

    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    run_flowgrpo(config, task_runner_class=OneStepDiffusionTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
