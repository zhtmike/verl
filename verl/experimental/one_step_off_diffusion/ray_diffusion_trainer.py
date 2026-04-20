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
One-step-off-policy trainer for diffusion FlowGRPO.
"""

import asyncio
import uuid
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl.trainer.diffusion.ray_diffusion_trainer import RayFlowGRPOTrainer, compute_advantage
from verl.trainer.ppo.metric_utils import compute_variance_proxy_metrics
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger


class SeparateRayFlowGRPOTrainer(RayFlowGRPOTrainer):
    """
    Support for the initialization and fit process of Ray Trainer in the resource-separated scenario:
        - One-step off-policy
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
        self.is_last_step = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self.checkpoint_manager = None

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, reference, reward)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        self._init_async_rollout_manager()

        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        raise NotImplementedError

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel], config=self.config.reward.reward_model
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _init_worker_groups(self):
        # initialize WorkerGroup
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        self.all_wg = all_wg

    def _init_models(self):
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = self.all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

    def _init_reward_loop(self):
        from verl.experimental.reward_loop import RewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

    def _init_async_rollout_manager(self):
        pass

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self.epoch = epoch
                self.fit_step(batch_dict)
                if self.is_last_step:
                    return

    def fit_step(self, batch_dict: Any = None):
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()

        with marked_timer("step", self.timing_raw):
            batch = self._fit_get_batch(batch_dict)
            batch = self._fit_generate(batch)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_weights()
            self._fit_dump_data(batch)

        self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_experimental(batch)
        self._fit_postprocess_step()

    def _fit_prepare_step(self):
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
        self.is_last_step = self.global_steps >= self.total_training_steps

    def _fit_get_batch(self, batch_dict: dict) -> DataProto:
        batch = DataProto.from_single_dict(batch_dict)
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        return batch

    def _fit_generate(self, batch: DataProto = None) -> DataProto:
        timing_raw = self.timing_raw
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
            self.checkpoint_manager.sleep_replicas()
            timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        return batch

    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                batch_reward = self._compute_reward_colocate(batch)
                batch = batch.union(batch_reward)
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
            self.reward_tensor = reward_tensor
            self.reward_extra_infos_dict = reward_extra_infos_dict
        return batch

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        bypass_recomputing_logprobs = self.config.algorithm.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self._compute_old_log_prob(batch)
                batch = batch.union(old_log_prob)
                if "rollout_log_probs" in batch.batch.keys():
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))

        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
        return batch

    def _fit_compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        if self.use_reference_policy:
            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        return batch

    def _fit_compute_advantage(self, batch) -> DataProto:
        timing_raw = self.timing_raw
        reward_tensor = self.reward_tensor
        reward_extra_infos_dict = self.reward_extra_infos_dict

        with marked_timer("adv", timing_raw, color="brown"):
            batch.batch["sample_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            num_timesteps = batch.batch["old_log_probs"].shape[1]
            batch.batch["sample_level_rewards"] = batch.batch["sample_level_scores"].expand(-1, num_timesteps)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                global_std=self.config.algorithm.global_std,
                config=self.config.algorithm,
            )
        return batch

    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self._update_actor(batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(actor_output_metrics)
        return batch

    def _fit_update_weights(self):
        timing_raw = self.timing_raw
        with marked_timer("update_weights", timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)

    def _fit_dump_data(self, batch: DataProto):
        timing_raw = self.timing_raw
        reward_extra_infos_dict = self.reward_extra_infos_dict
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    def _fit_validate(self):
        metrics = self.metrics
        timing_raw = self.timing_raw
        if self.config.trainer.test_freq > 0 and (
            self.is_last_step or self.global_steps % self.config.trainer.test_freq == 0
        ):
            with marked_timer("testing", timing_raw, color="green"):
                val_metrics: dict = self._validate()
                if self.is_last_step:
                    self.last_val_metrics = val_metrics
            metrics.update(val_metrics)

    def _fit_save_checkpoint(self):
        timing_raw = self.timing_raw
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if self.config.trainer.save_freq > 0 and (
            self.is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

    def _fit_collect_metrics(self, batch):
        metrics = self.metrics
        timing_raw = self.timing_raw
        metrics.update(compute_data_metrics_diffusion(batch=batch))
        num_images = batch.batch["advantages"].shape[0]
        metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_images))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

    def _fit_torch_memory(self):
        if (
            hasattr(self.config.actor_rollout_ref.actor, "profiler")
            and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
        ):
            self.actor_rollout_wg.dump_memory_snapshot(
                tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
            )

    def _fit_experimental(self, batch):
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)
        if hasattr(self.train_dataset, "on_batch_end"):
            self.train_dataset.on_batch_end(batch=batch)

    def _fit_postprocess_step(self):
        metrics = self.metrics
        timing_raw = self.timing_raw
        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)
        self.logger.log(data=metrics, step=self.global_steps)
        self.progress_bar.update(1)
        self.global_steps += 1
        if self.is_last_step:
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
            pprint(f"Final validation metrics: {self.last_val_metrics}")
            self.progress_bar.close()


class OneStepOffRayFlowGRPOTrainer(SeparateRayFlowGRPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        # Skip rollout worker mapping and let agentloop create it.
        role_worker_mapping.pop(Role.Rollout, None)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================

        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
        self.is_last_step = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.agent_loop import AgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config, reward_loop_worker_handles=reward_loop_worker_handles
        )

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        # Launch individual reward computations as each generation completes
        future_reward = None

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        reward_tensor, reward_extra_info = extract_reward(batch)
        return reward_tensor, reward_extra_info

    async def fit(self):
        """
        The training loop of FlowGRPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the FlowGRPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return

    async def fit_step(self, batch_data_future, continuous_iterator):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update actor -> 9. Post-step processing

        Args:
            batch_data_future: batch future
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()

        with marked_timer("step", self.timing_raw):
            batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_ref_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_advantage(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_actor(batch)
            await asyncio.sleep(0)
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_experimental(batch)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        # sync weights from actor to rollout
        with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
            self._fit_update_weights()
            await self.async_rollout_manager.clear_kv_cache()

        # async next generation
        if not self.is_last_step:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future
