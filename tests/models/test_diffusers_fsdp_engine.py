import os
import pytest
import ray
import torch
from functools import partial
import numpy as np

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.engine import DiffusersFSDPEngine
from verl.workers.config import ActorConfig, DiffusersModelConfig, FSDPEngineConfig, FSDPOptimizerConfig, TrainingWorkerConfig, PolicyLossConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss

def create_training_config(model_type, strategy, device_count, model):
    if device_count == 1:
        cp = fsdp_size = 1
    else:
        cp = 2
        fsdp_size = 4
    path = os.path.expanduser(model)
    tokenizer_path=os.path.joint(path, "tokenizer")
    model_config = DiffusersModelConfig(
        path=path,
        tokenizer_path=tokenizer_path,
        use_remove_padding=True,
    )

    kwargs = dict(
        param_offload=True,
        optimizer_offload=True,
        grad_offload=True,
        model_dtype="float16",
        dtype="float16"
    )
    if strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=fsdp_size, strategy=strategy, ulysses_sequence_parallel_size=cp, **kwargs
        )
        optimizer_config = FSDPOptimizerConfig(lr=1e-4, weight_decay=0.0001)
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    training_config = TrainingWorkerConfig(
        model_type=model_type,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )

    policy_loss = PolicyLossConfig(loss_mode="flow_grpo")
    actor_config = ActorConfig(
         strategy=strategy,
         clip_ratio=0.0001,
         clip_ratio_high=5.0,
         ppo_mini_batch_size=4,
         ppo_micro_batch_size_per_gpu=8,
         optim=optimizer_config,
         fsdp_config=engine_config,
         policy_loss=policy_loss,
         model_config=model_config,
    )
    return training_config, actor_config

def create_data_samples(tokenizer) -> DataProto:
    from tensordict import TensorDict
    batch_size = 8
    seq_len = 64
    img_size = 512
    latent_dim=16
    cached_steps=40
    torch.manual_seed(1)
    np.random.seed(1)

    data_td = TensorDict(
        {
            "input_ids": torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len)),
            "response_mask": torch.ones((batch_size, seq_len)),
            "old_log_probs": torch.randn((batch_size, seq_len)),
            "advantages": torch.randn((batch_size, seq_len)),
            "responses": torch.randn((batch_size, 3, img_size, img_size)),
            "latents": torch.randn((batch_size, latent_dim)),
            "rollout_log_probs": torch.randn((batch_size,)),
            "timesteps": torch.randn((batch_size, cached_steps)),
            "prompt_embeds": torch.randn((batch_size, latent_dim)),
            "prompt_embeds_mask": torch.ones((batch_size, latent_dim)),
            "pooled_prompt_embeds": torch.randn((batch_size, latent_dim)),
            "negative_prompt_embeds": torch.randn((batch_size, latent_dim)),
            "negative_prompt_embeds_mask": torch.ones((batch_size, latent_dim)),
            "negative_pooled_prompt_embeds": torch.randn((batch_size, latent_dim)),
            "loss_mask": torch.ones((batch_size, seq_len)),
        },
        batch_size=batch_size,
    )
    data = DataProto(
        batch=[data_td],
        non_tensor_batch={
            "height": img_size,
            "width": img_size,
            "vae_scale_factor": 8,
        },
    )
    data.meta_info["cached_steps"] = data.batch["timesteps"].shape[1]
    data.meta_info["global_token_num"] = torch.sum(data.batch["attention_mask"], dim=-1).tolist()

    return data

@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_diffusers_fsdp_engine(strategy):
    # Create configs
    ray.init()
    device_count = torch.cuda.device_count()
    training_config, actor_config = create_training_config(
        model_type="diffusion_model",
        strategy=strategy,
        device_count=device_count,
        model="~/models/Qwen/Qwen-Image",
    )
    # init model
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    assert isinstance(wg.engine, DiffusersFSDPEngine), "Engine is not an instance of DiffusersFSDPEngine"
    wg.reset()

    # set loss function
    loss_fn = partial(ppo_loss, config=actor_config)
    wg.set_loss_fn(loss_fn)

    # eval
    data_td = create_data_samples(wg.engine.module.tokenizer)
    output = wg.infer_batch(data_td)
    loss, output_dict = output.get()

    print("Output:", output_dict)
    print("Loss:", loss)
    assert "model_output" in output_dict.keys()