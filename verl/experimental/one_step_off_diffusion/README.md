# Recipe: One Step Off Diffusion Policy Async Trainer

Last updated: 04/20/2026.

## Introduction

### Background

Standard diffusion FlowGRPO training is synchronous: each step waits for rollout generation to finish, then computes
reward/log-prob/advantage, and finally updates the actor. This is simple and stable, but it can waste GPU time when
generation has long-tail latency.

### Solution

This module implements a **one-step-off async trainer** for diffusion FlowGRPO.
It overlaps generation of the next batch with optimization of the current batch:

1. Wait for the previous async generation result.
2. Immediately launch the next async generation.
3. Run reward/log-prob/reference/advantage/update on the current batch.

This keeps the training pipeline busy while preserving one-step-off-policy behavior.

## Implementation

### Async Pipeline

The key control flow is implemented in `ray_diffusion_trainer.py`:

- `_create_continuous_iterator`: keeps a continuous epoch-spanning data stream.
- `_async_gen_next_batch`: builds a batch, launches async rollout generation, and returns the future payload.
- `fit_step`: consumes the previous future, starts the next one, then runs the training sub-steps.

Minimal pseudocode:

```python
continuous_iterator = self._create_continuous_iterator()
batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))

while batch_data_future is not None:
    batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)
    batch = self._fit_compute_reward(batch)
    batch = self._fit_compute_log_prob(batch)
    batch = self._fit_compute_ref_log_prob(batch)
    batch = self._fit_compute_advantage(batch)
    batch = self._fit_update_actor(batch)
```

### Parameter Synchronization

After consuming one generated batch, the trainer synchronizes actor weights to rollout before launching the next async
generation:

- `_fit_update_weights()`
- `await self.async_rollout_manager.clear_kv_cache()`

This ensures rollout uses fresh parameters with one-step lag.

## Usage

### Entrypoint

```shell
python3 -m verl.experimental.one_step_off_diffusion_policy.main_flowgrpo \
    --config-path=config \
    --config-name='one_step_off_flowgrpo_trainer.yaml' \
    algorithm.adv_estimator=flow_grpo
```

### Example Scripts

- Training-style example: `examples/flowgrpo_trainer/run_flowgrpo_one_step_off.sh`
- CI smoke test: `tests/special_e2e/run_flowgrpo_trainer_one_step_off.sh`

The smoke test follows the same low-cost strategy as
`tests/special_e2e/run_flowgrpo_trainer_diffusers.sh`.

### Configuration Notes

- This recipe is diffusion-only and separated from `one_step_off_policy` (LLM).
- `actor_rollout_ref.hybrid_engine` must be `False` in this async setup.
- Rollout resource knobs are configured with:
  - `rollout.nnodes`
  - `rollout.n_gpus_per_node`

## Functional Support

| Category | Support Situation |
| --- | --- |
| Task type | Diffusion FlowGRPO |
| Training mode | One-step-off async |
| Entrypoint | `verl.experimental.one_step_off_diffusion_policy.main_flowgrpo` |
| Config | `config/one_step_off_flowgrpo_trainer.yaml` |

## Acknowledgement

This recipe borrows the core one-step-off async training idea from
`verl.experimental.one_step_off_policy`, and adapts it to diffusion FlowGRPO
training/runtime paths.
