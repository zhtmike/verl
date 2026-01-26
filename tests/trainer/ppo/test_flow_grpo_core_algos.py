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

import random
import unittest

import numpy as np
import pytest
import torch
import uuid
from verl.utils.config import omega_conf_to_dataclass

from verl.trainer.ppo.core_algos import (
    compute_flow_grpo_outcome_advantage,
    compute_policy_loss_flow_grpo,
)

@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """Test flow-GRPO advantage and return computation."""

    # prepere input
    batch_size = 8
    steps = 10
    token_level_rewards = torch.randn((batch_size, steps, ), dtype=torch.float32)
    uid = np.array([uuid.uuid4().hex for _ in range(batch_size)])

    advantages, returns = compute_flow_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=None,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)

def test_compute_policy_loss_flow_grpo() -> None:
    """Test flow-GRPO policy loss computation."""

    # prepare input
    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)
    from verl.trainer.config import ActorConfig
    minimal_config = {"clip_ratio": 0.0001, "clip_ratio_high": 5.0}
    config = omega_conf_to_dataclass(minimal_config, ActorConfig)

    for step in range(steps):
        pg_loss, pg_metrics = compute_policy_loss_flow_grpo(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            config=config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics.keys()

if __name__ == "__main__":
    unittest.main()
