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
from verl.workers.rollout.diffusion_sampling_utils import build_omni_diffusion_sampling_kwargs


def test_build_omni_diffusion_sampling_kwargs_keeps_generic_fields_distinct():
    sampling_kwargs = build_omni_diffusion_sampling_kwargs(
        sampling_params={
            "num_inference_steps": 10,
            "guidance_scale": 4.5,
            "height": 512,
            "width": 768,
            "max_model_len": 2048,
            "seed": 123,
            "noise_level": 0.25,
            "logprobs": True,
        },
        model_extra_configs={
            "noise_level": 0.7,
            "sde_type": "cps",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
        },
        omni_engine_kwargs={
            "true_cfg_scale": 5.0,
            "max_sequence_length": 1536,
        },
    )

    assert sampling_kwargs == {
        "num_inference_steps": 10,
        "height": 512,
        "width": 768,
        "seed": 123,
        "true_cfg_scale": 5.0,
        "max_sequence_length": 1536,
        "extra_args": {
            "noise_level": 0.25,
            "logprobs": True,
            "sde_type": "cps",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
        },
    }


def test_build_omni_diffusion_sampling_kwargs_request_overrides_model_extras():
    sampling_kwargs = build_omni_diffusion_sampling_kwargs(
        sampling_params={
            "noise_level": 0.1,
            "logprobs": False,
        },
        model_extra_configs={
            "noise_level": 0.7,
            "sde_type": "sde",
        },
        omni_engine_kwargs={},
    )

    assert sampling_kwargs == {
        "extra_args": {
            "noise_level": 0.1,
            "logprobs": False,
            "sde_type": "sde",
        }
    }
