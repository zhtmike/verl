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
from typing import Any

_OMNI_DIRECT_REQUEST_PARAM_NAMES = {
    "height",
    "width",
    "num_inference_steps",
    "seed",
}

_OMNI_GENERIC_REQUEST_EXTRA_ARG_NAMES = {
    "logprobs",
    "noise_level",
}


def build_diffusion_backend_sampling_params(
    sampling_params: dict[str, Any],
    *,
    model_extra_configs: dict[str, Any] | None,
    direct_param_names: set[str],
    rename_map: dict[str, str],
) -> dict[str, Any]:
    """Translate generic diffusion request params into backend sampling kwargs.

    Generic request fields stay in the agent loop. Backend/model-specific fields are
    attached here, where request-level overrides should win over model defaults.
    """
    backend_sampling_params: dict[str, Any] = {}
    extra_args = {
        key: value for key, value in (model_extra_configs or {}).items() if value is not None
    }

    for key, value in sampling_params.items():
        if value is None:
            continue

        backend_key = rename_map.get(key, key)
        if backend_key in direct_param_names:
            backend_sampling_params[backend_key] = value
        else:
            extra_args[backend_key] = value

    if extra_args:
        backend_sampling_params["extra_args"] = extra_args

    return backend_sampling_params


def build_omni_diffusion_sampling_kwargs(
    sampling_params: dict[str, Any],
    *,
    model_extra_configs: dict[str, Any] | None,
    omni_engine_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build Omni diffusion sampling kwargs without aliasing distinct fields.

    Generic rollout fields like `guidance_scale` and `max_model_len` stay generic at the
    agent-loop boundary. The Omni adapter only forwards request fields with matching
    semantics and reads Omni-native controls from backend-owned engine kwargs.
    """
    allowed_request_names = _OMNI_DIRECT_REQUEST_PARAM_NAMES | _OMNI_GENERIC_REQUEST_EXTRA_ARG_NAMES
    filtered_sampling_params = {
        key: value for key, value in sampling_params.items() if key in allowed_request_names and value is not None
    }
    sampling_kwargs = build_diffusion_backend_sampling_params(
        filtered_sampling_params,
        model_extra_configs=model_extra_configs,
        direct_param_names=_OMNI_DIRECT_REQUEST_PARAM_NAMES,
        rename_map={},
    )

    omni_engine_kwargs = omni_engine_kwargs or {}
    for key in ("true_cfg_scale", "max_sequence_length"):
        value = omni_engine_kwargs.get(key)
        if value is not None:
            sampling_kwargs[key] = value

    return sampling_kwargs
