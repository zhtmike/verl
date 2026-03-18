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
