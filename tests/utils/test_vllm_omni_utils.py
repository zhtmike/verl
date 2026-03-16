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
"""
Unit tests for verl/utils/vllm_omni/utils.py.

All external dependencies (vllm_omni, vllm) are mocked via unittest.mock.
The module is loaded directly from its source file
(importlib.util.spec_from_file_location) so that the full verl package
initialisation (which requires ray, torch, …) is bypassed.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — build lightweight fake module trees
# ---------------------------------------------------------------------------


class FakeOmniLoRARequest:
    """Minimal stand-in for vllm_omni.lora.request.LoRARequest."""

    def __init__(self, lora_name: str = "", lora_int_id: int = 1, lora_path: str = "dummy", **kwargs):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeDiffusionLoRAManagerClass:
    """Stand-in for vllm_omni.diffusion.lora.manager.DiffusionLoRAManager."""

    _load_adapter = MagicMock()


class FakeLoRAModel:
    """Stand-in for vllm.lora.lora_model.LoRAModel."""

    def __init__(self, lora_id=1):
        self.id = lora_id
        self.loras = {}

    @classmethod
    def from_lora_tensors(cls, tensors, peft_helper, lora_model_id, **kwargs):
        m = cls(lora_id=lora_model_id)
        return m

    @classmethod
    def from_local_checkpoint(cls, path, expected_lora_modules, peft_helper, lora_model_id, **kwargs):
        m = cls(lora_id=lora_model_id)
        return m


class FakePEFTHelper:
    """Stand-in for vllm.lora.peft_helper.PEFTHelper."""

    r = 4
    lora_alpha = 8
    target_modules = ["q_proj"]

    @classmethod
    def from_dict(cls, d):
        return cls()

    @classmethod
    def from_local_dir(cls, *args, **kwargs):
        return cls()


def _build_sys_modules_patch():
    """Return a dict of fake modules that satisfy all imports in utils.py."""
    # ---- vllm_omni ----
    vllm_omni = types.ModuleType("vllm_omni")

    lora_mod = types.ModuleType("vllm_omni.lora")
    lora_req_mod = types.ModuleType("vllm_omni.lora.request")
    lora_req_mod.LoRARequest = FakeOmniLoRARequest
    vllm_omni.lora = lora_mod
    lora_mod.request = lora_req_mod

    diffusion_mod = types.ModuleType("vllm_omni.diffusion")
    diffusion_lora_mod = types.ModuleType("vllm_omni.diffusion.lora")
    diffusion_lora_manager_mod = types.ModuleType("vllm_omni.diffusion.lora.manager")
    fake_manager = FakeDiffusionLoRAManagerClass
    fake_manager._load_adapter = MagicMock()  # reset for each test suite
    diffusion_lora_manager_mod.DiffusionLoRAManager = fake_manager
    diffusion_lora_manager_mod.logger = MagicMock()
    vllm_omni.diffusion = diffusion_mod
    diffusion_mod.lora = diffusion_lora_mod
    diffusion_lora_mod.manager = diffusion_lora_manager_mod

    # ---- vllm ----
    vllm_mod = types.ModuleType("vllm")
    vllm_lora_mod = types.ModuleType("vllm.lora")
    vllm_lora_model_mod = types.ModuleType("vllm.lora.lora_model")
    vllm_lora_model_mod.LoRAModel = FakeLoRAModel
    vllm_peft_helper_mod = types.ModuleType("vllm.lora.peft_helper")
    vllm_peft_helper_mod.PEFTHelper = FakePEFTHelper
    vllm_lora_utils_mod = types.ModuleType("vllm.lora.utils")
    vllm_lora_utils_mod.get_adapter_absolute_path = lambda p: p
    vllm_mod.lora = vllm_lora_mod

    return {
        "vllm_omni": vllm_omni,
        "vllm_omni.lora": lora_mod,
        "vllm_omni.lora.request": lora_req_mod,
        "vllm_omni.diffusion": diffusion_mod,
        "vllm_omni.diffusion.lora": diffusion_lora_mod,
        "vllm_omni.diffusion.lora.manager": diffusion_lora_manager_mod,
        "vllm": vllm_mod,
        "vllm.lora": vllm_lora_mod,
        "vllm.lora.lora_model": vllm_lora_model_mod,
        "vllm.lora.peft_helper": vllm_peft_helper_mod,
        "vllm.lora.utils": vllm_lora_utils_mod,
    }, fake_manager


def _load_utils_module(patch_dict, fake_manager_cls):
    """
    Load verl/utils/vllm_omni/utils.py directly from disk, bypassing the full
    verl package init (which would require ray, torch, etc.).
    """
    # Evict any previously cached version
    for key in list(sys.modules.keys()):
        if key in ("verl.utils.vllm_omni.utils",):
            del sys.modules[key]

    # Reset the manager mock so tests are isolated
    fake_manager_cls._load_adapter = MagicMock()

    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "verl", "utils", "vllm_omni", "utils.py")
    src_path = os.path.abspath(src_path)

    spec = importlib.util.spec_from_file_location("verl.utils.vllm_omni.utils", src_path)
    mod = importlib.util.module_from_spec(spec)

    old = {k: sys.modules.get(k) for k in patch_dict}
    sys.modules.update(patch_dict)
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod


# ---------------------------------------------------------------------------
# Module-level fixture — load once for the whole test session
# ---------------------------------------------------------------------------

_PATCH_DICT, _FAKE_MANAGER = _build_sys_modules_patch()
_MOD = _load_utils_module(_PATCH_DICT, _FAKE_MANAGER)


# ---------------------------------------------------------------------------
# Tests for OmniTensorLoRARequest
# ---------------------------------------------------------------------------


class TestOmniTensorLoRARequest:
    """Tests for the OmniTensorLoRARequest class."""

    def test_is_subclass_of_omni_lora_request(self):
        """OmniTensorLoRARequest must extend OmniLoRARequest (FakeOmniLoRARequest)."""
        assert issubclass(_MOD.OmniTensorLoRARequest, FakeOmniLoRARequest)

    def test_has_peft_config_annotation(self):
        """OmniTensorLoRARequest must declare a peft_config field annotation."""
        annotations = getattr(_MOD.OmniTensorLoRARequest, "__annotations__", {})
        assert "peft_config" in annotations

    def test_has_lora_tensors_annotation(self):
        """OmniTensorLoRARequest must declare a lora_tensors field annotation."""
        annotations = getattr(_MOD.OmniTensorLoRARequest, "__annotations__", {})
        assert "lora_tensors" in annotations

    def test_peft_config_is_dict_typed(self):
        """peft_config annotation should be dict."""
        annotations = _MOD.OmniTensorLoRARequest.__annotations__
        assert annotations["peft_config"] is dict

    def test_lora_tensors_is_dict_typed(self):
        """lora_tensors annotation should be dict."""
        annotations = _MOD.OmniTensorLoRARequest.__annotations__
        assert annotations["lora_tensors"] is dict


# ---------------------------------------------------------------------------
# Tests for VLLMOmniHijack
# ---------------------------------------------------------------------------


class TestVLLMOmniHijack:
    """Tests for VLLMOmniHijack.hijack()."""

    def setup_method(self):
        """Re-load the module fresh before each test so hijack state is isolated."""
        _, self._fake_manager = _build_sys_modules_patch()
        self._mod = _load_utils_module(_PATCH_DICT, self._fake_manager)

    def test_hijack_is_callable_without_instance(self):
        """hijack() is a static method — callable on the class directly."""
        assert callable(self._mod.VLLMOmniHijack.hijack)

    def test_hijack_replaces_load_adapter(self):
        """After hijack(), DiffusionLoRAManager._load_adapter is replaced."""
        original = self._fake_manager._load_adapter
        self._mod.VLLMOmniHijack.hijack()
        assert self._fake_manager._load_adapter is not original, (
            "hijack() must replace DiffusionLoRAManager._load_adapter"
        )

    def test_hijack_is_idempotent(self):
        """Calling hijack() twice must not raise."""
        self._mod.VLLMOmniHijack.hijack()
        self._mod.VLLMOmniHijack.hijack()  # second call should not raise

    def test_hijacked_adapter_returns_lora_model_and_peft_helper(self):
        """
        After hijack(), calling _load_adapter with an OmniTensorLoRARequest-like
        object should return (LoRAModel, PEFTHelper).
        """
        self._mod.VLLMOmniHijack.hijack()
        patched_fn = self._fake_manager._load_adapter

        fake_manager_instance = MagicMock()
        fake_manager_instance._expected_lora_modules = ["q_proj"]
        fake_manager_instance.dtype = "bfloat16"

        # Build an OmniTensorLoRARequest instance
        tensor_req = MagicMock(spec=self._mod.OmniTensorLoRARequest)
        tensor_req.peft_config = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
        tensor_req.lora_tensors = {"q_proj.weight": [0.1]}
        tensor_req.lora_int_id = 99

        result = patched_fn(fake_manager_instance, tensor_req)
        assert result is not None
        lora_model, peft_helper = result
        assert lora_model is not None
        assert peft_helper is not None

    def test_hijacked_adapter_raises_when_no_expected_modules(self):
        """
        _load_adapter should raise ValueError when _expected_lora_modules is empty.
        """
        self._mod.VLLMOmniHijack.hijack()
        patched_fn = self._fake_manager._load_adapter

        fake_manager_instance = MagicMock()
        fake_manager_instance._expected_lora_modules = []

        tensor_req = MagicMock(spec=self._mod.OmniTensorLoRARequest)

        with pytest.raises(ValueError, match="No supported LoRA modules"):
            patched_fn(fake_manager_instance, tensor_req)
