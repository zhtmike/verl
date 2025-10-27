# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from collections import OrderedDict
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    assert mcore.__version__ >= "0.13.0", "Fused forward patching requires mecore >= 0.13.0"
    model = _get_patching_model(model)
    if model is not None:
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model.forward = model.forward_backup


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else True
        )  # vision model always needs pre_process
        post_process: bool = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        output_orig: CausalLMOutputForPPO = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            labels=labels_rmpad,
            packed_seq_params=packed_seq_params,
            temperature=temperature,
            **model_kwargs,
        )

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    # Run decoder.
    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
        **kwargs,
    )

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        model.output_layer.weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
