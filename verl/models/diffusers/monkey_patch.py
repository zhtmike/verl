# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging

import torch

logger = logging.getLogger(__name__)


def _patch_sp_native_attention_backward():
    """Fix _native_attention_backward_op shape mismatch: pass grad_out directly
    (not permuted) since out is already in [B, S, H, D] format after the permute."""
    try:
        import diffusers.models.attention_dispatch as ad
    except ImportError:
        return

    if not hasattr(ad, "_native_attention_backward_op"):
        return

    def patched_bwd_fn(ctx, grad_out, *args, **kwargs):
        # Wrap in enable_grad: Function.backward runs under no-grad by default,
        # so SDPA would produce output with no grad_fn, making torch.autograd.grad fail.
        with torch.enable_grad():
            query, key, value = ctx.saved_tensors

            query = query.detach().requires_grad_(True)
            key = key.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)

            query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
            out = torch.nn.functional.scaled_dot_product_attention(
                query=query_t,
                key=key_t,
                value=value_t,
                attn_mask=ctx.attn_mask,
                dropout_p=ctx.dropout_p,
                is_causal=ctx.is_causal,
                scale=ctx.scale,
                enable_gqa=ctx.enable_gqa,
            )
            out = out.permute(0, 2, 1, 3)  # [B, H, S, D] -> [B, S, H, D]

            # grad_out is [B, S, H, D] matching `out` — do NOT permute it
            grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
                outputs=out,
                inputs=[query_t, key_t, value_t],
                grad_outputs=grad_out,
                retain_graph=False,
            )

        grad_query = grad_query_t.permute(0, 2, 1, 3)
        grad_key = grad_key_t.permute(0, 2, 1, 3)
        grad_value = grad_value_t.permute(0, 2, 1, 3)

        return grad_query, grad_key, grad_value

    ad._native_attention_backward_op = patched_bwd_fn
    logger.info("Patched diffusers._native_attention_backward_op for [B,S,H,D] grad_out shape mismatch.")


def apply_monkey_patch_for_ulysses_sp():
    """Apply all monkey patches required for Ulysses Sequence Parallel training."""
    logger.warning("Applying diffusers monkey-patches for Ulysses Sequence Parallel. ")

    _patch_sp_native_attention_backward()
