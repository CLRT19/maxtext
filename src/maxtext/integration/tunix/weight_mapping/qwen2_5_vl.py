# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HF→MaxText weight mapping for Qwen2.5-VL.

The text decoder mapping is identical to Qwen2 (see `qwen2.py` in this
directory) — Qwen2.5-VL's LM half *is* Qwen2.5-7B's LM. This module reuses
those entries verbatim and adds the vision-tower entries on top.

The HF state-dict for Qwen2.5-VL prefixes the LM with `model.` (same as
Qwen2.5-7B) and the vision tower with `visual.`. See the layout below.

Vision-tower entries are stubbed (Phase 1.3); the LM half is complete.
"""

from dataclasses import dataclass

from maxtext.integration.tunix.weight_mapping.qwen2 import QWEN2_VLLM_MAPPING


@dataclass
class QWEN2_5_VL_VLLM_MAPPING:
  """MaxText Qwen2.5-VL → HF Qwen2.5-VL weight mapping."""

  @staticmethod
  def to_hf_hook_fns():
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_mapping():
    """Mapping from MaxText to HF for Qwen2.5-VL.

    LM half: identical to Qwen2 mapping (28-layer dense decoder).
    Vision tower: TODO(Phase 1.3) — entries for visual.patch_embed,
    visual.blocks.{i}.{norm,attn,mlp,...}, visual.merger.
    """
    mapping = dict(QWEN2_VLLM_MAPPING.to_hf_mapping())
    # ------------------------------------------------------------------
    # TODO(Phase 1.3): vision tower entries.
    #   Reference HF tensor names (from safetensors index of
    #   /tmp/qwen/Qwen2.5-VL-7B-Instruct):
    #     visual.patch_embed.proj.weight                        (3D conv)
    #     visual.blocks.{i}.norm1.weight
    #     visual.blocks.{i}.norm2.weight
    #     visual.blocks.{i}.attn.qkv.weight
    #     visual.blocks.{i}.attn.qkv.bias
    #     visual.blocks.{i}.attn.proj.weight
    #     visual.blocks.{i}.attn.proj.bias
    #     visual.blocks.{i}.mlp.gate_proj.weight
    #     visual.blocks.{i}.mlp.up_proj.weight
    #     visual.blocks.{i}.mlp.down_proj.weight
    #     visual.merger.ln_q.weight
    #     visual.merger.mlp.0.weight
    #     visual.merger.mlp.0.bias
    #     visual.merger.mlp.2.weight
    #     visual.merger.mlp.2.bias
    # ------------------------------------------------------------------
    return mapping
