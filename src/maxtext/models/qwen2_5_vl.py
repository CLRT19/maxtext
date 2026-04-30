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

"""Qwen2.5-VL vision tower in MaxText.

The text decoder reuses `maxtext.models.qwen2.Qwen2DecoderLayer` with
`use_mrope=True` and `mrope_section=[16, 24, 24]`, configured via
`configs/models/qwen2.5-vl-7b.yml`.

This file ports the vision tower from
  /home/linrong/repo/vlm-tpu/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py
which is the production JAX inference path for Qwen2.5-VL on TPU. The math is
preserved verbatim; differences are:

  * `sharded_flash_attention` → `jax.nn.dot_product_attention` (first-cut;
    Pallas flash kernel is a follow-up perf optimization).
  * `Qwen2_5_VLVisionConfig` → MaxText `Config` with `*_for_vit` flat fields.
  * `nnx.initializers.uniform()` → MaxText's `nn.initializers.normal(stddev=1.0)`.

Numerical reference (Phase 1.4 parity gate):
  gs://linrong-vlm-tpu-us-central1-a/parity/hf_ref.npz
"""
# pylint: disable=arguments-differ

from __future__ import annotations

import math
from functools import partial
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config


_INIT = nnx.initializers.normal(stddev=0.02)


class _SegmentIds(NamedTuple):
  """Per-token segment ids; tokens with the same id can attend to each other."""

  q: jax.Array  # [B, T_q]
  kv: jax.Array  # [B, T_kv]


def _apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
  """Vision RoPE: split last dim in half, rotate as complex pairs.

  x: [B, T, N, H]
  rotary_pos_emb: [T, H/2]
  """
  half_dim = x.shape[-1] // 2
  x_real = x[..., :half_dim]
  x_imag = x[..., half_dim:]
  cos_emb = jnp.cos(rotary_pos_emb)[None, :, None, :]
  sin_emb = jnp.sin(rotary_pos_emb)[None, :, None, :]
  x_rotated_real = x_real * cos_emb - x_imag * sin_emb
  x_rotated_imag = x_real * sin_emb + x_imag * cos_emb
  return jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)


def _generate_window_segment_ids(
    cu_seqlens: jax.Array, seq_len: int, padded_seq_len: int
) -> _SegmentIds:
  """Segment ids for windowed attention; padding gets segment 0."""
  indices = jnp.arange(seq_len, dtype=jnp.int32)
  segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side="right") + 1
  padding_segment_ids = jnp.zeros(padded_seq_len - seq_len, dtype=jnp.int32)
  segment_ids = jnp.concatenate([segment_ids, padding_segment_ids]).reshape(1, -1)
  return _SegmentIds(q=segment_ids, kv=segment_ids)


def _attn_xla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: Optional[_SegmentIds],
    sm_scale: float,
) -> jax.Array:
  """Vanilla XLA attention with optional segment mask. Replaces flash kernel.

  q, k, v: [B, N, T, H]. Returns [B, N, T, H].
  """
  scores = jnp.einsum("bnth,bnsh->bnts", q, k) * sm_scale
  if segment_ids is not None:
    # segment_ids.q: [1, T], segment_ids.kv: [1, T]
    mask = segment_ids.q[..., None] == segment_ids.kv[:, None, :]
    # broadcast over heads dim: [1, T, T] -> [1, 1, T, T]
    mask = mask[:, None, :, :]
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
  weights = jax.nn.softmax(scores, axis=-1)
  return jnp.einsum("bnts,bnsh->bnth", weights, v)


class Qwen2_5_VisionMLP(nnx.Module):
  """Vision MLP: gate_proj * up_proj -> down_proj. SiLU activation."""

  def __init__(self, config: Config, dtype: jnp.dtype, rngs: nnx.Rngs):
    in_features = config.hidden_size_for_vit
    hidden_features = config.intermediate_size_for_vit
    self.gate_proj = nnx.Linear(
        in_features, hidden_features, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, (None, "model")),
        bias_init=nnx.with_partitioning(_INIT, ("model",)),
        rngs=rngs,
    )
    self.up_proj = nnx.Linear(
        in_features, hidden_features, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, (None, "model")),
        bias_init=nnx.with_partitioning(_INIT, ("model",)),
        rngs=rngs,
    )
    self.down_proj = nnx.Linear(
        hidden_features, in_features, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, ("model", None)),
        bias_init=nnx.with_partitioning(_INIT, (None,)),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jax.nn.silu(self.gate_proj(x))
    up = self.up_proj(x)
    return self.down_proj(gate * up)


class Qwen2_5_VisionAttention(nnx.Module):
  """Vision self-attention with optional windowed segments and 2D RoPE.

  Padded to a multiple of 128 internally for kernel-friendliness (kept for
  parity with tpu-inference even though we use XLA attention here).
  """

  _BLOCK_K_MAJOR = 128

  def __init__(self, config: Config, dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh):
    self.hidden_size = config.hidden_size_for_vit
    self.num_heads = config.num_attention_heads_for_vit
    self.head_dim = self.hidden_size // self.num_heads
    self.mesh = mesh
    self.qkv_proj = nnx.Linear(
        self.hidden_size, 3 * self.hidden_size, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, (None, "model")),
        bias_init=nnx.with_partitioning(_INIT, ("model",)),
        rngs=rngs,
    )
    self.proj = nnx.Linear(
        self.hidden_size, self.hidden_size, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, ("model", None)),
        bias_init=nnx.with_partitioning(_INIT, (None,)),
        rngs=rngs,
    )
    self._sm_scale = 1.0 / math.sqrt(self.head_dim)

  def __call__(
      self,
      x: jax.Array,
      rotary_pos_emb: jax.Array,
      cu_window_seqlens: Optional[jax.Array] = None,
      use_fullattn: bool = True,
  ) -> jax.Array:
    # x: [T, B, D] (B = 1 in current design)
    T, B, D = x.shape
    assert B == 1, "Vision attention currently only supports batch size 1"
    qkv = self.qkv_proj(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # [T, B, D] -> [B, T, N, H] -> [B, N, T, H]
    q = jnp.transpose(q.reshape(T, B, self.num_heads, self.head_dim), (1, 0, 2, 3))
    k = jnp.transpose(k.reshape(T, B, self.num_heads, self.head_dim), (1, 0, 2, 3))
    v = jnp.transpose(v.reshape(T, B, self.num_heads, self.head_dim), (1, 0, 2, 3))
    q = _apply_rotary_pos_emb_vision(q, rotary_pos_emb)
    k = _apply_rotary_pos_emb_vision(k, rotary_pos_emb)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Pad to a multiple of 128 (parity with tpu-inference; harmless for XLA attn).
    block = self._BLOCK_K_MAJOR
    T_attn = q.shape[2]
    padded_T = (T_attn + block - 1) // block * block
    pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))
    q = jnp.pad(q, pad_width)
    k = jnp.pad(k, pad_width)
    v = jnp.pad(v, pad_width)

    segment_ids = _generate_window_segment_ids(
        cu_window_seqlens, T_attn, padded_T
    )
    out = _attn_xla(q, k, v, segment_ids, self._sm_scale)
    out = out[:, :, :T_attn, :]
    out = jnp.transpose(out, (2, 0, 1, 3)).reshape(T, B, D)
    return self.proj(out)


class Qwen2_5_VisionBlock(nnx.Module):
  """Pre-norm transformer block: norm + attn + residual + norm + mlp + residual."""

  def __init__(self, config: Config, dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh):
    dim = config.hidden_size_for_vit
    norm_layer = partial(
        nnx.RMSNorm,
        epsilon=config.normalization_layer_epsilon,
        scale_init=nnx.with_partitioning(_INIT, (None,)),
    )
    self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
    self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
    self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
    self.mlp = Qwen2_5_VisionMLP(config=config, dtype=dtype, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      rotary_pos_emb: jax.Array,
      cu_window_seqlens: Optional[jax.Array] = None,
      use_fullattn: bool = True,
  ) -> jax.Array:
    x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens, use_fullattn)
    x = x + self.mlp(self.norm2(x))
    return x


class Qwen2_5_VisionPatchEmbed(nnx.Module):
  """3D conv patch embedding over (T, H, W) patches.

  Input: pixel_values of shape [L, C * temporal_patch_size * patch_size**2].
  Output: [L, hidden_size_for_vit].
  """

  def __init__(self, config: Config, dtype: jnp.dtype, rngs: nnx.Rngs):
    self.patch_size = config.patch_size_for_vit
    self.temporal_patch_size = config.temporal_patch_size_for_vit
    self.in_channels = config.num_channels_for_vit
    self.hidden_size = config.hidden_size_for_vit
    kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
    self.proj = nnx.Conv(
        in_features=self.in_channels,
        out_features=self.hidden_size,
        kernel_size=kernel_size,
        strides=kernel_size,
        use_bias=False,
        param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, (None, None, None, None, "model")),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    L, dim = x.shape
    expected = self.temporal_patch_size * self.patch_size * self.patch_size
    C = dim // expected
    x = x.reshape(L, C, self.temporal_patch_size, self.patch_size, self.patch_size)
    x = jnp.transpose(x, (0, 2, 3, 4, 1))  # L, T, H, W, C
    x = self.proj(x)
    return x.reshape(L, self.hidden_size)


class Qwen2_5_VisionPatchMerger(nnx.Module):
  """RMSNorm + reshape (spatial_merge_size**2 patches concatenated) + MLP-2 with GELU.

  Reduces the per-patch features by spatial_merge_size**2 and projects from
  hidden_size_for_vit (1280) to out_hidden_size_for_vit (3584 = LM hidden).
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    context_dim = config.hidden_size_for_vit
    spatial_merge_size = config.spatial_merge_size_for_vit
    out_dim = config.out_hidden_size_for_vit
    dtype = getattr(jnp, config.dtype_mm) if isinstance(config.dtype_mm, str) else config.dtype_mm

    merged_hidden = context_dim * (spatial_merge_size ** 2)
    self._merged_hidden = merged_hidden
    self.ln_q = nnx.RMSNorm(
        context_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        scale_init=nnx.with_partitioning(_INIT, (None,)),
        rngs=rngs,
    )
    self.mlp_fc1 = nnx.Linear(
        merged_hidden, merged_hidden, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, (None, "model")),
        bias_init=nnx.with_partitioning(_INIT, ("model",)),
        rngs=rngs,
    )
    self.mlp_fc2 = nnx.Linear(
        merged_hidden, out_dim, use_bias=True, param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_INIT, ("model", None)),
        bias_init=nnx.with_partitioning(_INIT, (None,)),
        rngs=rngs,
    )

  def __call__(self, embeddings: jax.Array) -> jax.Array:
    x = self.ln_q(embeddings)
    x = x.reshape(-1, self._merged_hidden)
    x = self.mlp_fc1(x)
    x = jax.nn.gelu(x, approximate=False)
    x = self.mlp_fc2(x)
    return x


class _Qwen2_5_VisionRotaryEmbedding(nnx.Module):
  """1D rotary frequencies; pos_ids index into this table per axis (h or w)."""

  def __init__(self, dim: int, theta: float = 10000.0):
    self.dim = dim
    self.theta = float(theta)

  def __call__(self, seqlen: int) -> jax.Array:
    inv_freq = 1.0 / (self.theta ** (
        jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim
    ))
    seq = jnp.arange(seqlen, dtype=jnp.float32)
    return jnp.outer(seq, inv_freq).astype(jnp.bfloat16)


class Qwen2_5_VisionEncoder(nnx.Module):
  """Top-level Qwen2.5-VL vision tower.

  Inputs:
    input_images: pixel_values of shape `[total_patches, C*T*P*P]`
                  (e.g. `[N, 1176]` for default 14x14, T=2, C=3).
    grid_thw:     tuple of (T, H, W) per image, static (must be hashable).
                  Passed via call_kwargs because MaxText's `VisionEncoder.__call__`
                  surface only forwards `input_images` + `deterministic`. We
                  attach grid_thw onto the encoder as `self._next_grid_thw` from
                  the Transformer call site (TODO: integration in Phase 1.4).

  Output: pre-merger hidden states of shape `[total_patches, hidden_size_for_vit]`.
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    dtype_mm = getattr(jnp, config.dtype_mm) if isinstance(config.dtype_mm, str) else config.dtype_mm

    self.window_size = config.window_size_for_vit
    self.patch_size = config.patch_size_for_vit
    self.spatial_merge_size = config.spatial_merge_size_for_vit
    self.spatial_merge_unit = self.spatial_merge_size ** 2
    self.fullatt_block_indexes = tuple(config.fullatt_block_indexes_for_vit)
    self.num_blocks = config.num_hidden_layers_for_vit

    self.patch_embed = Qwen2_5_VisionPatchEmbed(config=config, dtype=dtype_mm, rngs=rngs)

    head_dim = config.hidden_size_for_vit // config.num_attention_heads_for_vit
    self.rotary_pos_emb = _Qwen2_5_VisionRotaryEmbedding(
        head_dim // 2, theta=float(config.rope_theta_for_vit)
    )

    self.blocks = nnx.data([
        Qwen2_5_VisionBlock(config=config, dtype=dtype_mm, rngs=rngs, mesh=mesh)
        for _ in range(self.num_blocks)
    ])

    # grid_thw is plumbed via this attribute; the Transformer top-level forward
    # sets it before calling the encoder. See models.py:481 for the hook.
    self._next_grid_thw: Optional[tuple[tuple[int, int, int], ...]] = None

  # ---------- helpers (translated from tpu-inference) ----------
  def _rotary_pos_emb_thw(self, t: int, h: int, w: int) -> jax.Array:
    hpos_ids, wpos_ids = jnp.indices((h, w))
    s = self.spatial_merge_size
    hpos_ids = hpos_ids.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).flatten()
    wpos_ids = wpos_ids.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).flatten()
    pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
    pos_ids = jnp.tile(pos_ids, (t, 1))
    rotary_pos_emb_full = self.rotary_pos_emb(max(h, w))
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
    rotary_pos_emb = rotary_pos_emb.reshape(
        rotary_pos_emb.shape[0] // self.spatial_merge_unit,
        self.spatial_merge_unit, -1,
    )
    return rotary_pos_emb

  def _get_window_index_thw(self, t: int, h: int, w: int):
    vit_merger_window = self.window_size // self.spatial_merge_size // self.patch_size
    s = self.spatial_merge_size
    llm_h = h // s
    llm_w = w // s
    index = jnp.arange(t * llm_h * llm_w).reshape(t, llm_h, llm_w)
    pad_h = vit_merger_window - llm_h % vit_merger_window
    pad_w = vit_merger_window - llm_w % vit_merger_window
    nh = (llm_h + pad_h) // vit_merger_window
    nw = (llm_w + pad_w) // vit_merger_window
    index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
    index_padded = index_padded.reshape(t, nh, vit_merger_window, nw, vit_merger_window)
    index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
        t, nh * nw, vit_merger_window, vit_merger_window
    )
    seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
    index_padded = index_padded.reshape(-1)
    num_valid = t * llm_h * llm_w
    valid_indices = jnp.nonzero(index_padded != -100, size=num_valid)[0]
    index_new = index_padded[valid_indices]
    cu_seqlens = (jnp.cumsum(seqlens) * self.spatial_merge_unit).astype(jnp.int32)
    return index_new, cu_seqlens

  def _get_rope_by_thw(self, t: int, h: int, w: int):
    window_index_thw, cu_seqlens_window_thw = self._get_window_index_thw(t, h, w)
    rotary_pos_emb_thw = self._rotary_pos_emb_thw(t, h, w)
    rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :].reshape(
        -1, rotary_pos_emb_thw.shape[-1]
    )
    cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)
    return rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw, cu_seqlens_thw

  def _compute_aux_arrays(self, grid_thw):
    rotary_pos_emb = []
    window_index = []
    cu_window_seqlens = [jnp.array([0], dtype=jnp.int32)]
    cu_seqlens_pre = []
    window_index_id = 0
    cu_window_seqlens_last = 0
    for (t, h, w) in grid_thw:
      llm_h = h // self.spatial_merge_size
      llm_w = w // self.spatial_merge_size
      rope_thw, win_thw, cu_win_thw, cu_thw = self._get_rope_by_thw(t, h, w)
      window_index.append(win_thw + window_index_id)
      window_index_id += t * llm_h * llm_w
      cu_win_thw = cu_win_thw + cu_window_seqlens_last
      cu_window_seqlens_last = cu_win_thw[-1]
      cu_window_seqlens.append(cu_win_thw)
      rotary_pos_emb.append(rope_thw)
      cu_seqlens_pre.append(cu_thw)
    rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
    window_index = jnp.concatenate(window_index, axis=0)
    cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)
    cu_seqlens = jnp.concatenate(cu_seqlens_pre, axis=0)
    cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
    cu_seqlens = jnp.pad(cu_seqlens, ((1, 0),), mode="constant", constant_values=0)
    return window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens

  def _compute_hidden_states(
      self,
      x: jax.Array,
      window_index: jax.Array,
      rotary_pos_emb: jax.Array,
      cu_seqlens: jax.Array,
      cu_window_seqlens: jax.Array,
  ) -> jax.Array:
    h = self.patch_embed(x)
    seq_len = x.shape[0]
    h = h.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    h = h[window_index, :, :].reshape(seq_len, -1)
    h = jnp.expand_dims(h, axis=1)  # [T, B=1, D]
    for layer_num, blk in enumerate(self.blocks):
      if layer_num in self.fullatt_block_indexes:
        h = blk(h, rotary_pos_emb=rotary_pos_emb, cu_window_seqlens=cu_seqlens, use_fullattn=True)
      else:
        h = blk(h, rotary_pos_emb=rotary_pos_emb, cu_window_seqlens=cu_window_seqlens, use_fullattn=False)
    return h  # pre-merger; merger is `Qwen2_5_VisionMerger` (run separately by VisionEncoder)

  def __call__(self, input_images: jax.Array, *, deterministic: bool = False) -> jax.Array:
    del deterministic  # vision tower is deterministic in inference + frozen-train
    if self._next_grid_thw is None:
      raise RuntimeError(
          "Qwen2_5_VisionEncoder requires grid_thw to be set on the encoder via"
          " `encoder._next_grid_thw = ((t, h, w), ...)` before forward. The"
          " Transformer top-level forward must thread image_grid_thw through to"
          " the encoder. See models.py:481-485 (TODO Phase 1.4)."
      )
    grid_thw = self._next_grid_thw
    self._next_grid_thw = None
    window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = (
        self._compute_aux_arrays(grid_thw)
    )
    return self._compute_hidden_states(
        input_images, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens
    )


class Qwen2_5_VisionMerger(nnx.Module):
  """Wrapper around `Qwen2_5_VisionPatchMerger` that matches MaxText's projector
  signature (single positional `embeddings`).

  Also undoes the window-permutation applied by the encoder so that the
  downstream LM sees patches in original raster order. The encoder stashes the
  inverse permutation on `self._reverse_indices` for the merger to consume.
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self._merger = Qwen2_5_VisionPatchMerger(config=config, mesh=mesh, rngs=rngs)
    self._next_reverse_indices: Optional[jax.Array] = None

  def __call__(self, embeddings: jax.Array) -> jax.Array:
    out = self._merger(embeddings)
    if self._next_reverse_indices is not None:
      out = out[self._next_reverse_indices, :]
      self._next_reverse_indices = None
    return out
