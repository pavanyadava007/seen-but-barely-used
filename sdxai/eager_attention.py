"""Eager (fp32, torch-only) replacement for upstream `FlashMHA` that exposes attention weights.

Flash-attn returns `attn_weights=None`. `EagerMHA` reuses the wrapped module's
`in_proj_weight/in_proj_bias/out_proj` and computes softmax(QK^T/sqrt(d)) explicitly.

key_padding_mask semantics follow the flash path (True = key is VALID), because that is what
`FlashAttention.forward` passes to `unpad_input`.

Additional analysis-only hooks:
  * `key_valid`   : extra bool (B, S) mask ANDed into key validity (instance masking)
  * `record`      : callable(name, weights[B,H,Q,S]) invoked on every forward
  * `zero_values` : bool, replace all values by 0 (removes the information flow through this
                    attention; the residual connection and output bias remain)
"""
from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EagerMHA(nn.Module):
    def __init__(self, flash_mha: nn.Module, name: str):
        super().__init__()
        self.inner = flash_mha  # shares parameters; nothing copied
        self.name = name
        self.num_heads = flash_mha.num_heads
        self.head_dim = flash_mha.head_dim
        self.record: Optional[Callable[[str, torch.Tensor], None]] = None
        self.key_valid: Optional[torch.Tensor] = None
        self.zero_values = False

    def forward(self, q, k, v, key_padding_mask=None):
        m = self.inner
        w = m.in_proj_weight.float()
        b = m.in_proj_bias.float() if m.in_proj_bias is not None else None
        wq, wk, wv = w.chunk(3)
        bq, bk, bv = b.chunk(3) if b is not None else (None, None, None)
        q = F.linear(q.float(), wq, bq)
        k = F.linear(k.float(), wk, bk)
        v = F.linear(v.float(), wv, bv)
        if self.zero_values:
            v = torch.zeros_like(v)
        B, T, _ = q.shape
        S = k.shape[1]
        H, D = self.num_heads, self.head_dim
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)  # B,H,T,S
        valid = None
        if key_padding_mask is not None:
            valid = key_padding_mask.bool()
        if self.key_valid is not None:
            valid = self.key_valid if valid is None else (valid & self.key_valid)
        if valid is not None:
            logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        if self.record is not None:
            self.record(self.name, attn)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, H * D)
        out = F.linear(ctx, m.out_proj.weight.float(),
                       m.out_proj.bias.float() if m.out_proj.bias is not None else None)
        return out, attn


def flash_wrappers(model: nn.Module) -> Dict[str, nn.Module]:
    """name -> MultiheadFlashAttention module (the wrapper holding `.attn`)."""
    from projects.mmdet3d_plugin.models.attention import MultiheadFlashAttention
    return {n: m for n, m in model.named_modules() if isinstance(m, MultiheadFlashAttention)}


@contextlib.contextmanager
def use_eager_attention(model: nn.Module, prefixes: Optional[List[str]] = None,
                        record: Optional[Callable[[str, torch.Tensor], None]] = None):
    """Swap `.attn` of every MultiheadFlashAttention (optionally only modules equal to, or nested
    under, one of the dotted paths in `prefixes`) for an EagerMHA. Yields {name: EagerMHA}. Restores the flash modules on exit."""
    swapped = {}
    eagers = {}
    try:
        for name, wrapper in flash_wrappers(model).items():
            if prefixes and not any(name == p or name.startswith(p + ".") for p in prefixes):
                continue
            eager = EagerMHA(wrapper.attn, name)
            eager.record = record
            swapped[name] = wrapper.attn
            wrapper.attn = eager
            eagers[name] = eager
        yield eagers
    finally:
        for name, wrapper in flash_wrappers(model).items():
            if name in swapped:
                wrapper.attn = swapped[name]
        # flash_wrappers finds wrappers by type, so the loop above restores all of them


class AttentionRecorder:
    """Collects attention maps; `select` reduces each map before storing (e.g. ego row only)."""

    def __init__(self, select: Optional[Callable[[str, torch.Tensor], Optional[torch.Tensor]]] = None):
        self.select = select
        self.maps: Dict[str, List[torch.Tensor]] = {}

    def __call__(self, name: str, attn: torch.Tensor) -> None:
        x = attn if self.select is None else self.select(name, attn)
        if x is not None:
            self.maps.setdefault(name, []).append(x.detach())

    def clear(self) -> None:
        self.maps = {}
