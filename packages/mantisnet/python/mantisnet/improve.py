"""Closed-form policy improvement (``KLENT_FOR_HEXO.md`` section 2, eq. 3).

    pi'(a|s)  proportional to  exp[ (Q_score(s,a) + tau * log pi(a|s)) / (tau + lam) ]
    v_hat(s) = E_{pi'}[ Q(s, A) ]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .segments import segment_ids, segment_log_softmax, segment_sum


@dataclass
class ImprovedPolicy:
    """π′ and its per-position diagnostics, flat over the legal cells."""

    probs: Tensor  # (N,) π′, sums to 1 within each segment
    v_hat: Tensor  # (P,) E_{π′}[Q]
    kl: Tensor  # (P,) D_KL(π′ ‖ π_θ)
    norm_entropy: Tensor  # (P,) H(π′) / log|A|; defined as 0 where |A| = 1


@torch.no_grad()
def improved_policy(
    policy_logits: Tensor,
    q_score: Tensor,
    q_value: Tensor,
    offsets: Tensor,
    tau: float,
    lam: float,
) -> ImprovedPolicy:
    """Apply eq. 3 to the acting score while averaging the action value."""
    if tau < 0 or lam < 0 or tau + lam <= 0:
        raise ValueError(f"need tau, lam >= 0 with tau + lam > 0, got ({tau}, {lam})")
    if q_score.shape != q_value.shape:
        raise ValueError(
            "q_score and q_value must have the same shape, got "
            f"{tuple(q_score.shape)} and {tuple(q_value.shape)}"
        )
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)

    # At least fp32; float64 when inputs are, to preserve comparison precision.
    dtype = torch.promote_types(
        torch.promote_types(
            torch.promote_types(policy_logits.dtype, q_score.dtype), q_value.dtype
        ),
        torch.float32,
    )
    log_pi = segment_log_softmax(policy_logits.to(dtype), offsets)
    score = q_score.to(dtype)
    q = q_value.to(dtype)
    log_improved = segment_log_softmax(
        (score + tau * log_pi) / (tau + lam), offsets
    )
    probs = log_improved.exp()

    # Normalize by segment mass to keep |E[Q]| <= max|Q| despite fp32 ulps.
    mass = segment_sum(probs, seg, p)
    v_hat = segment_sum(probs * q, seg, p) / mass
    kl = segment_sum(probs * (log_improved - log_pi), seg, p) / mass
    entropy = segment_sum(-probs * log_improved, seg, p) / mass
    counts = (offsets[1:] - offsets[:-1]).to(dtype)
    norm_entropy = torch.where(counts > 1, entropy / counts.clamp(min=2).log(), entropy.new_zeros(p))
    return ImprovedPolicy(probs=probs, v_hat=v_hat, kl=kl, norm_entropy=norm_entropy)
