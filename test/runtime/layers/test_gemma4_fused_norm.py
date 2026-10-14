"""Fused Gemma 4 norm kernels against the unfused op sequence they replace."""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed_kernel.ops.layernorm import (  # noqa: E402
    qk_rmsnorm_rope_neox,
    rmsnorm_residual_rmsnorm,
)

from tokenspeed.runtime.layers.layernorm import RMSNorm  # noqa: E402
from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    Gemma4RotaryEmbedding,
    get_rope,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_BF16 = dict(rtol=1.6e-2, atol=2e-2)


def _norm(width: int) -> RMSNorm:
    norm = RMSNorm(width, eps=1e-6).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    return norm


@pytest.mark.parametrize("tokens", [1, 33])
@pytest.mark.parametrize("with_scale", [False, True])
def test_rmsnorm_residual_rmsnorm_matches_norm_add_scale_norm(tokens, with_scale):
    torch.manual_seed(0)
    width = 5376
    post, nxt = _norm(width), _norm(width)
    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) * 3
    res = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.37], device="cuda", dtype=torch.bfloat16)

    want_y = post(x) + res
    if with_scale:
        want_y = want_y * scale
    want_z = nxt(want_y)
    y, z = rmsnorm_residual_rmsnorm(
        x, res, post.weight, nxt.weight, 1e-6, scale=scale if with_scale else None
    )
    torch.testing.assert_close(y, want_y, **_BF16)
    torch.testing.assert_close(z, want_z, **_BF16)


@pytest.mark.parametrize(
    "head_dim,nq,nkv,full", [(256, 32, 16, False), (512, 32, 4, True)]
)
def test_qk_rmsnorm_rope_neox_matches_norms_then_rope(head_dim, nq, nkv, full):
    torch.manual_seed(0)
    tokens = 19
    with torch.device("cuda"):
        if full:
            rope = Gemma4RotaryEmbedding(
                head_dim, head_dim // 4, 4096, 1e6, True, torch.float32
            )
        else:
            rope = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=4096,
                base=1e4,
                is_neox_style=True,
                rope_scaling=None,
                dtype=torch.float32,
            )
    rope.cos_sin_cache = rope.cos_sin_cache.cuda()
    q_norm, k_norm = _norm(head_dim), _norm(head_dim)
    qkv = torch.randn(
        tokens, (nq + 2 * nkv) * head_dim, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    heads = qkv.view(tokens, -1, head_dim)
    q, k, _ = heads.split([nq, nkv, nkv], dim=1)

    want_q = q_norm(q).view(tokens, -1)
    want_k = k_norm(k).view(tokens, -1)
    want_q, want_k = rope(positions, want_q, want_k)
    got_q, got_k = qk_rmsnorm_rope_neox(
        q, k, q_norm.weight, k_norm.weight, 1e-6, positions, rope.cos_sin_cache
    )
    assert got_q.shape == want_q.shape and got_k.shape == want_k.shape
    torch.testing.assert_close(got_q, want_q, **_BF16)
    torch.testing.assert_close(got_k, want_k, **_BF16)

    # Same launch with weightless v_norm (Gemma 4 path).
    vv = heads.split([nq, nkv, nkv], dim=1)[2]
    from tokenspeed.runtime.layers.layernorm import RMSNormNoWeight

    v_norm = RMSNormNoWeight(head_dim, eps=1e-6).cuda()
    want_v = v_norm(vv).view(tokens, -1)
    got_q2, got_k2, got_v = qk_rmsnorm_rope_neox(
        q,
        k,
        q_norm.weight,
        k_norm.weight,
        1e-6,
        positions,
        rope.cos_sin_cache,
        v=vv,
    )
    torch.testing.assert_close(got_q2, want_q, **_BF16)
    torch.testing.assert_close(got_k2, want_k, **_BF16)
    torch.testing.assert_close(got_v, want_v, **_BF16)


def test_fused_kernels_handle_an_empty_batch():
    w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    y, z = rmsnorm_residual_rmsnorm(x, x, w, w, 1e-6)
    assert y.shape == (0, 256) and z.shape == (0, 256)
    heads = torch.empty(0, 3, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.ones(8, 256, device="cuda")
    pos = torch.empty(0, dtype=torch.long, device="cuda")
    q, k = qk_rmsnorm_rope_neox(heads[:, :2], heads[:, 2:], w, w, 1e-6, pos, cache)
    assert q.shape == (0, 512) and k.shape == (0, 256)
    heads4 = torch.empty(0, 4, 256, device="cuda", dtype=torch.bfloat16)
    q, k, v = qk_rmsnorm_rope_neox(
        heads4[:, :2], heads4[:, 2:3], w, w, 1e-6, pos, cache, v=heads4[:, 3:]
    )
    assert q.shape == (0, 512) and k.shape == (0, 256) and v.shape == (0, 256)
