"""MoE (gluon MXFP4) per-stage time and effective weight bandwidth at TP4 shapes."""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/workspace/.wt-wave/tokenspeed-kernel/test")
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from utils import make_mxfp4_moe_weights

E, H, I, TOPK = 384, 5120, 576, 6
MS = (
    [int(v) for v in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [8, 16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 1024]
)
g = torch.Generator(device="cuda").manual_seed(1)
raw = make_mxfp4_moe_weights(E, H, I, g)
m = torch.nn.Module()
m.w13_input_layout = "interleaved"
m.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
for k, s in [
    ("w13_weight", "w13_weight"),
    ("w13_weight_scale", "w13_scale"),
    ("w2_weight", "w2_weight"),
    ("w2_weight_scale", "w2_scale"),
]:
    setattr(m, k, torch.nn.Parameter(raw[s].clone(), requires_grad=False))
preprocess_gluon_mxfp4_gfx950_moe_weights({}, m)


def run(x, tw, ti):
    return gluon_mxfp_precomputed_mxfp4_fused_moe(
        x,
        tw,
        ti,
        m.w13_weight_triton_tensor,
        m.w2_weight_triton_tensor,
        w13_mx_scale=m.w13_precision_config.b_mx_scale,
        w2_mx_scale=m.w2_precision_config.b_mx_scale,
        expert_start=0,
        global_num_experts=E,
        do_finalize=True,
        x_quantized=None,
        out_dtype=torch.bfloat16,
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        swiglu_beta=0.0,
        out=None,
    )


# reference read bandwidth
big = torch.empty(2 * 1024**3 // 4, dtype=torch.float32, device="cuda")
dst = torch.empty_like(big)
for _ in range(3):
    dst.copy_(big)
st = torch.cuda.Event(True)
en = torch.cuda.Event(True)
st.record()
for _ in range(10):
    dst.copy_(big)
en.record()
torch.cuda.synchronize()
print(f"copy BW (r+w, 2GB): {4*10/(st.elapsed_time(en)/1e3)/1e3:.2f} TB/s")
w13b = 2 * I * H // 2 * (1 + 1 / 16)
w2b = H * I // 2 * (1 + 1 / 16)
for M in MS:
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=g)
    ti = (
        torch.rand(M, E, device="cuda", generator=g)
        .topk(TOPK, dim=1)
        .indices.to(torch.int32)
    )
    tw = torch.softmax(torch.randn(M, TOPK, device="cuda", generator=g), dim=1)
    ne = torch.unique(ti).numel()
    for _ in range(5):
        run(x, tw, ti)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(30):
            run(x, tw, ti)
        torch.cuda.synchronize()
    ks = {}
    for ev in p.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            ks[ev.key] = ev.device_time_total / ev.count
    if M == MS[0]:
        print(list(ks))
    s1 = sum(v for k, v in ks.items() if "stage1" in k)
    s2 = sum(v for k, v in ks.items() if "stage2" in k)
    tot = sum(ks.values())
    print(
        f"M={M:5d} experts={ne:3d} total={tot:7.1f}us s1={s1:7.1f}us({ne*w13b/s1/1e6:5.2f}TB/s) s2={s2:7.1f}us({ne*w2b/s2/1e6:5.2f}TB/s) other={tot-s1-s2:5.1f} | "
        + " ".join(
            f"{k[:28]}={v:.0f}"
            for k, v in sorted(ks.items(), key=lambda kv: -kv[1])[:6]
        ),
        flush=True,
    )
