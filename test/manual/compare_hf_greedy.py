# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Greedy TokenSpeed-vs-reference check for (tiny) checkpoints.

Two phases, run as separate processes so they never share a GPU context:

  engine: serve the checkpoint with ``Engine`` (TP/EP as given), greedy-decode
          fixed random prompts and dump output ids + per-token logprobs.
  ref:    teacher-force the engine's own outputs through the reference model
          (HF transformers, or a ``--ref-json`` dump from another engine run)
          and report, per prompt, how many leading tokens match the
          reference argmax and the max |logprob| difference on those tokens.

Usage:
  python test/manual/compare_hf_greedy.py engine --model M --tp 4 --out e.json
  python test/manual/compare_hf_greedy.py ref --model M --engine-json e.json
  python test/manual/compare_hf_greedy.py cmp --engine-json a.json --ref-json b.json
"""

import argparse
import json
import os
import random

PROMPT_LENS = (5, 37, 130, 700)


def make_prompts(
    vocab_size: int, seed: int, lens: tuple[int, ...] = PROMPT_LENS
) -> list[list[int]]:
    rng = random.Random(seed)
    # Skip the low ids: most tokenizers keep special tokens there.
    lo, hi = 1000, min(vocab_size, 100000) - 1
    return [[rng.randint(lo, hi) for _ in range(n)] for n in lens]


def text_vocab_size(model: str) -> int:
    # Raw config.json: some checkpoints (glm5_next) are unknown to transformers.
    with open(os.path.join(model, "config.json")) as f:
        cfg = json.load(f)
    return (cfg.get("text_config") or cfg)["vocab_size"]


def run_engine(args) -> None:
    from tokenspeed.runtime.entrypoints.engine import Engine

    lens = tuple(int(n) for n in args.lens.split(",")) if args.lens else PROMPT_LENS
    prompts = make_prompts(args.vocab or text_vocab_size(args.model), args.seed, lens)
    kwargs = dict(
        model=args.model,
        world_size=args.tp,
        ep_size=args.ep,
        dtype="bfloat16",
        port=args.port,
        gpu_memory_utilization=args.mem,
        trust_remote_code=True,
        enable_output_logprobs=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_total_tokens=32768,
        max_num_seqs=8,
        disable_kvstore=True,
        enforce_eager=args.eager,
        max_cudagraph_capture_size=8,
    )
    if args.lm_only:
        kwargs["language_model_only"] = True
    if args.extra:
        kwargs.update(json.loads(args.extra))
    engine = Engine(**kwargs)
    try:
        sp = {
            "max_new_tokens": args.new,
            "temperature": 0,
            "logprobs": 0,
            "ignore_eos": True,
        }
        outs = []
        single_idx = 1 if len(prompts) > 1 else 0
        # One batched call (exercises batched prefill/decode), then one prompt
        # alone so batching-dependent bugs show up as a mismatch between them.
        if args.serial:
            # Each prompt alone: a reference free of batching effects.
            batch = [engine.generate(input_ids=q, sampling_params=sp) for q in prompts]
        else:
            batch = engine.generate(input_ids=prompts, sampling_params=sp)
        single = engine.generate(input_ids=prompts[single_idx], sampling_params=sp)
        for prompt, resp in zip(prompts, batch):
            lps = []
            for pos in resp["meta_info"].get("logprobs", []):
                tid, lp = next(iter(pos.items()))
                lps.append([int(tid), float(lp.logprob)])
            outs.append(
                {"prompt": prompt, "output_ids": resp["output_ids"], "logprobs": lps}
            )
        result = {
            "model": args.model,
            "tp": args.tp,
            "ep": args.ep,
            "outs": outs,
            "single_index": single_idx,
            "single_output_ids": single["output_ids"],
        }
    finally:
        engine.shutdown()
    with open(args.out, "w") as f:
        json.dump(result, f)
    same = result["single_output_ids"] == outs[single_idx]["output_ids"]
    print(f"ENGINE_DONE tp={args.tp} ep={args.ep} batch_vs_single_match={same}")
    for o in outs:
        print(len(o["prompt"]), o["output_ids"])


def run_ref(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    with open(args.engine_json) as f:
        eng = json.load(f)
    load = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if args.ref_kwargs:
        load.update(json.loads(args.ref_kwargs))
    model = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(args.model, **load)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"{cls.__name__}: {type(exc).__name__}: {str(exc)[:200]}")
    if model is None:
        raise SystemExit("no HF class could load the checkpoint")
    model = model.to(args.device).eval()
    report = []
    for o in eng["outs"]:
        ids = o["prompt"] + o["output_ids"]
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=args.device)).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        n = len(o["prompt"])
        match, maxdiff = 0, 0.0
        for i, tid in enumerate(o["output_ids"]):
            row = lp[n - 1 + i]
            if int(row.argmax()) != tid:
                # A near-tie flips under bf16 noise; report the gap.
                gap = float(row.max() - row[tid])
                report.append(
                    f"  len={n} first mismatch at {i}: ref_argmax={int(row.argmax())} eng={tid} gap={gap:.4f}"
                )
                break
            match += 1
            if i < len(o["logprobs"]):
                maxdiff = max(maxdiff, abs(float(row[tid]) - o["logprobs"][i][1]))
        report.append(
            f"len={n} match={match}/{len(o['output_ids'])} max|dlogprob|={maxdiff:.4f}"
        )
    print("\n".join(report))


def run_cmp(args) -> None:
    a = json.load(open(args.engine_json))
    b = json.load(open(args.ref_json))
    for x, y in zip(a["outs"], b["outs"]):
        n = 0
        for t, u in zip(x["output_ids"], y["output_ids"]):
            if t != u:
                break
            n += 1
        d = max(
            [abs(p[1] - q[1]) for p, q in zip(x["logprobs"][:n], y["logprobs"][:n])]
            or [0]
        )
        print(
            f"len={len(x['prompt'])} match={n}/{len(x['output_ids'])} max|dlogprob|={d:.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["engine", "ref", "cmp"])
    p.add_argument("--model")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--port", type=int, default=31000)
    p.add_argument("--mem", type=float, default=0.3)
    p.add_argument("--new", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lens", default="", help="comma-separated prompt lengths")
    p.add_argument("--vocab", type=int, default=0)
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--serial", action="store_true", help="generate prompts one at a time"
    )
    p.add_argument("--device", default="cpu", help="reference model device")
    p.add_argument("--lm-only", action="store_true", help="--language-model-only")
    p.add_argument("--extra", default="", help="JSON dict of extra Engine kwargs")
    p.add_argument(
        "--ref-kwargs", default="", help="JSON dict of extra from_pretrained kwargs"
    )
    p.add_argument("--out", default="engine.json")
    p.add_argument("--engine-json")
    p.add_argument("--ref-json")
    args = p.parse_args()
    {"engine": run_engine, "ref": run_ref, "cmp": run_cmp}[args.phase](args)


if __name__ == "__main__":
    main()
