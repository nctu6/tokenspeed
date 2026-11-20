"""Long-context correctness check: hide facts in ~30K-token filler, ask for them.

Each case puts a unique code at a different depth of a long unique filler and
asks for it back. Prints how many answers contain the code.
Usage: python3 needle_test.py [--cases 6] [--tokens 30000]
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import urllib.request

URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash-1017")
WORDS = "river stone cloud lantern harbor meadow copper violet signal orchard canyon ember".split()


def case(seed: int, tokens: int) -> tuple[str, str]:
    rng = random.Random(seed)
    code = f"{rng.randint(100000, 999999)}-{rng.choice(WORDS).upper()}"
    lines = [
        " ".join(rng.choice(WORDS) for _ in range(16)) + "."
        for _ in range(int(tokens / 18))
    ]
    depth = int(len(lines) * (0.1 + 0.8 * (seed % 5) / 4))
    lines.insert(depth, f"IMPORTANT: the vault access code is {code}. Remember it.")
    prompt = (
        "\n".join(lines)
        + "\n\nWhat is the vault access code mentioned in the text above? Reply with the code only."
    )
    return prompt, code


def ask(seed: int, tokens: int) -> bool:
    prompt, code = case(seed, tokens)
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    ok = code in (msg.get("content") or "")
    print(
        f"seed={seed} depth_case={seed % 5} code={code} ok={ok} answer={(msg.get('content') or '')[:60]!r}",
        flush=True,
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=30000)
    args = ap.parse_args()
    with cf.ThreadPoolExecutor(2) as ex:
        results = list(ex.map(lambda s: ask(s, args.tokens), range(args.cases)))
    print(f"NEEDLE {sum(results)}/{len(results)} correct")


if __name__ == "__main__":
    main()
