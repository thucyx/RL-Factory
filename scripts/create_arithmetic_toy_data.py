import argparse
import os
import random

import pandas as pd


def make_example(index, rng):
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a = rng.randint(0, 20)
        b = rng.randint(0, 20)
        answer = a * b
    else:
        a = rng.randint(-100, 100)
        b = rng.randint(-100, 100)
        answer = a + b if op == "+" else a - b

    prompt = (
        "Solve the arithmetic problem. "
        "Put only the final integer inside <answer> and </answer>. "
        f"Problem: {a} {op} {b} = ?"
    )

    return {
        "data_source": "arithmetic_toy",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "arithmetic",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": str(answer)},
        },
        "extra_info": {
            "index": index,
            "expression": f"{a} {op} {b}",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/arithmetic_toy")
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    rng = random.Random(args.seed)

    train = [make_example(i, rng) for i in range(args.train_size)]
    test = [make_example(i, rng) for i in range(args.test_size)]

    pd.DataFrame(train).to_parquet(os.path.join(args.local_dir, "train.parquet"))
    pd.DataFrame(test).to_parquet(os.path.join(args.local_dir, "test.parquet"))

    print(f"Wrote {len(train)} train examples and {len(test)} test examples to {args.local_dir}")


if __name__ == "__main__":
    main()
