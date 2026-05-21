from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    acc = np.load(run_dir / "accuracy_matrix.npy")
    print("最後一列 metrics：")
    print(metrics.tail(1).to_string(index=False))
    print("accuracy matrix：")
    print(np.array2string(acc, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
