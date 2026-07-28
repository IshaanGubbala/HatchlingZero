#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_gdn2_reference import TinyHZ0AModel
from reference.hz0a_inference import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HZ-0A reference prefill and recurrent decode.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()
    model = TinyHZ0AModel.init(args.seed, 32, 16, 3, 4, 4, 4, 32, attention_layer_indices=[])
    tokens = np.arange(args.batch_size * args.sequence_length, dtype=np.int64).reshape(args.batch_size, args.sequence_length) % 32
    print(json.dumps(benchmark(model, tokens), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
