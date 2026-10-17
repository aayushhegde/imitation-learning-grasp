#!/usr/bin/env python3
"""Record the polished, strictly completed obstacle pick-and-place animation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ilgrasp.visualization import record_verified_expert_gif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/assets/verified_pick_cross_place.gif"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(record_verified_expert_gif(args.output, seed=args.seed).as_dict())


if __name__ == "__main__":
    main()
