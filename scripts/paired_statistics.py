"""Interval estimates for the held-out comparisons reported in the manuscript.

Inputs are the per-model success counts and the discordant counts of each paired comparison
on the shared 100-scene suite (40000-40099). For a comparison A -> B, b is the number of scenes
that only A completes and c the number that only B completes; the paired difference is
(c - b) / n, i.e. B minus A.

* Wilson score interval for a single completion rate (Wilson, 1927).
* Exact two-sided McNemar test on the discordant pairs (McNemar, 1947).
* Newcombe (1998) method 10 interval for the paired difference: Wilson limits for both
  marginal rates combined through the continuity-corrected correlation term.

Run:  python scripts/paired_statistics.py
"""

from __future__ import annotations

from math import comb, sqrt

Z = 1.959963984540054


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def newcombe10(a: int, b: int, c: int, d: int, z: float = Z) -> tuple[float, float, float]:
    """Difference (a + c)/n - (a + b)/n with Newcombe's method 10 limits.

    a: both succeed, b: A only, c: B only, d: neither.
    """
    n = a + b + c + d
    p_a, p_b = (a + b) / n, (a + c) / n
    l_a, u_a = wilson(a + b, n, z)
    l_b, u_b = wilson(a + c, n, z)
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    numerator = a * d - b * c
    if numerator > 0:
        numerator = max(numerator - n / 2, 0.0)
    phi = 0.0 if denominator == 0 else numerator / sqrt(denominator)
    diff = p_b - p_a
    lower = sqrt(max(0.0, (p_b - l_b) ** 2 - 2 * phi * (p_b - l_b) * (u_a - p_a) + (u_a - p_a) ** 2))
    upper = sqrt(max(0.0, (u_b - p_b) ** 2 - 2 * phi * (u_b - p_b) * (p_a - l_a) + (p_a - l_a) ** 2))
    return diff, diff - lower, diff + upper


# (A, B, training seed, successes of A, successes of B, b = A-only, c = B-only), n = 100 scenes.
COMPARISONS = [
    ("MLP22", "GRU33", 0, 18, 93, 2, 77), ("MLP22", "GRU33", 1, 24, 89, 1, 66),
    ("MLP22", "GRU33", 2, 23, 84, 7, 68),
    ("GRU33", "GRU33-F", 0, 93, 97, 3, 7), ("GRU33", "GRU33-F", 1, 89, 96, 3, 10),
    ("GRU33", "GRU33-F", 2, 84, 97, 2, 15),
    ("MLP33-F", "GRU33-F", 0, 2, 97, 0, 95), ("MLP33-F", "GRU33-F", 1, 45, 96, 2, 53),
    ("MLP33-F", "GRU33-F", 2, 17, 97, 1, 81),
    ("GRU33-F", "GRU33-F after aggregation", 0, 97, 85, 15, 3),
]


def main() -> None:
    # Worked examples from Newcombe (1998), Table III, method 10 (theta = (f - g)/n).
    for (e, f, g, h), expected in [((36, 12, 2, 0), (0.0569, 0.3404)),
                                   ((20, 12, 2, 16), (0.0562, 0.3292)),
                                   ((35, 14, 0, 1), (0.1461, 0.4175))]:
        _, lo, hi = newcombe10(e, g, f, h)
        assert abs(lo - expected[0]) < 5e-5 and abs(hi - expected[1]) < 5e-5, (lo, hi, expected)
    print("Newcombe (1998) Table III method-10 examples reproduced.\n")
    print(f"{'A -> B':34s} seed   a   b   c   d   diff (pp)  95% CI (pp)        exact McNemar p")
    for name_a, name_b, seed, k_a, k_b, b, c in COMPARISONS:
        a = k_a - b
        assert a + c == k_b
        d = 100 - a - b - c
        diff, lo, hi = newcombe10(a, b, c, d)
        print(f"{name_a + ' -> ' + name_b:34s} {seed:>4d} {a:3d} {b:3d} {c:3d} {d:3d}   {100 * diff:+5.0f}"
              f"      [{100 * lo:+6.1f}, {100 * hi:+6.1f}]    {mcnemar_exact(b, c):.3g}")


if __name__ == "__main__":
    main()
