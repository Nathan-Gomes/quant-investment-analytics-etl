"""A 32-bit PRNG that produces identical draws in Python and in JavaScript.

The research pipeline uses ``numpy.random.default_rng``. The web app runs the
same scenario engine in two places: on the FastAPI server, and inside the
browser for the offline demo build. NumPy's generator has no JavaScript twin,
so both engines share this mulberry32 implementation instead. The same seed and
the same inputs therefore produce the same scenario numbers on either side,
which keeps a result someone shares reproducible by whoever reviews it.

Bit-level equivalence with ``static/rng.js`` is covered by
``tests/test_app_engine.py::test_prng_matches_javascript``.
"""

from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """The low 32 bits of a product, matching JavaScript's Math.imul."""
    return (a * b) & MASK


class Mulberry32:
    """Deterministic uniform draws in [0, 1)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK

    def next_float(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & MASK
        a = self.state
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0

    def integers(self, high: int, count: int) -> np.ndarray:
        """``count`` draws from 0 (inclusive) to ``high`` (exclusive)."""
        if high < 1:
            raise ValueError("Upper bound must be at least 1")
        out = np.empty(count, dtype=np.int64)
        for i in range(count):
            out[i] = int(self.next_float() * high)
        return np.minimum(out, high - 1)
