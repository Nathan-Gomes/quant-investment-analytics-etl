"""A 32-bit PRNG that produces identical draws in Python and in JavaScript.

The research pipeline uses ``numpy.random.default_rng``. The web app runs the
same scenario engine in two places: on the FastAPI server, and inside the
browser for the offline demo build. NumPy's generator has no JavaScript twin,
so both engines share this mulberry32 implementation instead. The same seed and
the same inputs therefore produce the same scenario numbers on either side,
which keeps a result someone shares reproducible by whoever reviews it.

Bit-level equivalence with ``static/rng.js`` is covered by
``tests/application/test_app_engine.py::test_prng_matches_javascript``.
"""

from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """The low 32 bits of a product, matching JavaScript's Math.imul."""
    return (a * b) & MASK


def _imul_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise Math.imul. uint64 holds the full product of two 32-bit values,
    so the low 32 bits are exact with no overflow to guard against."""
    return (a * b) & np.uint64(MASK)


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

    def floats(self, count: int) -> np.ndarray:
        """``count`` draws, computed without stepping through them one at a time.

        The state update is a single addition of a constant, so the whole state
        sequence is ``seed + n * 0x6D2B79F5`` modulo 2^32 and can be produced in
        one vectorized step. Only the mixing function then has to be applied, and
        it is elementwise. The output is bit-identical to calling ``next_float``
        that many times, which the tests check; this is the same generator, not
        an approximation of it.
        """
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        steps = np.arange(1, count + 1, dtype=np.uint64)
        state = (np.uint64(self.state) + steps * np.uint64(0x6D2B79F5)) & np.uint64(MASK)
        self.state = int(state[-1])

        t = _imul_vector(state ^ (state >> np.uint64(15)), np.uint64(1) | state)
        t = (t + _imul_vector(t ^ (t >> np.uint64(7)), np.uint64(61) | t)) & np.uint64(MASK) ^ t
        return ((t ^ (t >> np.uint64(14))) & np.uint64(MASK)).astype(np.float64) / 4294967296.0

    def integers(self, high: int, count: int) -> np.ndarray:
        """``count`` draws from 0 (inclusive) to ``high`` (exclusive)."""
        if high < 1:
            raise ValueError("Upper bound must be at least 1")
        return np.minimum((self.floats(count) * high).astype(np.int64), high - 1)
