"""Random substreams: one per (specimen, purpose), independent of everything else.

SeedSequence(seed, spawn_key=(i, j)) equals the j-th child of the i-th child of
SeedSequence(seed), but is built without spawning state. A specimen's streams
therefore depend only on (seed, i): every arm sees the same truth and the same
noise (common random numbers), and results do not change with the number of
specimens or workers.
"""
import numpy as np

STREAMS = {"truth": 0, "sensor": 1, "scoring": 2, "estimator": 3, "forecast": 4}


def stream(seed, i, name):
    return np.random.SeedSequence(seed, spawn_key=(int(i), STREAMS[name]))


def child(seq, j):
    """The j-th child of `seq`, stateless (calling twice gives the same stream)."""
    return np.random.SeedSequence(seq.entropy, spawn_key=tuple(seq.spawn_key) + (int(j),))


def generator(seq):
    return np.random.default_rng(seq)
