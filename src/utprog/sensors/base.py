"""Observations and the sensor interface: observe(specimen, times, seed) -> [Observation]."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    N: float                    # cycle count of the reading
    kind: str                   # "sized", "censored" (detected, below the sizing limit) or "miss"
    value: float = np.nan       # sized depth (m)
    channel: str = "depth"


@dataclass(frozen=True)
class StressObservation:
    N: float
    values: tuple               # layer-average residual stress (MPa), one per layer
    depths: tuple               # layer depth sampled by each frequency (m)
    kind: str = "stress"
    channel: str = "stress"


class Sensor:
    channel = ""

    def observe(self, specimen, times, seed):
        """Readings at `times` (sorted cycles). `seed` is the specimen's sensor SeedSequence;
        the same seed gives the same error field to every arm (common random numbers)."""
        raise NotImplementedError
