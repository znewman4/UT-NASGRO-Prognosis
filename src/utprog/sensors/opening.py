"""Crack-opening channel (Approach 4, M4). On hold until the standard-variables rerun.

If kept: a feature-to-opening calibration against compliance and DIC
(Rokhlin & Kim 2003; Cheng et al. 2017), observing the opening level Kop/Kmax
that NASGRO's Newman closure function predicts from alpha and Smax/sigma0.
"""
from .base import Sensor


class OpeningSensor(Sensor):
    channel = "opening"

    def observe(self, specimen, times, seed):
        raise NotImplementedError("Opening channel: M4, only if the rerun supports it.")
