"""Crack-angle channel (Approach 5, M3): measured tilt for the mixed-mode switch rule."""
from .base import Sensor


class AngleSensor(Sensor):
    channel = "angle"

    def observe(self, specimen, times, seed):
        raise NotImplementedError("Angle channel: M3, with a realistic angle prior.")
