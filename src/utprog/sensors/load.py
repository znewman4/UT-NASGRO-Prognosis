"""Load channel (strain gauge or LCR): time-resolved load, overloads. Needed to give
continuous monitoring what periodic inspection cannot see; the sensor model is a
question for the supervisors (not in M0-M2)."""
from .base import Sensor


class LoadSensor(Sensor):
    channel = "load"

    def observe(self, specimen, times, seed):
        raise NotImplementedError("Load channel: after M2, sensor model to agree with supervisors.")
