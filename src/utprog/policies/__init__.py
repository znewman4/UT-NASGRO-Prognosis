"""Measurement policies: times(specimen, ctx) -> measurement schedule (cycles).

    none               no readings (the prior forecast)
    one_off(at)        a single reading
    periodic(T)        every T cycles from `offset`
    continuous(dt)     every dt cycles (a permanently installed sensor)
    event_triggered    a base policy plus a burst of readings after each load event

Spacings are given in cycles or as a fraction of the nominal life (`*_frac`)
and snapped to the sensor time grid, so readings of different arms at the same
cycle share their measurement error. A schedule runs to the specimen's failure
(no reading can follow it); `next_after` gives the next planned reading from
the policy alone, without looking at the specimen's life.
"""
import numpy as np


def snap(t, grid):
    t = np.asarray(t, float)
    return np.unique(np.rint(t / grid) * grid)


class Policy:
    kind = ""

    def schedule(self, specimen, ctx, horizon):
        raise NotImplementedError

    def times(self, specimen, ctx):
        h = specimen.life if np.isfinite(specimen.life) else ctx["horizon"]
        t = snap(self.schedule(specimen, ctx, h), ctx["grid"])
        return t[(t >= 0.0) & (t < h)]

    def next_after(self, N, specimen, ctx):
        t = snap(self.schedule(specimen, ctx, N + ctx["horizon"]), ctx["grid"])
        t = t[t > N]
        return float(t[0]) if t.size else np.inf


class NoMeasurement(Policy):
    kind = "none"

    def schedule(self, specimen, ctx, horizon):
        return np.empty(0)


class OneOff(Policy):
    kind = "one_off"

    def __init__(self, at=0.0):
        self.at = float(at)

    def schedule(self, specimen, ctx, horizon):
        return np.array([self.at])


class Periodic(Policy):
    kind = "periodic"

    def __init__(self, T, offset=0.0):
        self.T, self.offset = float(T), float(offset)

    def schedule(self, specimen, ctx, horizon):
        return np.arange(self.offset, horizon, self.T)


class Continuous(Policy):
    kind = "continuous"

    def __init__(self, dt):
        self.dt = float(dt)

    def schedule(self, specimen, ctx, horizon):
        return np.arange(0.0, horizon, self.dt)


class EventTriggered(Policy):
    kind = "event_triggered"

    def __init__(self, base, burst, dt):
        self.base, self.burst, self.dt = base, int(burst), float(dt)

    def schedule(self, specimen, ctx, horizon):
        t = self.base.schedule(specimen, ctx, horizon)
        ev = np.asarray(specimen.events, float)
        ev = ev[ev < horizon]
        extra = (ev[:, None] + self.dt * np.arange(1, self.burst + 1)[None, :]).ravel()
        return np.concatenate([t, ev, extra])


def _cycles(spec, key, ctx, spacing=True, default=None):
    """Cycles from `key` or `key_frac` (of the nominal life); spacings are whole grid steps, >= 1."""
    if key in spec:
        x = float(spec[key])
    elif f"{key}_frac" in spec:
        x = float(spec[f"{key}_frac"]) * ctx["N_nominal"]
    elif default is not None:
        return default
    else:
        raise KeyError(f"policy needs {key} or {key}_frac")
    return max(ctx["grid"], float(np.rint(x / ctx["grid"])) * ctx["grid"]) if spacing else x


def from_config(spec, ctx):
    kind = spec["kind"]
    if kind == "none":
        return NoMeasurement()
    if kind == "one_off":
        return OneOff(_cycles(spec, "at", ctx, spacing=False, default=0.0))
    if kind == "periodic":
        return Periodic(_cycles(spec, "T", ctx), _cycles(spec, "offset", ctx, spacing=False, default=0.0))
    if kind == "continuous":
        return Continuous(_cycles(spec, "dt", ctx))
    if kind == "event_triggered":
        return EventTriggered(from_config(spec["base"], ctx), spec.get("burst", 5), _cycles(spec, "dt", ctx))
    raise ValueError(f"unknown policy {kind}")
