import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utprog._external import nasgro  # noqa: E402
from utprog.estimators.base import Prior  # noqa: E402
from utprog.forward.toy_paris import ToyParis  # noqa: E402
from utprog.sensors.base import Observation  # noqa: E402
from utprog.truth.priors import InitialDepth  # noqa: E402


@pytest.fixture(scope="session")
def ns():
    return nasgro()


@pytest.fixture
def toy():
    return ToyParis(C=1e-11, m=3.0, dS=100.0, Y=1.12, a_c=10e-3)


def toy_prior(a0_cov=0.0, sd=0.4):
    return Prior(0.0, sd, InitialDepth(1e-3, a0_cov, 4.0))


def toy_record(model, log_lam, a0, times, sd, rng):
    """Sized readings of the toy with independent Gaussian error."""
    a = model.advance(a0, np.asarray(times, float), {"log_lambda": log_lam})
    y = a + sd * rng.standard_normal(a.shape)
    return [Observation(float(N), "sized", float(v)) for N, v in zip(times, y)]
