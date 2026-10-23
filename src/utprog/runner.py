"""Runs truth x sensor x policy x estimator from one config.

    python -m utprog.runner configs/01_depth_only.yaml [--variant a1mm] [--n 200] [--workers 8]

For each specimen: draw the truth; draw forecast times (random times in
[0, frac_max x life], plus fixed life fractions for the prognostic metrics);
then for every arm, generate the policy's readings, feed them to a fresh
estimator in time order and score a forecast at each forecast time. Scoring at
random times is fair to periodic inspection: its last reading is anywhere from
0 to T old, as in service.

Each specimen has its own random substreams (utprog.rng), so every arm sees the
same truth, the same sensor error field and the same forecast draws (common
random numbers), and results do not depend on the number of workers.

Output per run, in results/<experiment>/<variant>/<UTC time>_<config hash>/:
    manifest.json   provenance (required before any result is reported)
    summary.csv     per arm: U90, bias, coverage with Wilson interval, CRPS
    alpha_lambda.csv, prognostic_horizon.csv
    records.csv.gz, specimens.csv, decisions.csv.gz   raw (not committed)
"""
import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfgmod
from . import manifest, policies
from ._external import ROOT, nasgro
from .estimators.base import Prior
from .estimators.grid_bayes import GridBayes
from .estimators.likelihood import NoiseModel
from .estimators.model_selection import ModelSelection
from .estimators.particle_filter import ParticleFilter
from .forward import nasgro_adapter as adapter
from .forward.table_cache import NasgroTable
from .forward.toy_paris import ToyParis
from .rng import stream
from .scoring import prognostic
from .scoring.metrics import forecast_record
from .scoring.sbc import rank
from .scoring.summary import summarise
from .sensors.depth import DepthSensor
from .truth import priors, scenarios
from .truth.priors import InitialDepth
from .truth.specimen import NasgroTruth, ToyTruth

RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
GRID_OPTS = ("n_lambda", "n_a0", "width", "min_cells", "edge_mass")
PF_OPTS = ("n_particles", "ess_frac", "n_moves")
BY = ("arm", "sensor", "policy", "estimator")


@dataclass(frozen=True)
class Arm:
    name: str
    sensor: str
    policy: str
    estimator: str


def arms(cfg):
    a = cfg["arms"]
    out = []
    prod = a.get("product")
    if prod:
        for s, p, e in itertools.product(prod["sensor"], prod["policy"], prod["estimator"]):
            out.append(Arm(f"{s}|{p}|{e}", s, p, e))
    for name, spec in (a.get("list") or {}).items():
        out.append(Arm(name, spec["sensor"], spec["policy"], spec["estimator"]))
    return out


def _opts(spec, keys):
    return {k: spec[k] for k in keys if k in spec}


class Components:
    """Truth, sensors, policies and estimator factories, built once per process from a config."""

    def __init__(self, cfg, cache_dir=CACHE):
        self.cfg, self.cache_dir = cfg, cache_dir
        t, p = cfg["truth"], cfg.get("prior", {})
        self.kind = t.get("kind", "nasgro")
        self.a_start = InitialDepth.from_config(t["a_start"])
        a0p = p.get("a0", "same_as_truth")
        self.a0_prior = self.a_start if a0p == "same_as_truth" else InitialDepth.from_config(a0p)
        self._tables = {}
        if self.kind == "toy":
            self.case, self.nominal, self.spread, self.interval = "toy", {}, None, None
            self.toy = ToyParis.from_config(t["toy"])
            self.truth = ToyTruth(self.toy, self.a_start, t["log_lambda"]["mu"], t["log_lambda"]["sd"])
            n_nom = float(self.toy.rul(np.array([self.a_start.median]), {"log_lambda": np.array([0.0])})[0])
        else:
            self.case = t["case"]
            self.nominal, self.spread = dict(t.get("nominal") or {}), t.get("spread") or None
            nom = priors.nominal_inputs(self.case, self.nominal)
            a_lim = float(nasgro().cases.a_limit(self.case, nom)[0])
            nom["a_i"] = np.array([self.a_start.median])
            n_nom = float(adapter.trajectory(self.case, nom).life[0])
            if not np.isfinite(n_nom):
                raise ValueError("the nominal crack arrests from the median start depth; raise the depth or the load")
            rel = (p.get("log_lambda") or {}).get("derive", {}).get("interval_rel", [1.0, 3.0])
            m = self.a_start.median
            self.interval = (rel[0] * m, min(rel[1] * m, 0.5 * a_lim))
            classes = {}
            for name, spec in (t.get("residual_stress") or {}).items():
                spec = self._rs_spec(spec)
                classes[name] = (float(spec.get("prob", 1.0)), scenarios.rs_scenario(spec, self.case))
            self.truth = NasgroTruth(self.case, self.a_start, self.nominal, self.spread, classes,
                                     t.get("variability"), t.get("n_grid", 1000), self.interval)
            f = cfg.get("forward", {})
            self.a_min = min(float(f.get("a_min_mm", 0.02)) * 1e-3, 0.5 * min(self.a_start.lo, self.a0_prior.lo))
            self.n_grid = int(f.get("n_grid", 600))
        grid = max(1.0, float(np.rint(float(cfg.get("time_grid_frac", 0.001)) * n_nom)))
        self.ctx = dict(N_nominal=n_nom, grid=grid, horizon=10.0 * n_nom)
        self.prior, self.prior_info = self._prior()
        self.sensor_cfg = {k: cfgmod.deep_merge(cfg.get("sensor_common") or {}, v) for k, v in cfg["sensors"].items()}
        self.sensors = {k: DepthSensor(NoiseModel.from_config(v, n_nom), grid, self.ctx["horizon"])
                        for k, v in self.sensor_cfg.items()}
        self.policies = {k: policies.from_config(v, self.ctx) for k, v in cfg["policies"].items()}
        self.arms = arms(cfg)
        for a in self.arms:
            for part, pool in (("sensor", self.sensors), ("policy", self.policies), ("estimator", cfg["estimators"])):
                if getattr(a, part) not in pool:
                    raise KeyError(f"arm {a.name}: no {part} named {getattr(a, part)}")
            self.make_estimator(a.estimator, a.sensor)      # builds (and caches) every table now

    def _rs_spec(self, spec):
        spec = dict(spec or {})
        if spec.get("form") == "redistribution":
            spec.setdefault("a_ref_mm", self.a_start.median * 1e3)
        return spec

    def _prior(self):
        ll = self.cfg["prior"]["log_lambda"]
        if "derive" in ll:
            if self.kind == "toy":
                raise ValueError("the toy prior is given, not derived")
            d = ll["derive"]
            info = priors.derive_lambda_prior(self.case, self.interval, n=int(d.get("n", 4000)),
                                              seed=int(d.get("seed", 1)), nominal=self.nominal, spread=self.spread,
                                              cache_dir=self.cache_dir)
            mu, sd = info["mu"], info["sd"] * float(d.get("sd_inflation", 1.0))
        else:
            mu, sd = float(ll["mu"]), float(ll["sd"])
            info = dict(mu=mu, sd=sd)
        return Prior(mu, sd, self.a0_prior), dict(info, mu_used=mu, sd_used=sd)

    def forward(self, rs=None, axes=None):
        if self.kind == "toy":
            if rs is not None or axes:
                raise ValueError("the toy has no residual stress or table axes")
            return self.toy
        key = json.dumps([rs, axes], sort_keys=True, default=str)
        if key not in self._tables:
            self._tables[key] = NasgroTable(self.case, self.a_min, self.n_grid, self.nominal, axes, rs,
                                            self.cache_dir)
        return self._tables[key]

    def make_estimator(self, name, sensor):
        """(estimator, prior) for an arm; the noise model is the sensor's unless overridden."""
        spec = self.cfg["estimators"][name]
        scfg = self.sensor_cfg[sensor]
        if spec.get("noise_override"):
            scfg = cfgmod.deep_merge(scfg, spec["noise_override"])
        noise = NoiseModel.from_config(scfg, self.ctx["N_nominal"])
        kind = spec["kind"]
        if kind == "grid":
            return GridBayes(self.forward(), noise, **_opts(spec, GRID_OPTS)), self.prior
        if kind == "particle_filter":
            return ParticleFilter(self.forward(), noise, **_opts(spec, PF_OPTS)), self.prior
        if kind == "model_selection":
            members, prs, probs = {}, {}, {}
            for h, hs in spec["hypotheses"].items():
                hs = self._rs_spec(hs)
                sc = scenarios.rs_scenario(hs, self.case)
                pr, axes = self.prior, None
                if sc is not None:
                    lo, hi = sc["A"]
                    k = int(hs.get("amp_nodes", 11))
                    if k >= 2 and hi > lo:
                        nodes = lo + (np.arange(k) + 0.5) * (hi - lo) / k     # midpoint rule for A ~ U(lo, hi)
                        axes = {"rs_amp": nodes.tolist()}
                        pr = replace(self.prior, axes={"rs_amp": (nodes, np.full(k, 1.0 / k))})
                members[h] = GridBayes(self.forward(sc, axes), noise, **_opts(spec, GRID_OPTS))
                prs[h], probs[h] = pr, float(hs.get("prob", 1.0))
            return ModelSelection(members, prs, probs), None
        raise ValueError(f"unknown estimator {kind}")

    def observe(self, spec, arm, i):
        t = self.policies[arm.policy].times(spec, self.ctx)
        return self.sensors[arm.sensor].observe(spec, t, stream(self.cfg["seed"], i, "sensor"))


_C = None


def _init(cfg):
    global _C
    _C = Components(cfg)


def _new_estimator(arm, i):
    est, prior = _C.make_estimator(arm.estimator, arm.sensor)
    est.reset(prior, rng=np.random.default_rng(stream(_C.cfg["seed"], i, "estimator")))
    return est, np.random.default_rng(stream(_C.cfg["seed"], i, "forecast"))


def forecast_arm(spec, arm, times, kinds, i):
    sc = _C.cfg["scoring"]
    obs = _C.observe(spec, arm, i)
    est, rng_f = _new_estimator(arm, i)
    n_s, n_sbc = int(sc.get("n_samples", 2000)), int(sc.get("sbc_draws", 0))
    k = n_sized = 0
    out = []
    for N, kind in zip(times, kinds):
        while k < len(obs) and obs[k].N <= N:
            est.update(obs[k])
            n_sized += obs[k].kind == "sized"
            k += 1
        x = est.forecast(N, n_s, rng_f)
        rec = dict(specimen=i, arm=arm.name, sensor=arm.sensor, policy=arm.policy, estimator=arm.estimator,
                   kind=str(kind), N=float(N), life=spec.life, frac=float(N / spec.life),
                   true_rul=float(spec.life - N), n_obs=k, n_sized=n_sized,
                   last_obs_age=float(N - obs[k - 1].N) if k else np.nan,
                   true_log_lambda=spec.truth_theta["log_lambda"], true_a0_mm=spec.a0 * 1e3, true_class=spec.label)
        rec.update(forecast_record(x, spec.life - N, spec.life, sc.get("alpha_rel", 0.2), sc.get("alpha_abs", 0.05)))
        rec.update(est.summary())
        if n_sbc and not isinstance(est, ModelSelection):
            rec["sbc_rank"] = rank(est.posterior_draws(n_sbc, rng_f)["log_lambda"], spec.truth_theta["log_lambda"])
        out.append(rec)
    return out


def decide_arm(spec, arm, i):
    """Approach 4: act once P(failure before the next planned reading + lead) >= p_act."""
    dc = _C.cfg["decision"]
    n_nom = _C.ctx["N_nominal"]
    p_acts = [float(p) for p in dc["p_act"]]
    lead = float(dc.get("lead_frac", 0.0)) * n_nom
    pol = _C.policies[arm.policy]
    obs = _C.observe(spec, arm, i)
    est, rng_f = _new_estimator(arm, i)
    n_s = int(dc.get("n_samples", 1000))
    acted = {}
    for k, o in enumerate(obs):
        est.update(o)
        pending = [p for p in p_acts if p not in acted]
        if not pending:
            break
        window = pol.next_after(o.N, spec, _C.ctx) - o.N + lead
        p_fail = float(np.mean(est.forecast(o.N, n_s, rng_f) <= window))
        for p in pending:
            if p_fail >= p:
                acted[p] = (k + 1, o.N)
    rows = []
    for p in p_acts:
        if p in acted:
            n_r, Na = acted[p]
            r = dict(acted=1, failed=0, n_readings=n_r, service=Na, unused=spec.life - Na)
        else:
            r = dict(acted=0, failed=1, n_readings=len(obs), service=spec.life, unused=0.0)
        rows.append(dict(specimen=i, arm=arm.name, sensor=arm.sensor, policy=arm.policy, estimator=arm.estimator,
                         p_act=p, life=spec.life, service_frac=r["service"] / n_nom,
                         unused_frac=r["unused"] / n_nom, **r))
    return rows


def run_specimen(i):
    cfg = _C.cfg
    spec = _C.truth.draw(np.random.default_rng(stream(cfg["seed"], i, "truth")), index=i)
    srow = dict(specimen=i, a0_mm=spec.a0 * 1e3, life=spec.life, a_fail_mm=spec.a_fail * 1e3,
                true_log_lambda=spec.truth_theta["log_lambda"], label=spec.label, arrested=spec.arrested)
    if spec.arrested:
        return srow, [], []
    sc = cfg["scoring"]
    rng_s = np.random.default_rng(stream(cfg["seed"], i, "scoring"))
    n_rand = int(sc.get("n_random", 5))
    t_rand = rng_s.random(n_rand) * float(sc.get("frac_max", 0.95)) * spec.life
    t_fix = np.asarray(sc.get("fixed_fractions", []), float) * spec.life
    times = np.concatenate([t_rand, t_fix])
    kinds = np.array(["random"] * n_rand + ["fixed"] * t_fix.size)
    order = np.argsort(times, kind="stable")
    records, decisions = [], []
    for arm in _C.arms:
        records += forecast_arm(spec, arm, times[order], kinds[order], i)
        if (cfg.get("decision") or {}).get("enabled"):
            decisions += decide_arm(spec, arm, i)
    return srow, records, decisions


def run(cfg, out_dir, workers=None):
    started = time.time()
    _init(cfg)                               # builds priors and tables (disk-cached) before the workers start
    n = int(cfg["n_specimens"])
    workers = int(workers or cfg.get("workers") or 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(cfg,)) as ex:
            res = list(ex.map(run_specimen, range(n), chunksize=max(1, n // (8 * workers))))
    else:
        res = [run_specimen(i) for i in range(n)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specimens = pd.DataFrame([r[0] for r in res])
    records = pd.DataFrame([x for r in res for x in r[1]])
    decisions = pd.DataFrame([x for r in res for x in r[2]])
    specimens.to_csv(out / "specimens.csv", index=False)
    if not records.empty:
        records.to_csv(out / "records.csv.gz", index=False)
        summarise(records[records["kind"] == "random"], BY).to_csv(out / "summary.csv", index=False)
        if (records["kind"] == "fixed").any():
            beta = float(cfg["scoring"].get("beta", 0.5))
            prognostic.alpha_lambda(records, beta, BY).to_csv(out / "alpha_lambda.csv", index=False)
            prognostic.prognostic_horizon(records, beta, BY).to_csv(out / "prognostic_horizon.csv", index=False)
    if not decisions.empty:
        decisions.to_csv(out / "decisions.csv.gz", index=False)
    manifest.write(out, cfg, started, extra=dict(n_specimens=n, n_arrested=int(specimens["arrested"].sum()),
                                                  workers=workers, prior=_C.prior_info, ctx=_C.ctx))
    return out


def run_dir(cfg):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return RESULTS / cfg["experiment"] / cfg.get("variant", "base") / f"{stamp}_{cfgmod.config_hash(cfg)[:8]}"


def main(argv=None, default_config=None):
    ap = argparse.ArgumentParser(description="Run an experiment config (every variant unless --variant).")
    ap.add_argument("config", nargs="?", default=default_config)
    ap.add_argument("--variant", action="append", help="run only this variant (repeatable)")
    ap.add_argument("--n", type=int, help="override n_specimens (recorded in the manifest)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)
    if args.config is None:
        ap.error("a config file is required")
    done = []
    for name, cfg in cfgmod.variants(cfgmod.load(args.config)):
        if args.variant and name not in args.variant:
            continue
        if args.n:
            cfg["n_specimens"] = args.n
        out = run(cfg, run_dir(cfg), args.workers)
        print(f"{name}: {out}")
        done.append((name, cfg, out))
    return done


if __name__ == "__main__":
    main()
