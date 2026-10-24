"""Runs truth x sensor x policy x estimator from one config.

    python -m utprog.runner configs/01_depth_only.yaml [--variant a1mm] [--n 200] [--workers 8]

For each specimen: draw the truth; choose forecast times; then for every arm,
generate the policy's readings, feed them to a fresh estimator in time order and
score a forecast at each forecast time. Forecast times, all shared by every arm:
    random  uniform in [0, frac_max x life]: fair to periodic inspection, whose last
            reading is then anywhere from 0 to T old, as in service
    fixed   fixed fractions of life: the cutoffs for alpha-lambda and prognostic horizon
    depth   the moment the true crack reaches each of `at_depths_mm`: the sensitivity
            study's scoring point, so oracle arms can be held to its numbers
    time    fixed fractions of the nominal life (e.g. the end of a sizing window)
Specimens whose crack never grows are kept: their remaining life is infinite, only
the arrest forecast (p_arrest) and classifications are scored for them, and they
are scored over [0, horizon_frac x nominal life].

An arm's sensor may combine several sensors with "+", e.g. "sd0.2+lcr40". Oracle
arms (estimator kind study_mc) see the truth by design and use no readings.

Each specimen has its own random substreams (utprog.rng), so every arm sees the
same truth, the same sensor error fields and the same forecast draws (common
random numbers), and results do not depend on the number of workers.

Output per run, in results/<experiment>/<variant>/<UTC time>_<config hash>/:
    manifest.json   provenance (required before any result is reported)
    summary.csv     per arm and forecast kind: U90, bias, coverage with Wilson interval, CRPS
    summary_depth.csv  per arm at each scoring depth (the comparison with the study)
    alpha_lambda.csv, prognostic_horizon.csv
    records.csv.gz, specimens.csv, decisions.csv.gz   raw (not committed)
"""
import argparse
import itertools
import re
import json
import sys
import time
import warnings
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
from .estimators.likelihood import NoiseModel, StressNoise
from .estimators.model_selection import ModelSelection
from .estimators.nasgro_is import NasgroIS, Population
from .estimators.particle_filter import ParticleFilter
from .estimators.study_mc import StaticMC, StudyMC
from .forward import nasgro_adapter as adapter
from .forward.table_cache import NasgroTable
from .forward.toy_paris import ToyParis
from .rng import stream
from .scoring import prognostic
from .scoring.metrics import forecast_record
from .scoring.sbc import rank
from .scoring.summary import summarise, summarise_depth
from .sensors.depth import DepthSensor
from .sensors.stress import LCRSensor
from .truth import priors, scenarios
from .truth.priors import depth_prior
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
    prods = list((a.get("products") or {}).values()) + ([a["product"]] if a.get("product") else [])
    for prod in prods:
        for s, p, e in itertools.product(prod["sensor"], prod["policy"], prod["estimator"]):
            out.append(Arm(f"{s}|{p}|{e}", s, p, e))
    for name, spec in (a.get("list") or {}).items():
        out.append(Arm(name, spec["sensor"], spec["policy"], spec["estimator"]))
    return out


def _opts(spec, keys):
    return {k: spec[k] for k in keys if k in spec}


def _is_stress(spec):
    return spec.get("type") == "lcr"


class Components:
    """Truth, sensors, policies and estimator factories, built once per process from a config."""

    def __init__(self, cfg, cache_dir=CACHE):
        self.cfg, self.cache_dir = cfg, cache_dir
        t, p = cfg["truth"], cfg.get("prior", {})
        self.kind = t.get("kind", "nasgro")
        self.a_start = depth_prior(t["a_start"])
        a0p = p.get("a0", "same_as_truth")
        self.a0_prior = self.a_start if a0p == "same_as_truth" else depth_prior(a0p)
        self._tables, self._populations, self._oracles = {}, {}, {}
        self.rs_classes = {}
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
            for name, spec in (t.get("residual_stress") or {}).items():
                spec = self._rs_spec(spec)
                self.rs_classes[name] = (float(spec.get("prob", 1.0)), scenarios.rs_scenario(spec, self.case))
            self.truth = NasgroTruth(self.case, self.a_start, self.nominal, self.spread, self.rs_classes,
                                     t.get("variability"), t.get("n_grid", 1000), self.interval,
                                     extended=t.get("extended", False))
            f = cfg.get("forward", {})
            self.a_min = min(float(f.get("a_min_mm", 0.02)) * 1e-3, 0.5 * min(self.a_start.lo, self.a0_prior.lo))
            self.n_grid = int(f.get("n_grid", 600))
        grid = max(1.0, float(np.rint(float(cfg.get("time_grid_frac", 0.001)) * n_nom)))
        horizon = float((cfg.get("scoring") or {}).get("horizon_frac", 1.0)) * n_nom
        self.ctx = dict(N_nominal=n_nom, grid=grid, horizon=horizon, lookahead=10.0 * n_nom)
        self.prior, self.prior_info = self._prior() if "prior" in cfg else (None, {})
        self.sensor_cfg = {}
        self.sensors = {}
        for k, v in cfg["sensors"].items():
            if _is_stress(v):
                self.sensor_cfg[k] = dict(v)
                self.sensors[k] = LCRSensor.from_config(v, StressNoise.from_config(v), n_nom)
            else:
                v = cfgmod.deep_merge(cfg.get("sensor_common") or {}, v)
                self.sensor_cfg[k] = v
                self.sensors[k] = DepthSensor(NoiseModel.from_config(v, n_nom), grid, horizon,
                                              v.get("dropout"), n_nom)
        self.policies = {k: policies.from_config(v, self.ctx) for k, v in cfg["policies"].items()}
        self.arms = arms(cfg)
        for a in self.arms:
            for s in a.sensor.split("+"):
                if s not in self.sensors:
                    raise KeyError(f"arm {a.name}: no sensor named {s}")
            if a.policy not in self.policies:
                raise KeyError(f"arm {a.name}: no policy named {a.policy}")
            if a.estimator not in cfg["estimators"]:
                raise KeyError(f"arm {a.name}: no estimator named {a.estimator}")
            self.make_estimator(a.estimator, a.sensor)      # builds (and caches) tables and populations now

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

    def population(self, n, seed):
        key = (int(n), int(seed))
        if key not in self._populations:
            self._populations[key] = Population(self.case, n, seed, self.a0_prior, self.nominal, self.spread,
                                                self.rs_classes)
        return self._populations[key]

    def noise_models(self, spec, combo):
        """Depth NoiseModel and StressNoise for an arm's sensors, with the estimator's overrides."""
        noise = stress = None
        for s in combo.split("+"):
            scfg = self.sensor_cfg[s]
            if _is_stress(scfg):
                stress = StressNoise.from_config(cfgmod.deep_merge(scfg, spec.get("stress_noise_override") or {}))
            else:
                noise = NoiseModel.from_config(cfgmod.deep_merge(scfg, spec.get("noise_override") or {}),
                                               self.ctx["N_nominal"])
        return noise, stress

    def make_estimator(self, name, combo):
        """(estimator, prior) for an arm; noise models are the sensors' unless overridden."""
        spec = self.cfg["estimators"][name]
        noise, stress = self.noise_models(spec, combo)
        kind = spec["kind"]
        if kind == "study_mc":                 # stateless, so one instance per process keeps its cache
            if name not in self._oracles:
                self._oracles[name] = StudyMC(self.case, spec.get("known", ()), spec.get("n", 2000),
                                              spec.get("seed", 1), self.nominal, self.spread, self.rs_classes,
                                              spec.get("fix_tilt", False))
            return self._oracles[name], None
        if noise is None:
            raise ValueError(f"estimator {name} needs a depth sensor in its arm")
        if kind == "static_mc":
            return StaticMC(self.case, noise, spec.get("n", 2000), spec.get("seed", 1), self.nominal,
                            self.spread, self.rs_classes, spec.get("fix_tilt", False)), None
        if kind == "grid":
            return GridBayes(self.forward(), noise, stress_noise=stress, **_opts(spec, GRID_OPTS)), self.prior
        if kind == "particle_filter":
            return ParticleFilter(self.forward(), noise, **_opts(spec, PF_OPTS)), self.prior
        if kind == "nasgro_is":
            return NasgroIS(self.population(spec.get("n", 4000), spec.get("seed", 2)), noise, stress), None
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
                members[h] = GridBayes(self.forward(sc, axes), noise, stress_noise=stress, **_opts(spec, GRID_OPTS))
                prs[h], probs[h] = pr, float(hs.get("prob", 1.0))
            return ModelSelection(members, prs, probs), None
        raise ValueError(f"unknown estimator {kind}")

    def observe(self, spec, arm, i):
        t = self.policies[arm.policy].times(spec, self.ctx)
        obs = []
        for s in arm.sensor.split("+"):
            obs += self.sensors[s].observe(spec, t, stream(self.cfg["seed"], i, "sensor"))
        return sorted(obs, key=lambda o: o.N)


_C = None


def _init(cfg):
    global _C
    _C = Components(cfg)


def _new_estimator(arm, i):
    est, prior = _C.make_estimator(arm.estimator, arm.sensor)
    est.reset(prior, rng=np.random.default_rng(stream(_C.cfg["seed"], i, "estimator")))
    return est, np.random.default_rng(stream(_C.cfg["seed"], i, "forecast"))


def _forecast(est, spec, N, n, rng):
    if getattr(est, "sees_truth", False):
        return est.forecast_truth(spec, N)
    return est.forecast(N, n, rng)


def forecast_arm(spec, arm, points, i):
    """points: [(N, kind, depth_mm)] sorted by N."""
    sc = _C.cfg["scoring"]
    espec = _C.cfg["estimators"][arm.estimator]
    kinds = espec.get("score_kinds")
    est, rng_f = _new_estimator(arm, i)
    obs = [] if getattr(est, "sees_truth", False) else _C.observe(spec, arm, i)
    n_s, n_sbc = int(sc.get("n_samples", 2000)), int(sc.get("sbc_draws", 0))
    fresh = _C.policies[arm.policy].kind == "at_forecast"     # each forecast sees only its own fresh reading
    k = n_sized = 0
    out = []
    for N, kind, depth_mm in points:
        if kinds and kind not in kinds:
            continue
        if fresh:
            est, rng_f = _new_estimator(arm, i)
            while k < len(obs) and obs[k].N < N:
                k += 1
            n_sized = 0
        while k < len(obs) and obs[k].N <= N:
            est.update(obs[k])
            n_sized += obs[k].kind == "sized"
            k += 1
        x = _forecast(est, spec, N, n_s, rng_f)
        true_rul = spec.life - N
        rec = dict(specimen=i, arm=arm.name, sensor=arm.sensor, policy=arm.policy, estimator=arm.estimator,
                   oracle=bool(getattr(est, "sees_truth", False)), kind=kind, depth_mm=depth_mm,
                   N=float(N), life=spec.life, arrested=spec.arrested,
                   frac=float(N / spec.life) if not spec.arrested else np.nan,
                   true_rul=float(true_rul), n_obs=k, n_sized=n_sized,
                   last_obs_age=float(N - obs[k - 1].N) if k else np.nan,
                   true_log_lambda=spec.truth_theta["log_lambda"], true_a0_mm=spec.a0 * 1e3,
                   true_depth_mm=float(spec.depth(N)) * 1e3, true_class=spec.label)
        rec.update(forecast_record(x, true_rul, spec.life, sc.get("alpha_rel", 0.2), sc.get("alpha_abs", 0.05)))
        rec.update(est.summary())
        if n_sbc and isinstance(est, (GridBayes, ParticleFilter)):
            rec["sbc_rank"] = rank(est.posterior_draws(n_sbc, rng_f, N=N)["log_lambda"], spec.truth_theta["log_lambda"])
        out.append(rec)
    return out


def decide_arm(spec, arm, i):
    """Approach 3: act once P(failure before the next planned reading + lead) >= p_act."""
    dc = _C.cfg["decision"]
    n_nom = _C.ctx["N_nominal"]
    p_acts = [float(p) for p in dc["p_act"]]
    lead = float(dc.get("lead_frac", 0.0)) * n_nom
    pol = _C.policies[arm.policy]
    est, rng_f = _new_estimator(arm, i)
    times = pol.times(spec, _C.ctx)
    obs = [] if getattr(est, "sees_truth", False) else _C.observe(spec, arm, i)
    events = [(t, None) for t in times] if getattr(est, "sees_truth", False) else [(o.N, o) for o in obs]
    n_s = int(dc.get("n_samples", 1000))
    acted = {}
    for k, (N, o) in enumerate(events):
        if o is not None:
            est.update(o)
        pending = [p for p in p_acts if p not in acted]
        if not pending:
            break
        window = pol.next_after(N, spec, _C.ctx) - N + lead
        p_fail = float(np.mean(_forecast(est, spec, N, n_s, rng_f) <= window))
        for p in pending:
            if p_fail >= p:
                acted[p] = (k + 1, N)
    end = spec.life if not spec.arrested else _C.ctx["horizon"]
    rows = []
    for p in p_acts:
        if p in acted:
            n_r, Na = acted[p]
            r = dict(acted=1, failed=0, n_readings=n_r, service=Na, unused=end - Na)
        else:
            r = dict(acted=0, failed=int(not spec.arrested), n_readings=len(events), service=end, unused=0.0)
        rows.append(dict(specimen=i, arm=arm.name, sensor=arm.sensor, policy=arm.policy, estimator=arm.estimator,
                         p_act=p, life=spec.life, arrested=spec.arrested, service_frac=r["service"] / n_nom,
                         unused_frac=r["unused"] / n_nom, **r))
    return rows


def forecast_points(spec, i):
    """Forecast times shared by every arm: [(N, kind, depth_mm)] sorted by N."""
    sc = _C.cfg["scoring"]
    n_nom = _C.ctx["N_nominal"]
    span = spec.life if not spec.arrested else _C.ctx["horizon"]
    rng_s = np.random.default_rng(stream(_C.cfg["seed"], i, "scoring"))
    pts = [(float(u) * float(sc.get("frac_max", 0.95)) * span, "random", np.nan)
           for u in rng_s.random(int(sc.get("n_random", 5)))]
    pts += [(float(f) * span, "fixed", np.nan) for f in sc.get("fixed_fractions", [])]
    for d in sc.get("at_depths_mm", []):
        N = spec.time_at_depth(float(d) * 1e-3)
        if np.isfinite(N):
            pts.append((N, "depth", float(d)))
    g = _C.ctx["grid"]                     # snapped like the readings, so a reading at that cycle counts
    pts += [(float(np.rint(f * n_nom / g) * g), "time", np.nan) for f in map(float, sc.get("at_time_frac", []))
            if f * n_nom < span]
    return sorted(pts, key=lambda x: x[0])


def run_specimen(i):
    cfg = _C.cfg
    spec = _C.truth.draw(np.random.default_rng(stream(cfg["seed"], i, "truth")), index=i)
    srow = dict(specimen=i, a0_mm=spec.a0 * 1e3, life=spec.life, a_fail_mm=spec.a_fail * 1e3,
                true_log_lambda=spec.truth_theta["log_lambda"], label=spec.label, arrested=spec.arrested)
    points = forecast_points(spec, i)
    spec.forecast_times = [N for N, _, _ in points]       # for the at_forecast policy only
    records, decisions = [], []
    for arm in _C.arms:
        records += forecast_arm(spec, arm, points, i)
        if (cfg.get("decision") or {}).get("enabled") and _C.policies[arm.policy].kind not in ("none", "at_forecast"):
            decisions += decide_arm(spec, arm, i)
    return srow, records, decisions


def _progress(it, n, cfg, started):
    """Collect results, printing a line to stderr about every 10 % so long runs can be followed."""
    out, step = [], max(1, n // 10)
    for k, r in enumerate(it, 1):
        out.append(r)
        if k % step == 0 or k == n:
            print(f"[{cfg.get('experiment')}/{cfg.get('variant', 'base')}] {k}/{n} specimens, "
                  f"{time.time() - started:.0f} s", file=sys.stderr, flush=True)
    return out


def run(cfg, out_dir, workers=None):
    started = time.time()
    _init(cfg)                               # builds priors, tables and populations before the workers start
    n = int(cfg["n_specimens"])
    workers = int(workers or cfg.get("workers") or 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(cfg,)) as ex:
            res = _progress(ex.map(run_specimen, range(n), chunksize=max(1, n // (8 * workers))), n, cfg, started)
    else:
        res = _progress(map(run_specimen, range(n)), n, cfg, started)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specimens = pd.DataFrame([r[0] for r in res])
    records = pd.DataFrame([x for r in res for x in r[1]])
    decisions = pd.DataFrame([x for r in res for x in r[2]])
    specimens.to_csv(out / "specimens.csv", index=False)
    warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN groups (e.g. only arrested specimens)
    if not records.empty:
        records.to_csv(out / "records.csv.gz", index=False)
        summarise(records, BY + ("kind",)).to_csv(out / "summary.csv", index=False)
        if (records["kind"] == "depth").any():
            summarise_depth(records).to_csv(out / "summary_depth.csv", index=False)
        if (records["kind"] == "fixed").any():
            beta = float(cfg["scoring"].get("beta", 0.5))
            fin = records[~records["arrested"].astype(bool)]
            prognostic.alpha_lambda(fin, beta, BY).to_csv(out / "alpha_lambda.csv", index=False)
            prognostic.prognostic_horizon(fin, beta, BY).to_csv(out / "prognostic_horizon.csv", index=False)
    if not decisions.empty:
        decisions.to_csv(out / "decisions.csv.gz", index=False)
    manifest.write(out, cfg, started, extra=dict(n_specimens=n, n_arrested=int(specimens["arrested"].sum()),
                                                  workers=workers, prior=_C.prior_info, ctx=_C.ctx))
    return out


def run_dir(cfg):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    variant = re.sub(r"[^A-Za-z0-9._=-]+", "_", cfg.get("variant", "base"))     # filesystem-safe on every OS
    return RESULTS / cfg["experiment"] / variant / f"{stamp}_{cfgmod.config_hash(cfg)[:8]}"


def main(argv=None, default_config=None):
    ap = argparse.ArgumentParser(description="Run an experiment config (every variant unless --variant).")
    ap.add_argument("config", nargs="?", default=default_config)
    ap.add_argument("--variant", action="append", help="run only this variant (repeatable)")
    ap.add_argument("--n", type=int, help="override n_specimens (recorded in the manifest)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="write runs under this directory instead of results/ (e.g. smoke tests)")
    args = ap.parse_args(argv)
    if args.config is None:
        ap.error("a config file is required")
    done = []
    for name, cfg in cfgmod.variants(cfgmod.load(args.config)):
        if args.variant and name not in args.variant:
            continue
        if args.n:
            cfg["n_specimens"] = args.n
        out = run_dir(cfg)
        if args.out:
            out = Path(args.out) / out.relative_to(RESULTS)
        out = run(cfg, out, args.workers)
        print(f"{name}: {out}")
        done.append((name, cfg, out))
    return done


if __name__ == "__main__":
    main()
