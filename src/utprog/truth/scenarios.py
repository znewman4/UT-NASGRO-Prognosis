"""Effects on the virtual specimens beyond constant-amplitude NASGRO growth.

Residual stress (M2, Approach 3) reuses nasgro_scenarios.residual_stress unchanged.
Three forms:
    superposition    decaying profile A sigma_y (1 - z/L) e^(-z/L); K_res(a) from the
                     Shen & Glinka weight function, added to Kmax and Kmin
                     (Parker 1982; LaRue & Daniewicz 2007)
    bs7910_uniform   uniform background A sigma_y (BS 7910 Level-1 style)
    redistribution   superposition with K_res relaxing once the crack extends past a_ref:
                         K_res,eff(a) = K_res(a) [r_inf + (1 - r_inf) exp(-(a - a_ref)/ell)]
                     A phenomenological stand-in for the redistribution Edwards (2006)
                     reports; r_inf and ell are placeholders to agree with supervisors.
Sign classes (tensile, none, compressive) are scenarios with A > 0, no residual
stress, and A < 0.

Growth-rate variability along the path (optional, truth only): the rate is
multiplied by exp(sd Z - sd^2/2), Z a stationary Gaussian AR(1) in log depth,
after the lognormal random-factor model of Yang & Manning (1996).

Overloads, R changes and mixed mode belong to later milestones.
"""
import numpy as np

from .._external import nasgro

RS_FORMS = ("superposition", "bs7910_uniform", "redistribution")


def rs_scenario(spec, case):
    """Scenario dict in nasgro_scenarios format from a config entry; None for no residual stress.

    spec keys: from (a name in nasgro_scenarios.residual_stress.SCENARIOS), A [lo, hi]
    (fraction of sigma_y), L_mm [median, CoV], shakedown, form, label, and for
    redistribution r_inf, ell_mm, a_ref_mm.
    """
    spec = {k: v for k, v in (spec or {}).items() if k != "prob"}
    if not spec or spec.get("form") == "none":
        return None
    base = {}
    if "from" in spec:
        base = dict(nasgro().residual_stress.SCENARIOS[spec["from"]])
        if base["case"] != case:
            raise ValueError(f"scenario {spec['from']} is for the {base['case']}, not the {case}")
    form = spec.get("form", "bs7910_uniform" if base.get("shape") == "uniform" else "superposition")
    if form not in RS_FORMS:
        raise ValueError(f"unknown residual-stress form {form}")
    lo, hi = map(float, spec.get("A", base.get("A")))
    if form == "bs7910_uniform":
        L = None
    elif "L_mm" in spec:
        L = (float(spec["L_mm"][0]) * 1e-3, float(spec["L_mm"][1]))
    elif base.get("L") is not None:
        L = tuple(base["L"])
    else:
        raise ValueError(f"form {form} needs a decay length L_mm")
    sc = dict(case=case, A=(lo, hi), L=L, shape="uniform" if form == "bs7910_uniform" else "decaying",
              shakedown=bool(spec.get("shakedown", base.get("shakedown", False))),
              label=spec.get("label", base.get("label", form)), form=form)
    if form == "redistribution":
        sc.update(r_inf=float(spec.get("r_inf", 0.5)), ell=float(spec.get("ell_mm", 1.0)) * 1e-3,
                  a_ref=float(spec["a_ref_mm"]) * 1e-3)
    return sc


def add_rs_inputs(s, sc, seed, fixed=()):
    """Draw the residual-stress inputs (amplitude ~ U(A), L lognormal) with the validated sampler."""
    return nasgro().residual_stress.add_inputs(s, sc, int(seed), fixed)


def rs_inputs_at(s, sc, amp):
    """Residual-stress inputs at amplitude(s) `amp` (fraction of sigma_y), L at its median: table nodes."""
    amp = np.broadcast_to(np.asarray(amp, float), s["a_i"].shape)
    return nasgro().residual_stress.add_inputs(s, dict(sc, A=(amp, amp)), 0, fixed=("rs_amp", "rs_len"))


def relaxation(a, a_ref, r_inf, ell):
    x = np.maximum(np.asarray(a, float) - a_ref, 0.0) / ell
    return r_inf + (1.0 - r_inf) * np.exp(-x)


def rs_k_fn(case, sc):
    """k_res_fn(s, a) for the scenario's form."""
    base = nasgro().residual_stress.k_res(case, sc["shape"])
    if sc["form"] != "redistribution":
        return base
    a_ref, r_inf, ell = sc["a_ref"], sc["r_inf"], sc["ell"]

    def fn(s, a):
        return base(s, a) * relaxation(a, a_ref, r_inf, ell)
    return fn


def path_variability(sd, corr_log_a, z):
    """Rate multiplier along the path; z holds one standard normal per depth bin."""
    def fn(a):
        la = np.log(np.asarray(a, float))
        m = la.shape[-1]
        rho = np.exp(-np.abs(np.diff(la, axis=-1)) / corr_log_a)
        e = np.empty_like(la)
        e[..., 0] = z[0]
        for j in range(1, m):
            r = rho[..., j - 1]
            e[..., j] = r * e[..., j - 1] + np.sqrt(1.0 - r ** 2) * z[j]
        return np.exp(sd * e - 0.5 * sd ** 2)
    return fn


def overloads(*_, **__):
    raise NotImplementedError("Overload histories need a load-sensor model; not in M0-M2.")


def r_change(*_, **__):
    raise NotImplementedError("Stress-ratio changes are not in M0-M2.")


def mixed_mode(*_, **__):
    raise NotImplementedError("Mixed mode is M3 (Approach 5); nasgro_scenarios.mixed_mode is the starting point.")
