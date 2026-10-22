"""
Step 02 -- Table 1 and Figs. 4-6: the method on three RELAB mixtures of known make-up.

For each mixture the end-member lab spectra are turned into k(lambda) at the grain
size the paper uses for Eq. A1 (38, 45, 45 um; Table 1), then the mixture spectrum
is fitted for the component concentrations and one common grain size S with the
Nelder-Mead + S-corridor procedure (nm_fit.py). No nFe and no gray component --
"since all of the samples are immature, we did not consider space-weathering
factors" (Sec. 2.4). The Odessa metal powder is nontransparent: n = 2.0, tau from
A2 (Sec. 2.4, Appendix A.2).

Choices the paper does not state (PAPER_NOTES.md):
  D4  fit range 0.30-2.60 um on the native 5-nm RELAB grid; <A> in sigma_Rel over
      the same range. Chosen because the <A> this implies for the agglutinate of
      Table 7 (0.05674) equals the value implied by its printed sigma / sigma_Rel
      (0.05677); the other plausible ranges miss by >= 0.0025.
  D5  concentrations free (not forced to sum to 100 %) and renormalised for the
      "restored proportions" column -- the printed proportions sum to exactly 100.
      Sensitivity: the sum-constrained fit.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import korokhin_common as kc
import nm_fit
import shkuratov as sk

STAGE = "step02"
LAM = np.round(np.arange(0.30, 2.6001, 0.005), 3)

MIXTURES = [
    {"id": 1, "mixture": "c1ms16", "S_A1": 38.0, "truth": [50, 50],
     "members": [("OLV", "c1ms14", kc.N_TRANSPARENT), ("PYX", "c1ms15", kc.N_TRANSPARENT)]},
    {"id": 2, "mixture": "c1xt36", "S_A1": 45.0, "truth": [33.3, 33.3, 33.3],
     "members": [("OLV", "c2po81", kc.N_TRANSPARENT), ("PYX", "c1pe41", kc.N_TRANSPARENT),
                 ("ANO", "c1pa60", kc.N_TRANSPARENT)]},
    # the paper prints 40:30:30 (OLV:PYX:MET); RELAB's own label for C1SC07 reads
    # "OLV:PYX:MET 30:40:30" -- both are kept (PAPER_NOTES.md F2)
    {"id": 3, "mixture": "c1sc07", "S_A1": 45.0, "truth": [40, 30, 30], "truth_relab": [30, 40, 30],
     "members": [("OLV", "c1sc02", kc.N_TRANSPARENT), ("PYX", "c1sc09", kc.N_TRANSPARENT),
                 ("MET", "c1sc06", kc.N_OPAQUE)]},
]


def spectrum_on_grid(spec_id):
    l, a = kc.relab_spectrum(spec_id)
    return kc.resample(l, a, LAM)


def build(mix):
    comps = []
    for name, spec, n in mix["members"]:
        a = spectrum_on_grid(spec)
        if n == kc.N_OPAQUE:
            comps.append(sk.Component(name, n, tau=sk.tau_from_albedo(a, n)))
        else:
            comps.append(sk.Component(name, n, k=sk.k_from_albedo(a, LAM, n, mix["S_A1"])))
    obs = spectrum_on_grid(mix["mixture"])
    ok = np.isfinite(obs)
    for c in comps:
        ok &= np.isfinite(c.k if c.transparent else c.tau)
    lam = LAM[ok]
    comps = [sk.Component(c.name, c.n, k=c.k[ok] if c.transparent else None,
                          tau=None if c.transparent else c.tau[ok]) for c in comps]
    return sk.Mixture(lam, comps), obs[ok], lam


def fit(mix, constrained=False, warm=None):
    model, obs, lam = build(mix)
    n = len(mix["members"])
    if constrained:
        # fit n-1 concentrations, the last is 100 - sum; negative -> infeasible
        def fwd(th):
            c = np.r_[th[1:], 100.0 - th[1:].sum()]
            if c[-1] < 0:
                return np.full(lam.shape, np.nan)
            return model.forward(th[0], c / 100.0)
        k = n - 1
    else:
        def fwd(th):
            return model.forward(th[0], th[1:] / 100.0)
        k = n
    prob = nm_fit.Problem(fwd, obs, scale=[100.0] + [100.0] * k, lower=[0.0] + [0.0] * k,
                          upper=[500.0] + [200.0] * k, start=[30.0] + [100.0 / n] * k,
                          facet=[nm_fit.S_FACET] + [10.0] * k)
    r = prob.corridor()
    if warm is not None:
        # also start from the (renormalised) free solution, so an infeasible-edge
        # stall of the constrained simplex is not mistaken for its optimum
        th_w, f_w = prob.run_from(np.r_[warm[0], (100.0 * warm[1:] / warm[1:].sum())[:k]])
        if f_w < r["sigma"]:
            r.update({"theta": th_w, "sigma": f_w, "s0": "warm"})
    th = r["theta"]
    conc = np.r_[th[1:], 100.0 - th[1:].sum()] if constrained else th[1:]
    return {"S": th[0], "conc": conc, "sigma": r["sigma"],
            "sigma_rel": 100.0 * r["sigma"] / float(np.mean(obs)), "model": fwd(th),
            "obs": obs, "lam": lam, "nfev": r["nfev"], "seconds": r["seconds"], "s0": r["s0"]}


def main() -> int:
    kc.banner(STAGE, "Table 1: restore the make-up of three RELAB mixtures")
    kc.verify_manifest(STAGE)
    t1 = pd.read_csv(kc.PAPER_TABLES / "t01_lab_mixtures.csv")
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    for mix in MIXTURES:
        pub = t1[t1["mixture"] == mix["id"]].iloc[0]
        pub_pct = [float(v) for v in pub["restored_pct"].split(":")]
        free = None
        for constrained in (False, True):
            r = fit(mix, constrained, warm=free)
            if not constrained:
                free = np.r_[r["S"], r["conc"]]
            norm = 100.0 * r["conc"] / r["conc"].sum()
            row = {"mixture": mix["id"], "spectrum": mix["mixture"],
                   "variant": "sum-constrained" if constrained else "free (primary)",
                   "S_um": r["S"], "sigma": r["sigma"], "sigma_rel_pct": r["sigma_rel"],
                   "sum_c_raw_pct": float(r["conc"].sum()), "published_S_um": pub["restored_S_um"],
                   "published_sigma_rel_pct": pub["sigma_rel_pct"], "best_S0_um": r["s0"],
                   "nfev": r["nfev"], "seconds": r["seconds"]}
            for j, (name, _, _) in enumerate(mix["members"]):
                row["c%d_%s_pct" % (j + 1, name)] = norm[j]
                row["c%d_published_pct" % (j + 1)] = pub_pct[j]
                row["c%d_true_pct" % (j + 1)] = mix["truth"][j]
                row["c%d_true_relab_label_pct" % (j + 1)] = mix.get("truth_relab", mix["truth"])[j]
            rows.append(row)
            print("  mixture %d %-16s S %6.1f um (paper %.1f)  proportions %s (paper %s, true %s)"
                  "  sigma_rel %.2f%% (paper %.1f)  [%.0f s]"
                  % (mix["id"], row["variant"], r["S"], pub["restored_S_um"],
                     ":".join("%.0f" % v for v in norm), pub["restored_pct"],
                     ":".join("%.0f" % v for v in mix["truth"]), r["sigma_rel"],
                     pub["sigma_rel_pct"], r["seconds"]))
            if not constrained:
                i = mix["id"] - 1
                ax = axes[i, 0]
                for name, spec, _ in mix["members"]:
                    l, a = kc.relab_spectrum(spec)
                    ax.plot(l, a, lw=1, label="%s (%s)" % (name, spec.upper()))
                ax.set_xlim(0.3, 2.6)
                ax.set_ylabel("Reflectance")
                ax.legend(fontsize=8)
                ax.set_title("Fig. %d(a) end-members" % (mix["id"] + 3), fontsize=9)
                ax = axes[i, 1]
                ax.plot(r["lam"], r["obs"], "b", lw=1.2, label="measured %s" % mix["mixture"].upper())
                ax.plot(r["lam"], r["model"], "r", lw=1, label="model")
                ax.plot(r["lam"], r["model"] - r["obs"] + 0.1, "m", lw=1, label="model - measured + 0.1")
                ax.set_xlim(0.3, 2.6)
                ax.legend(fontsize=8)
                ax.set_title("Fig. %d(b) fit: sigma_Rel %.2f%%" % (mix["id"] + 3, r["sigma_rel"]),
                             fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength, um")
    fig.tight_layout()
    fig.savefig(kc.FIGURES / "fig04-06_lab_mixtures.png", dpi=130)
    pd.DataFrame(rows).to_csv(kc.TABLES / "step02_table1.csv", index=False)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
