"""
Step 03 -- the basis set (Table 4, Fig. 9), the rejected sets (Figs. 7-8), the
end-member identities, and Table 7 (the lunar agglutinate LS-CMP-045 fitted with
the new basis).

Table 7 uses the full 8-parameter model (Sec. 4.1) on the lab spectrum of C2LS45,
fitted over 0.30-2.60 um on its native 5-nm grid (D4), gray albedo 0.15,
alpha_Fe as selected by step01.

Starting point and simplex facets for the non-S parameters are not given in the
paper (D6): nFe 0.05, c_Gr 30 %, CPX/OPX/OLV/ILM 10 %, PLG 20 % (Table 10 shows
the printed fits start from c_PLG = 20 %, since its 20 % row equals Table 9 site 1);
facets 0.02 for nFe and 10 % for the concentrations; S corridor 1-100 um, facet 1 um.
"""
from __future__ import annotations

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import korokhin_common as kc
import nm_fit
import shkuratov as sk

STAGE = "step03"
LAM = np.round(np.arange(0.30, 2.6001, 0.005), 3)

START = np.array([30.0, 0.05, 30.0, 10.0, 10.0, 10.0, 20.0, 10.0])      # PARAMS order
SCALE = np.array([100.0, 0.2, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
FACET = np.array([nm_fit.S_FACET, 0.02, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
LOWER = np.zeros(8)
UPPER = np.array([500.0, 1.0, 400.0, 300.0, 300.0, 300.0, 300.0, 300.0])


def fit_8param(model, obs, s0_grid=nm_fit.S0_GRID, start=START):
    prob = nm_fit.Problem(model.fast, obs, SCALE, LOWER, UPPER, start, FACET, s_index=0)
    r = prob.corridor(s0_grid)
    th = r["theta"]
    return {"theta": th, "sigma": r["sigma"], "model": model.forward(th),
            "seconds": r["seconds"], "nfev": r["nfev"], "s0": r["s0"]}


def identities() -> pd.DataFrame:
    rows = []
    for name, b in kc.BASIS.items():
        t = (kc.RELAB_DIR / ("%s.xml" % b["spectrum"])).read_text(errors="replace")
        m = re.search(r"<speclib:specimen_description>(.*?)</speclib:specimen_description>", t, re.S)
        desc = m.group(1).strip() if m else ""
        klass = ("orthopyroxene" if "Orthopyroxene" in desc else
                 "clinopyroxene" if "Clinopyroxene" in desc else "")
        rows.append({"paper_label": name, "spectrum": b["spectrum"], "relab_id": b["relab_id"],
                     "specimen": kc.read_relab_xml(kc.RELAB_DIR / ("%s.xml" % b["spectrum"]))["specimen"],
                     "relab_description": desc, "pyroxene_class_per_relab": klass,
                     "label_consistent": not (name == "OPX" and klass == "clinopyroxene")})
    return pd.DataFrame(rows)


def main() -> int:
    kc.banner(STAGE, "basis set, end-member identities, Table 7 (agglutinate)")
    kc.verify_manifest(STAGE)

    # ---- end-member identities ------------------------------------------
    ident = identities()
    ident.to_csv(kc.TABLES / "step03_endmember_identity.csv", index=False)
    for _, r in ident.iterrows():
        print("  %-3s %-8s %-40s | %s" % (r["paper_label"], r["spectrum"], r["specimen"][:40],
                                          r["relab_description"][:70]))

    # ---- Fig. 9: A(lambda) and k(lambda) of the basis --------------------
    comps = sk.lunar_basis_components(LAM, gray=False)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 9))
    colors = {"CPX": "g", "OPX": "b", "OLV": "orange", "PLG": "m", "ILM": "k"}
    krows = {"lam_um": LAM}
    ax_t = a2.twinx()
    for c in comps:
        l, a = kc.relab_spectrum(kc.BASIS[c.name]["spectrum"])
        a1.plot(l, a, color=colors[c.name], lw=1, label=c.name)
        if c.transparent:
            a2.plot(LAM, c.k, color=colors[c.name], lw=1, label=c.name)
            krows["k_" + c.name] = c.k
        else:
            ax_t.plot(LAM, c.tau, color="k", lw=1, label="tau(ILM)")
            krows["tau_ILM"] = c.tau
    pd.DataFrame(krows).to_csv(kc.TABLES / "step03_basis_k.csv", index=False)
    a1.set_xlim(0.3, 2.6); a1.set_ylabel("Reflectance"); a1.legend(); a1.set_title("Fig. 9(a)")
    a2.set_xlim(0.3, 2.6); a2.set_ylim(0, 0.001); a2.set_ylabel("k (S = 30 um, n = 1.6)")
    ax_t.set_ylim(0, 1.5); ax_t.set_ylabel("tau(ILM)")
    a2.legend(loc="upper center"); a2.set_title("Fig. 9(b)"); a2.set_xlabel("Wavelength, um")
    fig.tight_layout(); fig.savefig(kc.FIGURES / "fig09_basis_set.png", dpi=130); plt.close(fig)

    # ---- Figs. 7-8: the rejected sets, where the spectra exist ------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, title, members in (
            (axes[0], "Fig. 7 Cloutis & Gaffey (1993) set (CASC02 unavailable)",
             [("PLG", "c1sc37"), ("OPX", "c1sc36"), ("CPX", "c1sc35"), ("ILM", "c1sc34")]),
            (axes[1], "Fig. 8 Noble et al. (2006) set (4 of 7 unavailable)",
             [("AGG", "c2ls45"), ("OPX", "c1sb06"), ("PLG", "c1sw12")])):
        for name, spec in members:
            l, a = kc.relab_spectrum(spec)
            ax.plot(l, a, lw=1, label="%s %s" % (name, spec.upper()))
        ax.set_xlim(0.3, 2.6); ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
        ax.set_xlabel("Wavelength, um")
    fig.tight_layout(); fig.savefig(kc.FIGURES / "fig07-08_rejected_sets.png", dpi=120); plt.close(fig)

    # ---- Table 7: the agglutinate ---------------------------------------
    l, a = kc.relab_spectrum("c2ls45")
    obs = kc.resample(l, a, LAM)
    model = sk.LunarModel(LAM)
    r = fit_8param(model, obs)
    th = r["theta"]
    mean_a = float(np.mean(obs))
    row = dict(zip(kc.PARAMS, th))
    row.update({"sum_c": float(th[2:].sum()), "sigma": r["sigma"],
                "sigma_rel": 100 * r["sigma"] / mean_a, "mean_albedo": mean_a,
                "best_S0": r["s0"], "nfev": r["nfev"], "seconds": r["seconds"]})
    pd.DataFrame([row]).to_csv(kc.TABLES / "step03_table7.csv", index=False)
    pub = pd.read_csv(kc.PAPER_TABLES / "t07_agglutinate_fit.csv").iloc[0]
    print("\n  Table 7 (LS-CMP-045)   ours     paper")
    for k in kc.PARAMS + ["sum_c", "sigma", "sigma_rel"]:
        print("    %-9s %10.4f %9.4f" % (k, row[k], pub[k]))
    print("  <A> over 0.30-2.60 um: %.5f (implied by the paper's sigma/sigma_Rel: 0.05677)" % mean_a)
    print("  %.0f s, %d function evaluations" % (r["seconds"], r["nfev"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(LAM, obs, "b", lw=1.2, label="measured C2LS45")
    ax.plot(LAM, r["model"], "r", lw=1, label="model")
    ax.plot(LAM, r["model"] - obs + 0.1, "m", lw=1, label="model - measured + 0.1")
    ax.set_xlim(0.3, 2.6); ax.set_xlabel("Wavelength, um"); ax.set_ylabel("Reflectance")
    ax.set_title("Fig. 10 / Table 7: sigma_Rel %.2f%% (paper 1.55%%)" % row["sigma_rel"], fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(kc.FIGURES / "fig10_agglutinate_fit.png", dpi=130); plt.close(fig)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
