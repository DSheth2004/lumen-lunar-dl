"""
Step 10 -- does the method recover the known make-up of real lunar soils?

The abstract claims the retrieved parameters "are in agreement with the values found
through lunar sample analysis"; the paper never tests this quantitatively on
samples of known composition. Here the paper's own procedure (the 8-parameter model,
Nelder-Mead + S corridor, on the 21-point grid) is applied to the LSCC spectra of the
nine mare soils whose modal mineralogy was measured (Taylor et al. 2001), each size fraction separately (27 spectra, n = 9 independent soils) -- and
to the ten Apollo 14/16 highland soils of Taylor et al. (2010)
(30 spectra, n = 10). Summaries are reported for the mare set (n = 9), the
highland set (n = 10) and all soils (n = 19), each against its own baseline.

What is compared (D14):
  * mineral PROPORTIONS within the mineral subtotal, on both sides: the model's gray
    component stands for glass and agglutinates (Sec. 3.2), so only the relative
    amounts of the five minerals are commensurable with the modal analyses.
    Model CPX  <-> Mg- + Fe-clinopyroxene (high-Ca, augite);
    model "OPX" <-> orthopyroxene + pigeonite (low-Ca pyroxene -- the basis
    spectrum itself is a pigeonite, PAPER_NOTES F3);
  * grain size S against the sieve fraction (<10, 10-20, 20-45 um);
  * c_nFe against the ferromagnetic maturity index Is/FeO.
Control: each soil's proportions predicted as the mean of the
other soils of the same set (same size fraction) -- a zero-information baseline.
Agreement only counts if the spectral retrieval beats it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import korokhin_common as kc
import shkuratov as sk
from step03_basis_set import fit_8param
from step06_m3_site_fits import S0_GRID

STAGE = "step10"
SOILS = ["10084", "12001", "12030", "15041", "15071", "70181", "71061", "71501", "79221"]
HIGHLAND_SOILS = ["14141", "14163", "14259", "14260", "61141", "61221", "62231", "64801", "67461", "67481"]
FRACTIONS = ["<10", "10-20", "20-45"]
FRACTION_MEAN_UM = {"<10": 5.0, "10-20": 15.0, "20-45": 32.5}
MAP = {"CPX": ["mg_clinopyroxene", "fe_clinopyroxene"], "OPX": ["orthopyroxene", "pigeonite"],
       "OLV": ["olivine"], "PLG": ["plagioclase"], "ILM": ["ilmenite"]}


def truth_props(modal: pd.DataFrame, soil: str, frac: str) -> dict | None:
    m = modal[(modal["sample"].astype(str) == soil) & (modal["size_fraction_um"] == frac)]
    if m.empty:
        return None
    v = m.set_index("phase")["modal_pct"]
    raw = {k: float(sum(v.get(p, np.nan) for p in ps)) for k, ps in MAP.items()}
    if not all(np.isfinite(list(raw.values()))):
        return None
    tot = sum(raw.values())
    return {k: 100 * x / tot for k, x in raw.items()}


def summarise(df: pd.DataFrame):
    """Per-mineral proportions vs truth (per-soil means) and the leave-one-soil-out
    mean-of-the-others baseline, within the soils of df."""
    for m in kc.MINERALS:
        base = []
        for _, r in df.iterrows():
            others = df[(df["soil"] != r["soil"]) & (df["fraction"] == r["fraction"])]
            base.append(others["prop_true_" + m].mean())
        df["prop_base_" + m] = base
    per_soil = df.groupby("soil").mean(numeric_only=True)
    summ = []
    for m in kc.MINERALS:
        t, f, b = per_soil["prop_true_" + m], per_soil["prop_fit_" + m], per_soil["prop_base_" + m]
        rho = spearmanr(t, f).statistic if f.std() > 0 else np.nan
        summ.append({"mineral": m, "n_soils": len(per_soil), "mae_fit": float((f - t).abs().mean()),
                     "mae_mean_of_others": float((b - t).abs().mean()), "bias_fit": float((f - t).mean()),
                     "spearman_fit_vs_true": float(rho), "true_mean": float(t.mean()),
                     "fit_mean": float(f.mean())})
    summ = pd.DataFrame(summ)
    summ["fit_beats_baseline"] = summ["mae_fit"] < summ["mae_mean_of_others"]
    mono = []
    for soil, g in df.groupby("soil"):
        g = g.sort_values("fraction_mean_um")
        mono.append(bool(np.all(np.diff(g["fit_S"].to_numpy()) > 0)) if len(g) == 3 else np.nan)
    rho_s = spearmanr(df["fraction_mean_um"], df["fit_S"]).statistic
    rho_m = {fr: spearmanr(g["is_feo"], g["fit_nFe"]).statistic for fr, g in df.groupby("fraction")}
    extra = pd.DataFrame([{"soils_with_S_increasing_with_fraction": int(np.nansum(mono)),
                           "soils_tested": int(np.sum(np.isfinite(mono))),
                           "spearman_S_vs_fraction": float(rho_s),
                           **{"spearman_nFe_vs_IsFeO_%s" % k: float(v) for k, v in rho_m.items()}}])
    return summ, extra


def main() -> int:
    kc.banner(STAGE, "LSCC mare + highland soils: retrieved vs measured mineralogy and maturity")
    kc.verify_manifest(STAGE)
    modal = pd.concat([pd.read_csv(kc.GROUND_TRUTH / "lscc_modal_mineralogy.csv"),
                       pd.read_csv(kc.GROUND_TRUTH / "lscc_highland_modal_mineralogy.csv")])
    isfeo = pd.concat([pd.read_csv(kc.GROUND_TRUTH / "lscc_is_feo.csv"),
                       pd.read_csv(kc.GROUND_TRUTH / "lscc_highland_is_feo.csv")])
    model = sk.LunarModel(kc.M3_GRID_UM)
    rows = []
    for soil in SOILS + HIGHLAND_SOILS:
        kd = kc.read_kdata(soil)
        for frac in FRACTIONS:
            truth = truth_props(modal, soil, frac)
            if truth is None or frac not in kd:
                continue
            obs = kc.gauss_to_grid(kd["lam_um"], kd[frac])
            r = fit_8param(model, obs, s0_grid=S0_GRID)
            th = dict(zip(kc.PARAMS, r["theta"]))
            mins = np.array([max(th[m], 0.0) for m in kc.MINERALS])
            props = 100 * mins / mins.sum() if mins.sum() > 0 else np.full(5, np.nan)
            ifo = isfeo[(isfeo["sample"].astype(str) == soil) & (isfeo["size_fraction_um"] == frac)]
            row = {"soil": soil, "set": "mare" if soil in SOILS else "highland",
                   "fraction": frac, "fraction_mean_um": FRACTION_MEAN_UM[frac],
                   "is_feo": float(ifo["is_feo"].iloc[0]) if len(ifo) else np.nan,
                   "sigma_rel": 100 * r["sigma"] / obs.mean()}
            row.update({"fit_" + k: v for k, v in th.items()})
            for j, m in enumerate(kc.MINERALS):
                row["prop_fit_" + m] = props[j]
                row["prop_true_" + m] = truth[m]
            rows.append(row)
            print("  %s %-6s S %6.1f  nFe %.3f  props fit %s | true %s  (sigma_Rel %.2f%%)"
                  % (soil, frac, th["S"], th["nFe"], " ".join("%4.0f" % p for p in props),
                     " ".join("%4.0f" % truth[m] for m in kc.MINERALS), row["sigma_rel"]))
    df = pd.DataFrame(rows)
    df.to_csv(kc.TABLES / "step10_lscc_fits.csv", index=False)

    summ_all, extra_all = [], []
    for set_name, sub in (("mare", df[df["set"] == "mare"]), ("highland", df[df["set"] == "highland"]),
                          ("all", df)):
        summ, extra = summarise(sub.copy())
        summ_all.append(summ.assign(set=set_name))
        extra_all.append(extra.assign(set=set_name))
        print("\n  [%s] mineral proportions, n = %d soils (per-soil mean over fractions):"
              % (set_name, summ["n_soils"].iloc[0]))
        print(summ.round(2).to_string(index=False))
        e = extra.iloc[0]
        print("  [%s] grain size: S rises with the sieve fraction in %d of %d soils; Spearman(S, fraction) = %.2f"
              % (set_name, e["soils_with_S_increasing_with_fraction"], e["soils_tested"],
                 e["spearman_S_vs_fraction"]))
        print("  [%s] maturity: Spearman(c_nFe, Is/FeO) per fraction: %s"
              % (set_name, ", ".join("%s %.2f" % (k.split("_")[-1], e[k]) for k in extra.columns
                                     if k.startswith("spearman_nFe"))))
    summ = pd.concat(summ_all)
    summ.to_csv(kc.TABLES / "step10_lscc_summary_by_set.csv", index=False)
    summ[summ["set"] == "mare"].drop(columns=["set"]).to_csv(kc.TABLES / "step10_lscc_summary.csv", index=False)
    extra = pd.concat(extra_all)
    extra.to_csv(kc.TABLES / "step10_lscc_grain_maturity_by_set.csv", index=False)
    extra[extra["set"] == "mare"].drop(columns=["set"]).rename(
        columns={"spearman_S_vs_fraction": "spearman_S_vs_fraction_all27"}).to_csv(
        kc.TABLES / "step10_lscc_grain_maturity.csv", index=False)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
