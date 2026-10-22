"""
Physics for LUMEN: Hapke radiative transfer, nanophase-iron space weathering, and the
domain-randomised synthetic-soil simulator used for pretraining.

Hapke core (Chandrasekhar H, reflectance <-> SSA, equivalent-slab SSA from n, k, D) is
adapted from the Jung et al. (2026) reproduction in this repository
(../Jung2026_SpaceWeatheredMaria/scripts/hapke.py), where it was self-tested against the
authors' released code; the photon mean free path uses the standard Hapke (1993) form
(that reproduction's deviation D1). Internal scattering s = 0, as in both released Jung
implementations.

Space weathering follows Hapke (2001) and Lucey & Riner (2011): nanophase iron (npFe0)
adds an absorption coefficient

    alpha_np(lambda) = 36 pi z(lambda) phi / lambda,
    z = n_h^3 n_Fe k_Fe / [ (n_Fe^2 - k_Fe^2 + 2 n_h^2)^2 + (2 n_Fe k_Fe)^2 ],

phi = volume fraction of Fe = (rho_host / rho_Fe) * mass fraction. Larger "microphase"
iron (mpFe0, > ~50 nm) darkens without reddening and is modelled as a grey absorber with
geometric cross-section, alpha_mp = 0.75 phi_mp / r_mp (r_mp = 0.1 um).

Everything is vectorised over a leading sample axis; one call simulates ~10^5 soils.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import lumen_common as lc

RHO_FE = 7.87
R_MP_UM = 0.10
IS_FEO_PER_NPFE = 2270.0       # Is/FeO = K * npFe0(wt%) / FeO(wt%): ~0.4 wt% npFe0 in a
                               # FeO = 15 % mature soil -> Is/FeO ~ 60 (Morris 1978 scale);
                               # only sets the pretraining target's scale (D4).

# --------------------------------------------------------------------------
# Hapke core (Jung reproduction, Appendices D/E)
# --------------------------------------------------------------------------
def chandrasekhar_h(w, x):
    w = np.asarray(w, dtype=float)
    gamma = np.sqrt(np.clip(1.0 - w, 0.0, 1.0))
    r0 = (1.0 - gamma) / (1.0 + gamma)
    return 1.0 / (1.0 - w * x * (r0 + (1.0 - 2.0 * r0 * x) / 2.0 * np.log((1.0 + x) / x)))


def ssa_to_reflectance(w, mu0=lc.MU0, mu=lc.MU):
    """Reflectance factor (P = 1, B = 0): r = w / (4 (mu0 + mu)) H(mu0) H(mu)."""
    w = np.asarray(w, dtype=float)
    return w / 4.0 / (mu0 + mu) * chandrasekhar_h(w, mu0) * chandrasekhar_h(w, mu)


_SSA_TAB = {}


def reflectance_to_ssa(R, mu0=lc.MU0, mu=lc.MU, n_table=200001):
    """Vectorised inverse of ssa_to_reflectance (monotone; tabulated once)."""
    key = (mu0, mu, n_table)
    if key not in _SSA_TAB:
        w = np.linspace(1e-9, 1.0 - 1e-12, n_table)
        _SSA_TAB[key] = (ssa_to_reflectance(w, mu0, mu), w)
    r_tab, w_tab = _SSA_TAB[key]
    R = np.asarray(R, dtype=float)
    out = np.interp(R, r_tab, w_tab)
    return np.where((R > r_tab[0]) & (R < r_tab[-1]), out, np.nan)


def s_e(n, k):
    return ((n - 1.0) ** 2 + k ** 2) / ((n + 1.0) ** 2 + k ** 2) + 0.05


def s_i(n):
    return 1.014 - 4.0 / (n * (n + 1.0) ** 2)


def d_mean_factor(n):
    """<D> / D for the equivalent slab (standard Hapke 1993 eq. 6.20 form)."""
    n = np.asarray(n, dtype=float)
    return 2.0 / 3.0 * (n ** 2 - 1.0 / n * (n ** 2 - 1.0) ** 1.5)


def slab_ssa(alpha_um, n, k, D_um):
    """SSA of a grain from total absorption coefficient alpha (1/um), real index n,
    imaginary index k (only enters the surface reflection), grain size D (um).
    Broadcasts over all arguments."""
    se = s_e(n, k)
    si = s_i(n)
    th = np.exp(-alpha_um * d_mean_factor(n) * D_um)
    return se + (1.0 - se) * (1.0 - si) * th / (1.0 - si * th)


def k_from_reflectance(R, lam_um, n, D_um, mu0=lc.MU0, mu=lc.MU, lo=1e-9, hi=1.0):
    """Per-band bisection for k such that the slab model reproduces R (Appendix E)."""
    R = np.atleast_1d(np.asarray(R, float))
    lam_um = np.atleast_1d(np.asarray(lam_um, float))
    out = np.full(R.shape, np.nan)
    for i, (r, lam) in enumerate(zip(R, lam_um)):
        def f(kk):
            a = 4.0 * np.pi * kk / lam
            return float(ssa_to_reflectance(slab_ssa(a, n, kk, D_um), mu0, mu)) - r
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            # brighter than a lossless grain, or darker than k = 1: clamp to the bound
            out[i] = lo if abs(f_lo) < abs(f_hi) else hi
            continue
        out[i] = brentq(f, lo, hi, xtol=1e-14, rtol=1e-12)
    return out


# --------------------------------------------------------------------------
# metallic iron optical constants and the npFe0 absorption
# --------------------------------------------------------------------------
def load_fe_nk():
    """Johnson & Christy (1974) to 1.937 um, Ordal et al. (1988) beyond, scaled to join
    continuously (the same construction as the Korokhin reproduction)."""
    import yaml

    def read(name):
        d = yaml.safe_load((lc.OPTICAL_DIR / name).read_text())
        rows = np.array([list(map(float, l.split()))
                         for l in d["DATA"][0]["data"].strip().splitlines()])
        return rows[:, 0], rows[:, 1], rows[:, 2]

    lj, nj, kj = read("Fe_JohnsonChristy1974.yml")
    lo, no, ko = read("Fe_Ordal1988.yml")
    edge = lj[-1]
    n_o_edge = np.interp(np.log(edge), np.log(lo), no)
    k_o_edge = np.interp(np.log(edge), np.log(lo), ko)
    keep = lo > edge
    lam = np.concatenate([lj, lo[keep]])
    n = np.concatenate([nj, no[keep] * nj[-1] / n_o_edge])
    k = np.concatenate([kj, ko[keep] * kj[-1] / k_o_edge])
    return lam, n, k


def z_hapke(lam_um, n_host):
    lam_fe, n_fe, k_fe = load_fe_nk()
    nf = np.interp(np.log(lam_um), np.log(lam_fe), n_fe)
    kf = np.interp(np.log(lam_um), np.log(lam_fe), k_fe)
    nh = float(n_host)
    return nh ** 3 * nf * kf / ((nf ** 2 - kf ** 2 + 2 * nh ** 2) ** 2 + (2 * nf * kf) ** 2)


def npfe_alpha_unit(lam_um, n_host, rho_host):
    """alpha_np per 1 wt% npFe0 (1/um), so alpha_np = f_wt% * this."""
    phi_per_wtpct = 0.01 * rho_host / RHO_FE
    return 36.0 * np.pi * z_hapke(lam_um, n_host) * phi_per_wtpct / np.asarray(lam_um)


def mpfe_alpha_unit(rho_host):
    """grey alpha_mp per 1 wt% mpFe0 (1/um)."""
    return 0.75 * (0.01 * rho_host / RHO_FE) / R_MP_UM


# --------------------------------------------------------------------------
# end-member library: k(lambda) of every staged mineral separate / glass
# --------------------------------------------------------------------------
OXIDES = ["FeO", "TiO2", "Al2O3", "CaO", "MgO"]
M_CAO, M_MGO, M_FEO, M_SIO2 = 56.08, 40.30, 71.85, 60.08


def pyroxene_oxides(wo, mg, tio2, al2o3):
    """Oxide wt% of a pyroxene from Wo fraction, Mg# (molar), TiO2 and Al2O3 wt%."""
    en, fs = (1 - wo) * mg, (1 - wo) * (1 - mg)
    tot = wo * M_CAO + en * M_MGO + fs * M_FEO + M_SIO2
    scale = (100.0 - tio2 - al2o3) / 100.0
    return {"FeO": 100 * fs * M_FEO / tot * scale, "TiO2": tio2, "Al2O3": al2o3,
            "CaO": 100 * wo * M_CAO / tot * scale, "MgO": 100 * en * M_MGO / tot * scale}


def olivine_oxides(fo):
    tot = 2 * fo * M_MGO + 2 * (1 - fo) * M_FEO + M_SIO2
    return {"FeO": 100 * 2 * (1 - fo) * M_FEO / tot, "TiO2": 0.05, "Al2O3": 0.05, "CaO": 0.2,
            "MgO": 100 * 2 * fo * M_MGO / tot}


# Nominal compositions of each end-member's parent phase (D3): approximate literature
# values for the source rock types (Lunar Sourcebook ch. 5; Papike et al. 1991), jittered
# per sample by the simulator. They only set the chemistry labels of synthetic soils.
_PLG_MARE = {"FeO": 0.6, "TiO2": 0.05, "Al2O3": 33.8, "CaO": 18.0, "MgO": 0.3}
_PLG_HIGH = {"FeO": 0.2, "TiO2": 0.02, "Al2O3": 35.3, "CaO": 19.2, "MgO": 0.15}
NOMINAL_CHEM = {
    "c1lr171": _PLG_MARE, "c1lr174": _PLG_MARE, "c1lr177": _PLG_MARE, "c1lr181": _PLG_MARE,
    "c1lr210": _PLG_MARE, "c1lr214": _PLG_MARE, "c1lr217": _PLG_MARE, "c1lr221": _PLG_MARE,
    "c1lr183": _PLG_HIGH, "c1lr223": _PLG_HIGH, "c1lr224": _PLG_HIGH, "c1lr226": _PLG_HIGH,
    "c1lr229": _PLG_HIGH, "c1lr240c": _PLG_HIGH, "c1lr241c": _PLG_HIGH, "c1lr242c": _PLG_HIGH,
    "c1lr243c": _PLG_HIGH,
    "c1lr173": pyroxene_oxides(0.08, 0.65, 0.6, 1.5), "c1lr209": pyroxene_oxides(0.08, 0.65, 0.6, 1.5),
    "c1lr176": pyroxene_oxides(0.10, 0.60, 1.0, 2.0), "c1lr216": pyroxene_oxides(0.10, 0.60, 1.0, 2.0),
    "c1lr180": pyroxene_oxides(0.10, 0.60, 1.0, 2.0), "c1lr220": pyroxene_oxides(0.10, 0.60, 1.0, 2.0),
    "c1ls12": pyroxene_oxides(0.03, 0.80, 0.4, 1.5),
    "c1lr168": pyroxene_oxides(0.30, 0.45, 1.5, 3.0), "c1lr211": pyroxene_oxides(0.30, 0.45, 1.5, 3.0),
    "c1lr170": pyroxene_oxides(0.25, 0.65, 0.8, 2.5), "c1lr213": pyroxene_oxides(0.25, 0.65, 0.8, 2.5),
    "c1lr172": pyroxene_oxides(0.35, 0.55, 1.0, 2.5), "c1lr208": pyroxene_oxides(0.35, 0.55, 1.0, 2.5),
    "c1lr175": pyroxene_oxides(0.40, 0.65, 3.0, 5.0), "c1lr215": pyroxene_oxides(0.40, 0.65, 3.0, 5.0),
    "c1lr179": pyroxene_oxides(0.40, 0.65, 3.0, 5.0), "c1lr219": pyroxene_oxides(0.40, 0.65, 3.0, 5.0),
    "c1ls09": pyroxene_oxides(0.30, 0.50, 1.5, 3.0),
    "c1lr169": olivine_oxides(0.65), "c1lr212": olivine_oxides(0.65),
    "c1lr225": olivine_oxides(0.87), "c1ls05": olivine_oxides(0.87), "c1lr14": olivine_oxides(0.87),
    "c1lr227": olivine_oxides(0.87),
    "c1lr178": {"FeO": 44.5, "TiO2": 53.5, "Al2O3": 0.1, "CaO": 0.1, "MgO": 1.5},
    "c1lr182": {"FeO": 44.5, "TiO2": 53.5, "Al2O3": 0.1, "CaO": 0.1, "MgO": 1.5},
    "c1lr218": {"FeO": 44.5, "TiO2": 53.5, "Al2O3": 0.1, "CaO": 0.1, "MgO": 1.5},
    "c1lr222": {"FeO": 44.5, "TiO2": 53.5, "Al2O3": 0.1, "CaO": 0.1, "MgO": 1.5},
    "c1lr51": {"FeO": 22.0, "TiO2": 8.8, "Al2O3": 6.3, "CaO": 7.4, "MgO": 14.7},
    "c1lr52": {"FeO": 19.7, "TiO2": 0.4, "Al2O3": 7.5, "CaO": 8.6, "MgO": 17.5},
    # synthetic lunar glasses: feldspathic (L19) and mafic (L22) bulk-glass approximations
    "c1sc50": {"FeO": 5.0, "TiO2": 0.4, "Al2O3": 26.0, "CaO": 15.0, "MgO": 6.0},
    "c1sc53": {"FeO": 18.0, "TiO2": 2.0, "Al2O3": 9.0, "CaO": 10.0, "MgO": 12.0},
}


class EndmemberLibrary:
    """k(lambda) on the model grid for every library spectrum, from Hapke inversion of
    its measured reflectance at the separate's nominal effective grain size."""

    def __init__(self, lam_um: np.ndarray):
        self.lam = np.asarray(lam_um, float)
        lib = lc.ENDMEMBER_LIBRARY.copy()
        k, R, rt = [], [], []
        for _, r in lib.iterrows():
            lam_src, refl = lc.read_relab_tab(lc.RELAB_DIR / (r["spectrum_id"] + ".tab"))
            Rm = lc.srf_resample(lam_src, refl, self.lam)
            if np.isnan(Rm).any():                     # a few separates stop at 2.5 um
                ok = np.isfinite(Rm)
                Rm = np.interp(self.lam, self.lam[ok], Rm[ok])
            n = lc.EM_N[r["em_class"]]
            kk = k_from_reflectance(Rm, self.lam, n, r["d_eff_um"])
            back = ssa_to_reflectance(slab_ssa(4 * np.pi * kk / self.lam, n, kk, r["d_eff_um"]))
            k.append(kk)
            R.append(Rm)
            rt.append(np.max(np.abs(back - Rm)))
        self.table = lib.assign(roundtrip_max_abs=rt)
        self.k = np.asarray(k)
        self.R = np.asarray(R)
        self.cls = lib["em_class"].to_numpy()
        self.n = np.array([lc.EM_N[c] for c in self.cls])
        self.chem = np.array([[NOMINAL_CHEM[s][o] for o in OXIDES] for s in lib["spectrum_id"]])
        self.index = {c: np.flatnonzero(self.cls == c) for c in np.unique(self.cls)}


# --------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------
# prior over the crystalline proportions (PLG, LCP, HCP, OLV, ILM): three petrological
# regimes from general lunar knowledge (D2) -- not fitted to the evaluation soils.
REGIMES = {
    "feldspathic": (0.35, np.array([12.0, 1.5, 0.6, 0.6, 0.15])),
    "basaltic": (0.35, np.array([4.0, 1.5, 2.5, 0.6, 1.0])),
    "broad": (0.30, np.array([1.2, 0.8, 0.8, 0.5, 0.4])),
}
CRYST = ["PLG", "LCP", "HCP", "OLV", "ILM"]


class Simulator:
    def __init__(self, lib: EndmemberLibrary):
        self.lib = lib
        lam = lib.lam
        self.lam = lam
        # npFe0 / mpFe0 absorption per wt%, by host class
        self.np_unit = {c: npfe_alpha_unit(lam, lc.EM_N[c], lc.EM_RHO.get(c, 3.0))
                        for c in ["PLG", "LCP", "HCP", "OLV", "ILM", "VGL", "HGL"]}
        self.mp_unit = mpfe_alpha_unit(3.0)

    def sample(self, n: int, rng: np.random.Generator, randomize_output: bool = True) -> dict:
        lib, lam, L = self.lib, self.lam, self.lam.size
        # ---- composition ---------------------------------------------------------
        names = list(REGIMES)
        probs = np.array([REGIMES[k][0] for k in names])
        reg = rng.choice(len(names), size=n, p=probs / probs.sum())
        cryst = np.zeros((n, 5))
        for j, nm in enumerate(names):
            m = reg == j
            cryst[m] = rng.dirichlet(REGIMES[nm][1], size=int(m.sum()))
        maturity = rng.uniform(0, 1, n)
        agg = np.clip(0.65 * maturity + rng.normal(0, 0.12, n), 0.0, 0.75)
        oth = 0.25 * rng.uniform(0, 1, n) ** 2
        s = agg + oth
        over = s > 0.9
        agg[over] *= 0.9 / s[over]
        oth[over] *= 0.9 / s[over]
        comp = np.column_stack([cryst * (1 - agg - oth)[:, None], agg, oth])     # (n, 7) vol

        # ---- end-member choice (domain randomisation) ------------------------------
        def pick(c):
            return rng.choice(lib.index[c], size=n)
        em = {c: pick(c) for c in CRYST}
        vgl, hgl = pick("VGL"), pick("HGL")
        oth_is_vgl = rng.uniform(0, 1, n) < 0.6                                  # OTH host
        oth_em = np.where(oth_is_vgl, vgl, pick("HGL"))

        # ---- grain sizes -----------------------------------------------------------
        d_soil = np.exp(rng.uniform(np.log(4.0), np.log(150.0), n))
        D = d_soil[:, None] * np.exp(rng.normal(0, 0.2, (n, 7)))
        D[:, 5] = d_soil * np.exp(rng.normal(0.15, 0.25, n))

        # ---- space weathering ------------------------------------------------------
        f_rim = 0.25 * maturity ** 1.5 * np.exp(rng.normal(0, 0.3, n))           # wt% npFe0
        f_agg = (0.2 + 1.0 * maturity) * np.exp(rng.normal(0, 0.3, n))           # wt% npFe0
        f_mp = maturity * rng.uniform(0.3, 1.5, n)                                # wt% mpFe0

        # ---- per-component SSA -------------------------------------------------------
        ssa = np.empty((n, 7, L))
        for j, c in enumerate(CRYST):
            k = lib.k[em[c]]                                                      # (n, L)
            nn = lc.EM_N[c]
            a = 4 * np.pi * k / lam + f_rim[:, None] * self.np_unit[c][None, :]
            ssa[:, j] = slab_ssa(a, nn, k, D[:, j:j + 1])
        # agglutinate: glass host mixed feldspathic/mafic by the crystalline make-up
        w_feld = cryst[:, 0] / np.clip(cryst[:, :4].sum(1), 1e-9, None)
        k_plg_glass = lib.k[lib.index["HGL"][0]]                                 # c1sc50 (L19)
        k_pyx_glass = lib.k[lib.index["HGL"][1]]                                 # c1sc53 (L22)
        k_host = w_feld[:, None] * k_plg_glass[None] + (1 - w_feld)[:, None] * k_pyx_glass[None]
        a = (4 * np.pi * k_host / lam + f_agg[:, None] * self.np_unit["HGL"][None, :]
             + f_mp[:, None] * self.mp_unit)
        ssa[:, 5] = slab_ssa(a, lc.EM_N["HGL"], k_host, D[:, 5:6])
        k = lib.k[oth_em]
        a = 4 * np.pi * k / lam + f_rim[:, None] * self.np_unit["VGL"][None, :]
        ssa[:, 6] = slab_ssa(a, lc.EM_N["VGL"], k, D[:, 6:7])

        # ---- intimate mixture: cross-section weights vol_i / D_i --------------------
        xs = comp / D
        xs = xs / xs.sum(1, keepdims=True)
        w_mix = np.einsum("nc,ncl->nl", xs, ssa)
        R = ssa_to_reflectance(np.clip(w_mix, 1e-6, 1 - 1e-9))

        # ---- labels: chemistry by mass; agglutinate carries the bulk composition ----
        rho = np.array([lc.EM_RHO[c] for c in lc.COMP])
        mass = comp * rho
        chem_em = np.zeros((n, 7, 5))
        for j, c in enumerate(CRYST):
            chem_em[:, j] = lib.chem[em[c]]
        chem_em[:, 6] = lib.chem[oth_em]
        # per-sample jitter of phase chemistry (+-15 %, FeO/MgO anticorrelated)
        jit = np.exp(rng.normal(0, 0.15, (n, 7, 1)))
        chem_em[:, :, [0, 1]] *= jit                                              # FeO, TiO2
        chem_em[:, :, 4] *= 1.0 / jit[:, :, 0]                                    # MgO
        non_agg = [0, 1, 2, 3, 4, 6]
        m_na = mass[:, non_agg]
        chem = np.einsum("nc,nco->no", m_na, chem_em[:, non_agg]) / m_na.sum(1, keepdims=True)
        mfrac = mass / mass.sum(1, keepdims=True)
        npfe_total = (mfrac[:, :5].sum(1) + mfrac[:, 6]) * f_rim + mfrac[:, 5] * f_agg
        isfeo = IS_FEO_PER_NPFE * npfe_total / np.clip(chem[:, 0], 0.3, None)

        # ---- output randomisation (calibration / photometric / noise) ---------------
        if randomize_output:
            x = lam - 1.5
            tilt = 1 + rng.normal(0, 0.03, (n, 1)) * x + rng.normal(0, 0.02, (n, 1)) * x ** 2
            scale = rng.uniform(0.93, 1.07, (n, 1))
            noise = rng.normal(0, 1, (n, L)) * rng.uniform(0.002, 0.008, (n, 1))
            R = R * tilt * scale * (1 + noise)
        R = np.clip(R, 1e-4, None)
        return {
            "R": R.astype(np.float32),
            "comp": (100 * comp).astype(np.float32),
            "chem": chem.astype(np.float32),
            "logIsFeO": np.log10(np.clip(isfeo, 0.1, None)).astype(np.float32),
            "logD": np.log10(d_soil).astype(np.float32),
            "npfe_wt": npfe_total.astype(np.float32),
            "maturity": maturity.astype(np.float32),
            "regime": reg.astype(np.int8),
        }
