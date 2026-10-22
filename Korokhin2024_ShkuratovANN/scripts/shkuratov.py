"""
The Shkuratov et al. (1999) spectral mixing model, exactly as summarised in the
Appendix of Korokhin et al. (2024). Equation numbers below are the paper's.

Units: wavelength and grain size in micrometres, so k is dimensionless and
alpha_Fe is in um^-1. Concentrations enter the mixture as FRACTIONS here; the
paper quotes them in percent, and the callers convert.

Two kinds of component:
  * transparent (CPX, OPX, OLV, PLG, and the lab minerals of Table 1):
    k(lambda) is derived once from the lab spectrum at the lab grain size (A1-A7);
    in a mixture tau = 4 pi k S / lambda + alpha_Fe(lambda) c_nFe d   (A9, A16);
  * nontransparent (ilmenite, metal, the gray component, agglutinate):
    tau(lambda) comes straight from the lab albedo via A2 and is NOT scaled by
    S nor modified by nFe (Sec. 2.2 step 2, Sec. 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import korokhin_common as kc


# --------------------------------------------------------------------------
# single-particle optics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fresnel:
    """Average reflection/transmission coefficients for real index n, Eqs. A3-A7."""
    n: float
    r0: float = field(init=False)
    Re: float = field(init=False)
    Rb: float = field(init=False)
    Rf: float = field(init=False)
    Ri: float = field(init=False)
    Te: float = field(init=False)
    Ti: float = field(init=False)

    def __post_init__(self):
        n = self.n
        r0 = ((n - 1.0) / (n + 1.0)) ** 2           # normal-incidence Fresnel coefficient
        Re = r0 + 0.055                             # A3
        Rb = (0.28 * n - 0.20) * Re                 # A4
        Ri = 1.04 - 1.0 / n ** 2                    # A5
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "Re", Re)
        object.__setattr__(self, "Rb", Rb)
        object.__setattr__(self, "Rf", Re - Rb)     # below A8
        object.__setattr__(self, "Ri", Ri)
        object.__setattr__(self, "Te", 1.0 - Re)    # A6
        object.__setattr__(self, "Ti", 1.0 - Ri)    # A7


def tau_from_albedo(A, n: float, q: float = kc.Q) -> np.ndarray:
    """Eq. A2: optical density between two internal reflections from the albedo."""
    f = Fresnel(n)
    A = np.asarray(A, dtype=float)
    y = (1.0 - A) ** 2 / (2.0 * A)
    a = f.Te * f.Ti * (y * f.Ri + q * f.Te)
    b = y * f.Rb * f.Ri + q * f.Te ** 2 * (1.0 + f.Ti) / 2.0 - f.Te * (1.0 - q * f.Rb)
    c = 2.0 * y * f.Rb - 2.0 * f.Te * (1.0 - q * f.Rb) + q * f.Te ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = b / a + np.sqrt((b / a) ** 2 - c / a)
        return -np.log(arg)


def k_from_albedo(A, lam_um, n: float, S_um: float, q: float = kc.Q) -> np.ndarray:
    """Eq. A1: k = tau lambda / (4 pi S)."""
    return tau_from_albedo(A, n, q) * np.asarray(lam_um) / (4.0 * np.pi * S_um)


def particle_rb_rf(tau, n: float):
    """Eq. A8: backward and forward particle albedo from the optical density."""
    f = Fresnel(n)
    e = np.exp(-np.asarray(tau, dtype=float))
    H = 0.5 * f.Te * f.Ti * f.Ri * e * e / (1.0 - f.Ri * e)
    return f.Rb + H, f.Rf + f.Te * f.Ti * e + H


def albedo_from_rho(rho_b, rho_f) -> np.ndarray:
    """Eq. A11: albedo of a half-infinite pile of layers. NaN where unphysical."""
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (1.0 + rho_b ** 2 - rho_f ** 2) / (2.0 * rho_b)
        ok = (rho_b > 0) & (P >= 1.0 - 1e-9)     # P = 1 exactly for a lossless medium
        P = np.maximum(P, 1.0)
        A = P - np.sqrt(P * P - 1.0)
    return np.where(ok, A, np.nan)


def albedo_floor(n: float, q: float = kc.Q) -> float:
    """Darkest albedo the model can represent for index n (tau -> infinity: surface
    reflection only). Below it Eq. A2 has no solution."""
    return float(single_albedo(np.inf, n, q))


def single_albedo(tau, n: float, q: float = kc.Q) -> np.ndarray:
    """A single-component medium: A8 -> A10 -> A11."""
    rb, rf = particle_rb_rf(tau, n)
    return albedo_from_rho(q * rb, q * rf + 1.0 - q)


# --------------------------------------------------------------------------
# nanophase iron (Appendix A.4)
# --------------------------------------------------------------------------
def load_fe_nk():
    """Metallic-iron n,k: Johnson & Christy (1974) up to 1.937 um, Ordal et al. (1988)
    beyond, scaled to join continuously at 1.937 um (Weaver 1979, which the paper
    used for the infrared, is not openly available -- deviation D3)."""
    import yaml

    def read(name):
        d = yaml.safe_load((kc.OPTICAL / name).read_text())
        rows = np.array([list(map(float, l.split()))
                         for l in d["DATA"][0]["data"].strip().splitlines()])
        return rows[:, 0], rows[:, 1], rows[:, 2]

    lj, nj, kj = read("Fe_JohnsonChristy1974.yml")
    lo, no, ko = read("Fe_Ordal1988.yml")
    edge = lj[-1]
    # scale Ordal so it meets J&C at the last J&C point (log-lambda interpolation)
    n_o_edge = np.interp(np.log(edge), np.log(lo), no)
    k_o_edge = np.interp(np.log(edge), np.log(lo), ko)
    keep = lo > edge
    lam = np.concatenate([lj, lo[keep]])
    n = np.concatenate([nj, no[keep] * nj[-1] / n_o_edge])
    k = np.concatenate([kj, ko[keep] * kj[-1] / k_o_edge])
    return lam, n, k


def alpha_fe(lam_um, n_host: float = kc.N_TRANSPARENT, fe=None) -> np.ndarray:
    """Eqs. A14-A15: alpha_Fe = 6 pi (n/lambda) Im{(m^2-1)/(m^2+2)}, m = n_Fe/n.

    tau_Fe = alpha_Fe * c_nFe * d.  Units um^-1 for lambda in um.
    """
    lam_fe, n_fe, k_fe = load_fe_nk() if fe is None else fe
    lam_um = np.asarray(lam_um, dtype=float)
    nn = np.interp(np.log(lam_um), np.log(lam_fe), n_fe)
    kk = np.interp(np.log(lam_um), np.log(lam_fe), k_fe)
    m = (nn + 1j * kk) / n_host
    lorentz = (m * m - 1.0) / (m * m + 2.0)
    return 6.0 * np.pi * (n_host / lam_um) * lorentz.imag


# --------------------------------------------------------------------------
# mixtures
# --------------------------------------------------------------------------
@dataclass
class Component:
    name: str
    n: float
    k: np.ndarray | None = None      # transparent: absorption spectrum on the grid
    tau: np.ndarray | None = None    # nontransparent: fixed optical density on the grid

    @property
    def transparent(self) -> bool:
        return self.k is not None


class Mixture:
    """Intimate mixture of coarse particles (Appendix A.3, type 1), Eq. A12.

    forward(S, conc, nFe) -> albedo on `lam`; broadcasting over leading axes of
    S / conc / nFe, so a whole training set is evaluated in one call.
    """

    def __init__(self, lam_um, components: list[Component], q: float = kc.Q,
                 alpha_fe_um=None, d_um: float = kc.D_RIND_UM):
        self.lam = np.asarray(lam_um, dtype=float)
        self.components = components
        self.names = [c.name for c in components]
        self.q = q
        self.d = d_um
        self.alpha = np.zeros_like(self.lam) if alpha_fe_um is None else np.asarray(alpha_fe_um)
        self._fixed = {}
        for c in components:
            if not c.transparent:
                tau = np.broadcast_to(np.asarray(c.tau, dtype=float), self.lam.shape)
                self._fixed[c.name] = particle_rb_rf(tau, c.n)

    def particle_albedos(self, S, nFe=0.0):
        """rb, rf for every component: arrays (..., n_comp, n_lam)."""
        S = np.asarray(S, dtype=float)[..., None]
        nFe = np.asarray(nFe, dtype=float)[..., None]
        rbs, rfs = [], []
        for c in self.components:
            if c.transparent:
                tau = 4.0 * np.pi * c.k * S / self.lam + self.alpha * nFe * self.d   # A9 + A16
                rb, rf = particle_rb_rf(tau, c.n)
            else:
                rb, rf = self._fixed[c.name]
                shape = np.broadcast_shapes(S.shape[:-1], nFe.shape[:-1]) + self.lam.shape
                rb, rf = np.broadcast_to(rb, shape), np.broadcast_to(rf, shape)
            rbs.append(rb)
            rfs.append(rf)
        return np.stack(rbs, axis=-2), np.stack(rfs, axis=-2)

    def forward(self, S, conc, nFe=0.0):
        """conc: (..., n_comp) FRACTIONS (not required to sum to 1, as in the paper)."""
        rb, rf = self.particle_albedos(S, nFe)
        conc = np.asarray(conc, dtype=float)[..., None]
        rho_b = self.q * np.sum(conc * rb, axis=-2)                       # A12
        rho_f = self.q * np.sum(conc * rf, axis=-2) + 1.0 - self.q
        return albedo_from_rho(rho_b, rho_f)                              # A11


# --------------------------------------------------------------------------
# the lunar basis of Table 4 + gray component: the 8-parameter model of Sec. 4.1
# --------------------------------------------------------------------------
def lunar_basis_components(lam_um, a_gray: float = kc.A_GRAY, gray: bool = True,
                           agglutinate: np.ndarray | None = None) -> list[Component]:
    """Gray + CPX, OPX, OLV, PLG, ILM on `lam_um` (Table 4, Sec. 3.2).

    With `agglutinate` (albedo on the grid) the gray component is replaced by the
    agglutinate, treated as nontransparent (Sec. 2.2 step 2) -- Table 8.
    """
    comps = []
    if agglutinate is not None:
        # n for the agglutinate is not given (the paper fixes 2.0 only for ilmenite and the
        # gray component). With n = 2.0 its 0.54-um albedo (0.0405) lies below the darkest
        # value that index can represent (0.0431) and A2 has no solution; n = 1.6, the
        # paper's value for its other silicates, is used instead (D19).
        comps.append(Component("AGG", kc.N_TRANSPARENT,
                               tau=tau_from_albedo(agglutinate, kc.N_TRANSPARENT)))
    elif gray:
        tau_gr = float(tau_from_albedo(a_gray, kc.N_OPAQUE))
        comps.append(Component("Gr", kc.N_OPAQUE, tau=np.full(np.shape(lam_um), tau_gr)))
    for name in kc.MINERALS:
        b = kc.BASIS[name]
        lam, a = kc.relab_spectrum(b["spectrum"])
        a_grid = kc.resample(lam, a, lam_um)
        if name == "ILM":
            comps.append(Component(name, b["n"], tau=tau_from_albedo(a_grid, b["n"])))
        else:
            comps.append(Component(name, b["n"], k=k_from_albedo(a_grid, lam_um, b["n"], kc.S_LAB_UM)))
    return comps


try:
    from numba import njit as _njit
except ImportError:                                     # pragma: no cover
    _njit = None


def _lunar_kernel(P, k4, alpha, lam, rb_fix, rf_fix, Rb, Rf, Te, Ti, Ri, q, d):
    """Compiled twin of LunarModel.forward for 4 transparent + 2 fixed components.

    P (n, 8) = S, nFe, c_first, c_CPX, c_OPX, c_OLV, c_PLG, c_ILM (percent);
    fixed components are the gray/AGG (index 0) and ILM (index 1) in rb_fix/rf_fix.
    """
    n, L = P.shape[0], lam.shape[0]
    out = np.empty((n, L))
    for p in range(n):
        S = P[p, 0]
        nfe = P[p, 1]
        for l in range(L):
            sb = P[p, 2] * rb_fix[0, l] + P[p, 7] * rb_fix[1, l]
            sf = P[p, 2] * rf_fix[0, l] + P[p, 7] * rf_fix[1, l]
            for j in range(4):
                tau = 4.0 * np.pi * k4[j, l] * S / lam[l] + alpha[l] * nfe * d
                e = np.exp(-tau)
                H = 0.5 * Te * Ti * Ri * e * e / (1.0 - Ri * e)
                sb += P[p, 3 + j] * (Rb + H)
                sf += P[p, 3 + j] * (Rf + Te * Ti * e + H)
            rho_b = q * sb / 100.0
            rho_f = q * sf / 100.0 + 1.0 - q
            Pv = (1.0 + rho_b * rho_b - rho_f * rho_f) / (2.0 * rho_b)
            if rho_b <= 0.0 or Pv < 1.0 - 1e-9:
                out[p, l] = np.nan
            else:
                if Pv < 1.0:
                    Pv = 1.0
                out[p, l] = Pv - np.sqrt(Pv * Pv - 1.0)
    return out


if _njit is not None:
    _lunar_kernel = _njit(cache=True)(_lunar_kernel)


class LunarModel:
    """The 8-parameter model: S, nFe, c_Gr, c_CPX, c_OPX, c_OLV, c_PLG, c_ILM (Table 11 order).

    Concentrations in PERCENT at this interface, matching the paper's tables.
    `forward` is the numpy reference; `fast` is the compiled twin (step01 gate G6
    checks they agree), used where millions of evaluations are needed.
    """

    def __init__(self, lam_um=kc.M3_GRID_UM, a_gray: float = kc.A_GRAY,
                 agglutinate: np.ndarray | None = None, alpha_fe_um=None):
        self.lam = np.asarray(lam_um, dtype=float)
        if alpha_fe_um is None:
            alpha_fe_um = alpha_fe_primary(self.lam)
        self.mix = Mixture(self.lam, lunar_basis_components(self.lam, a_gray, agglutinate=agglutinate),
                           alpha_fe_um=alpha_fe_um)
        self.first = self.mix.names[0]      # "Gr" or "AGG"
        comps = {c.name: c for c in self.mix.components}
        self._k4 = np.stack([comps[m].k for m in kc.TRANSPARENT]).astype(np.float64)
        fixed = [self.first, "ILM"]
        self._rb_fix = np.stack([np.broadcast_to(self.mix._fixed[f][0], self.lam.shape) for f in fixed])
        self._rf_fix = np.stack([np.broadcast_to(self.mix._fixed[f][1], self.lam.shape) for f in fixed])
        self._alpha = np.asarray(self.mix.alpha, dtype=np.float64)
        f = Fresnel(kc.N_TRANSPARENT)
        self._fr = (f.Rb, f.Rf, f.Te, f.Ti, f.Ri)

    def forward(self, params) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        S, nFe = p[..., 0], p[..., 1]
        conc = p[..., 2:] / 100.0            # Gr/AGG, CPX, OPX, OLV, PLG, ILM
        return self.mix.forward(S, conc, nFe)

    def fast(self, params) -> np.ndarray:
        p = np.asarray(params, dtype=np.float64)
        flat = np.ascontiguousarray(p.reshape(-1, 8))
        out = _lunar_kernel(flat, self._k4, self._alpha, self.lam, self._rb_fix, self._rf_fix,
                            *self._fr, self.mix.q, self.mix.d)
        return out.reshape(p.shape[:-1] + (self.lam.size,))


def alpha_fe_primary(lam_um) -> np.ndarray:
    """alpha_Fe as selected by step01 (computed, or the digitized Fig. A1 if the
    computed curve failed the pre-registered +-15% check)."""
    choice = kc.TABLES / "step01_alpha_fe_choice.csv"
    if choice.exists():
        c = pd.read_csv(choice)
        if str(c["primary"].iloc[0]) == "digitized":
            dig = pd.read_csv(kc.DIGITIZED / "figA1_alpha_fe.csv")
            return np.interp(lam_um, dig["lam_um"], dig["alpha_fe"])
    return alpha_fe(lam_um)
