"""
Hapke radiative transfer core -- Jung et al. (2026) Appendices C, D and E.

Equation numbers refer to the paper.  Where the paper's printed equation and the
authors' released code disagree, the code is followed and the disagreement is
recorded in DEVIATIONS below, because the released code is what produced the
published numbers.

DEVIATIONS from the paper text
------------------------------
E9 (mean free path).  The paper prints

        <D> = (2/3)[ n^2 + (1/n^2)(n^2 - 1)^(3/2) ] D

  but BOTH released implementations -- hapkeRTM.py::_D_mean and mcmc/nkDtowi.m --
  use

        <D> = (2/3)[ n^2 - (1/n)(n^2 - 1)^(3/2) ] D

  which is the standard Hapke (1993, eq. 6.20) form.  Two independent files
  agreeing against the printed equation makes this a typesetting error in the
  paper, not a modelling choice.  We use the code form; `d_mean` exposes a
  `paper_form` switch so the audit can quantify the difference.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# Internal scattering coefficient. Both released implementations hard-code s = 0.
S_SCATTER = 0.0


# --------------------------------------------------------------------------
# Appendix D -- reflectance <-> single scattering albedo
# --------------------------------------------------------------------------
def chandrasekhar_h(ssa, x):
    """Chandrasekhar H function, Hapke's approximation (Eq. D3).

    H(x) = [ 1 - ssa*x*( r0 + (1 - 2*r0*x)/2 * ln((1+x)/x) ) ]^-1
    with r0 = (1-gamma)/(1+gamma), gamma = sqrt(1-ssa).
    """
    ssa = np.asarray(ssa, dtype=float)
    gamma = np.sqrt(1.0 - ssa)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    return 1.0 / (1.0 - ssa * x * (r0 + (1.0 - 2.0 * r0 * x) / 2.0 * np.log((1.0 + x) / x)))


def ssa_to_reflectance(ssa, mu0, mu):
    """Radiance coefficient from SSA (Eq. D4), with B = 0 and P = 1.

        r = ssa / (4*(mu0 + mu)) * H(ssa, mu0) * H(ssa, mu)
    """
    ssa = np.asarray(ssa, dtype=float)
    return ssa / 4.0 / (mu0 + mu) * chandrasekhar_h(ssa, mu0) * chandrasekhar_h(ssa, mu)


def reflectance_to_ssa(refl, mu0, mu, lo=1e-6, hi=1.0 - 1e-12, xtol=1e-12):
    """Invert Eq. D4 for SSA by bisection on f(ssa) = r_Hapke(ssa) - r (Eq. D1).

    The authors bracket on [0.1, 1]; we widen the lower bound to 1e-6 so that
    dark spectra (M3 shadowed pixels, ilmenite shortward of 1 um) do not fall
    outside the bracket and silently become NaN.  Within [0.1, 1] the two agree
    to bisection tolerance -- checked by the audit.
    """
    refl = np.atleast_1d(np.asarray(refl, dtype=float))
    out = np.full(refl.shape, np.nan)
    f_lo = ssa_to_reflectance(lo, mu0, mu)
    f_hi = ssa_to_reflectance(hi, mu0, mu)
    for i, r in enumerate(refl):
        if not np.isfinite(r) or r <= f_lo or r >= f_hi:
            continue
        out[i] = brentq(lambda w: ssa_to_reflectance(w, mu0, mu) - r, lo, hi, xtol=xtol)
    return out


_SSA_TABLE: dict = {}


def reflectance_to_ssa_fast(refl, mu0, mu, n_table=200001):
    """Vectorised inverse of Eq. D4, for per-pixel work.

    r(ssa) is smooth and strictly increasing on (0, 1) -- proved by the
    monotonicity self-test in step01 -- so the inverse can be tabulated once and
    evaluated by interpolation instead of running a bisection per band.  Needed
    for image cubes, where the bisection route would mean millions of root
    solves.  Agreement with `reflectance_to_ssa` is gated in step01.
    """
    key = (mu0, mu, n_table)
    if key not in _SSA_TABLE:
        w = np.linspace(1e-9, 1.0 - 1e-12, n_table)
        _SSA_TABLE[key] = (ssa_to_reflectance(w, mu0, mu), w)
    r_tab, w_tab = _SSA_TABLE[key]

    refl = np.asarray(refl, dtype=float)
    out = np.interp(refl, r_tab, w_tab, left=np.nan, right=np.nan)
    return np.where(np.isfinite(refl) & (refl > r_tab[0]) & (refl < r_tab[-1]), out, np.nan)


# --------------------------------------------------------------------------
# Appendix E -- optical constants -> single scattering albedo
# --------------------------------------------------------------------------
def s_e(n, k):
    """Surface reflection coefficient for externally incident light (Eq. E4)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    return ((n - 1.0) ** 2 + k ** 2) / ((n + 1.0) ** 2 + k ** 2) + 0.05


def s_i(n):
    """Surface reflection coefficient for internally scattered light (Eq. E5)."""
    n = np.asarray(n, dtype=float)
    return 1.014 - 4.0 / (n * (n + 1.0) ** 2)


def alpha(k, lam):
    """Internal absorption coefficient (Eq. E8): alpha = 4*pi*k/lambda."""
    return 4.0 * np.pi * np.asarray(k, dtype=float) / np.asarray(lam, dtype=float)


def r_internal(a, s=S_SCATTER):
    """Internal diffusive reflectance inside a particle (Eq. E7).

    With s = 0 this is identically 0; kept in full form so a non-zero internal
    scattering coefficient can be explored.
    """
    a = np.asarray(a, dtype=float)
    q = np.sqrt(a / (a + s)) if s else np.ones_like(a)
    return (1.0 - q) / (1.0 + q)


def d_mean(n, D, paper_form=False):
    """Mean free path of a photon in a grain of diameter D (Eq. E9).

    See DEVIATIONS in the module docstring: default is the released-code form.
    """
    n = float(np.ravel(n)[0])
    if paper_form:
        return 2.0 / 3.0 * (n ** 2 + 1.0 / n ** 2 * (n ** 2 - 1.0) ** 1.5) * D
    return 2.0 / 3.0 * (n ** 2 - 1.0 / n * (n ** 2 - 1.0) ** 1.5) * D


def theta_internal(ri, a, d_bar, s=S_SCATTER):
    """Particle internal transmission coefficient (Eq. E6)."""
    a = np.asarray(a, dtype=float)
    ex = np.exp(-np.sqrt(a * (a + s)) * d_bar)
    return (ri + ex) / (1.0 + ri * ex)


def nk_to_ssa(n, k, D, lam, s=S_SCATTER, paper_form=False):
    """SSA of one end-member from its optical constants and grain size (Eq. E3).

        ssa = Se + (1 - Se)*(1 - Si)/(1 - Si*theta) * theta

    n, k, lam are arrays of equal length; D is a scalar in the same length unit
    as lam.  Mirrors mcmc/nkDtowi.m, which evaluates <D> with n[0].
    """
    a = alpha(k, lam)
    ri = r_internal(a, s)
    th = theta_internal(ri, a, d_mean(n, D, paper_form=paper_form), s)
    se = s_e(n, k)
    si = s_i(n)
    return se + (1.0 - se) * (1.0 - si) / (1.0 - si * th) * th


def reflectance_to_k(refl, lam, n, D, mu0, mu, lo=1e-9, hi=1.0, xtol=1e-12):
    """Solve Eq. E2 for k(lambda) by bisection (Appendix E).

    f(k) = r_Hapke( nk_to_ssa(n, k, D, lam) ) - r_measured

    The authors bracket k on [1e-7, 1]; we go to 1e-9 so that very transparent
    plagioclase continuum bands still bracket a root instead of returning NaN.
    """
    refl = np.atleast_1d(np.asarray(refl, dtype=float))
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    n = np.broadcast_to(np.asarray(n, dtype=float), refl.shape)
    out = np.full(refl.shape, np.nan)
    for i, r in enumerate(refl):
        if not np.isfinite(r):
            continue

        def f(kk, i=i, r=r):
            w = nk_to_ssa(np.array([n[i]]), np.array([kk]), D, np.array([lam[i]]))
            return float(ssa_to_reflectance(w, mu0, mu)[0]) - r

        try:
            f_lo, f_hi = f(lo), f(hi)
        except (ValueError, FloatingPointError):
            continue
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            continue
        out[i] = brentq(f, lo, hi, xtol=xtol)
    return out


# --------------------------------------------------------------------------
# Appendix C -- intimate mixture
# --------------------------------------------------------------------------
def mass_to_cross_section(rho, D, wt):
    """Fractional relative cross-sections from mass abundance (Eqs. C2, C3).

        sigma_i = m_i / (rho_i * D_i);   f_i = sigma_i / sum(sigma)

    Vectorised over a leading ensemble axis, so an (Nsample, Nendmember) block
    of proposals can be converted in one call.  Mirrors mcmc/wtperctof.m.
    """
    rho = np.asarray(rho, dtype=float)
    D = np.asarray(D, dtype=float)
    wt = np.asarray(wt, dtype=float)
    sig = wt / rho / D
    return sig / sig.sum(axis=-1, keepdims=True)


def mix_ssa(f, ssa_stack):
    """SSA of the mixture (Eq. C1): omega_mix = sum_i f_i * omega_i.

    f          : (..., N) fractional cross-sections
    ssa_stack  : (N, Nlam) per-end-member SSA on a common wavelength grid
    returns    : (..., Nlam)
    """
    return np.asarray(f, dtype=float) @ np.asarray(ssa_stack, dtype=float)
