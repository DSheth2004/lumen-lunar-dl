"""
Vectorised Hapke forward model for the CATMIP likelihood.

Given an ensemble of (grain size, abundance) proposals, returns the mixture SSA
on a fixed wavelength grid.  Equations C1-C3 and E3-E9 of the paper; identical in
content to mcmc/Hapke_opt_forward.m + nkDtowi.m + wtperctof.m, but evaluated for
the whole ensemble at once.

With the internal scattering coefficient s = 0 (as both released implementations
hard-code) Eqs. E6/E7 collapse to

    r_i   = 0
    theta = exp(-alpha * <D>)

so the only grain-size-dependent term is <D>, and everything else can be
precomputed once per end-member.
"""
from __future__ import annotations

import numpy as np

import hapke


class MixtureForward:
    """Precomputes per-end-member optical terms on a fixed wavelength grid.

    Parameters
    ----------
    lam_um : array[Nlam]
    n : array[Nem] or array[Nem, Nlam]
        Real refractive index per end-member.
    k : array[Nem, Nlam]
        Imaginary index per end-member on lam_um.
    rho : array[Nem]
        Solid densities.  Units cancel in Eq. C2, so kg/m3 and g/cm3 agree.
    """

    def __init__(self, lam_um, n, k, rho):
        self.lam = np.asarray(lam_um, dtype=float)
        self.k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        self.n = np.broadcast_to(n[:, None], self.k.shape) if n.ndim == 1 else n
        self.rho = np.asarray(rho, dtype=float)
        self.nem, self.nlam = self.k.shape

        # Terms independent of grain size.
        self.alpha = hapke.alpha(self.k, self.lam[None, :])          # (Nem, Nlam)
        self.se = hapke.s_e(self.n, self.k)                          # (Nem, Nlam)
        self.si = hapke.s_i(self.n[:, 0])                            # (Nem,)
        # <D> = shape_factor * D, with the shape factor fixed by n (Eq. E9).
        n0 = self.n[:, 0]
        self.dshape = 2.0 / 3.0 * (n0 ** 2 - 1.0 / n0 * (n0 ** 2 - 1.0) ** 1.5)

        self.valid = np.isfinite(self.alpha) & np.isfinite(self.se)

    def endmember_ssa(self, D):
        """SSA per end-member for an ensemble of grain sizes.

        D : array[Nens, Nem] in microns (same unit as lam_um)
        returns array[Nens, Nem, Nlam]
        """
        D = np.asarray(D, dtype=float)
        dbar = D * self.dshape[None, :]                              # (Nens, Nem)
        th = np.exp(-self.alpha[None, :, :] * dbar[:, :, None])      # (Nens, Nem, Nlam)
        se = self.se[None, :, :]
        si = self.si[None, :, None]
        return se + (1.0 - se) * (1.0 - si) / (1.0 - si * th) * th

    def mixture_ssa(self, D, m, chunk=400):
        """Mixture SSA for an ensemble.

        D : array[Nens, Nem] grain sizes, microns
        m : array[Nens, Nem] mass abundances (sum to 1)
        returns array[Nens, Nlam]
        """
        D = np.atleast_2d(np.asarray(D, dtype=float))
        m = np.atleast_2d(np.asarray(m, dtype=float))
        out = np.empty((len(D), self.nlam))
        for lo in range(0, len(D), chunk):
            hi = min(lo + chunk, len(D))
            f = hapke.mass_to_cross_section(self.rho[None, :], D[lo:hi], m[lo:hi])
            ssa = self.endmember_ssa(D[lo:hi])
            out[lo:hi] = np.einsum("ni,nil->nl", f, ssa)
        return out

    def mixture_ssa_excluding(self, D, m, drop_names, names, chunk=400):
        """Mixture SSA with some end-members removed and the rest renormalised.

        Used for the paper's final step: forward-model the fresh spectrum
        "omitting the contribution of agglutinates that was estimated from the
        inverse model".
        """
        keep = np.array([nm not in set(drop_names) for nm in names])
        D = np.atleast_2d(np.asarray(D, dtype=float))[:, keep]
        m = np.atleast_2d(np.asarray(m, dtype=float))[:, keep]
        m = m / m.sum(axis=1, keepdims=True)
        sub = MixtureForward(self.lam, self.n[keep, 0], self.k[keep], self.rho[keep])
        return sub.mixture_ssa(D, m, chunk=chunk)


def make_log_likelihood(fwd: MixtureForward, data_ssa: np.ndarray, cinv: float):
    """Gaussian data log-likelihood, per mcmc/likelihood.m.

        L = -0.5 * res' * Cinv * res,  Cinv = 1/Cov  (a scalar in the released
                                                      code, Cov = 2e-4)

    NOTE, deviation: paper Eq. B2 describes a full 2Nd x 2Nd covariance matrix
    C_x. The released implementation uses a single scalar variance, i.e. i.i.d.
    Gaussian errors with sigma = sqrt(2e-4) = 0.0141 in SSA. We follow the code.
    """
    obs = np.asarray(data_ssa, dtype=float)
    good = np.isfinite(obs)

    def loglike(theta):
        theta = np.atleast_2d(theta)
        nem = fwd.nem
        model = fwd.mixture_ssa(theta[:, :nem], theta[:, nem:])
        res = model[:, good] - obs[None, good]
        bad = ~np.isfinite(res).all(axis=1)
        out = -0.5 * cinv * np.einsum("nl,nl->n", res, res)
        out[bad] = -np.inf
        return out

    return loglike
