"""
The paper's numerical-optimization inverse (Sec. 2.3), used for Tables 1, 5-10.

  * objective: sigma, the RMS deviation of Eq. (1);
  * Nelder-Mead simplex with a user-set initial simplex ("initial facet lengths");
  * every free parameter normalised to a common range by a scaling factor;
  * an exhaustive search over the starting grain size S0 = 1, 2, ..., 100 um
    (to 500 um for young areas) with a small initial S facet (1 um), keeping the
    run with the lowest sigma -- the "narrow corridor of S" that avoids local minima.

What the paper leaves unstated, fixed here (PAPER_NOTES.md D5-D7):
  * concentrations, S and nFe are bounded below by 0 (the printed fits contain
    exact 0.00 values, which only a bound produces);
  * starting concentrations and facets for the non-S parameters (below);
  * convergence tolerances (below).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

S0_GRID = np.arange(1.0, 101.0, 1.0)        # um, step 1 um (Sec. 2.3)
S_FACET = 1.0                               # um
XATOL = 1e-4                                # in normalised parameter units
FATOL = 1e-8                                # on sigma
MAXFEV = 3000


class Problem:
    """A fit: forward(theta) -> spectrum, observed spectrum, parameter scaling.

    theta is the physical parameter vector; index `s_index` is the grain size.
    `scale` maps physical -> normalised (theta / scale), so one simplex step means
    a comparable change in every parameter.
    """

    def __init__(self, forward, observed, scale, lower, upper, start, facet, s_index=0):
        self.forward = forward
        self.obs = np.asarray(observed, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.start = np.asarray(start, dtype=float)
        self.facet = np.asarray(facet, dtype=float)
        self.s = s_index
        self.nfev = 0

    def sigma(self, theta) -> float:
        self.nfev += 1
        m = self.forward(theta)
        d = m - self.obs
        v = float(np.sqrt(np.mean(d * d)))
        return v if np.isfinite(v) else 1e3

    def run_from(self, theta0) -> tuple[np.ndarray, float]:
        z0 = theta0 / self.scale
        steps = self.facet / self.scale
        simplex = np.vstack([z0] + [z0 + np.eye(z0.size)[i] * steps[i] for i in range(z0.size)])
        res = minimize(lambda z: self.sigma(z * self.scale), z0, method="Nelder-Mead",
                       bounds=list(zip(self.lower / self.scale, self.upper / self.scale)),
                       options={"initial_simplex": simplex, "xatol": XATOL, "fatol": FATOL,
                                "maxfev": MAXFEV})
        return res.x * self.scale, float(res.fun)

    def corridor(self, s0_grid=S0_GRID, keep_all=False):
        """Exhaustive search over the starting grain size; best sigma wins."""
        best, allrun = None, []
        t = time.perf_counter()
        for s0 in s0_grid:
            theta0 = self.start.copy()
            theta0[self.s] = s0
            th, f = self.run_from(theta0)
            allrun.append((s0, f, th))
            if best is None or f < best[1]:
                best = (th, f, s0)
        out = {"theta": best[0], "sigma": best[1], "s0": best[2], "nfev": self.nfev,
               "seconds": time.perf_counter() - t}
        if keep_all:
            out["runs"] = allrun
        return out
