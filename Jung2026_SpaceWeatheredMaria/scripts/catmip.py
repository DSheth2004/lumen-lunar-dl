"""
CATMIP -- Cascading Adaptive Transitional Metropolis In Parallel.

Minson, S. E., Simons, M., & Beck, J. L. (2013), Geophys. J. Int. 194, 1701.
Applied to Hapke unmixing by Lapotre, Ehlmann & Minson (2017), JGR Planets 122,
983, and modified for lunar mare basalts by Jung et al. (2026) Appendix B.

Python reimplementation of the authors' released MATLAB (reference/
space-weathering-main/mcmc/), which is the specification followed here.

Model vector theta = [D_1..D_N, m_1..m_N]: N grain sizes then N mass abundances.

The transitional structure is preserved exactly -- adaptive dbeta chosen so the
plausibility weights have coefficient of variation 1, multinomial resampling,
then a short Metropolis chain per particle -- but the per-particle chains are run
VECTORISED across the ensemble rather than in a parfor loop.  The chains are
independent given the stage, so this is the same algorithm, not an approximation.

PROPOSAL MODES
--------------
"released"  reproduces catmip_metro_sample_dirichlet.m verbatim, including

                Sm0 = Sm;  Sm = eye(Nparam);  Sm(1:2,1:2) = Sm0(1:2,1:2);

            i.e. the adaptive covariance is applied ONLY to the first two grain
            sizes and every other grain size random-walks with unit variance
            (1 um steps) regardless of the sample covariance.  That looks like
            residue from a two-end-member test case, but it is what produced the
            published results, so it is the default for reproduction.

"adaptive"  applies the full adaptive covariance to all N grain sizes, which is
            what the algorithm as described in Minson et al. (2013) calls for.

In both of those modes the abundance half of the proposal is an INDEPENDENCE
sampler: a fresh draw from the Dirichlet prior, per drchrnd in the released code.
It is a valid proposal (same distribution for every particle, so the released
code can omit the Hastings correction) but it does not scale with beta.  Once the
tempered likelihood is sharp, an independent prior draw is essentially never
accepted, acceptance decays to zero, the Metropolis stage stops diversifying, and
the ensemble degenerates onto a handful of duplicated particles.  When that
happens LLK - median(LLK) is identically zero, so catmip_calc_beta sees
COV(w) = 0 <= 1 and sets dbeta = 1 - beta, jumping straight to the target
posterior.  This is reproduced faithfully here and is measured by
`unique_fraction` in the stage history.

"rw"        adds the missing piece: a symmetric-in-spirit random walk ON THE
            SIMPLEX, m' ~ Dirichlet(conc * m), carrying the Hastings correction
            log q(m|m') - log q(m'|m) that such a proposal requires, with `conc`
            adapted alongside Cm^2 to hold acceptance near 0.25.  This is the
            mode to use when the posterior width itself is the quantity of
            interest, since the released modes report a degenerate ensemble.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

# --------------------------------------------------------------------------
# Constants from the released catmip_dirichlet.m
# --------------------------------------------------------------------------
BTOL = 0.005          # stop when 1 - beta < BTOL
CMSQR0 = 0.1 * 0.1    # initial proposal scale factor
DBETA0 = 5.0e-5       # initial guess for the beta increment
COV_TARGET = 1.0      # target coefficient of variation of the weights


def _cov_of_weights(llk: np.ndarray, dbeta: float) -> float:
    """Coefficient of variation of exp(llk*dbeta), computed stably."""
    x = llk * dbeta
    x = x - x.max()
    w = np.exp(x)
    mean = w.mean()
    if mean <= 0 or not np.isfinite(mean):
        return np.inf
    return float(w.std() / mean)


def _solve_dbeta(llk: np.ndarray, beta: float, dbeta: float) -> float:
    """Choose the beta increment so COV(weights) == COV_TARGET.

    Mirrors catmip_calc_beta.m: centre the log-likelihoods, test whether the
    whole remaining interval is acceptable, otherwise bracket and bisect in
    log-space.
    """
    llk = llk - np.median(llk)
    remaining = 1.0 - beta
    if _cov_of_weights(llk, remaining) <= COV_TARGET:
        return remaining

    lo, hi = 1e-12, max(dbeta, 1e-12)
    n = 0
    while _cov_of_weights(llk, hi) < COV_TARGET and hi < remaining and n < 200:
        hi *= 2.0
        n += 1
    hi = min(hi, remaining)

    for _ in range(200):                       # bisection in log(dbeta)
        mid = np.sqrt(lo * hi)
        if _cov_of_weights(llk, mid) < COV_TARGET:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-10:
            break
    return float(min(max(hi, 1e-12), remaining))


def _normalised_weights(llk: np.ndarray, dbeta: float) -> np.ndarray:
    x = llk * dbeta
    x = x - x.max()
    w = np.exp(x)
    return w / w.sum()


def dirichlet_logpdf_unnorm(m: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """sum_i (alpha_i - 1) * log(m_i), per llk_prior.m (normaliser dropped)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum((alpha - 1.0) * np.log(m), axis=-1)


def dirichlet_logpdf(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Full log Dirichlet density, needed for the Hastings correction."""
    x = np.clip(x, 1e-300, None)
    return (gammaln(alpha.sum(axis=-1))
            - gammaln(alpha).sum(axis=-1)
            + np.sum((alpha - 1.0) * np.log(x), axis=-1))


class Catmip:
    """CATMIP sampler for the 2N-parameter Hapke unmixing problem.

    Parameters
    ----------
    log_likelihood : callable (theta[N_ens, 2*Nem]) -> array[N_ens]
        Data log-likelihood.  Must return -inf for models it cannot evaluate.
    n_endmembers : int
    prior_alpha : array[Nem]
        Dirichlet concentration vector (component_means * alpha0).
    size_low, size_high : float
        Uniform grain-size prior bounds, microns.
    n_samples : int
        Ensemble size (Nm in load_data.m; released value 2000).
    n_steps : int
        Metropolis chain length per cooling step (Nsteps; released value 10).
    proposal : {"released", "adaptive", "rw"}
    """

    def __init__(self, log_likelihood, n_endmembers, prior_alpha,
                 size_low=0.0, size_high=990.0, n_samples=2000, n_steps=10,
                 proposal="released", seed=0, max_stages=200, verbose=True,
                 rw_concentration=400.0, target_accept=0.25):
        self.loglike = log_likelihood
        self.nem = int(n_endmembers)
        self.nparam = 2 * self.nem
        self.alpha = np.asarray(prior_alpha, dtype=float)
        self.size_low = float(size_low)
        self.size_high = float(size_high)
        self.n = int(n_samples)
        self.nsteps = int(n_steps)
        self.proposal = proposal
        self.rng = np.random.default_rng(seed)
        self.max_stages = int(max_stages)
        self.verbose = verbose
        self.conc = float(rw_concentration)
        self.target_accept = float(target_accept)
        self.history: list[dict] = []

    # ---- prior ----------------------------------------------------------
    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        D = theta[:, :self.nem]
        m = theta[:, self.nem:]
        out = np.zeros(len(theta))
        inside = np.all((D >= self.size_low) & (D <= self.size_high), axis=1)
        # uniform density is constant inside the box; -inf outside
        out[~inside] = -np.inf
        bad_m = np.any(m <= 0.0, axis=1) | (np.abs(m.sum(axis=1) - 1.0) > 1e-9)
        out[bad_m] = -np.inf
        ok = inside & ~bad_m
        out[ok] += dirichlet_logpdf_unnorm(m[ok], self.alpha)
        return out

    def sample_prior(self, n: int) -> np.ndarray:
        D = self.rng.uniform(self.size_low, self.size_high, size=(n, self.nem))
        m = self.rng.dirichlet(self.alpha, size=n)
        return np.hstack([D, m])

    # ---- main loop ------------------------------------------------------
    def run(self):
        theta = self.sample_prior(self.n)
        llk = self.loglike(theta)
        lpr = self.log_prior(theta)

        beta = 0.0
        dbeta = DBETA0
        cmsqr = CMSQR0
        stage = 0

        while True:
            stage += 1
            finite = np.isfinite(llk)
            if not finite.any():
                raise RuntimeError("CATMIP: no finite likelihoods at stage %d" % stage)
            work = np.where(finite, llk, -np.inf)
            safe = np.where(finite, llk, np.min(llk[finite]))

            dbeta = _solve_dbeta(safe, beta, dbeta)
            w = _normalised_weights(safe - np.median(safe), dbeta)
            w = np.where(finite, w, 0.0)
            w = w / w.sum()
            beta = min(beta + dbeta, 1.0)

            # multinomial resampling (histc(rand, cumsum(w)) in the original)
            idx = self.rng.choice(self.n, size=self.n, p=w)
            theta = theta[idx].copy()
            llk = llk[idx].copy()
            lpr = lpr[idx].copy()

            # adaptive proposal covariance over the grain-size block
            Sm = np.cov(theta[:, :self.nem].T, aweights=None) * cmsqr
            Sm = np.atleast_2d(Sm)
            Sm = 0.5 * (Sm + Sm.T)

            theta, llk, lpr, acc, rej = self._metropolis(theta, llk, lpr, beta, Sm)

            acc_ratio = acc / max(acc + rej, 1)
            kc = (8.0 * acc_ratio + 1.0) / 9.0
            cmsqr = float(np.clip(kc, 0.1, 1.0) ** 2)
            if self.proposal == "rw":
                # higher concentration => smaller step. Adapt to target_accept.
                self.conc = float(np.clip(
                    self.conc * np.exp(2.0 * (self.target_accept - acc_ratio)),
                    5.0, 1e7))

            uniq = len(np.unique(theta, axis=0)) / self.n
            self.history.append({"stage": stage, "beta": beta, "dbeta": dbeta,
                                 "cmsqr": cmsqr, "concentration": self.conc,
                                 "accept_ratio": acc_ratio,
                                 "unique_fraction": uniq,
                                 "max_llk": float(np.nanmax(llk))})
            if self.verbose:
                print("    stage %3d  beta = %.6f  acc = %.3f  uniq = %.3f  maxLLK = %.2f"
                      % (stage, beta, acc_ratio, uniq, np.nanmax(llk)))

            if (1.0 - beta) < BTOL or stage >= self.max_stages:
                break

        self.theta = theta
        self.llk = llk
        self.log_prior_final = lpr
        return theta, llk

    def _metropolis(self, theta, llk, lpr, beta, Sm):
        n, nem = self.n, self.nem
        acc = rej = 0

        if self.proposal == "released":
            # Sm = eye(Nparam); Sm(1:2,1:2) = Sm0(1:2,1:2)  -- see class docstring
            cov = np.eye(nem)
            cov[:2, :2] = Sm[:2, :2]
        elif self.proposal in ("adaptive", "rw"):
            cov = Sm + 1e-12 * np.eye(nem)
        else:
            raise ValueError("unknown proposal mode %r" % self.proposal)

        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(cov)
            chol = evecs @ np.diag(np.sqrt(np.clip(evals, 1e-12, None)))

        post = lpr + beta * llk

        eps = 1e-12
        for _ in range(self.nsteps - 1):
            step = self.rng.standard_normal((n, nem)) @ chol.T
            cand = theta.copy()
            cand[:, :nem] = theta[:, :nem] + step

            if self.proposal == "rw":
                # m' ~ Dirichlet(conc * m): a random walk on the simplex.
                a_fwd = self.conc * np.clip(theta[:, nem:], eps, None)
                prop = self.rng.gamma(a_fwd)
                prop = np.clip(prop, eps, None)
                prop /= prop.sum(axis=1, keepdims=True)
                cand[:, nem:] = prop
                # Hastings: log q(m|m') - log q(m'|m)   (asymmetric proposal)
                a_rev = self.conc * np.clip(prop, eps, None)
                hastings = (dirichlet_logpdf(theta[:, nem:], a_rev)
                            - dirichlet_logpdf(prop, a_fwd))
            else:
                cand[:, nem:] = self.rng.dirichlet(self.alpha, size=n)
                hastings = np.zeros(n)

            cand_lpr = self.log_prior(cand)
            ok = np.isfinite(cand_lpr)
            cand_llk = np.full(n, -np.inf)
            if ok.any():
                cand_llk[ok] = self.loglike(cand[ok])
            good = ok & np.isfinite(cand_llk) & np.isfinite(hastings)

            cand_post = np.where(good, cand_lpr + beta * cand_llk, -np.inf)
            u = self.rng.random(n)
            take = good & (u <= np.exp(np.clip(cand_post - post + hastings, -700, 0)))

            theta[take] = cand[take]
            llk[take] = cand_llk[take]
            lpr[take] = cand_lpr[take]
            post[take] = cand_post[take]
            acc += int(take.sum())
            rej += int(n - take.sum())

        return theta, llk, lpr, acc, rej

    # ---- summaries ------------------------------------------------------
    def map_model(self):
        """Maximum a posteriori: the accepted model with the highest data llk.

        Matches MAIN.m, which selects on LLK(2,:,end) -- the data likelihood.
        """
        i = int(np.nanargmax(self.llk))
        return self.theta[i], i

    def mode_model(self, bins=60, kde=True):
        """Per-parameter mode of the final ensemble (the paper's "mode" estimate).

        The paper defines this as "the most likely value for each of the 18
        parameters, which do not necessarily correspond to the MAP" -- i.e. the
        marginal mode of each parameter taken independently.  A vector built from
        independent marginals is not itself constrained to be a good model when
        the parameters are correlated, which is exactly what we observe on a
        well-mixed ensemble.  A Gaussian KDE is used instead of a raw histogram
        so the estimate does not jump between adjacent bins.
        """
        out = np.empty(self.nparam)
        for j in range(self.nparam):
            x = self.theta[:, j]
            if kde and np.ptp(x) > 0:
                try:
                    from scipy.stats import gaussian_kde
                    grid = np.linspace(x.min(), x.max(), 512)
                    out[j] = float(grid[np.argmax(gaussian_kde(x)(grid))])
                    continue
                except Exception:
                    pass
            hist, edges = np.histogram(x, bins=bins)
            k = int(np.argmax(hist))
            out[j] = 0.5 * (edges[k] + edges[k + 1])
        return out
