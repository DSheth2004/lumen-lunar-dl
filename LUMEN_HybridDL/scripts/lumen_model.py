"""
LUMEN network, losses, SSA-space mixup and the training loops (METHOD_NOTES.md D8-D14).

  input   (2, 71): standardised log reflectance + convex-hull continuum-removed R, the
          standardisation taken from the synthetic set only (no real-data statistics);
  body    compact 1-D CNN with GroupNorm (fine-tuning batches are small);
  heads   Dirichlet over the 7-part modal composition; heteroscedastic Gaussians (beta-NLL)
          for FeO, TiO2, Al2O3, CaO, MgO and log10(Is/FeO); auxiliary Gaussians for
          log10 grain size and log10 npFe0 (pretraining only).

Everything here is deterministic given the seeds passed in.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lumen_common as lc
import physics as ph

N_COMP = len(lc.COMP)
SCALARS = lc.CHEM + [lc.MAT]                         # 5 oxides + logIsFeO
AUX = ["logD", "log10_npfe"]
# fixed a-priori target scaling (general lunar ranges, not fitted to any data set)
SCALAR_CENTRE = np.array([10.0, 3.0, 18.0, 12.0, 9.0, 1.6], np.float32)
SCALAR_SCALE = np.array([5.0, 3.0, 7.0, 3.0, 4.0, 0.5], np.float32)
AUX_CENTRE = np.array([1.4, -0.7], np.float32)
AUX_SCALE = np.array([0.5, 0.6], np.float32)
COMP_EPS = 0.005                                     # Dirichlet label smoothing (fraction units)
BETA_NLL = 0.5


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def _block(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.GELU())


class LumenNet(nn.Module):
    def __init__(self, n_bands: int = 71, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            _block(2, 32, 7), _block(32, 32, 5), nn.MaxPool1d(2),
            _block(32, 64, 5), _block(64, 64, 3), nn.MaxPool1d(2),
            _block(64, 64, 3),
        )
        L = n_bands // 2 // 2
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(64 * L, 256), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU())
        self.comp = nn.Linear(128, N_COMP)
        self.scal = nn.Linear(128, 2 * len(SCALARS))
        self.aux = nn.Linear(128, 2 * len(AUX))

    def forward(self, x):
        h = self.trunk(self.body(x))
        alpha = F.softplus(self.comp(h)) + 0.05
        alpha = alpha.clamp(max=1e4)
        s = self.scal(h)
        a = self.aux(h)
        nS, nA = len(SCALARS), len(AUX)
        return {"alpha": alpha,
                "mu": s[:, :nS], "logvar": s[:, nS:].clamp(-10, 6),
                "aux_mu": a[:, :nA], "aux_logvar": a[:, nA:].clamp(-10, 6)}


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def dirichlet_nll(alpha, y):
    """y: proportions (rows sum to 1), smoothed away from 0. Mean NLL per part."""
    y = (y + COMP_EPS) / (1.0 + N_COMP * COMP_EPS)
    a0 = alpha.sum(1)
    ll = torch.lgamma(a0) - torch.lgamma(alpha).sum(1) + ((alpha - 1) * torch.log(y)).sum(1)
    return -ll / N_COMP


def beta_nll(mu, logvar, y, mask, beta=BETA_NLL):
    var = logvar.exp()
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)
    w = var.detach() ** beta
    l = nll * w * mask
    return l.sum() / mask.sum().clamp(min=1.0)


def loss_fn(out, batch, comp_l1: float, aux_weight: float):
    cm = batch["comp_mask"]
    if cm.sum() > 0:
        a = out["alpha"][cm]
        y = batch["comp"][cm]
        l_comp = dirichlet_nll(a, y).mean()
        if comp_l1 > 0:
            l_comp = l_comp + comp_l1 * (a / a.sum(1, keepdim=True) - y).abs().mean() * 10.0
    else:
        l_comp = out["alpha"].sum() * 0.0
    l_s = beta_nll(out["mu"], out["logvar"], batch["scal"], batch["scal_mask"])
    loss = l_comp + l_s
    if aux_weight > 0 and "aux" in batch:
        loss = loss + aux_weight * beta_nll(out["aux_mu"], out["aux_logvar"], batch["aux"],
                                            torch.ones_like(batch["aux"]))
    return loss


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------
class FeatureScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __call__(self, F_):
        return ((F_ - self.mean) / self.std).astype(np.float32)


def scale_scalars(Y):
    return ((Y - SCALAR_CENTRE) / SCALAR_SCALE).astype(np.float32)


def unscale_scalars(mu, var=None):
    m = mu * SCALAR_SCALE + SCALAR_CENTRE
    return m if var is None else (m, var * SCALAR_SCALE ** 2)


def output_randomisation(R, rng, lam):
    """The simulator's calibration / photometric / noise randomisation (D11)."""
    n = len(R)
    x = lam - 1.5
    tilt = 1 + rng.normal(0, 0.03, (n, 1)) * x + rng.normal(0, 0.02, (n, 1)) * x ** 2
    scale = rng.uniform(0.93, 1.07, (n, 1))
    noise = rng.normal(0, 1, R.shape) * rng.uniform(0.002, 0.008, (n, 1))
    return np.clip(R * tilt * scale * (1 + noise), 1e-4, None)


def ssa_mixup(R, comp, chem, log_isfeo, d_nom, n, rng):
    """SSA-space mixtures of 2-3 training spectra (D10).

    Mass weights m ~ Dirichlet(1); spectra mix in single-scattering albedo with
    cross-section weights m_i / D_i (nominal fraction size); labels mix exactly:
    composition by volume (equal grain density assumed, so by m), chemistry by mass,
    Is/FeO_mix = sum m_i Is_i / sum m_i FeO_i with Is_i = (Is/FeO)_i FeO_i.
    Composition is labelled only when every source has a measured composition."""
    k = np.where(rng.uniform(0, 1, n) < 0.5, 2, 3)
    src = np.full((n, 3), -1)
    for i in range(n):
        src[i, :k[i]] = rng.choice(len(R), size=k[i], replace=False)
    m = np.zeros((n, 3))
    for kk in (2, 3):
        sel = k == kk
        m[sel, :kk] = rng.dirichlet(np.ones(kk), size=int(sel.sum()))
    s = np.where(src >= 0, src, 0)
    W = ph.reflectance_to_ssa(R)[s]                              # (n, 3, L)
    xs = np.where(src >= 0, m / d_nom[s], 0.0)
    xs = xs / xs.sum(1, keepdims=True)
    R_mix = ph.ssa_to_reflectance(np.einsum("nk,nkl->nl", xs, W))
    comp_mix = np.einsum("nk,nkc->nc", m, np.nan_to_num(comp[s]))
    comp_ok = ~np.any((src >= 0) & np.isnan(comp[s]).any(-1), axis=1)
    comp_mix[~comp_ok] = np.nan
    chem_mix = np.einsum("nk,nkc->nc", m, chem[s])
    feo = chem[s][:, :, 0]
    is_ = (10.0 ** log_isfeo[s]) * feo
    isfeo_mix = (m * is_).sum(1) / (m * feo).sum(1)
    return R_mix, comp_mix, chem_mix, np.log10(isfeo_mix)


def to_tensor_batch(Xf, comp, scal, scal_mask, aux=None):
    comp_mask = ~np.isnan(comp).any(1)
    b = {"x": torch.from_numpy(Xf),
         "comp": torch.from_numpy(np.nan_to_num(comp / 100.0).astype(np.float32)),
         "comp_mask": torch.from_numpy(comp_mask),
         "scal": torch.from_numpy(np.nan_to_num(scal).astype(np.float32)),
         "scal_mask": torch.from_numpy(scal_mask.astype(np.float32))}
    if aux is not None:
        b["aux"] = torch.from_numpy(aux.astype(np.float32))
    return b


# --------------------------------------------------------------------------
# pretraining on the synthetic set
# --------------------------------------------------------------------------
def synthetic_targets(syn, idx):
    comp = syn["comp"][idx].astype(np.float32)
    scal = np.column_stack([syn["chem"][idx], syn["logIsFeO"][idx]]).astype(np.float32)
    aux = np.column_stack([syn["logD"][idx], np.log10(np.clip(syn["npfe_wt"][idx], 1e-3, None))])
    aux = ((aux - AUX_CENTRE) / AUX_SCALE).astype(np.float32)
    return comp, scale_scalars(scal), aux


def pretrain(feat, syn, train_idx, val_idx, seed, epochs, batch=512, lr=2e-3, wd=1e-4,
             comp_l1=0.0, aux_weight=0.5, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LumenNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = len(train_idx) // batch
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    comp, scal, aux = synthetic_targets(syn, np.arange(len(syn["R"])))
    ones = np.ones_like(scal)
    hist = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(train_idx)
        tot = 0.0
        for s in range(steps_per_epoch):
            j = np.sort(perm[s * batch:(s + 1) * batch])
            b = to_tensor_batch(np.asarray(feat[j]), comp[j], scal[j], ones[j], aux[j])
            out = model(b["x"])
            loss = loss_fn(out, b, comp_l1, aux_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        ev = evaluate_synthetic(model, feat, syn, val_idx)
        ev.update({"epoch": ep + 1, "train_loss": tot / steps_per_epoch, "seconds": time.time() - t0})
        hist.append(ev)
        log("    seed %d epoch %2d  loss %.3f  val comp MAE %.2f  FeO %.2f  TiO2 %.2f  logIsFeO %.3f  (%.0f s)"
            % (seed, ev["epoch"], ev["train_loss"], ev["comp_mae"], ev["FeO_mae"], ev["TiO2_mae"],
               ev["logIsFeO_mae"], ev["seconds"]))
    return model, hist


@torch.no_grad()
def raw_predict(model, Xf, batch=4096):
    model.eval()
    outs = []
    for s in range(0, len(Xf), batch):
        o = model(torch.from_numpy(np.asarray(Xf[s:s + batch])))
        outs.append({k: v.numpy() for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def member_outputs(model, Xf):
    """Per-member predictive moments in physical units."""
    o = raw_predict(model, Xf)
    a = o["alpha"].astype(np.float64)
    a0 = a.sum(1, keepdims=True)
    cm = a / a0
    cv = a * (a0 - a) / (a0 ** 2 * (a0 + 1))
    mu, var = unscale_scalars(o["mu"].astype(np.float64), np.exp(o["logvar"].astype(np.float64)))
    return {"comp_mean": 100 * cm, "comp_var": 1e4 * cv, "scal_mean": mu, "scal_var": var}


def ensemble(members_out):
    """Mixture moments: mean of means; variance = mean of variances + variance of means."""
    keys = [("comp_mean", "comp_var"), ("scal_mean", "scal_var")]
    res = {}
    for km, kv in keys:
        M = np.stack([m[km] for m in members_out])
        V = np.stack([m[kv] for m in members_out])
        res[km] = M.mean(0)
        res[kv] = V.mean(0) + M.var(0)
        res[km + "_members"] = M
    return res


def evaluate_synthetic(model, feat, syn, idx):
    o = member_outputs(model, feat[idx])
    comp = syn["comp"][idx]
    scal = np.column_stack([syn["chem"][idx], syn["logIsFeO"][idx]])
    ev = {"comp_mae": float(np.abs(o["comp_mean"] - comp).mean())}
    for j, nm in enumerate(SCALARS):
        ev[nm + "_mae"] = float(np.abs(o["scal_mean"][:, j] - scal[:, j]).mean())
    for j, nm in enumerate(lc.COMP):
        ev[nm + "_mae"] = float(np.abs(o["comp_mean"][:, j] - comp[:, j]).mean())
    return ev


# --------------------------------------------------------------------------
# fine-tuning on real spectra (+ SSA mixup + synthetic replay)
# --------------------------------------------------------------------------
CONFIGS = {
    # name: (init, n_real, n_mixup, n_replay, lr)
    "A1_scratch_real": ("scratch", 16, 0, 0, 1e-3),
    "A1m_scratch_mixup": ("scratch", 16, 32, 0, 1e-3),
    "A2_pretrain_real": ("pretrained", 16, 0, 0, 3e-4),
    "A3_pretrain_mixup": ("pretrained", 16, 32, 0, 3e-4),
    "A4_full": ("pretrained", 16, 32, 16, 3e-4),
}
FT_STEPS = 1500
FT_WARMUP = 100
FT_WD = 1e-4


def finetune(config, init_state, real, syn_feat, syn_targets, syn_pool, scaler, seed, lam,
             steps=FT_STEPS):
    """real: dict with R (n, L), comp (n, 7, NaN where absent), chem (n, 5), logIsFeO (n,),
    d_nom (n,). Returns the fine-tuned model."""
    init, n_real, n_mix, n_rep, lr = CONFIGS[config]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LumenNet()
    if init == "pretrained":
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=FT_WD)

    def lr_at(s):
        if s < FT_WARMUP:
            return (s + 1) / FT_WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - FT_WARMUP) / max(1, steps - FT_WARMUP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # pre-generate every real / mixup draw for the whole run (vectorised)
    n = len(real["R"])
    j = rng.integers(0, n, steps * n_real)
    R_parts = [real["R"][j]]
    comp_parts = [real["comp"][j]]
    scal_parts = [np.column_stack([real["chem"][j], real["logIsFeO"][j]])]
    if n_mix:
        Rm, cm, chm, im = ssa_mixup(real["R"], real["comp"], real["chem"], real["logIsFeO"],
                                    real["d_nom"], steps * n_mix, rng)
        R_parts.append(Rm)
        comp_parts.append(cm)
        scal_parts.append(np.column_stack([chm, im]))
    R_all = output_randomisation(np.vstack(R_parts), rng, lam)
    X_all = scaler(lc.spectral_features(R_all, lam))
    comp_all = np.vstack(comp_parts).astype(np.float32)
    scal_all = np.vstack(scal_parts).astype(np.float32)
    smask_all = np.isfinite(scal_all)
    scal_all = scale_scalars(np.nan_to_num(scal_all))
    off_mix = steps * n_real
    if n_rep:
        rep = rng.choice(syn_pool, size=steps * n_rep)
        rep_comp, rep_scal, _ = syn_targets
    model.train()
    for s in range(steps):
        ir = np.arange(s * n_real, (s + 1) * n_real)
        im_ = off_mix + np.arange(s * n_mix, (s + 1) * n_mix)
        idx = np.r_[ir, im_].astype(int)
        Xb, cb, sb, mb = X_all[idx], comp_all[idx], scal_all[idx], smask_all[idx]
        if n_rep:
            jr = np.sort(rep[s * n_rep:(s + 1) * n_rep])
            Xb = np.vstack([Xb, np.asarray(syn_feat[jr])])
            cb = np.vstack([cb, rep_comp[jr]])
            sb = np.vstack([sb, rep_scal[jr]])
            # replay supervises composition and chemistry; its Is/FeO is on the D4
            # pretraining scale and is therefore masked
            mrep = np.ones((len(jr), len(SCALARS)), bool)
            mrep[:, -1] = False
            mb = np.vstack([mb, mrep])
        b = to_tensor_batch(Xb, cb, sb, mb)
        out = model(b["x"])
        loss = loss_fn(out, b, comp_l1=COMP_L1, aux_weight=0.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    return model


COMP_L1 = 1.0          # pre-registered (METHOD_NOTES.md D9), chosen on synthetic data only
