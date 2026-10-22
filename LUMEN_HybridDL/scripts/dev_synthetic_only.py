"""
Development checks on SYNTHETIC data only, run before the pre-registration was frozen
(METHOD_NOTES.md section "How the schedule was fixed"). No LSCC spectrum or label is
read here, so nothing in this file can leak evaluation data into a design choice.

  1. pretraining throughput (s / epoch on this machine);
  2. composition loss: Dirichlet NLL alone vs + L1 on the mean, on a synthetic
     validation set, short schedule, two seeds;
  3. the fine-tuning loop end to end on a synthetic "pseudo-real" set of 75 spectra
     (shape / timing check of every ablation configuration).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

import lumen_common as lc
import lumen_model as lm

STAGE = "dev"


def main() -> int:
    t0 = lc.banner(STAGE, "development checks on synthetic data only")
    torch.set_num_threads(4)
    syn = dict(np.load(lc.PRODUCTS / "synthetic.npz"))
    lam = syn["lam_um"]
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    n = 120_000
    feat = scaler(lc.spectral_features(syn["R"][:n], lam))
    sub = {k: v[:n] for k, v in syn.items() if k in ("R", "comp", "chem", "logIsFeO", "logD", "npfe_wt")}
    tr, va = np.arange(0, 100_000), np.arange(100_000, 110_000)

    rows = []
    for comp_l1 in (0.0, 1.0):
        for seed in (0, 1):
            tt = time.time()
            model, hist = lm.pretrain(feat, sub, tr, va, seed=seed, epochs=3, comp_l1=comp_l1)
            ev = hist[-1]
            rows.append({"comp_l1": comp_l1, "seed": seed, "seconds": time.time() - tt,
                         **{k: ev[k] for k in ev if k.endswith("_mae")}})
    df = pd.DataFrame(rows)
    df.to_csv(lc.TABLES / "dev_synthetic_comp_loss.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("  seconds per 100k-sample epoch: %.1f" % (df["seconds"].mean() / 3))

    # fine-tuning dry run on a synthetic pseudo-real set
    rng = np.random.default_rng(5)
    pr = rng.choice(np.arange(110_000, 120_000), 75, replace=False)
    comp = sub["comp"][pr].astype(float)
    comp[rng.uniform(0, 1, 75) < 0.25] = np.nan
    real = {"R": sub["R"][pr].astype(float), "comp": comp, "chem": sub["chem"][pr].astype(float),
            "logIsFeO": sub["logIsFeO"][pr].astype(float),
            "d_nom": rng.choice([5.0, 15.0, 32.5, 17.0], 75)}
    syn_targets = lm.synthetic_targets(sub, np.arange(n))
    pool = np.arange(0, 100_000)
    state = model.state_dict()
    torch.set_num_threads(1)
    for cfg in lm.CONFIGS:
        tt = time.time()
        lm.finetune(cfg, state, real, feat, syn_targets, pool, scaler, seed=0, lam=lam, steps=300)
        print("  fine-tune %-18s 300 steps: %.1f s (1 thread)" % (cfg, time.time() - tt))
    print("[%s] done in %.1f s" % (STAGE, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
