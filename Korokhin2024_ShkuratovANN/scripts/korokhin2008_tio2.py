"""
The TiO2 networks of Korokhin et al. (2008), PSS 56:1063-1078 -- the comparison of the
paper's Fig. 30 ("TiO2 abundance obtained by using a moderately nonlinear ANN (Korokhin
et al., 2008)"). doi:10.1016/j.pss.2008.02.001.

Transcribed here:
  Table 1  the LSCC training set: A(415, 750, 900, 950, 1000) (fractions) and TiO2 (wt%)
           for 19 soils x 4 size classes (<10, 10-20, 20-45 um, bulk) = 76 rows;
  Table 2  input standardization x0, xk (six spectral parameters, four class inputs);
           output: TiO2 = yk * Y + y0 with y0 = -0.178, yk = 11.396 (Sec. 3);
  Tables 3-5  the "significantly non-linear" net: 6-4-2-1, alpha = 1.5 (Sec. 4.1);
  Tables 6-9  the two-net system: classification 6-3-4 and prognosis 10-3-1, alpha = 1.2
           (Sec. 4.2). No bias terms are printed for any layer, so none are used.
Inputs (Sec. 2): AR = A(750), CBR = A(415)/A(750), CIR1 = A(900)/A(750), CIR2 =
A(950)/A(750), CIR3 = A(1000)/A(750), D = A(750) A(1000) / A(900)^2; neuron output
f(s) = 1 / (1 + exp(-alpha s)).

The "moderately non-linear" net that Fig. 30 names -- one hidden layer of 3 neurons,
alpha = 1.0, 500 epochs (Sec. 4.1) -- has no published weights; retrain() rebuilds it on
Table 1 with the published standardization. Choices the source leaves open (D22): no
bias (as every published layer), per-sample back-propagation with learning rate 0.1 and
momentum 0.9, weights uniform in +-0.5, samples reshuffled every epoch, all 76 rows.

reading_test() settles what the extracted text leaves ambiguous -- whether
X = (x - x0) / xk or (x - x0) * xk, and whether AR enters as a fraction or in % -- by
the published LSCC correlation of the Tables 3-5 net (k = 0.99, Fig. 18).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Table 1: sample, size, A415, A750, A900, A950, A1000, TiO2
TABLE1 = """
10084 <10 0.0462 0.0793 0.0900 0.0928 0.0958 7.25
10084 10-20 0.0431 0.0691 0.0746 0.0751 0.0767 7.94
10084 20-45 0.0437 0.0645 0.0638 0.0626 0.0627 8.30
10084 bulk 0.0448 0.0726 0.0790 0.0794 0.0813 7.54
12001 <10 0.0566 0.1093 0.1235 0.1274 0.1328 2.78
12001 10-20 0.0456 0.0822 0.0848 0.0847 0.0882 2.96
12001 20-45 0.0465 0.0759 0.0719 0.0702 0.0725 3.20
12001 bulk 0.0496 0.0868 0.0912 0.0919 0.0954 2.96
12030 <10 0.1357 0.2066 0.2165 0.2176 0.2252 3.01
12030 10-20 0.0994 0.1541 0.1350 0.1295 0.1359 3.32
12030 20-45 0.0995 0.1474 0.1211 0.1148 0.1193 3.74
12030 bulk 0.1101 0.1683 0.1572 0.1535 0.1607 3.23
15041 <10 0.0534 0.1121 0.1274 0.1317 0.1377 1.79
15041 10-20 0.0418 0.0762 0.0792 0.0793 0.0830 1.88
15041 20-45 0.0460 0.0760 0.0728 0.0729 0.0764 2.03
15041 bulk 0.0461 0.0867 0.0917 0.0934 0.0973 1.83
15071 <10 0.0558 0.1137 0.1285 0.1321 0.1375 1.57
15071 10-20 0.0472 0.0876 0.0880 0.0876 0.0924 1.88
15071 20-45 0.0496 0.0856 0.0798 0.0799 0.0835 2.33
15071 bulk 0.0501 0.0961 0.1003 0.1024 0.1068 1.81
70181 <10 0.0554 0.0989 0.1156 0.1207 0.1257 6.54
70181 10-20 0.0513 0.0803 0.0879 0.0893 0.0923 7.88
70181 20-45 0.0472 0.0705 0.0722 0.0724 0.0743 8.11
70181 bulk 0.0488 0.0783 0.0862 0.0878 0.0908 7.57
71061 <10 0.0933 0.1551 0.1942 0.2031 0.2096 7.89
71061 10-20 0.0753 0.1162 0.1256 0.1278 0.1316 8.94
71061 20-45 0.0769 0.1151 0.1007 0.0975 0.0983 9.48
71061 bulk 0.0882 0.1325 0.1524 0.1569 0.1610 8.76
71501 <10 0.0464 0.0842 0.0986 0.1018 0.1050 8.27
71501 10-20 0.0463 0.0753 0.0812 0.0810 0.0831 9.83
71501 20-45 0.0469 0.0749 0.0738 0.0716 0.0727 10.70
71501 bulk 0.0465 0.0767 0.0838 0.0846 0.0867 9.31
79221 <10 0.0520 0.0897 0.1041 0.1082 0.1131 5.83
79221 10-20 0.0457 0.0711 0.0776 0.0794 0.0823 7.21
79221 20-45 0.0469 0.0687 0.0690 0.0697 0.0714 7.38
79221 bulk 0.0491 0.0791 0.0887 0.0916 0.0949 6.39
14141 <10 0.2182 0.3280 0.3352 0.3366 0.3475 1.51
14141 10-20 0.1702 0.2590 0.2393 0.2385 0.2559 1.71
14141 20-45 0.1370 0.2009 0.1598 0.1565 0.1726 1.96
14141 bulk 0.1780 0.2623 0.2444 0.2441 0.2596 1.65
14163 <10 0.1134 0.1970 0.2175 0.2227 0.2313 2.07
14163 10-20 0.0804 0.1391 0.1442 0.1459 0.1542 1.88
14163 20-45 0.0613 0.1022 0.0937 0.0946 0.1000 2.00
14163 bulk 0.0903 0.1540 0.1632 0.1675 0.1741 1.90
14259 <10 0.0769 0.1564 0.1789 0.1862 0.1937 2.02
14259 10-20 0.0594 0.1184 0.1301 0.1340 0.1420 1.96
14259 20-45 0.0492 0.0809 0.0819 0.0826 0.0870 1.99
14259 bulk 0.0604 0.1146 0.1253 0.1291 0.1359 1.80
14260 <10 0.0659 0.1375 0.1588 0.1649 0.1715 1.94
14260 10-20 0.0575 0.1179 0.1293 0.1330 0.1400 1.98
14260 20-45 0.0465 0.0793 0.0788 0.0799 0.0849 1.86
14260 bulk 0.0574 0.1109 0.1210 0.1241 0.1321 1.85
61141 <10 0.1361 0.2464 0.2746 0.2827 0.2921 0.59
61141 10-20 0.1032 0.1818 0.1963 0.2012 0.2090 0.64
61141 20-45 0.0763 0.1265 0.1350 0.1388 0.1448 0.58
61141 bulk 0.1138 0.1960 0.2131 0.2190 0.2271 0.59
61221 <10 0.3171 0.4216 0.4351 0.4377 0.4440 0.50
61221 10-20 0.2976 0.3887 0.3844 0.3860 0.3939 0.54
61221 20-45 0.2455 0.3181 0.2846 0.2814 0.2896 0.56
61221 bulk 0.3020 0.3866 0.3821 0.3825 0.3906 0.52
62231 <10 0.1333 0.2350 0.2641 0.2721 0.2810 0.58
62231 10-20 0.1156 0.1952 0.2114 0.2164 0.2238 0.61
62231 20-45 0.1043 0.1492 0.1590 0.1628 0.1680 0.58
62231 bulk 0.1110 0.1858 0.2037 0.2096 0.2169 0.60
64801 <10 0.1403 0.2551 0.2854 0.2939 0.3038 0.61
64801 10-20 0.1124 0.2014 0.2202 0.2265 0.2350 0.68
64801 20-45 0.1023 0.1752 0.1865 0.1911 0.1986 0.63
64801 bulk 0.1139 0.1965 0.2143 0.2205 0.2288 0.65
67461 <10 0.2780 0.4133 0.4307 0.4350 0.4475 0.34
67461 10-20 0.2275 0.3382 0.3347 0.3375 0.3518 0.39
67461 20-45 0.1934 0.2801 0.2773 0.2796 0.2922 0.44
67461 bulk 0.2493 0.3596 0.3607 0.3651 0.3795 0.35
67481 <10 0.2550 0.3899 0.4138 0.4200 0.4324 0.42
67481 10-20 0.2116 0.3192 0.3261 0.3309 0.3444 0.40
67481 20-45 0.1633 0.2336 0.2368 0.2392 0.2490 0.49
67481 bulk 0.2264 0.3243 0.3343 0.3403 0.3525 0.44
"""
SPEC = ["AR", "CBR", "CIR1", "CIR2", "CIR3", "D"]
CLASSES = ["SCA16", "SCA14", "SCMlow", "SCMhigh"]
X0 = dict(zip(SPEC + CLASSES, (0.2431, 0.6288, 0.9832, 0.9996, 1.0402, 1.1336, 0.5, 0.5, 0.5, 0.5)))
XK = dict(zip(SPEC + CLASSES, (5.6007, 6.5615, 5.3248, 4.5291, 4.3326, 4.4592, 2.0, 2.0, 2.0, 2.0)))
Y0, YK = -0.178, 11.396

# Tables 3-5: significantly non-linear net, alpha = 1.5
SIG_W = [np.array([[0.4986, 2.3223, 0.5409, 1.2126, -0.1447, -0.0355],
                   [0.2062, 1.8351, 0.9909, -0.2891, -1.0505, -0.6925],
                   [1.5316, 1.6740, -0.7028, 0.3696, 1.6691, 1.2761],
                   [0.7294, -3.3990, -1.8817, -0.5255, 1.2135, 4.7384]]),
         np.array([[1.4344, -1.2085, 3.2410, 5.3733],
                   [2.0864, -1.0830, 0.0737, -0.8752]]),
         np.array([[-2.8466, -2.3900]])]
SIG_ALPHA = 1.5
# Tables 6-7: classification net (outputs SCA16, SCA14, SCMlow, SCMhigh), alpha = 1.2
CLS_W = [np.array([[4.3693, 2.0469, 1.1328, 1.7261, 0.0593, -1.2040],
                   [1.1415, -4.6267, -1.2505, -1.1236, 0.4143, 3.5966],
                   [2.5235, 0.9834, -0.2454, 0.5972, 2.1901, 1.5583]]),
         np.array([[3.1614, -0.3404, 2.0586],
                   [-1.4904, 2.4493, 2.2257],
                   [-2.4729, 2.6110, -2.8712],
                   [-3.0804, -5.5366, -1.3075]])]
# Tables 8-9: prognosis net (6 spectral + 4 class inputs), alpha = 1.2
PRG_W = [np.array([[0.4536, 0.7971, 0.6376, 0.3895, -0.3593, -0.0899, -0.6776, 0.0313, 0.6993, 1.6563],
                   [-0.0529, 1.0228, 0.0240, 1.1849, 1.0340, 0.6085, 0.5118, 0.3060, 0.1295, -0.5089],
                   [-0.5791, 1.0474, 0.6103, 0.4715, 0.6090, 0.1860, -0.7571, 0.4387, 0.0831, 1.7581]]),
         np.array([[1.0080, -2.0689, 1.5649]])]
SYS_ALPHA = 1.2
MODERATE = {"hidden": 3, "alpha": 1.0, "epochs": 500, "lr": 0.1, "momentum": 0.9}
PUBLISHED_K = {"significant": 0.99, "moderate": 0.95}


def table1() -> pd.DataFrame:
    rows = [l.split() for l in TABLE1.strip().splitlines()]
    df = pd.DataFrame(rows, columns=["sample", "size", "A415", "A750", "A900", "A950", "A1000", "TiO2"])
    for c in df.columns[2:]:
        df[c] = df[c].astype(float)
    return df


def check_table1(df: pd.DataFrame) -> dict:
    """Shape and plausibility of the transcription: 19 soils x 4 sizes, albedos in (0, 1),
    and the size classes positionally in the printed order for every soil."""
    order_ok = all(list(g["size"]) == ["<10", "10-20", "20-45", "bulk"] for _, g in df.groupby("sample", sort=False))
    alb = df[["A415", "A750", "A900", "A950", "A1000"]].to_numpy()
    return {"rows": len(df), "soils": df["sample"].nunique(), "size_order_ok": bool(order_ok),
            "albedo_in_0_1": bool(((alb > 0) & (alb < 1)).all()),
            "tio2_range": (float(df["TiO2"].min()), float(df["TiO2"].max()))}


def spectral_params(A415, A750, A900, A950, A1000, ar_percent: bool = False) -> np.ndarray:
    """(..., 6): AR, CBR, CIR1, CIR2, CIR3, D. A are fractions."""
    AR = A750 * (100.0 if ar_percent else 1.0)
    return np.stack([AR, A415 / A750, A900 / A750, A950 / A750, A1000 / A750, A750 * A1000 / A900 ** 2], axis=-1)


def standardize(x: np.ndarray, names: list, mode: str) -> np.ndarray:
    x0 = np.array([X0[n] for n in names]); xk = np.array([XK[n] for n in names])
    return (x - x0) * xk if mode == "multiply" else (x - x0) / xk


def _f(s, alpha):
    return 1.0 / (1.0 + np.exp(-alpha * s))


def forward(X: np.ndarray, weights: list, alpha: float) -> np.ndarray:
    h = X
    for W in weights:
        h = _f(h @ W.T, alpha)
    return h


def significant(P: np.ndarray, mode: str) -> np.ndarray:
    return YK * forward(standardize(P, SPEC, mode), SIG_W, SIG_ALPHA)[..., 0] + Y0


def two_net_system(P: np.ndarray, mode: str) -> np.ndarray:
    Xs = standardize(P, SPEC, mode)
    sc = forward(Xs, CLS_W, SYS_ALPHA)
    X = np.concatenate([Xs, standardize(sc, CLASSES, mode)], axis=-1)
    return YK * forward(X, PRG_W, SYS_ALPHA)[..., 0] + Y0


def reading_test(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for mode in ("multiply", "divide"):
        for pct in (False, True):
            P = spectral_params(*(df[c].to_numpy() for c in ("A415", "A750", "A900", "A950", "A1000")),
                                ar_percent=pct)
            for name, fn in (("significant", significant), ("two_net_system", two_net_system)):
                pred = fn(P, mode)
                rows.append({"net": name, "standardization": mode, "AR_percent": pct,
                             "k_vs_measured": float(np.corrcoef(pred, df["TiO2"])[0, 1]),
                             "rmse_wt_pct": float(np.sqrt(np.mean((pred - df["TiO2"]) ** 2)))})
    g = pd.DataFrame(rows)
    s = g[g["net"] == "significant"].sort_values("k_vs_measured", ascending=False).iloc[0]
    return {"standardization": s["standardization"], "AR_percent": bool(s["AR_percent"]),
            "k_significant": float(s["k_vs_measured"])}, g


def retrain(df: pd.DataFrame, reading: dict, seed: int) -> tuple[list, float]:
    """The 'moderately non-linear' net: 6-3-1, alpha 1.0, no bias, 500 epochs."""
    P = spectral_params(*(df[c].to_numpy() for c in ("A415", "A750", "A900", "A950", "A1000")),
                        ar_percent=reading["AR_percent"])
    X = standardize(P, SPEC, reading["standardization"])
    Y = (df["TiO2"].to_numpy() - Y0) / YK
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-0.5, 0.5, (MODERATE["hidden"], 6))
    W2 = rng.uniform(-0.5, 0.5, (1, MODERATE["hidden"]))
    V1, V2 = np.zeros_like(W1), np.zeros_like(W2)
    a, lr, mu = MODERATE["alpha"], MODERATE["lr"], MODERATE["momentum"]
    for _ in range(MODERATE["epochs"]):
        for i in rng.permutation(len(X)):
            h = _f(W1 @ X[i], a)
            o = _f(W2 @ h, a)
            d2 = (o - Y[i]) * a * o * (1 - o)
            d1 = (W2.T @ d2) * a * h * (1 - h)
            V2 = mu * V2 - lr * np.outer(d2, h)
            V1 = mu * V1 - lr * np.outer(d1, X[i])
            W2 += V2
            W1 += V1
    pred = YK * forward(X, [W1, W2], a)[:, 0] + Y0
    return [W1, W2], float(np.corrcoef(pred, df["TiO2"])[0, 1])


def moderate(P: np.ndarray, weights: list, reading: dict) -> np.ndarray:
    return YK * forward(standardize(P, SPEC, reading["standardization"]), weights, MODERATE["alpha"])[..., 0] + Y0
