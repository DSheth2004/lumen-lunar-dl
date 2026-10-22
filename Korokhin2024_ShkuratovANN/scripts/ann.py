"""
The paper's multilayer perceptron (Sec. 5.3, Fig. 13, Table 12), trained the way
it describes: per-sample (online) error back-propagation with an inertia
(momentum) term, sigmoid units throughout.

  layers        21 -> 42 -> 12 -> 8
  activation    f(y) = 1 / (1 + exp(-alpha y)), alpha = 1, every non-input neuron
  inputs        each wavelength scaled to [-1, +1] by the training set's min/max
  outputs       each parameter scaled to [0.05, 0.95] by the training set's min/max
  update        dw(t) = -eta dE/dw + 0.9 dw(t-1),  E = 1/2 sum (y - t)^2
  learning rate iteration 1: eta = 1.0 for the first epoch, then
                eta = k * sigma_Train(previous epoch), k = 0.01 (Table 12)
                iterations 2, 3: constant 0.002 / 0.010

Not stated in the paper (PAPER_NOTES.md D10-D12): every hidden and output neuron
has a bias (the "bias neuron" of each layer); weights start uniform in +-0.5;
samples are visited in a fresh random order each epoch; sigma_Train is the RMS
error of the normalised outputs over the whole training set.

The inner loop is compiled with numba; a float64 numpy forward pass (`predict`)
is used for mapping and is checked against torch in step07.
"""
from __future__ import annotations

import numpy as np
from numba import njit

LAYERS = (21, 42, 12, 8)
OUT_LO, OUT_HI = 0.05, 0.95


@njit(cache=True, fastmath=False)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def train_epoch(X, T, order, W1, b1, W2, b2, W3, b3, V1, c1, V2, c2, V3, c3, lr, mom):
    """One pass of online back-propagation with momentum. Returns sum of squared errors."""
    n_in, n1 = W1.shape
    n2 = W2.shape[1]
    n3 = W3.shape[1]
    h1 = np.empty(n1)
    h2 = np.empty(n2)
    y = np.empty(n3)
    d3 = np.empty(n3)
    d2 = np.empty(n2)
    d1 = np.empty(n1)
    sse = 0.0
    for idx in range(order.shape[0]):
        p = order[idx]
        x = X[p]
        # forward
        for j in range(n1):
            s = b1[j]
            for i in range(n_in):
                s += x[i] * W1[i, j]
            h1[j] = _sig(s)
        for j in range(n2):
            s = b2[j]
            for i in range(n1):
                s += h1[i] * W2[i, j]
            h2[j] = _sig(s)
        for j in range(n3):
            s = b3[j]
            for i in range(n2):
                s += h2[i] * W3[i, j]
            y[j] = _sig(s)
        # output error and deltas (sigmoid' = y (1 - y))
        for j in range(n3):
            e = y[j] - T[p, j]
            sse += e * e
            d3[j] = e * y[j] * (1.0 - y[j])
        for i in range(n2):
            s = 0.0
            for j in range(n3):
                s += W3[i, j] * d3[j]
            d2[i] = s * h2[i] * (1.0 - h2[i])
        for i in range(n1):
            s = 0.0
            for j in range(n2):
                s += W2[i, j] * d2[j]
            d1[i] = s * h1[i] * (1.0 - h1[i])
        # momentum updates
        for i in range(n2):
            for j in range(n3):
                V3[i, j] = -lr * h2[i] * d3[j] + mom * V3[i, j]
                W3[i, j] += V3[i, j]
        for j in range(n3):
            c3[j] = -lr * d3[j] + mom * c3[j]
            b3[j] += c3[j]
        for i in range(n1):
            for j in range(n2):
                V2[i, j] = -lr * h1[i] * d2[j] + mom * V2[i, j]
                W2[i, j] += V2[i, j]
        for j in range(n2):
            c2[j] = -lr * d2[j] + mom * c2[j]
            b2[j] += c2[j]
        for i in range(n_in):
            for j in range(n1):
                V1[i, j] = -lr * x[i] * d1[j] + mom * V1[i, j]
                W1[i, j] += V1[i, j]
        for j in range(n1):
            c1[j] = -lr * d1[j] + mom * c1[j]
            b1[j] += c1[j]
    return sse


class MLP:
    def __init__(self, seed: int, layers=LAYERS, init=0.5):
        rng = np.random.default_rng(seed)
        self.W = [rng.uniform(-init, init, (a, b)) for a, b in zip(layers[:-1], layers[1:])]
        self.b = [rng.uniform(-init, init, b) for b in layers[1:]]
        self.V = [np.zeros_like(w) for w in self.W]
        self.c = [np.zeros_like(b) for b in self.b]
        self.rng = rng
        self.x_lo = self.x_hi = self.y_lo = self.y_hi = None

    # ---- normalisation (factors from the training set, Sec. 5.3 (7), (8)) ----
    def fit_scaling(self, X, Y):
        self.x_lo, self.x_hi = X.min(axis=0), X.max(axis=0)
        self.y_lo, self.y_hi = Y.min(axis=0), Y.max(axis=0)

    def norm_x(self, X):
        return 2.0 * (X - self.x_lo) / (self.x_hi - self.x_lo) - 1.0

    def norm_y(self, Y):
        return OUT_LO + (OUT_HI - OUT_LO) * (Y - self.y_lo) / (self.y_hi - self.y_lo)

    def denorm_y(self, Yn):
        return self.y_lo + (Yn - OUT_LO) / (OUT_HI - OUT_LO) * (self.y_hi - self.y_lo)

    # ---- training / prediction ----
    def epoch(self, Xn, Tn, lr, mom=0.9) -> float:
        order = self.rng.permutation(Xn.shape[0])
        sse = train_epoch(Xn, Tn, order, self.W[0], self.b[0], self.W[1], self.b[1],
                          self.W[2], self.b[2], self.V[0], self.c[0], self.V[1], self.c[1],
                          self.V[2], self.c[2], lr, mom)
        return float(np.sqrt(sse / Tn.size))            # sigma_Train (RMS, normalised)

    def predict_norm(self, Xn):
        h = Xn
        for W, b in zip(self.W, self.b):
            h = 1.0 / (1.0 + np.exp(-(h @ W + b)))
        return h

    def predict(self, X):
        return self.denorm_y(self.predict_norm(self.norm_x(X)))

    def state(self) -> dict:
        d = {"W%d" % i: w.copy() for i, w in enumerate(self.W)}
        d.update({"b%d" % i: b.copy() for i, b in enumerate(self.b)})
        d.update({"x_lo": self.x_lo, "x_hi": self.x_hi, "y_lo": self.y_lo, "y_hi": self.y_hi})
        return d

    def load(self, d: dict):
        self.W = [d["W%d" % i].copy() for i in range(3)]
        self.b = [d["b%d" % i].copy() for i in range(3)]
        self.x_lo, self.x_hi, self.y_lo, self.y_hi = d["x_lo"], d["x_hi"], d["y_lo"], d["y_hi"]
