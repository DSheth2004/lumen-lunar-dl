"""
Consolidated Model Architectures and Transforms.

Contains:
- 2-level db4 Discrete Wavelet Transform (DWT) in periodization mode (A2=21, D2=21 -> 42-dim).
- 7-Layer Wavelet Autoencoder PyTorch Architecture (42 -> 32 -> 16 -> 6 -> 16 -> 32 -> 42).
- Supervised Regression Models (Lasso, SVR-Linear, SVR-RBF, Random Forest) with Table 5 grids.
"""
import pywt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict, Any


# ==============================================================================
# 1. Discrete Wavelet Transform (DWT) — 2-Level db4
# ==============================================================================

def dwt_db4_2level(spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply 2-level db4 DWT with periodization mode to an 83-band spectrum.
    
    Returns:
        tuple (A2, D2, D1) with shapes ((21,), (21,), (42,)).
    """
    coeffs = pywt.wavedec(spectrum, 'db4', level=2, mode='periodization')
    return coeffs[0], coeffs[1], coeffs[2]


def extract_retained_wavelet_features(spectra: np.ndarray) -> np.ndarray:
    """
    Extract 42-dim retained feature vector [A2, D2] for a batch of 83-band spectra.
    
    Args:
        spectra: Array of shape (N, 83) or (83,).
        
    Returns:
        Array of shape (N, 42) or (42,) with [A2, D2].
    """
    if spectra.ndim == 1:
        A2, D2, _ = dwt_db4_2level(spectra)
        return np.concatenate([A2, D2]).astype(np.float32)

    N = spectra.shape[0]
    out = np.zeros((N, 42), dtype=np.float32)
    for i in range(N):
        A2, D2, _ = dwt_db4_2level(spectra[i])
        out[i, :21] = A2
        out[i, 21:] = D2
    return out


# ==============================================================================
# 2. Deep Wavelet Autoencoder (42 -> 32 -> 16 -> 6 -> 16 -> 32 -> 42)
# ==============================================================================

class WaveletAutoencoder(nn.Module):
    """
    Exact 7-layer symmetric autoencoder matching Fernández-Díaz et al. (2026) §2.3.
    """
    def __init__(self, input_dim: int = 42, latent_dim: int = 6, leaky_slope: float = 0.01, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 42 -> 32 -> 16 -> 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(16, latent_dim)   # Bottleneck: Linear activation, no BN/dropout
        )

        # Decoder: 6 -> 16 -> 32 -> 42
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(32, input_dim)    # Output: Linear activation
        )
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ==============================================================================
# 3. Supervised Regression Models & Hyperparameter Search Grids
# ==============================================================================

def get_regression_models_and_grids(seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    """Return regression models and hyperparameter grids matching Table 5."""
    models = {
        "Lasso": Lasso(random_state=seed, max_iter=10000),
        "SVR (Linear)": SVR(kernel="linear"),
        "SVR (RBF)": SVR(kernel="rbf"),
        "Random Forest": RandomForestRegressor(random_state=seed, n_jobs=-1)
    }
    grids = {
        "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        "SVR (Linear)": {"C": [0.01, 0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.5]},
        "SVR (RBF)": {"C": [0.1, 1.0, 10.0, 100.0], "epsilon": [0.01, 0.1, 0.5], "gamma": ["scale", "auto"]},
        "Random Forest": {"n_estimators": [50, 100, 200, 500], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
    }
    return models, grids


def get_optimal_paper_models(seed: int = 42) -> Dict[str, Any]:
    """Return regressors initialized with optimal hyperparameters from Table 5."""
    return {
        "Lasso": Lasso(alpha=0.001, random_state=seed, max_iter=10000),
        "SVR (Linear)": SVR(kernel="linear", C=0.1, epsilon=0.1),
        "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=seed, n_jobs=-1)
    }
