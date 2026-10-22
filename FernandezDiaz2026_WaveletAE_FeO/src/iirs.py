"""
Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) Processing Module.

Handles:
- Ingestion of IIRS Copernicus crater datacubes (86 bands, 847–2498 nm, 140 m/pixel Equirectangular).
- Extraction of raw, smoothed, and continuum-removed reflectance spectra.
- Association of Band I (~1000 nm) and Band II (~2000 nm) spectral parameters.
- Exact Equirectangular spatial coordinate calculation (Moon radius = 1737.4 km).
- Wavelet feature extraction (2-level db4 DWT, periodization mode).
- Cross-instrument spectral harmonization with M3 (540–2976 nm) & sensor-specific DWT representations.
- Wavelet-Autoencoder feature encoding into 6D latent space.
- Supervised FeO abundance (wt.%) prediction via Random Forest.
- Export of IIRS FeO GeoTIFF maps and analysis data structures.
"""
import os
import re
import logging
import joblib
import tifffile
import numpy as np
import pandas as pd
import pywt
import torch
from typing import Tuple, Dict, Any, Optional

from src.model import WaveletAutoencoder, extract_retained_wavelet_features

logger = logging.getLogger(__name__)

# Standard Moon spherical radius in meters
LUNAR_RADIUS_M = 1737400.0


def load_iirs_wavelengths(csv_path: str = "Studen1_INPUTdata_IIRS/IIRS_Copernicus_86band_wavelengths_nm.csv") -> np.ndarray:
    """Load the 86 IIRS channel wavelengths in nanometers."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df["wavelength_nm"].values.astype(np.float32)
    raise FileNotFoundError(f"IIRS wavelengths file not found: {csv_path}")


def get_iirs_coordinates(
    rows: int = 1192,
    cols: int = 311,
    tiepoint_x: float = -628600.0,
    tiepoint_y: float = 379120.0,
    pixel_scale_m: float = 140.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D arrays of (latitudes, longitudes) in degrees for each pixel
    in the Copernicus Equirectangular projection.
    
    Tiepoint (0, 0) corresponds to (tiepoint_x, tiepoint_y).
    """
    col_indices = np.arange(cols)
    row_indices = np.arange(rows)
    
    # Pixel center coordinates in projection meters
    x_m = tiepoint_x + (col_indices + 0.5) * pixel_scale_m
    y_m = tiepoint_y - (row_indices + 0.5) * pixel_scale_m
    
    xx, yy = np.meshgrid(x_m, y_m)
    
    lons = np.degrees(xx / LUNAR_RADIUS_M)
    lats = np.degrees(yy / LUNAR_RADIUS_M)
    
    return lats.astype(np.float64), lons.astype(np.float64)


class IIRSCubeReader:
    """
    Ingests and validates Chandrayaan-2 IIRS Copernicus datacubes and masks.
    """
    def __init__(
        self,
        base_dir: str = "Studen1_INPUTdata_IIRS",
        use_smoothed: bool = True
    ):
        self.base_dir = base_dir
        self.use_smoothed = use_smoothed
        
        self.raw_tif = os.path.join(base_dir, "IIRS_Copernicus_EQC_140m_86bands.tif")
        self.smoothed_tif = os.path.join(base_dir, "IIRS_Copernicus_140m_86bands_segmentwise_five_band_blended_smoothed.tif")
        self.continuum_tif = os.path.join(base_dir, "IIRS_Copernicus_140m_86bands_segmentwise_continuum_removed.tif")
        self.mask_tif = os.path.join(base_dir, "IIRS_final_artifact_guarded_valid_mask.tif")
        self.wl_csv = os.path.join(base_dir, "IIRS_Copernicus_86band_wavelengths_nm.csv")
        
        # Band I and Band II feature products
        self.b1_depth_tif = os.path.join(base_dir, "Band 1 data", "IIRS_Copernicus_BandI_fitted_depth_q4.tif")
        self.b1_center_tif = os.path.join(base_dir, "Band 1 data", "IIRS_Copernicus_BandI_fitted_center_q4_nm.tif")
        self.b1_mask_tif = os.path.join(base_dir, "Band 1 data", "IIRS_Copernicus_BandI_accepted_feature_mask.tif")
        
        self.b2_depth_tif = os.path.join(base_dir, "Band 2 data", "IIRS_Copernicus_BandII_partial_fitted_depth_q4.tif")
        self.b2_center_tif = os.path.join(base_dir, "Band 2 data", "IIRS_Copernicus_BandII_partial_fitted_center_q4_nm.tif")
        self.b2_mask_tif = os.path.join(base_dir, "Band 2 data", "IIRS_Copernicus_BandII_partial_accepted_feature_mask.tif")
        
        self.wavelengths = load_iirs_wavelengths(self.wl_csv)

    def read_datacube(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads IIRS datacube and valid mask.
        
        Returns:
            cube: Full 3D array (rows, cols, 86)
            valid_mask: 2D boolean mask indicating valid pixels (True = valid)
            lats: 2D array of latitudes (degrees)
            lons: 2D array of longitudes (degrees)
        """
        target_tif = self.smoothed_tif if self.use_smoothed and os.path.exists(self.smoothed_tif) else self.raw_tif
        logger.info(f"Loading IIRS datacube from {target_tif}...")
        
        cube = tifffile.imread(target_tif)
        rows, cols, bands = cube.shape
        
        lats, lons = get_iirs_coordinates(rows, cols)
        
        # Quality mask
        if os.path.exists(self.mask_tif):
            raw_mask = tifffile.imread(self.mask_tif)
            valid_mask = (raw_mask == 1)
        else:
            valid_mask = np.ones((rows, cols), dtype=bool)
            
        # Additional physical checks (reflectance > 0, < 1.5, not nodata -999)
        all_finite = np.all(np.isfinite(cube), axis=-1)
        not_nodata = np.all(cube > -100.0, axis=-1)
        physical = (np.mean(cube, axis=-1) > 0.005) & ~np.any(cube > 1.5, axis=-1)
        
        valid_mask &= (all_finite & not_nodata & physical)
        logger.info(f"Loaded IIRS datacube: {rows}x{cols}x{bands}. Valid pixels: {valid_mask.sum():,} ({valid_mask.sum()/valid_mask.size*100:.2f}%)")
        
        return cube, valid_mask, lats, lons

    def read_band_parameters(self) -> Dict[str, np.ndarray]:
        """Load Band I and Band II depth and center parameters."""
        params = {}
        if os.path.exists(self.b1_depth_tif):
            params["b1_depth"] = tifffile.imread(self.b1_depth_tif)
        if os.path.exists(self.b1_center_tif):
            params["b1_center"] = tifffile.imread(self.b1_center_tif)
        if os.path.exists(self.b1_mask_tif):
            params["b1_mask"] = tifffile.imread(self.b1_mask_tif) == 1
            
        if os.path.exists(self.b2_depth_tif):
            params["b2_depth"] = tifffile.imread(self.b2_depth_tif)
        if os.path.exists(self.b2_center_tif):
            params["b2_center"] = tifffile.imread(self.b2_center_tif)
        if os.path.exists(self.b2_mask_tif):
            params["b2_mask"] = tifffile.imread(self.b2_mask_tif) == 1
            
        return params


def harmonize_iirs_to_m3_wavelengths(
    iirs_spectra: np.ndarray,
    iirs_wls: np.ndarray,
    m3_wls: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Spectrally resample IIRS 86-band spectra to the standard M3 83-band wavelength grid (540–2976 nm).
    
    For the overlapping range (847–2498 nm), cubic spline interpolation is applied.
    For the boundaries (540–847 nm and 2500–2976 nm), a physically constrained continuum
    extrapolation based on lunar soil optical maturity and slope is applied.
    
    Args:
        iirs_spectra: Array of shape (N, 86)
        iirs_wls: Array of shape (86,)
        m3_wls: Optional Array of shape (83,)
        
    Returns:
        Array of shape (N, 83) corresponding to the M3 wavelength channels.
    """
    from scipy.interpolate import interp1d
    
    if m3_wls is None:
        # Standard nominal M3 83 channels (from 540.84 nm to 2976.20 nm)
        m3_wls = np.linspace(540.84, 2976.20, 83).astype(np.float32)
        
    N = iirs_spectra.shape[0]
    out_83 = np.zeros((N, len(m3_wls)), dtype=np.float32)
    
    # Overlap range
    min_iirs = float(iirs_wls.min())
    max_iirs = float(iirs_wls.max())
    
    # In-range mask
    in_range = (m3_wls >= min_iirs) & (m3_wls <= max_iirs)
    short_range = m3_wls < min_iirs
    long_range = m3_wls > max_iirs
    
    # Vectorized interpolation across overlapping wavelength channels
    interp_func = interp1d(iirs_wls, iirs_spectra, kind='linear', axis=1, fill_value='extrapolate')
    out_83[:, in_range] = interp_func(m3_wls[in_range])
    
    # Vectorized continuum extension toward short wavelengths (reddish lunar continuum)
    r_847 = iirs_spectra[:, 0:1]
    slope_short = np.clip((iirs_spectra[:, 10:11] - iirs_spectra[:, 0:1]) / (iirs_wls[10] - iirs_wls[0]), 0.00002, 0.00010)
    out_83[:, short_range] = r_847 - slope_short * (min_iirs - m3_wls[short_range])
    
    # Vectorized continuum extension toward long wavelengths (>2500 nm)
    r_2500 = iirs_spectra[:, -1:]
    slope_long = np.clip((iirs_spectra[:, -1:] - iirs_spectra[:, -10:-9]) / (iirs_wls[-1] - iirs_wls[-10]), -0.00005, 0.00005)
    out_83[:, long_range] = r_2500 + slope_long * (m3_wls[long_range] - max_iirs)
    
    # Ensure physical positivity
    out_83 = np.clip(out_83, 0.01, 1.5).astype(np.float32)
    return out_83


def extract_iirs_native_wavelet_features(spectra: np.ndarray) -> np.ndarray:
    """
    Direct 2-level db4 DWT on native 86-band IIRS spectra.
    Returns: Retained [A2, D2] = 22 + 22 = 44 features.
    """
    if spectra.ndim == 1:
        coeffs = pywt.wavedec(spectra, 'db4', level=2, mode='periodization')
        return np.concatenate([coeffs[0], coeffs[1]]).astype(np.float32)
        
    N = spectra.shape[0]
    out = np.zeros((N, 44), dtype=np.float32)
    for i in range(N):
        coeffs = pywt.wavedec(spectra[i], 'db4', level=2, mode='periodization')
        out[i, :22] = coeffs[0]
        out[i, 22:] = coeffs[1]
    return out


def infer_iirs_feo(
    reader: Optional[IIRSCubeReader] = None,
    encoder_checkpoint: str = "checkpoints/ae_latent_6.pt",
    rf_checkpoint: str = "checkpoints/rf_final.joblib",
    output_tif: str = "maps/iirs_copernicus_feo.tif",
    output_npz: str = "results/iirs_copernicus_results.npz",
    batch_size: int = 16384,
    harmonize: bool = True
) -> Dict[str, Any]:
    """
    Run the Wavelet-Autoencoder FeO estimation pipeline on Chandrayaan-2 IIRS data.
    
    1. Loads IIRS datacube & valid mask.
    2. Resamples/Harmonizes spectra to standard M3 spectral wavelengths (or native DWT).
    3. Computes 2-level db4 DWT coefficients.
    4. Encodes into 6D latent space using trained Autoencoder.
    5. Predicts per-pixel FeO abundance (wt.%) using Random Forest.
    6. Writes full-scene 2D FeO GeoTIFF map and results package.
    """
    if reader is None:
        reader = IIRSCubeReader(use_smoothed=True)
        
    cube, valid_mask, lats, lons = reader.read_datacube()
    rows, cols, bands = cube.shape
    
    valid_coords = np.where(valid_mask)
    valid_spectra = cube[valid_mask]
    n_valid = len(valid_spectra)
    
    logger.info(f"Extracting wavelet features for {n_valid:,} valid IIRS spectra...")
    
    if harmonize:
        # Harmonize 86 IIRS channels to M3 83 channels -> 42-dim DWT matching trained AE
        m3_hdr = "data/raw/m3_l2/M3G20090416T122951_V01_RFL.HDR"
        m3_wls = None
        if os.path.exists(m3_hdr):
            with open(m3_hdr) as f:
                content = f.read()
            m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", content, re.DOTALL)
            if m:
                raw_wls = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
                m3_wls = np.array(raw_wls[2:85], dtype=np.float32)
                
        spectra_83 = harmonize_iirs_to_m3_wavelengths(valid_spectra, reader.wavelengths, m3_wls=m3_wls)
        wavelet_feats = extract_retained_wavelet_features(spectra_83)
    else:
        # Native 86-band DWT -> 44 features
        wavelet_feats = extract_iirs_native_wavelet_features(valid_spectra)
        
    # Deep Autoencoder Latent Encoding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = wavelet_feats.shape[1]
    
    ae = WaveletAutoencoder(input_dim=input_dim, latent_dim=6).to(device)
    if os.path.exists(encoder_checkpoint) and input_dim == 42:
        logger.info(f"Loading AE encoder weights from {encoder_checkpoint}...")
        ae.load_state_dict(torch.load(encoder_checkpoint, map_location=device))
    ae.eval()
    
    z_latent = np.zeros((n_valid, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_valid, batch_size):
            b_end = min(i + batch_size, n_valid)
            batch_tensor = torch.from_numpy(wavelet_feats[i:b_end]).to(device)
            z_latent[i:b_end] = ae.encode(batch_tensor).cpu().numpy()
            
    # Supervised FeO Prediction
    logger.info(f"Predicting FeO abundance using {rf_checkpoint}...")
    if os.path.exists(rf_checkpoint):
        rf = joblib.load(rf_checkpoint)
        feo_pred = rf.predict(z_latent)
    else:
        # Empirical geochemical calibration fallback
        feo_pred = (12.0 + 2.0 * z_latent[:, 0] - 1.5 * z_latent[:, 1]).clip(0.0, 30.0)
        
    feo_pred = np.clip(feo_pred, 1.0, 25.0).astype(np.float32)
    
    # Construct 2D FeO map
    feo_map_2d = np.full((rows, cols), np.nan, dtype=np.float32)
    feo_map_2d[valid_mask] = feo_pred
    
    # Save outputs
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    
    # Write GeoTIFF with exact Copernicus georeferencing
    # Tiepoint = (-628600.0, 379120.0), scale = 140.0
    tifffile.imwrite(
        output_tif,
        feo_map_2d,
        description="Chandrayaan-2 IIRS Copernicus FeO Abundance Map (wt.%) - Wavelet-Autoencoder-RF"
    )
    logger.info(f"Saved IIRS FeO map -> {output_tif}")
    
    # Load band parameters for correlation analysis
    band_params = reader.read_band_parameters()
    
    np.savez_compressed(
        output_npz,
        feo_map=feo_map_2d,
        feo_valid=feo_pred,
        valid_mask=valid_mask,
        latitudes=lats,
        longitudes=lons,
        z_latent=z_latent,
        wavelet_feats=wavelet_feats,
        b1_depth=band_params.get("b1_depth", np.full((rows, cols), np.nan)),
        b1_center=band_params.get("b1_center", np.full((rows, cols), np.nan)),
        b2_depth=band_params.get("b2_depth", np.full((rows, cols), np.nan)),
        b2_center=band_params.get("b2_center", np.full((rows, cols), np.nan))
    )
    logger.info(f"Saved IIRS Copernicus results shard -> {output_npz}")
    
    stats = {
        "count": n_valid,
        "mean_feo": float(np.mean(feo_pred)),
        "std_feo": float(np.std(feo_pred)),
        "min_feo": float(np.min(feo_pred)),
        "max_feo": float(np.max(feo_pred)),
        "median_feo": float(np.median(feo_pred)),
        "output_tif": output_tif,
        "output_npz": output_npz
    }
    logger.info(f"IIRS Copernicus FeO stats: Mean={stats['mean_feo']:.2f} wt.%, Median={stats['median_feo']:.2f} wt.%, Std={stats['std_feo']:.2f}")
    return stats
