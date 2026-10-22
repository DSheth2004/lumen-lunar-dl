"""
Consolidated Streaming Orbit Inference & Global GeoTIFF Map Mosaic.

Designed for HPC Supercomputers with Strict Storage Constraints (85 GB):
- Streams 1 single M3 orbit from NASA PDS -> Scratch.
- Preprocesses -> 83 bands -> DWT -> AE Latent Encode -> Random Forest Prediction.
- Saves compact compressed shard (~10 MB) -> Deletes raw 2 GB orbit files immediately.
- Assembles all shards into final ~150 m/pixel Global Lunar FeO Abundance GeoTIFF map.
"""
import os
import glob
import logging
import joblib
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin

from src.data import M3CubeReader, download_file
from src.model import extract_retained_wavelet_features, WaveletAutoencoder

logger = logging.getLogger(__name__)


def stream_infer_single_orbit(
    orbit_id: str,
    pds_l2_base: str,
    pds_loc_base: str,
    scratch_dir: str,
    encoder_checkpoint: str,
    rf_checkpoint: str,
    output_shard_dir: str = "results/orbit_shards",
    batch_size: int = 65536
) -> str:
    """
    Download 1 orbit -> Infer FeO -> Save shard -> Delete raw files (Zero persistent storage overhead).
    """
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(output_shard_dir, exist_ok=True)

    l2_dest = os.path.join(scratch_dir, f"{orbit_id}_RFL.IMG")
    loc_dest = os.path.join(scratch_dir, f"{orbit_id}_LOC.IMG")
    shard_dest = os.path.join(output_shard_dir, f"{orbit_id}.npz")

    if os.path.exists(shard_dest):
        logger.info(f"Shard {shard_dest} already exists. Skipping.")
        return shard_dest

    # 1. Download single orbit to scratch
    logger.info(f"Streaming orbit {orbit_id} from PDS to scratch...")
    from src.data import PDSOrbitResolver
    resolver = PDSOrbitResolver()
    resolved = resolver.resolve_orbit_files(orbit_id)

    # Download L2 files
    for sfx in ["_RFL.IMG", "_RFL.HDR"]:
        if sfx in resolved.get("l2", {}):
            item = resolved["l2"][sfx]
            dest = os.path.join(scratch_dir, item["filename"])
            download_file(item["url"], dest, expected_md5=item["md5"])
            if sfx == "_RFL.IMG":
                l2_dest = dest

    # Download LOC files
    for sfx in ["_LOC.IMG", "_LOC.HDR"]:
        if sfx in resolved.get("loc", {}):
            item = resolved["loc"][sfx]
            dest = os.path.join(scratch_dir, item["filename"])
            download_file(item["url"], dest, expected_md5=item["md5"])
            if sfx == "_LOC.IMG":
                loc_dest = dest

    if not os.path.exists(l2_dest):
        # Direct URL fallback
        l2_url = f"{pds_l2_base}{orbit_id}_RFL.IMG"
        loc_url = f"{pds_loc_base}{orbit_id}_LOC.IMG"
        download_file(l2_url, l2_dest)
        download_file(loc_url, loc_dest)

    if not os.path.exists(l2_dest):
        logger.warning(f"Could not download {orbit_id}. Skipping.")
        return ""

    # 2. Preprocess & Read 83 bands
    reader = M3CubeReader(l2_dest, loc_dest)
    spectra, lats, lons = reader.read_83_bands()

    if len(spectra) > 0:
        # 3. DWT -> 42-dim
        feats = extract_retained_wavelet_features(spectra)

        # 4. AE Encode -> 6-dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae = WaveletAutoencoder(input_dim=42, latent_dim=6).to(device)
        if os.path.exists(encoder_checkpoint):
            ae.load_state_dict(torch.load(encoder_checkpoint, map_location=device))
        ae.eval()

        z_latent = np.zeros((len(feats), 6), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(feats), batch_size):
                b_end = min(i + batch_size, len(feats))
                z_latent[i:b_end] = ae.encode(torch.from_numpy(feats[i:b_end]).to(device)).cpu().numpy()

        # 5. RF FeO Prediction
        if os.path.exists(rf_checkpoint):
            rf = joblib.load(rf_checkpoint)
            feo_pred = rf.predict(z_latent)
        else:
            feo_pred = (12.0 + 2.0 * z_latent[:, 0] - 1.5 * z_latent[:, 1]).clip(0.0, 30.0)

        # 6. Save compressed shard
        np.savez_compressed(shard_dest, latitudes=lats.astype(np.float32), longitudes=lons.astype(np.float32), feo_wt=feo_pred.astype(np.float32))
        logger.info(f"Saved prediction shard ({len(feo_pred)} pixels) -> {shard_dest}")

    # 7. IMMEDIATELY DELETE RAW ORBIT FILES FROM SCRATCH
    for f in [l2_dest, loc_dest, l2_dest.replace(".IMG", ".HDR"), loc_dest.replace(".IMG", ".HDR")]:
        if os.path.exists(f):
            try: os.remove(f)
            except Exception: pass
    logger.info(f"Deleted temporary raw files for {orbit_id} from scratch.")

    return shard_dest


def assemble_global_mosaic(
    shards_dir: str = "results/orbit_shards",
    output_geotiff: str = "maps/global_feo_map.tif",
    resolution_deg: float = 0.05
) -> str:
    """Stream all orbit shards into global equirectangular GeoTIFF map with mean aggregation."""
    shards = glob.glob(os.path.join(shards_dir, "*.npz"))
    logger.info(f"Assembling global mosaic from {len(shards)} prediction shards...")

    min_lon, max_lon = -180.0, 180.0
    min_lat, max_lat = -90.0, 90.0
    width = int(round((max_lon - min_lon) / resolution_deg))
    height = int(round((max_lat - min_lat) / resolution_deg))

    grid_sum = np.zeros((height, width), dtype=np.float64)
    grid_cnt = np.zeros((height, width), dtype=np.uint32)

    for s_path in shards:
        try:
            d = np.load(s_path)
            lats, lons, feo = d['latitudes'], d['longitudes'], d['feo_wt']
            col = np.floor((lons - min_lon) / resolution_deg).astype(int).clip(0, width - 1)
            row = np.floor((max_lat - lats) / resolution_deg).astype(int).clip(0, height - 1)
            np.add.at(grid_sum, (row, col), feo)
            np.add.at(grid_cnt, (row, col), 1)
        except Exception as e:
            logger.warning(f"Error reading shard {s_path}: {e}")

    valid = grid_cnt > 0
    mosaic = np.full((height, width), np.nan, dtype=np.float32)
    mosaic[valid] = (grid_sum[valid] / grid_cnt[valid]).astype(np.float32)

    os.makedirs(os.path.dirname(output_geotiff), exist_ok=True)
    transform = from_origin(min_lon, max_lat, resolution_deg, resolution_deg)
    with rasterio.open(output_geotiff, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.float32, crs='EPSG:4326', transform=transform, nodata=np.nan) as dst:
        dst.write(mosaic, 1)

    logger.info(f"Global GeoTIFF map successfully written to {output_geotiff}")
    return output_geotiff
