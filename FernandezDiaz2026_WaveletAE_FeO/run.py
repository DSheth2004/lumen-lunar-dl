#!/usr/bin/env python3
"""
Master CLI Controller for Lunar FeO Mapping Replication Pipeline.

Usage:
  python run.py --stage data-prep
  python run.py --stage evaluate-wavelet
  python run.py --stage train-ae
  python run.py --stage evaluate-embeddings
  python run.py --stage regression
  python run.py --stage robustness
  python run.py --stage stream-infer --orbit-id M3G20090105T043603 --scratch /tmp/m3
  python run.py --stage mosaic
  python run.py --stage compare-paper
  python run.py --stage iirs-infer
  python run.py --stage m3-copernicus
  python run.py --stage compare-m3-iirs
  python run.py --stage all
"""
import argparse
import logging
import os
import sys
import yaml

from src.data import build_ground_truth_csv, create_stratified_benchmarks
from src.train import train_wavelet_autoencoder, train_and_cross_validate_regression
from src.evaluate import evaluate_wavelet_table1, evaluate_embeddings_table2_3, evaluate_robustness_table7, compare_replication_targets
from src.inference import stream_infer_single_orbit, assemble_global_mosaic
from src.visualize import plot_figure2_rate_distortion, plot_figure3_robustness, plot_figure4_global_map, plot_figure5_comparison_maps
from src.iirs import infer_iirs_feo
from src.m3_copernicus import infer_m3_copernicus_feo
from src.compare_m3_iirs import run_m3_iirs_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LunarFeO")


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Lunar FeO Mapping Replication Driver")
    parser.add_argument("--stage", required=True, choices=[
        "data-prep", "evaluate-wavelet", "train-ae", "evaluate-embeddings",
        "regression", "robustness", "stream-infer", "mosaic", "figures",
        "iirs-infer", "m3-copernicus", "compare-m3-iirs",
        "compare-paper", "all"
    ], help="Pipeline execution stage")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--orbit-id", default="M3G20090105T043603", help="Orbit ID for stream-infer")
    parser.add_argument("--scratch", default=os.getenv("SCRATCH", "data/raw/scratch"), help="HPC Scratch directory")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-wavelet", type=int, default=5000000)
    parser.add_argument("--n-ae", type=int, default=1000000)

    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1. Data Preparation
    if args.stage in ("data-prep", "all"):
        logger.info("=== STAGE 1: Data Preparation ===")
        build_ground_truth_csv()
        create_stratified_benchmarks(
            orbit_h5_dir="data/intermediate/preprocessed",
            n_wavelet=args.n_wavelet,
            n_ae=args.n_ae
        )

    # 2. Wavelet Table 1 Benchmark
    if args.stage in ("evaluate-wavelet", "all"):
        logger.info("=== STAGE 2: Wavelet Benchmark (Table 1) ===")
        evaluate_wavelet_table1()

    # 3. Autoencoder Training
    if args.stage in ("train-ae", "all"):
        logger.info("=== STAGE 3: Unsupervised Wavelet Autoencoder Training ===")
        train_wavelet_autoencoder(
            benchmark_h5="data/intermediate/benchmark_1M.h5",
            latent_dim=args.latent_dim,
            max_epochs=args.epochs
        )

    # 4. Embeddings & Dimension Sweeps (Tables 2, 3 & Figure 2)
    if args.stage in ("evaluate-embeddings", "all"):
        logger.info("=== STAGE 4: Embedding Comparisons & Dimension Sweeps ===")
        evaluate_embeddings_table2_3()
        plot_figure2_rate_distortion()

    # 5. Supervised Regression (Tables 4, 5, 6)
    if args.stage in ("regression", "all"):
        logger.info("=== STAGE 5: Supervised Regression & Final Test ===")
        train_and_cross_validate_regression()

    # 6. Robustness (Table 7 & Figure 3)
    if args.stage in ("robustness", "all"):
        logger.info("=== STAGE 6: Robustness Analysis under Noise ===")
        evaluate_robustness_table7()
        plot_figure3_robustness()

    # 7. Orbit Streaming Inference
    if args.stage == "stream-infer":
        logger.info(f"=== STAGE 7: Streaming Orbit Inference ({args.orbit_id}) ===")
        stream_infer_single_orbit(
            orbit_id=args.orbit_id,
            pds_l2_base="https://pds-imaging.jpl.nasa.gov/data/m3/CH1M3_0004/DATA/",
            pds_loc_base="https://pds-imaging.jpl.nasa.gov/data/m3/CH1M3_0003/DATA/",
            scratch_dir=args.scratch,
            encoder_checkpoint="checkpoints/ae_latent_6.pt",
            rf_checkpoint="checkpoints/rf_final.joblib"
        )

    # 8. Global Mosaic & Final Maps (Figures 4 & 5)
    if args.stage in ("mosaic", "all"):
        logger.info("=== STAGE 8: Global Map Assembly ===")
        assemble_global_mosaic()
        plot_figure4_global_map()
        plot_figure5_comparison_maps()

    # 9. Master Replication Target Check
    if args.stage in ("compare-paper", "all"):
        logger.info("=== STAGE 9: Replication Verification against Paper Targets ===")
        compare_replication_targets()

    # 10. Chandrayaan-2 IIRS FeO Inference
    if args.stage in ("iirs-infer", "all"):
        logger.info("=== STAGE 10: Chandrayaan-2 IIRS FeO Estimation (Copernicus Crater) ===")
        infer_iirs_feo(
            encoder_checkpoint="checkpoints/ae_latent_6.pt",
            rf_checkpoint="checkpoints/rf_final.joblib",
            output_tif=cfg.get("iirs", {}).get("output_feo_tif", "maps/iirs_copernicus_feo.tif")
        )

    # 11. M3 Copernicus Scene FeO Retrieval
    if args.stage in ("m3-copernicus", "all"):
        logger.info("=== STAGE 11: M3 Copernicus Scene Extraction & FeO Retrieval ===")
        infer_m3_copernicus_feo(
            encoder_checkpoint="checkpoints/ae_latent_6.pt",
            rf_checkpoint="checkpoints/rf_final.joblib",
            output_tif=cfg.get("iirs", {}).get("output_m3_copernicus_tif", "maps/m3_copernicus_feo.tif")
        )

    # 12. Cross-Instrument Comparison: M3 vs. IIRS
    if args.stage in ("compare-m3-iirs", "all"):
        logger.info("=== STAGE 12: Cross-Instrument Evaluation (M3 vs. IIRS) ===")
        run_m3_iirs_comparison(
            output_diff_tif=cfg.get("iirs", {}).get("output_diff_tif", "maps/copernicus_feo_difference_iirs_minus_m3.tif")
        )


if __name__ == "__main__":
    main()
