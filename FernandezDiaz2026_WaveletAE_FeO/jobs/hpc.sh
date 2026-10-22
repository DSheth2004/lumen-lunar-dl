#!/bin/bash
# ==============================================================================
# Parameterized SLURM Execution Script — Supercomputer / HPC Cluster
# Usage:
#   sbatch --export=STAGE=benchmark jobs/hpc.sh
#   sbatch --export=STAGE=train jobs/hpc.sh
#   sbatch --export=STAGE=evaluate jobs/hpc.sh
#   sbatch --export=STAGE=regression jobs/hpc.sh
#   sbatch --export=STAGE=robustness jobs/hpc.sh
#   sbatch --export=STAGE=mosaic jobs/hpc.sh
#   sbatch --export=STAGE=iirs jobs/hpc.sh
#   sbatch --export=STAGE=m3-copernicus jobs/hpc.sh
#   sbatch --export=STAGE=compare-m3-iirs jobs/hpc.sh
#   sbatch --export=STAGE=all jobs/hpc.sh
# ==============================================================================
#SBATCH --job-name=lunar_feo
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hpc_%j.out
#SBATCH --error=logs/hpc_%j.err

echo "=== Lunar FeO HPC Job Started on $(date) ==="
echo "Node: $(hostname) | Working Dir: $(pwd)"
echo "Home: $HOME | Scratch: $SCRATCH"

mkdir -p logs data/raw/scratch checkpoints results figures maps manifests

# Default stage if not exported
STAGE=${STAGE:-all}

case $STAGE in
    benchmark)
        echo "Running Step: Benchmark Preparation & Table 1 Evaluation"
        python run.py --stage data-prep
        python run.py --stage evaluate-wavelet
        ;;
    train)
        echo "Running Step: Wavelet Autoencoder GPU Training"
        python run.py --stage train-ae --epochs 300
        ;;
    evaluate)
        echo "Running Step: Embedding Comparison (Tables 2, 3 & Figure 2)"
        python run.py --stage evaluate-embeddings
        ;;
    regression)
        echo "Running Step: Supervised Regression (Tables 4, 5, 6)"
        python run.py --stage regression
        ;;
    robustness)
        echo "Running Step: Noise Robustness Analysis (Table 7 & Figure 3)"
        python run.py --stage robustness
        ;;
    mosaic)
        echo "Running Step: Global GeoTIFF Map Mosaic & Figures 4, 5"
        python run.py --stage mosaic
        python run.py --stage compare-paper
        ;;
    iirs)
        echo "Running Step: Chandrayaan-2 IIRS FeO Inference (Copernicus Crater)"
        python run.py --stage iirs-infer
        ;;
    m3-copernicus)
        echo "Running Step: M3 Copernicus Scene FeO Extraction"
        python run.py --stage m3-copernicus
        ;;
    compare-m3-iirs)
        echo "Running Step: Cross-Instrument Evaluation (M3 vs. IIRS)"
        python run.py --stage iirs-infer
        python run.py --stage m3-copernicus
        python run.py --stage compare-m3-iirs
        ;;
    all)
        echo "Running Full End-to-End Pipeline including M3 and IIRS Cross-Comparison"
        python run.py --stage all
        ;;
    *)
        echo "Unknown stage: $STAGE"
        exit 1
        ;;
esac

echo "=== Lunar FeO HPC Job Finished on $(date) ==="
