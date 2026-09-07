#!/bin/bash

export CHECKPOINT=/scratch/obed
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY=${WANDB_API_KEY:?set your own wandb API key, e.g. via `wandb login` or exporting it here}
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export DINOV3_LIBRARY_PATH=/path/to/your/dinov3/clone  # git clone https://github.com/ObedDzik/dinov3
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store

module load python/3.12 cuda/12.2
module load opencv/4.12.0
source /path/to/your/venv/bin/activate  # set up your own venv per README.md
export CUDA_LAUNCH_BLOCKING=1

exec python -m dino_pca.train -c dino_pca/cfg.yaml