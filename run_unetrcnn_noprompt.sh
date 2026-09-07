#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=dino_512_auxdec1024_unetrcnn_noprompt_f0
#SBATCH --output=logs/dino_pca/%x-%A-%a.log

export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY=${WANDB_API_KEY:?set your own wandb API key, e.g. via `wandb login` or exporting it here}
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export DINOV3_LIBRARY_PATH=/path/to/your/dinov3/clone  # git clone https://github.com/ObedDzik/dinov3
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store/dinov3

module load python/3.12 cuda/12.2
module load opencv/4.12.0

source /path/to/your/venv/bin/activate  # set up your own venv per README.md
srun python -m dino_pca.train_unetr_cnn -c dino_pca/cfg.yaml \
  fold=0 name=dino_512_auxdec1024_unetrcnn_noprompt_0
