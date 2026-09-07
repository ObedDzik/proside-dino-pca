#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:30:00
#SBATCH --job-name=dino_256_auxdec1024_f0
#SBATCH --output=logs/dino_pca/%x-%A-%a.log

#s30ant_heatmap_tr_nct_val_OL_test_ua
# Load modules or activate environment
# module load python/3.12 cuda/12.2
# export DATA_ROOT=/datasets/exactvu_pca/nct2013
# export PYTHONPATH=/home/obed/projects/aip-medilab/obed/medproj/medAI:$PYTHONPATH
# export PYTHONPATH
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
srun python -m dino_pca.train -c dino_pca/cfg_256.yaml
