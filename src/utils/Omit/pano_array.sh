#!/bin/bash
#SBATCH --account=ms-rthornton42
#SBATCH --job-name=panopticon
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-6   # 7 jobs: 0,1,2,3,4,5,6


source  /work/projects/ms-rthornton42/rthornton42/DatGen/bin/activate


cd   /work/projects/ms-rthornton42/rthornton42/DatGen/PANOPTICON-DataGen/

# SLURM_ARRAY_TASK_ID will be 0..6
python src/prompt_gen.py \
  --num-chunks 7 \
  --chunk-index ${SLURM_ARRAY_TASK_ID}
