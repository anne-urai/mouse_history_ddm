#!/bin/bash

# https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/pages/37028013/Your+first+GPU+job
#SBATCH --output /home/uraiae/jobs/hddmnn_fit-%A_%a.out
#SBATCH --mail-user=a.e.urai@fsw.leidenuniv.nl # mail when done
#SBATCH --mail-type=END,FAIL # mail when done
#SBATCH --partition=gpu-short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1GB

# load necessary modules
module purge
module load Miniconda3/4.9.2
source activate hddmnn_env  # for all installed packages (hddm_env gives a kabuki bug for some reason)
# export PYTHONUNBUFFERED=TRUE # use -u to continually show output in logfile (unbuffered, bad when writing to home or data)

# are we using the GPU?
echo "[$SHELL] This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] Starting at "$(date)
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES
echo "[$SHELL] Conda env: "$CONDA_DEFAULT_ENV

# Actually run the file with input args, only one trace_id for now
python /home/uraiae/code/int-brain-lab/mouse_history_ddm/hddmnn_fit.py -d $1 -m $2

# Wrap up
echo "[$SHELL] Finished at "$(date)