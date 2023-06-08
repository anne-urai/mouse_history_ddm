#!/bin/bash -l

# https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/pages/37028013/Your+first+GPU+job
#SBATCH --output /home/uraiae/jobs/hddmnn_fit-%A_%a.out
#SBATCH --mail-user=a.e.urai@fsw.leidenuniv.nl # mail when done
#SBATCH --mail-type=END,FAIL # mail when done
#SBATCH --partition=gpu-medium
#SBATCH --time=1-00:00:00 # one day to fit, should be enough for simple HDDMnn models
#SBATCH --ntasks=1 # submit one job per task
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=90G

# load necessary modules
module purge
module load Miniconda3/4.9.2
module load gcc/8.2.0 # activate GCC for Fortran compiler
source activate hddmnn_env  # for all installed packages 
# export PYTHONUNBUFFERED=TRUE # use -u to continually show output in logfile (unbuffered, bad when writing to home or data)

# are we using the GPU?
echo "[$SHELL] This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] Starting at "$(date)
echo "[$SHELL] Running on node $HOSTNAME"
echo "[$SHELL] Conda env: "$CONDA_DEFAULT_ENV
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

# Actually run the file with input args, only one trace_id for now
python /home/uraiae/code/int-brain-lab/mouse_history_ddm/hddmnn_fit.py -d $1 -m $2

# Wrap up
echo "[$SHELL] Finished at "$(date)