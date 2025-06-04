#!/bin/bash
#SBATCH --ntasks=1                                                             # Number of tasks (see below)
#SBATCH --cpus-per-task=8                                                      # Number of CPU cores per task
#SBATCH --nodes=1                                                              # Ensure that all cores are on one machine
#SBATCH --partition=h100-ferranti                                              # Partition to use
#SBATCH --time=3-00:00                                                         # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                           # Request 1 GPU
#SBATCH --mem=100G                                                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/weka/bethge/dziadzio08/opencompass/slurm/outputs/job_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/weka/bethge/dziadzio08/opencompass/slurm/outputs/job_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                        # Email to which notifications will be sent

scontrol show job $SLURM_JOB_ID
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD:$PYTHONPATH

source $HOME/.bashrc
source $HOME/wild-llamas/.venv/bin/activate

python main.py