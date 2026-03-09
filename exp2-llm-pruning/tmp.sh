#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition a30
#SBATCH --mem=4G
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=sc2124 # required to send email notifcations - please replace <your_u

# install taker_mmd version
# cd to main directory of the project (/media/mmaheri/mohamad_ssd/selective-pruning)
# clone taker_mmd -> create directory selective-pruning/taker_mmd

export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
export POETRY_HOME="/vol/bitbucket/sc2124/poetry"
export PATH="/vol/bitbucket/sc2124/miniconda/my_conda/bin:$PATH"
export TMPDIR="$POETRY_HOME/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export POETRY_CACHE_DIR="/opt/poetry/cache"
export HF_HOME="$TMPDIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"    
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_CACHE="/vol/bitbucket/mm6322/sunil/datasets"
export WANDB_CACHE_DIR="$TMPDIR/wandb/cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb/config"
export WANDB_DIR="$TMPDIR/wandb/run" 
export WANDB_ARTIFACT_DIR="$TMPDIR/wandb/artifacts"
export WANDB_DATA_DIR="$TMPDIR/wandb/data"
source ~/.bashrc
source activate ../../venv-3.10/
pip install ../taker_mmd/


# activate the virtual environment created by Poetry
# source $(poetry env info --path)/bin/activate


# rm -rf outputs/
# rm -rf saved_dataset/
umask 0000
./prune-vit.sh


# export GLOG_minloglevel=2
# export PYTHONWARNINGS="ignore"
# export TORCH_LOGS="-dynamo"
# rm -rf cache_personalization/
# python ../taker_mmd/src/taker/fine_tune.py
