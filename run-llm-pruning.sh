#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition a40
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=sc2124 # required to send email notifcations - please replace <your_u

source /vol/bitbucket/mm6322/miniconda3/bin/activate /vol/bitbucket/mm6322/miniconda3/envs/myenv
pip install -r new_requirements.txt
pip install ./taker_mmd/
cd exp2-llm-pruning

python run_weight_pruning_llm_cli.py nickypro/tinyllama-15m \
  --retain_coefficient 1.5 \
  --hessian_coefficient 0.5 \
  --forget_fraction 0.04 \
  --fisher_block_size_mask 256 \
  --mask_num_grads 480 \
  --damp_masking 1e-6 \
  --signs "Second Order Neg" \
  --switch_m_mask True \
  --fisher_block_size_comp 256 \
  --comp_num_grads 480 \
  --damp_compensate 1e-6 \
  --switch_m_comp False \
  --fisher_m_multiplier 5 \
  --langs "C++,C,Scala,Java,Rust" \
  --base_model_path model-combined-c++-c-scala-java \
  --lora_path models/opt125-lora-epoch-5-max_steps-None-20250914-202918-langs-Rust
