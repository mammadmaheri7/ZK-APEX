pip install ./taker_mmd/
cd exp1-vit-pruning/

python run_weight_pruning_cli.py google/vit-base-patch16-224 --retain_coefficient 2.0 --hessian_coefficient 0.4484965300937087 --forget_fraction 0.04 --fisher_block_size_mask 64 --mask_num_grads 480 --damp_masking 1e-08 --signs "Second Order Neg" --switch_m_mask True --fisher_block_size_comp 576 --comp_num_grads 480 --damp_compensate 1e-06 --switch_m_comp False --fisher_m_multiplier 10 --print_personalization_scores False