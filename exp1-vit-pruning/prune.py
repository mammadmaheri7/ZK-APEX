from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_snip_pruning, run_pruning, run_snip_pruning_flags
from taker.fine_tune import prune_personalize_and_compensate, prune_personalize_and_compensate_hp_tuning
from taker.weight_pruning import run_weight_based_masking, get_midlayaer_scoring, per_weight_prune, try_get_pruned_grads
from huggingface_hub import login
import random
import optuna
import gc
import numpy as np


import torch

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
    model_repo    = "nickypro/tinyllama-15m",
    token_limit   = 1000,
    # run_pre_test  = True,
    run_pre_test= False,
    # Removals parameters
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_frac = 0.00,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    misc = 1,
    additional_datasets = tuple(),
    recalculate_activations = True, # iterative vs non-iterative pruning
)

# Parse CLI for arguments
c, args = cli_parser(c)

# TODO: set prpoer values
c.num_grads = 2048
c.fisher_block_size = 50



# path = run_weight_based_masking(c, retain_dataloader = False, forget_frac = 0.05, hessian_coefficient = 0)
# try_col_major(c, path)


def objective(trial):
    print("in objective")
    # Sample hyperparameters

    retain_coefficient = trial.suggest_categorical("retain_coefficient", [0, 0.5, 1.5, 2])
    hessian_coefficient = trial.suggest_float("hessian_coefficient", 0.0, 1)
    fisher_block_size_mask = trial.suggest_categorical("fisher_block_size_mask", [64, 256, 200, 576, 1152, 1536])
    fisher_block_size_comp = trial.suggest_categorical("fisher_block_size_comp", [64, 256, 200, 576, 1152, 1536])
    num_grads = trial.suggest_categorical("num_grads", [128, 256, 480])
    damp_compensate = trial.suggest_categorical("damp_compensate", [1e-8, 1e-7, 1e-6, 1e-3])
    damp_masking = trial.suggest_categorical("damp", [1e-8, 1e-7, 1e-6, 1e-3])
    signs = trial.suggest_categorical("signs", ["ABS", "All Neg", "Second Order Neg", "First Order Neg", "All positive"])
    switch_m_comp = trial.suggest_categorical("switch_m_comp", [True, False])
    switch_m_mask = trial.suggest_categorical("switch_m_mask", [True, False])
    forget_fraction = trial.suggest_categorical("forget_fraction", [0.01, 0.02, 0.04, 0.08, 0.1, 0.16])
    fisher_m_multiplier = trial.suggest_categorical("fisher_m_multiplier", [0.5, 1, 5, 10,])
   
   



    p_hps = {
        "num_epochs": 30,
        "base_lr": 8e-4,
        "layer_decay": 0.8,
        "freeze_layers": 0,
    }

    path = run_weight_based_masking(c, retain_dataloader = True, forget_frac = forget_fraction, hessian_coefficient = hessian_coefficient, 
                                    retain_coefficient = retain_coefficient, eval_pruned = False, 
                                    weight_based = True, switch_m = switch_m_mask, damp = damp_masking, signs = signs, fisher_block_size = fisher_block_size_mask)
    score = per_weight_prune(c, path, personalization_hps = p_hps, damp = damp_compensate, print_personalization_scores = False, 
                             fisher_block_size = fisher_block_size_comp, num_grads = num_grads, print_pruned_only_scores = True, switch_m = switch_m_comp, 
                             compensation_lr = 1, fisher_m_multiplier = fisher_m_multiplier)
    


    
    return (abs(score[0] -0.50), 0.5*score[1] + 0.5*score[2])


def objective_compensate_only(trial):
    path = "./outputs/vit_intermediate_row_masks__retaindl_False_retaincoefficient_1_-20250904-202244_forgetfrac_0.02.pt"
    p_hps = {
            "num_epochs": 30,
            "base_lr": 8e-4,
            "layer_decay": 0.8,
            "freeze_layers": 0,
    }
    fisher_block_size = trial.suggest_categorical("fisher_block_size", [64, 256, 576, 1152, 1536])
    num_grads = trial.suggest_categorical("num_grads", [128, 256, 480])
    damp_compensate = trial.suggest_categorical("damp_compensate", [1e-9, 1e-7, 1e-5, 1e-3, 1e-2])
    fisher_m_multiplier = trial.suggest_categorical("fisher_m_multiplier", [1, 5, 10, 20])
    switch_m = trial.suggest_categorical("switch_m", [True, False])

    score = per_weight_prune(c, path, personalization_hps = p_hps,  print_personalization_scores = False, fisher_block_size = fisher_block_size, 
                                 num_grads =  num_grads, print_pruned_only_scores = True, switch_m = switch_m, damp = damp_compensate, compensation_lr = 1, 
                                 use_basis = False, fisher_m_multiplier = fisher_m_multiplier)
    
    return score


def objective_prune_only(trial):
    p_hps = {
            "num_epochs": 30,
            "base_lr": 8e-4,
            "layer_decay": 0.8,
            "freeze_layers": 0,
    }
    fisher_block_size = trial.suggest_categorical("fisher_block_size", [36, 48, 64, 72])
    hessian_coefficient = trial.suggest_float("hessian_coefficient", 0.0, 3)
    num_grads = trial.suggest_categorical("num_grads", [128, 256, 480])
    damp = trial.suggest_categorical("damp", [1e-9, 1e-7, 1e-5, 1e-3, 1e-2])
    switch_m = trial.suggest_categorical("switch_m", [True, False])
    negative_first_order = trial.suggest_categorical("negative_first_order", [True, False])
    negative_second_order = trial.suggest_categorical("negative_second_order", [True, False])

    score = run_weight_based_masking(c, fisher_block_size = fisher_block_size, retain_dataloader = False, forget_frac = 0.02, hessian_coefficient = hessian_coefficient, 
            retain_coefficient = 1, eval_pruned = True, weight_based = True, switch_m = switch_m, damp = damp, num_grads = num_grads, normalize_by_layer = False, 
            negative_first_order = negative_first_order, negative_second_order = negative_second_order) 

    
    return score

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision('high')
    np.random.seed(SEED)
    random.seed(SEED)


    import os
    os.umask(0)

    p_hps = {
        "num_epochs": 30,
        "base_lr": 8e-4,
        "layer_decay": 0.8,
        "freeze_layers": 0,
    }

    local_db = "/homes/sc2124/optuna_databases/optuna_instance_based_full_pipeline_afterdebug.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{local_db}",
        engine_kwargs={
            "connect_args": {"timeout": 60}
        },
    )

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        storage=storage,
        study_name="optuna_instance_based_full_pipeline_afterdebug",
        load_if_exists=True,
    )

    print("after create study")
    study.optimize(objective, n_trials=1) 

