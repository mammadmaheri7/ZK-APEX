from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_snip_pruning, run_pruning, run_snip_pruning_flags
from taker.fine_tune import prune_personalize_and_compensate, prune_personalize_and_compensate_hp_tuning
from taker.weight_scoring_advanced import per_weight_score_personalize_and_prune_layer
from taker.weight_pruning_llm import per_weight_prune_llm, run_weight_based_masking_llm, personalize_and_save_model, load_finetuned_for_eval, save_codeparrot_loaders, personalize_and_save_model_lora
from huggingface_hub import login
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import random
import optuna
import gc
import numpy as np
import sys
import copy

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



def objective(trial):
    device = torch.device("cuda:0")
    base  = AutoModelForCausalLM.from_pretrained("model-combined-c++-c-scala-java", torch_dtype=torch.bfloat16)
    base.to(device)
    peft  = PeftModel.from_pretrained(base, "models/opt125-lora-epoch-5-max_steps-None-20250914-202918-langs-Rust", is_trainable=False)
    peft.to(device)
    model = peft.merge_and_unload() 
    model.to(device)

    
    # retain_coefficient = trial.suggest_categorical("retain_coefficient", [0, 0.5, 1.5, 2])
    retain_coefficient = trial.suggest_categorical("retain_coefficient", [0.5, 1.5, 2])

    fisher_block_size_mask = trial.suggest_categorical("fisher_block_size_mask", [64, 256, 200, 576, 1152, 1536])

    
    fisher_block_size_comp = trial.suggest_categorical("fisher_block_size_comp", [64, 256, 200, 576, 1152, 1536])
    damp_compensate = trial.suggest_categorical("damp_compensate", [1e-8, 1e-7, 1e-6, 1e-3, 1e-2])
    damp_masking = trial.suggest_categorical("damp_masking", [1e-8, 1e-7, 1e-6, 1e-3])
    switch_m_comp = trial.suggest_categorical("switch_m_comp", [True, False])
    switch_m_mask = trial.suggest_categorical("switch_m_mask", [True, False])
    forget_fraction = trial.suggest_categorical("forget_fraction", [0.01, 0.02, 0.04, 0.08, 0.1, 0.16])
    fisher_m_multiplier = trial.suggest_categorical("fisher_m_multiplier", [0.5, 1, 5, 10])
    # signs = trial.suggest_categorical("signs", ["Second Order Neg", "First Order Neg", "Abs Hessian Neg"])
    signs = trial.suggest_categorical("signs", ["Second Order Neg", "Abs Hessian Neg"])


    num_grads_compensate_fisher = trial.suggest_categorical("num_grads_compensate_fisher", [240, 480, 960, 1920])

    
    hessian_coefficient = trial.suggest_float("hessian_coefficient", 0.0, 1)
    # switch_m = trial.suggest_categorical("switch_m", [True, False])

    langs = ["C++", "C", "Scala", "Java", "Rust"]
    
    
    path = run_weight_based_masking_llm(c, hessian_coefficient = hessian_coefficient, forget_frac = forget_fraction, retain_dataloader = True, 
                                        retain_coefficient = retain_coefficient, fisher_block_size = fisher_block_size_mask, 
                                        num_grads=480, damp = damp_masking, switch_m = switch_m_mask, signs = signs)
    # path = "./outputs/llm_intermediate_row_masks__retaindl_True_retaincoefficient_2_-20251029-071814_forgetfrac_0.01.pt"
    
    
    score = per_weight_prune_llm(c, path, langs = langs, model = model,  personalization_hps = p_hps,  
                                 print_personalization_scores = False, fisher_block_size = fisher_block_size_comp, num_grads = num_grads_compensate_fisher, 
                                 print_pruned_only_scores = True, damp = damp_compensate, switch_m = switch_m_comp, 
                                 fisher_m_multiplier = fisher_m_multiplier)

    
    return (score[0], abs(score[1] - 0.6149846552997772))

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision('high')
    np.random.seed(SEED)
    random.seed(SEED)
    p_hps = {
        "num_epochs": 30,
        "base_lr": 8e-4,
        "layer_decay": 0.8,
        "freeze_layers": 0,
    }

    # if str(c.misc)[-1] == "1":
    #     forget_frac = c.misc / 10000
    #     normalized = True
    # else:
    #     forget_frac = c.misc / 20000
    #     normalized = False
    
    # tokenizer
    langs = ["C++", "C", "Scala", "Java", "Rust"]
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # save_codeparrot_loaders(langs, tokenizer)

    # base  = AutoModelForCausalLM.from_pretrained("model-combined-c++-c-scala-java", torch_dtype=torch.bfloat16)
    # peft  = PeftModel.from_pretrained(base, "models/opt125-lora-epoch-0-max_steps-None-20250914-195611-langs-Rust", is_trainable=False)
    # merged = peft.merge_and_unload()                     # now weights contain W_base + ΔW_lora
    # merged.save_pretrained("models/merged-rust-only-0-epchs")
    # model = load_finetuned_for_eval("models/merged-rust-only-0-epchs")
    # personalize_and_save_model_lora(langs, model = model)
    # print(f"forget frac is {forget_frac} normalized is {normalized}")

    # sys.exit()

    # model = load_finetuned_for_eval("models/opt125-epoch-5-max_steps-None-20250912-200406-langs-Rust")
    # path = run_weight_based_masking_llm(c,  forget_frac = c.misc/100, retain_dataloader = True, retain_coefficient = 2, switch_m = False, num_grads = 1000)
    # score = per_weight_prune_llm(c, path, model = model, langs = langs,  personalization_hps = p_hps,  
    #                              print_personalization_scores = True, fisher_block_size = 128, num_grads = 1000, 
    #                              print_pruned_only_scores = True, switch_m = True, damp = 1e-2,)
    # model, tokenizer = load_finetuned_for_eval("models/opt125-epoch-0-max_steps-None-20250906-070307")
    # ["C", "C++", "Java", "Scala"]
    # model, tokenizer = load_finetuned_for_eval("models/opt125-epoch-4-max_steps-None-20250907-062013-langs-Rust")
    # model.config.pad_token_id = tokenizer.pad_token_id
    # personalize_and_save_model(["Rust"], num_epochs = 6, model = None, tokenizer = None)

    


   
    #path = "./outputs/llm_intermediate_row_masks__retaindl_True_retaincoefficient_0.5_-20251028-002111_forgetfrac_0.1.pt"
    #path = run_weight_based_masking_llm(c,  hessian_coefficient = 0.5, forget_frac = 0.1, retain_dataloader = True, retain_coefficient = 0.5, switch_m = True, num_grads = 480)
    #score = per_weight_prune_llm(c, path, langs = langs, model = copy.deepcopy(merged),  personalization_hps = p_hps,  
     #                        print_personalization_scores = False, fisher_block_size = 256, num_grads = 250, 
      #                       print_pruned_only_scores = True, switch_m = True, damp = 1e-05)


    local_db = "/homes/sc2124/optuna_databases/optuna_llm_final_ft_fixed.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{local_db}",
        engine_kwargs={
            "connect_args": {"timeout": 60}
        },
    )
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=storage,
        study_name="optuna_llm_final_ft_fixed",
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=1) 
