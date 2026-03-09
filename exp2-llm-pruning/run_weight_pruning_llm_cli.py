import os
import random
from typing import Tuple, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from taker.data_classes import PruningConfig
from taker.parser import cli_parser, is_true, split_list
from taker.weight_pruning_llm import run_weight_based_masking_llm, per_weight_prune_llm, load_finetuned_for_eval


def _add_hparam_args(parser) -> None:
    parser.add_argument(
        "--retain_coefficient",
        type=float,
        required=True,
        help="Coefficient used when combining forget and retain scores during mask generation.",
    )
    parser.add_argument(
        "--hessian_coefficient",
        type=float,
        required=True,
        help="Weight applied to the Hessian term inside the masking score.",
    )
    parser.add_argument(
        "--forget_fraction",
        type=float,
        required=True,
        help="Fraction of weights to prune when building the mask.",
    )
    parser.add_argument(
        "--fisher_block_size_mask",
        type=int,
        required=True,
        help="Block size for Fisher inverse estimation during masking.",
    )
    parser.add_argument(
        "--mask_num_grads",
        type=int,
        required=True,
        help="Number of gradient samples for Fisher estimation during masking.",
    )
    parser.add_argument(
        "--damp_masking",
        type=float,
        required=True,
        help="Damping used while estimating Fisher information for masking.",
    )
    parser.add_argument(
        "--signs",
        type=str,
        choices=["Second Order Neg", "First Order Neg", "Abs Hessian Neg"],
        required=True,
        help="Scoring mode used inside the masking routine.",
    )
    parser.add_argument(
        "--switch_m_mask",
        type=str,
        required=True,
        help="Whether to switch the Fisher M-matrix term during masking (true/false).",
    )
    parser.add_argument(
        "--retain_dataloader",
        type=str,
        default="True",
        help="Enable retain dataloader usage for weighting retain scores (true/false).",
    )
    parser.add_argument(
        "--eval_pruned",
        type=str,
        default="False",
        help="Run the pruning-only evaluation before compensation (true/false).",
    )
    parser.add_argument(
        "--fisher_block_size_comp",
        type=int,
        required=True,
        help="Block size for Fisher inverse estimation during compensation.",
    )
    parser.add_argument(
        "--comp_num_grads",
        type=int,
        required=True,
        help="Number of gradient samples for Fisher estimation during compensation.",
    )
    parser.add_argument(
        "--damp_compensate",
        type=float,
        required=True,
        help="Damping used while estimating Fisher information for compensation.",
    )
    parser.add_argument(
        "--switch_m_comp",
        type=str,
        required=True,
        help="Whether to switch the Fisher M-matrix term during compensation (true/false).",
    )
    parser.add_argument(
        "--fisher_m_multiplier",
        type=float,
        required=True,
        help="Multiplier applied to the Fisher M-matrix during compensation.",
    )
    parser.add_argument(
        "--print_personalization_scores",
        type=str,
        default="False",
        help="Print scores for the personalized (pre-pruned) model (true/false).",
    )
    parser.add_argument(
        "--print_pruned_only_scores",
        type=str,
        default="True",
        help="Print scores for the pruned-only model before compensation (true/false).",
    )
    parser.add_argument(
        "--before_masking_evaluation",
        type=str,
        default="True",
        help="Print scores for the personalized model before masking (true/false).",
    )
    parser.add_argument(
        "--personalization_epochs",
        type=int,
        default=30,
        help="Number of epochs for the personalization fine-tuning stage.",
    )
    parser.add_argument(
        "--personalization_base_lr",
        type=float,
        default=8e-4,
        help="Base learning rate for personalization fine-tuning.",
    )
    parser.add_argument(
        "--personalization_layer_decay",
        type=float,
        default=0.8,
        help="Layer-wise learning rate decay for personalization.",
    )
    parser.add_argument(
        "--personalization_freeze_layers",
        type=int,
        default=0,
        help="Number of encoder layers to freeze during personalization.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="C++,C,Scala,Java,Rust",
        help="Comma-separated list of languages to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path or repo id of a merged/standalone model to load.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path or repo id to use when applying a LoRA adapter.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA adapter path to load and merge into the base model.",
    )


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _load_llm_model(args, device: torch.device) -> torch.nn.Module:
    if args.lora_path is not None:
        if args.base_model_path is None:
            raise ValueError("--base_model_path is required when --lora_path is provided.")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
        )
        base.to(device)
        peft = PeftModel.from_pretrained(base, args.lora_path, is_trainable=False)
        peft.to(device)
        model = peft.merge_and_unload()
        model.to(device)
        return model

    if args.model_path is None:
        raise ValueError("Provide either --model_path or --lora_path/--base_model_path to load a model.")

    return load_finetuned_for_eval(args.model_path, device=str(device))


def _run_experiment() -> Tuple[float, float]:
    c = PruningConfig(
        wandb_project="testing",
        model_repo="nickypro/tinyllama-15m",
        token_limit=1000,
        run_pre_test=False,
        ff_frac=0.02,
        ff_eps=0.001,
        attn_frac=0.0,
        attn_eps=1e-4,
        focus="pile_codeless",
        cripple="code",
        misc=1,
        additional_datasets=tuple(),
        recalculate_activations=True,
    )

    c, args = cli_parser(c, add_args_fn=_add_hparam_args)

    c.num_grads = args.comp_num_grads
    c.fisher_block_size = args.fisher_block_size_mask

    retain_dataloader = is_true(args.retain_dataloader)
    eval_pruned = is_true(args.eval_pruned)
    switch_m_mask = is_true(args.switch_m_mask)
    switch_m_comp = is_true(args.switch_m_comp)
    print_personalization_scores = is_true(args.print_personalization_scores)
    print_pruned_only_scores = is_true(args.print_pruned_only_scores)
    before_masking_evaluation = is_true(args.before_masking_evaluation)

    personalization_hps = {
        "num_epochs": args.personalization_epochs,
        "base_lr": args.personalization_base_lr,
        "layer_decay": args.personalization_layer_decay,
        "freeze_layers": args.personalization_freeze_layers,
    }

    langs: List[str] = split_list(args.langs)

    mask_path = run_weight_based_masking_llm(
        c,
        retain_dataloader=retain_dataloader,
        forget_frac=args.forget_fraction,
        hessian_coefficient=args.hessian_coefficient,
        retain_coefficient=args.retain_coefficient,
        eval_pruned=eval_pruned,
        switch_m=switch_m_mask,
        damp=args.damp_masking,
        signs=args.signs,
        fisher_block_size=args.fisher_block_size_mask,
        num_grads=args.mask_num_grads,
    )

    if isinstance(mask_path, tuple):
        mask_path, pruning_eval = mask_path
        print("Pruning-only evaluation:", pruning_eval)

    print(f"Saved mask artifacts to: {mask_path}")

    device = torch.device(c.model_device) if c.model_device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    base  = AutoModelForCausalLM.from_pretrained("model-combined-c++-c-scala-java", torch_dtype=torch.bfloat16)
    base.to(device)
    peft  = PeftModel.from_pretrained(base, "../models/opt125-lora-epoch-5-max_steps-None-20250914-202918-langs-Rust", is_trainable=False)
    peft.to(device)
    model = peft.merge_and_unload() 
    model.to(device)

    retain_acc, forget_acc = per_weight_prune_llm(
        c,
        mask_path,
        langs=langs,
        model=model,
        personalization_hps=personalization_hps,
        print_personalization_scores=print_personalization_scores,
        fisher_block_size=args.fisher_block_size_comp,
        damp=args.damp_compensate,
        num_grads=args.comp_num_grads,
        print_pruned_only_scores=print_pruned_only_scores,
        switch_m=switch_m_comp,
        fisher_m_multiplier=args.fisher_m_multiplier,
        before_masking_evaluation=before_masking_evaluation,
    )

    print(f"Compensated retain accuracy: {retain_acc:.6f}")
    print(f"Compensated forget accuracy: {forget_acc:.6f}")

    return retain_acc, forget_acc


def main() -> None:
    SEED = 42
    _set_random_seeds(SEED)

    # Ensure shared artifacts retain permissive permissions when running multiple jobs.
    os.umask(0)

    _run_experiment()


if __name__ == "__main__":
    print("Starting LLM weight pruning experiment...")
    main()
