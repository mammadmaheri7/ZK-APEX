import os
import random
from typing import Tuple

import numpy as np
import torch
from huggingface_hub import login

from taker.data_classes import PruningConfig
from taker.parser import cli_parser, is_true
from taker.weight_pruning import run_weight_based_masking, per_weight_prune


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
        choices=["ABS", "All Neg", "Second Order Neg", "First Order Neg", "All positive"],
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
        "--weight_based",
        type=str,
        default="True",
        help="Use weight-based masking instead of neuron-based masking (true/false).",
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
        "--compensation_lr",
        type=float,
        default=1.0,
        help="Learning rate used when applying compensation updates.",
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
        "--personalize_model",
        type=str,
        default="False",
        help="Re-train the personalization model instead of loading the cached checkpoint (true/false).",
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
        "--labels",
        type=int,
        nargs='+',
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146],
        help="Forget Set Labels",
    )


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _run_experiment() -> Tuple[float, float, float]:
    c = PruningConfig(
        wandb_project="testing",
        model_repo="google/vit-base-patch16-224",
        token_limit=1000,
        run_pre_test=False,
        ff_frac=0.02,
        ff_eps=0.001,
        attn_frac=0.0,
        attn_eps=1e-4,
        focus="imagenet-1k-birdless",
        cripple="imagenet-1k-birds",
        # misc=1,
        misc=0,
        additional_datasets=tuple(),
        # recalculate_activations=True,
        recalculate_activations=False,

        dtype="fp32", 
        n_steps=1,
        collection_sample_size=1e5,
        eval_sample_size=1e5,



    )

    c, args = cli_parser(c, add_args_fn=_add_hparam_args)

    c.num_grads = args.comp_num_grads
    c.fisher_block_size = args.fisher_block_size_mask

    print(c.model_repo)
    retain_dataloader = is_true(args.retain_dataloader)
    eval_pruned = is_true(args.eval_pruned)
    weight_based = is_true(args.weight_based)
    switch_m_mask = is_true(args.switch_m_mask)
    switch_m_comp = is_true(args.switch_m_comp)
    print_personalization_scores = is_true(args.print_personalization_scores)
    print_pruned_only_scores = is_true(args.print_pruned_only_scores)
    personalize_model = is_true(args.personalize_model)

    personalization_hps = {
        "num_epochs": args.personalization_epochs,
        "base_lr": args.personalization_base_lr,
        "layer_decay": args.personalization_layer_decay,
        "freeze_layers": args.personalization_freeze_layers,
    }

    mask_path = run_weight_based_masking(
        c,
        retain_dataloader=retain_dataloader,
        forget_frac=args.forget_fraction,
        hessian_coefficient=args.hessian_coefficient,
        retain_coefficient=args.retain_coefficient,
        eval_pruned=eval_pruned,
        weight_based=weight_based,
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

    forget_acc, sketch_acc, imagenet_acc = per_weight_prune(
        c,
        mask_path,
        personalization_hps=personalization_hps,
        print_personalization_scores=print_personalization_scores,
        fisher_block_size=args.fisher_block_size_comp,
        damp=args.damp_compensate,
        num_grads=args.comp_num_grads,
        print_pruned_only_scores=print_pruned_only_scores,
        switch_m=switch_m_comp,
        compensation_lr=args.compensation_lr,
        fisher_m_multiplier=args.fisher_m_multiplier,
        personalize_model=personalize_model,
    )

    print(f"Compensated forget accuracy:  {forget_acc:.6f}")
    print(f"Compensated sketch accuracy:  {sketch_acc:.6f}")
    print(f"Compensated imagenet accuracy:{imagenet_acc:.6f}")

    objective_primary = abs(forget_acc - 0.50)
    objective_secondary = 0.5 * sketch_acc + 0.5 * imagenet_acc
    print(f"Objective metrics -> abs(forget-0.5): {objective_primary:.6f}, blended score: {objective_secondary:.6f}")

    return forget_acc, sketch_acc, imagenet_acc


def main() -> None:
    SEED = 42
    _set_random_seeds(SEED)

    # Ensure shared artifacts retain permissive permissions when running multiple jobs.
    os.umask(0)
    _run_experiment()


if __name__ == "__main__":
    print("Starting weight pruning experiment...")
    main()
