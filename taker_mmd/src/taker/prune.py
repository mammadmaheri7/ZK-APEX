import math
import os
from typing import Optional, List

import numpy as np
import torch
import wandb
import copy

from .model import Model
from .data_classes import PruningConfig, RunDataHistory, \
                          RunDataItem, ActivationOverview
from .eval import evaluate_all
from .scoring import score_indices_by, score_indices
from .activations import get_midlayer_data, get_top_frac, \
    choose_attn_heads_by, save_timestamped_tensor_dict
from .texts import prepare
from torch import Tensor
from .texts import prepare_dataset, infer_dataset_config
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from torch.utils.data import RandomSampler, SequentialSampler
from huggingface_hub import login
import torch.nn as nn
def prune_and_evaluate(
        opt: Model,
        original_opt: Model, # Only used for compensation which require that the model is not pruned
        pruning_config: PruningConfig,
        focus_out: Optional[dict] = None,
        cripple_out: Optional[dict] = None,
        iteration: Optional[int] = None,
        compensation: bool = True,
        compensation_lr: float = 1.0,
    ):
    """
    Prune and evaluate the model

    Args:
        opt (Model): model to prune and evaluate
        pruning_config (PruningConfig): config for pruning
        focus_out (dict): output of get_midlayer_data for focus dataset
        cripple_out (dict): output of get_midlayer_data for cripple dataset
        iteration (int): iteration number for when activations are not recalculated

    Returns:
        output (RunDataItem): Eval data to add to RunDataHistory.
    """
    c = copy.deepcopy(pruning_config)

    # Find out what we are doing
    do_ff   = pruning_config.ff_frac > 0
    do_attn = pruning_config.attn_frac > 0
    if not do_ff and not do_attn:
        raise ValueError("Must prune at least one of FF or Attention")
    if do_attn and pruning_config.attn_mode not in ["pre-out", "value"]:
        raise NotImplementedError("attn_mode must be 'pre-out' or 'value'")

    # Get midlayer activations of FF and ATTN
    if pruning_config.recalculate_activations:
        focus_out   = get_midlayer_data( opt, pruning_config.focus,
            pruning_config.collection_sample_size, pruning_config.attn_mode )
        cripple_out = get_midlayer_data( opt, pruning_config.cripple,
            pruning_config.collection_sample_size, pruning_config.attn_mode )

    # Otherwise, import activation data, and adjust the "pruning fraction"
    else:
        c["ff_frac"]   = min( 1.0, c["ff_frac"]*(iteration+1) )
        c["attn_frac"] = min( 1.0, c["attn_frac"]*(iteration+1) )
        assert not (focus_out is None or cripple_out is None or iteration is None), \
            "Must provide focus_out and cripple_out if not recalculate_activations"
        
    # Prune the model using the activation data
    data = score_and_prune(opt, focus_out, cripple_out, c, c.save)

    # print("======== START ========")
    # # check the percentages of 1 and 0 values in the prune_mask
    # print(f'prune mask 1 percentage: {np.sum(data.raw["ff_criteria"]==1)/data.raw["ff_criteria"].size}') # .numel() is not a numpy function
    # print(f'prune mask 0 percentage: {np.sum(data.raw["ff_criteria"]==0)/data.raw["ff_criteria"].size}') # .numel() is not a numpy function
    # print("======== FISHER INVS ========")

    # save the prune_mask to file
    directory = f"outputs/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save/update the prune_mask (it will be used in fine_tune.py as the unlearning mask)
    if iteration == 0:
        torch.save(data.raw["ff_criteria"], f"{directory}/prune_mask.pt")
        print(f"Saved prune_mask to {directory}/prune_mask.pt" , f'sparsity: {np.sum(data.raw["ff_criteria"]) / data.raw["ff_criteria"].size}')
        combined_mask = torch.tensor(data.raw["ff_criteria"])
    else:
        # Load the existing prune_mask
        existing_mask = torch.load(f"{directory}/prune_mask.pt")
        # Combine the new mask with the existing mask
        combined_mask = torch.logical_or(torch.tensor(existing_mask), torch.tensor(data.raw["ff_criteria"]))
        # Save the combined mask
        torch.save(combined_mask, f"{directory}/prune_mask.pt")
        print(f"Updated prune_mask and saved to {directory}/prune_mask.pt" , f'sparsity: {torch.sum(combined_mask) / combined_mask.numel()}')

    if not compensation:
        pass
    else:
        print(f'--------------- compensation stage ---------------')

        # TODO: undo this one
         # Evaluate the model 
        with torch.no_grad():
            print(f'********* starting evalution after pruning / before compensation stage *********')
            eval_out = evaluate_all(opt, c.eval_sample_size, c.datasets,
                                    dataset_tokens_to_skip=c.collection_sample_size)
            data.update(eval_out)
        print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
        print(f'----- debug - model performance after pruning / before compensation -> eval_out: {print_eval_out} ----- \n\n')
        

        model = original_opt
        model.model.train() # TODO: maybe eval works better

        # create the dataset which used for calculation of fisher matrix
        dataset_name = pruning_config.focus
        eval_config = infer_dataset_config(dataset_name)
        eval_config.dataset_split = "train"
        eval_config.is_train_mode = True
        # if dataset_texts_to_skip is not None:
        #     eval_config.num_texts_to_skip = dataset_texts_to_skip
        # if "MaskedLM" in opt.cfg.architecture and masked_mode:
        #     eval_config.masked_model = True
        print(f'---- debug eval.config.masked_model: {eval_config.masked_model} ----')
        dataset    = prepare_dataset(eval_config)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3‑channel
            transforms.ToTensor(),
        ])
        if hasattr(dataset, 'transform'):
            dataset.transform = transform
        else:
            # fallback: wrap dataset, choose wrapper based on base type
            if isinstance(dataset, IterableDataset):
                dataset = WrappedIterableDataset(dataset,transform)
            else:
                dataset = WrappedDataset(dataset,transform)

        try:
            _ = len(dataset)
            has_len = True
        except TypeError:
            has_len = False
            
        sampler = SequentialSampler(dataset) if has_len else None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,          # <‑‑ one sample ⇒ per‑sample grad
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )

        device = next(model.model.parameters()).device
        loss_fn=nn.CrossEntropyLoss()

        # register hooks to capture gradients
        layer_grads = [None] * 12  # One for each encoder layer
        def make_hook(layer_idx):
            def hook(grad):
                layer_grads[layer_idx] = grad.detach().cpu()
            return hook

        print("=== START FISHER INVS COMPUTATION ===")
        # compute the gradients
        num_grads = pruning_config.num_grads
        # build one EmpiricalBlockFisherInverse per layer to stream grads into
        fisher_invs = [
            EmpiricalBlockFisherInverse(
                num_grads         = num_grads,
                fisher_block_size = pruning_config.fisher_block_size,
                num_weights       = 3072 * 768,   # one MLP matrix per layer
                damp              = 1e-3,
                device            = device,
            )
            for _ in range(12)
        ]

        counter = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.model.zero_grad()
            
            # Register hooks
            handles = []
            for i in range(12):
                layer = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
                handles.append(layer.register_hook(make_hook(i)))

            # Backward pass
            with torch.set_grad_enabled(True):
                logits = model.get_logits(pixel_values=inputs)
                logits = logits[:, 0, :]  # take only CLS token for classification
                loss = loss_fn(logits, targets)
                # print("Acc:", (logits.argmax(dim=-1) == targets).float().mean().item())
                loss.backward()

            # Add the grads to fisher inverses
            if all(g is not None for g in layer_grads):
                # Stream each layer's gradient into its own inverse
                for i, g_i in enumerate(layer_grads):
                    v = g_i.flatten().to(device)
                    v = v / (v.norm() + 1e-12)
                    fisher_invs[i].add_grad(v)
                    del v  # free memory
                layer_grads = [None] * 12

            # Remove hooks
            for h in handles:
                h.remove()

            counter += 1
            if counter >= num_grads:
                break

        torch.cuda.empty_cache()
        print("=== END FISHER INVS COMPUTATION ===")

        # print("======== START ========")
        # # check the percentages of 1 and 0 values in the prune_mask
        print(f'prune mask 1 percentage: {np.sum(data.raw["ff_criteria"]==1)/data.raw["ff_criteria"].size}') # .numel() is not a numpy function
        print(f'prune mask 0 percentage: {np.sum(data.raw["ff_criteria"]==0)/data.raw["ff_criteria"].size}') # .numel() is not a numpy function
        # print("======== FISHER INVS ========")

        # compute and apply the compensation (optimal brain surgery - obs) on the model
        prune_and_compensate_all_layers(
            model.model,
            prune_mask = combined_mask,
            fisher_invs = fisher_invs,
            learning_rate=compensation_lr,
            block_size=pruning_config.fisher_block_size, damp=1e-3, device=device
        )

        # evaluation the model after pruning and compensation
        with torch.no_grad():
            print(f'********* starting evalution after pruning+compensation *********')
            eval_out = evaluate_all(model, c.eval_sample_size, c.datasets,
                                    dataset_tokens_to_skip=c.collection_sample_size)
            data.update(eval_out)
        # exclude 'token_count' from eval_out to print
        print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
        print(f'----- debug - model performance after pruning+compensation -> eval_out: {print_eval_out} ----- \n\n')


    return data

def score_and_prune( opt: Model,
            focus_activations_data: ActivationOverview,
            cripple_activations_data: ActivationOverview,
            pruning_config: PruningConfig,
            save=False,
        ):
    # Get the top fraction FF activations and prune
    ff_frac, ff_eps     = pruning_config.ff_frac,   pruning_config.ff_eps
    attn_frac, attn_eps = pruning_config.attn_frac, pruning_config.attn_eps
    do_ff   = ff_frac > 0
    do_attn = attn_frac > 0

    act_subset = pruning_config.scoring_normalization
    if do_ff > 0:
        ff_focus_data   = focus_activations_data.mlp[act_subset]
        ff_cripple_data = cripple_activations_data.mlp[act_subset]
        ff_scoring_fn = score_indices_by(pruning_config.ff_scoring)

        ff_scores = ff_scoring_fn(opt, ff_focus_data, ff_cripple_data, ff_eps)
        ff_criteria, ff_threshold = get_top_frac(ff_scores, ff_frac)

        # TODO: uncomment this one - I've commment to compute the fisher without bug
        # opt.hooks.delete_mlp_neurons(ff_criteria)
        

    # Get the top fraction of Attention activations and prune
    if do_attn > 0:
        attn_focus_data   = focus_activations_data.attn[act_subset]
        attn_cripple_data = cripple_activations_data.attn[act_subset]
        # scoring for attention
        attn_scoring_fn = score_indices_by(pruning_config.attn_scoring)
        attn_scores = attn_scoring_fn(opt, attn_focus_data, attn_cripple_data, attn_eps)

        # offset by means if desired (probably bad?)
        means = None
        if pruning_config.do_attn_mean_offset:
            means = attn_focus_data["mean"]

        # get criteria for "neurons", or for "heads" if using full heads
        if pruning_config.attn_prune_heads:
            attn_head_scoring_fn = \
                choose_attn_heads_by(pruning_config.attn_prune_heads_mode)
            attn_criteria, attn_threshold = \
                attn_head_scoring_fn(opt, attn_scores, attn_frac)
            attn_criteria = opt.expand_remove_heads_to_remove_indices(attn_criteria)
        else:
            attn_criteria, attn_threshold = get_top_frac(attn_scores, attn_frac)
            _shape = (opt.cfg.n_layers, opt.cfg.n_heads, opt.cfg.d_head)
            attn_criteria = attn_criteria.reshape(_shape)

        # get criteria and prune if using only attention neurons
        if pruning_config.attn_mode == "pre-out":
            opt.hooks.delete_attn_neurons( attn_criteria) # TODO: add option for means
        elif pruning_config.attn_mode == "value":
            raise NotImplementedError("'value' pruning not yet implemented in v1")
            opt.delete_attn_values( attn_criteria, means )
        else:
            raise NotImplementedError("attn_mode must be 'pre-out' or 'value'")

    # Save the removals to file
    tensor_data = {
        "ff_scores": ff_scores if do_ff else None,
        # FIXME: doesn't return attn_std_mean
        "attn_scores": attn_scores if do_attn else None,
        "ff_criteria": ff_criteria if do_ff else None,
        "attn_criteria": attn_criteria if do_attn else None,
    }

    if save:
        subdirectory = f"{pruning_config.save_subdirectory}/" or ""
        path = f"{subdirectory}saved_tensors/{opt.model_size}"
        filename = f"{pruning_config.cripple}-{pruning_config.focus}-{opt.model_size}-{pruning_config.ff_frac}-recent.pt"
        save_timestamped_tensor_dict( opt, tensor_data, "activation_metrics", path, filename )

    # Initialize the output dictionary
    data = RunDataItem()

    data.update({'deletions': {
        "ff_threshold": ff_threshold if do_ff else 0,
        "attn_threshold": attn_threshold if do_attn else 0,
        "ff_del": float( torch.sum(ff_criteria) ) if do_ff else 0,
        "attn_del": float( torch.sum(attn_criteria) ) if do_attn else 0,
    }})

    data.update({'deletions_per_layer': {
        'ff': ff_criteria.sum(dim=-1).tolist() if do_ff else [],
        'attn': attn_criteria.sum(dim=-1).tolist() if do_attn else [],
    }})

    # Save removals and scores to history
    _numpify = lambda x: x.cpu().numpy() if x is not None else None
    data.update({'raw': {
        k: _numpify(v) for k,v in tensor_data.items()
    }})

    return data

def prune_random( opt: Model,
        ff_frac: float,
        attn_frac: float,
        ff_pruned: Optional[np.ndarray] = None,
        attn_pruned: Optional[np.ndarray] = None,
        ):
    """Prune a random fraction of FF and Attention weights
    Args:
        opt (Model): model to prune and evaluate
        ff_frac (float): fraction of FF to prune
        attn_frac (float): fraction of Attention to prune
        ff_pruned: list of which mlp neurons have already been pruned
        attn_pruned: list of which attn neurons have already been pruned

    Returns:
        ff_pruned: updated list of which mlp neurons have already been pruned
        attn_pruned: updated list of which attn neurons have already been pruned
        data_out: summary info on which neurons have been pruned
    """
    if ff_pruned is None:
        ff_pruned = np.zeros( (opt.cfg.n_layers, opt.cfg.d_mlp), dtype=np.bool_ )
    if attn_pruned is None:
        attn_pruned = np.zeros( (opt.cfg.n_layers, opt.cfg.d_model ), dtype=np.bool_ )

    n_ff_to_prune   = int( ff_frac   * opt.cfg.d_mlp )
    n_attn_to_prune = int( attn_frac * opt.cfg.d_model )

    # First prune the FF
    if not ff_frac == 0:
        for layer in range( opt.cfg.n_layers ):
            # choose new ff neurons to prune
            indices = np.where(ff_pruned[layer] == 0)[0]
            random_indices = np.random.choice(indices, n_ff_to_prune, replace=False)
            ff_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_ff_keys( ff_pruned )

    if not attn_frac == 0:
        for layer in range( opt.cfg.n_layers ):
            # choose new attention heads to prune
            indices = np.where(attn_pruned[layer] == 0)[0]
            random_indices = np.random.choice(indices, n_attn_to_prune, replace=False)
            attn_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_attn_pre_out( attn_pruned )

    data_out = {
        "ff_del": n_ff_to_prune*opt.cfg.n_layers,
        "attn_del": n_attn_to_prune*opt.cfg.n_layers
    }
    return ff_pruned, attn_pruned, data_out

def prune_random_and_evaluate( opt: Model,
        c: PruningConfig,
        ff_pruned: Optional[np.ndarray] = None,
        attn_pruned: Optional[np.ndarray] = None,
        ):
    """
    To use, run once with ff_pruned=None and attn_pruned=None, then run again
    with the parameters given as output passed back in.

    Args:
        opt (Model): The model to prune and evaluate
        c (PruningConfig): The pruning configuration
        ff_pruned (Optional[np.ndarray]): Bool list of FF neurons, default None.
        attn_pruned (Optional[np.ndarray], optional: Bool list of ATTN neurons, default None.

    Returns:
        ff_pruned (Optional[np.ndarray]):
        attn_pruned (Optional[np.ndarray]):
        data (RunDataItem):
    """


    # Prune the model randomly
    ff_pruned, attn_pruned, data_out = \
        prune_random( opt, c.ff_frac, c.attn_frac, ff_pruned, attn_pruned )

    # Initialize the output dictionary
    data = RunDataItem()

    # Evaluate the model
    data.update(
        evaluate_all( opt, c.eval_sample_size, c.datasets,
                      dataset_tokens_to_skip=c.collection_sample_size )
    )

    data.update({'deletions': data_out })

    data.update({'deletions_per_layer': {
        'ff': ff_pruned.sum(axis=-1).tolist() if (not ff_pruned is None) else 0,
        'attn': attn_pruned.sum(axis=-1).tolist() if (not attn_pruned is None) else 0,
    }})

    return ff_pruned, attn_pruned, data

import torch, numpy as np
from typing import List, Tuple



# @torch.no_grad()
# def prune_and_compensate_all_layers_CAP_global_quota(
#     model: torch.nn.Module,
#     prune_mask: torch.Tensor,                                # [num_layers, out], True => this row is ELIGIBLE
#     fisher_invs: List["EmpiricalBlockFisherInverseCAP"],
#     layer_params: Optional[List[Tuple[str, torch.nn.Parameter]]] = None,
#     blocks_in_parallel: int = 1024,                          # batch size in blocks; tune for speed/memory
#     prune_frac: float = 0.5,                                 # global fraction over eligible weights in the layer
#     eps: float = 1e-12,
# ):
#     """
#     CAP-style OBS with rank-1 downdates using a GLOBAL quota over all eligible weights in a layer.
#     `prune_mask[L, row] == True` marks rows that are *eligible* to be pruned. We will prune a global
#     fraction `prune_frac` of the eligible weights across those rows, allocating per-block quotas by
#     saliency (cheapest extra removals get one of the leftovers).

#     Requirements:
#       • fisher_invs[i].B divides in_features (e.g., 48/64/96/128 for ViT-B’s 768)
#       • fisher_invs[i].F_inv is block-diagonal (num_blocks = out * (in/B), BxB blocks)
#     """
#     # 0) discover target layers EXACTLY like the Fisher pass (use model.model)
#     if layer_params is None:
#         layer_params = [
#             (n, p) for n, p in model.model.named_parameters()
#             if n.endswith(".intermediate.dense.weight")
#         ]
#         def _idx(n):
#             try: return int(n.split("encoder.layer.")[1].split(".")[0])
#             except Exception: return 10**9
#         layer_params.sort(key=lambda kv: _idx(kv[0]))

#     assert len(layer_params) == len(fisher_invs), \
#         f"layers={len(layer_params)} vs finvs={len(fisher_invs)}"

#     # ensure boolean CPU row mask
#     if not isinstance(prune_mask, torch.Tensor):
#         prune_mask = torch.tensor(prune_mask)
#     prune_mask = prune_mask.bool().cpu()
#     assert prune_mask.shape[0] == len(layer_params), \
#         f"row mask L={prune_mask.shape[0]} != {len(layer_params)}"

#     for L, ((name, p), finv) in enumerate(zip(layer_params, fisher_invs)):
#         W = p.data                                    # (out, in) on (likely) CUDA
#         out, in_features = W.shape
#         B = finv.B
#         assert in_features % B == 0, f"B={B} must divide in={in_features}"
#         blocks_per_row = in_features // B
#         Nblocks = out * blocks_per_row

#         # reshape so each length-B block stays within a row
#         W_blocks = W.view(Nblocks, B)                 # on W.device
#         Hinv = finv.F_inv                              # (Nblocks, B, B) on finv.dev (often CPU)

#         # collect ALL block indices that correspond to eligible rows
#         eligible_rows = torch.nonzero(prune_mask[L], as_tuple=False).flatten()
#         if eligible_rows.numel() == 0:
#             continue

#         all_blk_idx = []
#         for r in eligible_rows.tolist():
#             start = r * blocks_per_row
#             all_blk_idx.extend(range(start, start + blocks_per_row))

#         blk_indices_cpu = torch.tensor(all_blk_idx, dtype=torch.long, device=Hinv.device)
#         blk_indices_gpu = blk_indices_cpu.to(W_blocks.device)

#         # Choose compute device (GPU for speed) while keeping master Hinv on CPU
#         compute_dev = torch.device("cuda") if torch.cuda.is_available() else Hinv.device

#         # -----------------------------
#         # Pass 1: PLAN per-block quotas
#         # -----------------------------
#         # We need for each block i:
#         #   C_i  = number of candidate (nonzero) weights (initially)
#         #   cost_i = cost of the next cheapest removal (min w^2 / diag(Hinv))
#         # Do this in mini-batches to limit memory.
#         step = blk_indices_cpu.numel() if (blocks_in_parallel is None or blocks_in_parallel <= 0) \
#                else int(blocks_in_parallel)

#         C_list = []
#         cost_list = []

#         for s in range(0, blk_indices_cpu.numel(), step):
#             sel_cpu = blk_indices_cpu[s:s + step]
#             sel_gpu = blk_indices_gpu[s:s + step]

#             # take slices and move to compute device
#             Wb   = W_blocks.index_select(0, sel_gpu).to(compute_dev)   # (Bz, B)
#             Hinb = Hinv.index_select(0, sel_cpu).to(compute_dev)       # (Bz, B, B)

#             # Candidate set = ALL positions in these blocks (full rows eligible),
#             # but treat pre-existing zeros as already-pruned and not candidates.
#             M0 = (Wb == 0)                                            # pre-zeroed
#             C_i = (B - M0.sum(dim=1)).to(torch.int64)                 # (Bz,)
#             C_list.append(C_i.cpu())

#             # next cheapest score per block (inf if no candidates left)
#             Hdiag  = torch.diagonal(Hinb, dim1=1, dim2=2).clamp_min(eps)   # (Bz, B)
#             scores = (Wb * Wb) / Hdiag
#             scores[M0] = float('inf')
#             next_cost, _ = torch.min(scores, dim=1)                        # (Bz,)
#             cost_list.append(next_cost.cpu())

#         C_all = torch.cat(C_list, dim=0)               # (M,) where M = #eligible blocks
#         cost_all = torch.cat(cost_list, dim=0)         # (M,)
#         total_candidates = int(C_all.sum().item())
#         if total_candidates == 0:
#             # nothing to prune in this layer
#             continue

#         K = int(math.floor(prune_frac * total_candidates))     # global target removals
#         base = torch.floor(prune_frac * C_all.float()).to(torch.int64)    # (M,)
#         assigned = int(base.sum().item())
#         R = max(0, K - assigned)

#         # extras go to R blocks with SMALLEST next_cost (where pruning one more is "cheapest")
#         extra = torch.zeros_like(base)
#         if R > 0:
#             # blocks that can still accept extras (i.e., C_i > base_i)
#             can_accept = (C_all > base)
#             # rank by next_cost among can_accept
#             # use +inf for blocks that cannot accept or have no candidates
#             cost_rank = cost_all.clone()
#             cost_rank[~can_accept] = float('inf')
#             if (cost_rank != float('inf')).any():
#                 # pick up to R cheapest
#                 R_eff = min(R, int((cost_rank != float('inf')).sum().item()))
#                 vals, idx = torch.topk(-cost_rank, k=R_eff, largest=True)  # negative for smallest cost
#                 extra[idx] = 1
#         quota = base + extra                                             # (M,)
#         # clamp to available candidates
#         quota = torch.minimum(quota, C_all)

#         # Keep a global counter of how many have already been pruned per block (start with pre-zeros)
#         pruned_so_far = (C_all * 0).to(torch.int64)    # (M,), we’ll recompute per batch from M0

#         # ------------------------------------------------
#         # Pass 2: PRUNE until each block meets its quota
#         # ------------------------------------------------
#         # We process blocks in the same batches as planning; in each batch
#         # we keep pruning only blocks that still need removals.
#         offset = 0
#         for s in range(0, blk_indices_cpu.numel(), step):
#             sel_cpu = blk_indices_cpu[s:s + step]
#             sel_gpu = blk_indices_gpu[s:s + step]
#             Bz = sel_cpu.numel()

#             # slice & move to compute device
#             Wb   = W_blocks.index_select(0, sel_gpu).to(compute_dev)   # (Bz, B)
#             Hinb = Hinv.index_select(0, sel_cpu).to(compute_dev)       # (Bz, B, B)

#             # initial masks/counters in this batch
#             M = (Wb == 0)                                             # already pruned
#             C_i = (B - M.sum(dim=1)).to(torch.int64)                  # candidates left initially
#             # local targets = quota for these blocks
#             q_local = quota[offset:offset + Bz].to(compute_dev)       # (Bz,)
#             # how many are already pruned in this batch (pre-zeros)
#             pruned_local = (B - C_i).to(torch.int64)                  # (Bz,)
#             # remaining removals needed in this batch
#             need_left = (q_local - pruned_local).clamp_min(0)         # (Bz,)
#             # active rows (need_left > 0)
#             active = (need_left > 0)

#             while active.any():
#                 act = torch.nonzero(active, as_tuple=False).flatten()     # indices in 0..Bz-1

#                 Hdiag  = torch.diagonal(Hinb, dim1=1, dim2=2).clamp_min(eps)  # (Bz, B)
#                 scores = (Wb * Wb) / Hdiag
#                 # Only allow indices not yet pruned in this batch
#                 remaining = ~M
#                 # Mask out completed rows entirely
#                 mask_done = torch.ones_like(scores, dtype=torch.bool)
#                 mask_done[act] = False
#                 scores[mask_done] = float('inf')
#                 scores[~remaining] = float('inf')

#                 # pick argmin per active row
#                 j = torch.argmin(scores[act], dim=1)                         # (|act|,)

#                 # OBS closed form on active rows
#                 col  = Hinb[act, j, :]                                       # (|act|, B)
#                 d    = Hdiag[act, j].clamp_min(eps)                          # (|act|,)
#                 stepv = (Wb[act, j] / d).unsqueeze(1)                        # (|act|, 1)
#                 Wb[act] -= col * stepv

#                 # mark & zero the chosen indices
#                 M[act, j] = True
#                 Wb[act, j] = 0.0

#                 # rank-1 downdate
#                 u = col / d.sqrt().unsqueeze(1)
#                 Hinb[act] -= torch.bmm(u.unsqueeze(2), u.unsqueeze(1))

#                 # update remaining need per active row
#                 need_left[act] -= 1
#                 active = need_left > 0

#             # write back this batch
#             W_blocks.index_copy_(0, sel_gpu, Wb.to(W_blocks.dtype).to(W_blocks.device))
#             Hinv.index_copy_(0, sel_cpu, Hinb.to(Hinv.dtype).to(Hinv.device))

#             offset += Bz

#         # restore original shape
#         p.data = W_blocks.view(out, in_features)

        
# @torch.no_grad()
# def prune_and_compensate_all_layers_CAP(
#     model: nn.Module,
#     prune_mask: torch.Tensor,                 # shape: [num_layers, out]; True => prune this row completely
#     fisher_invs: List["EmpiricalBlockFisherInverseCAP"],
#     layer_params: Optional[List[tuple[str, torch.nn.Parameter]]] = None,
#     blocks_in_parallel: int = -1,            # CAPHandle.blocks_in_parallel; -1 => all rows at once
#     eps: float = 1e-12,
#     device: Optional[torch.device] = None,
#     prune_frac = 0.9
# ):
#     """
#     Apply CAP-style OBS compensation given a row-level mask.
#     For each layer L and each row marked for pruning, we iteratively:
#       - choose the next scalar weight to remove by minimizing (w^2 / diag(H^-1))
#       - apply OBS closed-form update to that row’s weights
#       - rank-1 downdate the row’s inverse block (Sherman–Morrison)
#     until the entire row is zeroed.

#     Notes:
#       • Expects fisher_invs[i].F_inv to have shape (out, in, in): one inverse block per row.
#       • We operate ONLY on rows where prune_mask[L, row] == True; other rows remain unchanged.
#     """
#      # discover layers if not passed
#     if layer_params is None:
#         layer_params = [
#             (n, p) for n, p in model.named_parameters()
#             if n.endswith(".intermediate.dense.weight")
#         ]
#         def _idx(n):
#             try: return int(n.split("encoder.layer.")[1].split(".")[0])
#             except: return 10**9
#         layer_params.sort(key=lambda kv: _idx(kv[0]))

#     assert len(layer_params) == len(fisher_invs)

#     for L, ((name, p), finv) in enumerate(zip(layer_params, fisher_invs)):
#         W = p.data  # (out, in)
#         out, in_features = W.shape
#         B = finv.B
#         assert in_features % B == 0, f"Block size {B} must divide in_features {in_features}"
#         blocks_per_row = in_features // B

#         # reshape so each length-B block lies wholly inside a row
#         W_blocks = W.view(out * blocks_per_row, B)               # (Nblocks, B)
#         Hinv = finv.F_inv                                        # (Nblocks, B, B)
#         assert Hinv.shape[0] == out * blocks_per_row

#        # --- collect block indices covering the rows to prune ---
#         rows = torch.nonzero(prune_mask[L].bool(), as_tuple=False).flatten()
#         if rows.numel() == 0:
#             continue

#         blk_idx_list = []
#         for r in rows.tolist():
#             start = r * blocks_per_row
#             blk_idx_list.extend(range(start, start + blocks_per_row))

#         # Build indices on CPU first (for Hinv), then mirror to W's device when needed
#         blk_indices_cpu = torch.tensor(blk_idx_list, dtype=torch.long, device=Hinv.device)
#         blk_indices_gpu = blk_indices_cpu.to(W.device)   # same values, GPU device
        
#         step = blk_indices_cpu.numel() if (blocks_in_parallel is None or blocks_in_parallel <= 0) else int(blocks_in_parallel)
        
#         # --- process in mini-batches to cap working set ---
#         for s in range(0, blk_indices_cpu.numel(), step):
#             print(f'step {s} of f{blk_indices_cpu.numel()}')
#             sel_cpu = blk_indices_cpu[s:s + step]   # use for Hinv (CPU)
#             sel_gpu = blk_indices_gpu[s:s + step]   # use for W_blocks (GPU)

#             # Work on the inverse's device (CPU by default)
#             dev = Hinv.device

#             # Pull the selected blocks of weights with GPU indices, then move to CPU
#             Wb   = W_blocks.index_select(0, sel_gpu).to(dev)            # (Bz, B)
#             Hinb = Hinv.index_select(0, sel_cpu)                        # (Bz, B, B) already on CPU

#             # ===== CAP inner loop (unchanged) =====
#             M = (Wb == 0)
#             zeros_per_row = M.sum(dim=1)
#             min_zeros = int(zeros_per_row.min().item())
#             if min_zeros > 0:
#                 for i in range(Wb.size(0)):
#                     zero_ids = torch.nonzero(Wb[i] == 0, as_tuple=True)[0]
#                     if zero_ids.numel() > 0:
#                         take = min(min_zeros, zero_ids.numel())
#                         M[i, zero_ids[:take]] = True

#             Bz = Wb.size(0)
#             row_idx = torch.arange(Bz, device=dev)
#             print(f'using prune_frac {prune_frac}')
#             # Quota per row: prune a fraction of what's currently nonzero
#             available = (B - zeros_per_row).to(torch.int64)              # how many nonzeros left in each row
#             target_per_row = torch.floor(prune_frac * available.float()).to(torch.int64)
#             need_left = target_per_row.clamp_min(0)                      # (Bz,)

#             # Eligible positions are the ones not already pruned in this batch
#             remaining_mask = ~M                                          # (Bz, B) boolean

#             while (need_left > 0).any():
#                 active_rows = (need_left > 0)                            # (Bz,)
#                 # 1) score all rows at once
#                 Hdiag  = torch.diagonal(Hinb, dim1=1, dim2=2).clamp_min(eps)  # (Bz, B)
#                 scores = (Wb * Wb) / Hdiag                                   # (Bz, B)
#                 # ban ineligible coords and rows that are done
#                 scores[~remaining_mask] = float('inf')
#                 scores[~active_rows.unsqueeze(1)] = float('inf')

#                 # 2) pick one index per row (argmin across B)
#                 j = torch.argmin(scores, dim=1)                               # (Bz,)

#                 # 3) OBS update for all rows (gated by active_rows)
#                 col  = Hinb[row_idx, j, :]                                    # (Bz, B)
#                 d    = Hdiag[row_idx, j]                                      # (Bz,)
#                 wj   = Wb.gather(1, j.unsqueeze(1)).squeeze(1)                # (Bz,)
#                 stepv = torch.where(active_rows, wj / d, torch.zeros_like(wj))
#                 Wb   -= col * stepv.unsqueeze(1)

#                 # 4) mark & zero just-chosen coords for active rows
#                 act_idx = torch.nonzero(active_rows, as_tuple=False).flatten()
#                 M[act_idx, j[act_idx]] = True
#                 Wb[act_idx, j[act_idx]] = 0.0
#                 remaining_mask = ~M

#                 # 5) rank-1 downdate for all rows (gated)
#                 u = col / d.clamp_min(eps).sqrt().unsqueeze(1)                # (Bz, B)
#                 u *= active_rows.float().unsqueeze(1)
#                 Hinb -= torch.bmm(u.unsqueeze(2), u.unsqueeze(1))

#                 # 6) consume one quota per active row
#                 need_left = need_left - active_rows.to(need_left.dtype)
#             # ===== write back to original tensors on their native devices =====
#             W_blocks.index_copy_(0, sel_gpu, Wb.to(W_blocks.dtype).to(W_blocks.device))
#             Hinv.index_copy_(0, sel_cpu, Hinb.to(Hinv.dtype).to(Hinv.device))  # keep inverse consistent (optional)

#         # restore original shape
#         p.data = W_blocks.view(out, in_features)
######################################################################################
# Run Whole Pruning Procedure from Config
######################################################################################

def run_pruning(c: PruningConfig):
    # Initialise Model and show details about model
    print('in run_pruning')
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    # Prepare data logging
    # history = RunDataHistory(c.datasets)
    # wandb.init(
    #     project=c.wandb_project,
    #     entity=c.wandb_entity,
    #     name=c.wandb_run_name,
    #     )
    # wandb.config.update(c.to_dict(), allow_val_change=True)

    # Evaluate model before removal of any neurons
    assert c.run_pre_test==False, "run_pre_test is not supported in this version"
    if c.run_pre_test:
        print(f'********* starting evalution before pruning *********')
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        # history.add(data)
        # print(history.df.T)

    # If pruning randomly, no need to get activations
    if c.ff_scoring == "random" and c.attn_scoring == "random":
        ff_pruned, attn_pruned = None, None
        for i in range(c.n_steps):
            ff_pruned, attn_pruned, data = \
                prune_random_and_evaluate(opt, c, ff_pruned, attn_pruned)
            # history.add(data)

    # Iteratively prune neurons and evaluate
    elif c.recalculate_activations:
        original_opt = Model(
            c.model_size,
            limit=c.token_limit,
            dtype=c.dtype,
            svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator,
            model_device=c.model_device,
            mask_fn=c.mask_fn,
        )

        for i in range(c.n_steps):
            print(f'\n\n\n ----- START OF PRUNING STEP : {i} -----')

            # only compensate in the last step
            compensation = True if i == c.n_steps - 1 else False
            compensation_lr = 1.0 if compensation else 0.0
            
            # if not compensation:
            data = prune_and_evaluate(opt=opt, original_opt=None, pruning_config=c, iteration=i, compensation=compensation, compensation_lr=compensation_lr)
            # else:
            #     data = prune_and_evaluate(opt=opt, original_opt=original_opt, pruning_config=c, iteration=i, compensation=compensation, compensation_lr=compensation_lr)
            # history.add(data)

    # Non-iteratively get activations, then iteratively prune and evaluate
    else:
        print(f'----- debug - run_pruning - getting activations in else -----')
        focus_out   = get_midlayer_data(opt, c.focus,
                        c.collection_sample_size, c.attn_mode)
        cripple_out = get_midlayer_data(opt, c.cripple,
                        c.collection_sample_size, c.attn_mode)
        original_opt = Model(
            c.model_size,
            limit=c.token_limit,
            dtype=c.dtype,
            svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator,
            model_device=c.model_device,
            mask_fn=c.mask_fn,
        )
        
        for i in range(c.n_steps):
            print(f'\n\n\n ----- START OF PRUNING STEP : {i} -----')
            # only compensate in the last step
            compensation = True if i == c.n_steps - 1 else False
            compensation_lr = 1.0 if compensation else 0.0

            # if not compensation:
            data = prune_and_evaluate(opt=opt, original_opt=None, pruning_config=c, focus_out=focus_out, cripple_out=cripple_out, iteration=i, compensation=compensation, compensation_lr=compensation_lr)
            # else:
                # data = prune_and_evaluate(opt=opt, original_opt=original_opt, pruning_config=c, focus_out=focus_out, cripple_out=cripple_out, iteration=i, compensation=compensation, compensation_lr=compensation_lr)
            # history.add(data)

    # Format history to print
    # print(history.history[-1])
    # print(history.df.T)
    # print(history.df.T.to_csv())

    return opt, None
    #  history

######################################################################################
# "Forsaken"-style pruning
######################################################################################

def forsaken_pruning(c: PruningConfig,
        num_texts: int = 1,
        lr: float = 0.1,
        sigmoid_offset: float = 2.0,
        l1_norm_coeff: float = 1.0,
        ):
    # Initilaise Model and show details about model
    c.mask_fn = "sigmoid"
    c.misc = {
        "num_texts": num_texts,
        "lr": lr,
        "sigmoid_offset": sigmoid_offset,
        "l1_norm_coeff": l1_norm_coeff,
    }

    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    # Prepare data logging
    history = RunDataHistory(c.datasets)
    wandb.init(
        project=c.wandb_project,
        entity=c.wandb_entity,
        name=c.wandb_run_name,
        )
    wandb.config.update(c.to_dict())

    # Evaluate model before removal of any neurons
    if c.run_pre_test:
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)
        print(history.df.T)

    # Get activations
    focus_out   = get_midlayer_data(opt, c.focus,
                    c.collection_sample_size, c.attn_mode)
    cripple_out = get_midlayer_data(opt, c.cripple,
                    c.collection_sample_size, c.attn_mode)

    def normalize_scores(scores):
        normed_scores = []
        for score in scores:
            normed_scores.append(
                (score - score.mean()) / score.std()
            )
        return torch.stack(normed_scores)

    # Set masks for feed-forward layers
    ff_scores   = score_indices(c.ff_scoring,
        opt, focus_out.mlp.orig,   cripple_out.mlp.orig)
    ff_masks    = sigmoid_offset - normalize_scores(ff_scores)
    for layer_index in range(opt.cfg.n_layers):
        mask = opt.masks["mlp_pre_out"][layer_index]
        mask.set_mask(ff_masks[layer_index])

    # Set masks for attention heads
    attn_scores = score_indices(c.attn_scoring,
        opt, focus_out.attn.orig, cripple_out.attn.orig)
    attn_masks  = sigmoid_offset - normalize_scores(attn_scores)
    for layer_index in range(opt.cfg.n_layers):
        mask = opt.masks["attn_pre_out"][layer_index]
        mask.set_mask(attn_masks[layer_index])

    # Evaluate again now that we have adjusted the masks
    if True:
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)

    # Get parameters for back propagation
    mask_params = [
        *[p for mask in opt.masks["mlp_pre_out"] for p in mask.parameters()],
        *[p for mask in opt.masks["attn_pre_out"] for p in mask.parameters()],
    ]
    mask_l1_norm = torch.stack([
        *[(1-mask.get_mask()).mean() for mask in opt.masks["mlp_pre_out"]],
        *[(1-mask.get_mask()).mean() for mask in opt.masks["attn_pre_out"]],
    ]).mean()

    # Generate Inputs
    n_iter = 4
    optim = torch.optim.LBFGS(mask_params, lr, max_iter=n_iter)
    kl_loss_fn = torch.nn.KLDivLoss()
    #ce_loss_fn = torch.nn.CrossEntropyLoss()

    # Load datasets
    def gen_texts(num_texts=1):
        _cripple_texts, _focus_texts = [], []

        cripple_dataset, cripple_label, _skip50 = prepare(c.cripple)
        i = 0
        for data in cripple_dataset:
            i += 1
            if i > num_texts:
                break
            _cripple_texts.append(data[cripple_label])

        focus_dataset, focus_label, _skip50     = prepare(c.focus)
        i = 0
        for data in focus_dataset:
            i += 1
            if i > num_texts:
                break
            _focus_texts.append(data[focus_label])

        return _cripple_texts, _focus_texts

    # Begin calculating loss for with LBGFS
    def get_new_ids(n_batches = None):
        batches = []
        cripple_texts, focus_texts = gen_texts()
        bad_ids, junk_ids, good_ids = [], [], []
        with torch.no_grad():
            for text in cripple_texts:
                bad_ids.append( opt.get_ids(text) )
                junk_ids.append(
                    torch.randint_like(bad_ids[-1], 5, opt.tokenizer.vocab_size)
                )
            for text in focus_texts:
                good_ids.append( opt.get_ids(text) )
        return bad_ids, junk_ids, good_ids

    for j in range(c.n_steps//n_iter):
        bad_ids, junk_ids, good_ids = get_new_ids()

        # Begin LBGFS
        def closure():
            loss = 0
            optim.zero_grad()

            # Generate loss
            loss += mask_l1_norm * l1_norm_coeff

            for i in range(num_texts):
                # Get junk loss L_kl(gamma,P)
                with torch.no_grad():
                    junk_logits = opt.get_logits(input_ids=junk_ids[i])[..., :-1, :]
                bad_logits = opt.get_logits(input_ids=bad_ids[i])[..., :-1, :]
                loss += kl_loss_fn(bad_logits, junk_logits).mean()

                # Get good loss L_kl(gamma,Q)
                with torch.no_grad():
                    opt.masking_enabled = False
                    orig_logits = opt.get_logits(input_ids=good_ids[i])[..., :-1, :]
                    opt.masking_enabled = True
                new_logits = opt.get_logits(input_ids=good_ids[i])[..., :-1, :]
                loss += kl_loss_fn(new_logits, orig_logits).mean()

            # Backpropagate
            loss.backward(retain_graph=True)
            return loss

        # loss step
        for i in range(n_iter):
            optim.step(closure)
            data = evaluate_all(opt, c.eval_sample_size,
                c.datasets, c.collection_sample_size)
            history.add(data)

    # Format history to print
    print(history.history[-1])
    print(history.df.T)
    print(history.df.T.to_csv())

    return opt, history


class EmpiricalBlockFisherInverseCap:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
        perm: torch.Tensor|None=None, 
        invperm: torch.Tensor|None=None
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device
        self.perm = perm          
        self.invperm = invperm  

        self.num_blocks = math.ceil(self.d / self.B)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory
    def fisher_diag(self) -> Tensor:
        """
        Returns the diagonal of the *Fisher* matrix 
        (≈ Hessian) by inverting each block of F_inv.
        """
        # F_inv has shape (num_blocks, B, B)
        # invert each block
        # using torch.linalg.inv over the last two dims:
        F_blocks = torch.linalg.inv(self.F_inv)           # -> (num_blocks, B, B)
        # grab the block‐diagonals, flatten, and trim padding
        return F_blocks.diagonal(dim1=1, dim2=2).flatten()[: self.d]
    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched F_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g)

        # scalar denominator for each batch: (batch)
        alpha =  (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).unsqueeze(1)
        # (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze_(1)
        Finv_g = Finv_g / alpha.sqrt()


        # update F_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)
        # self.m += 1  
    
    def downdate_removed_index(self, idx: int, eps: float = 1e-12):
        """
        Rank-1 downdate of the inverse for eliminating weight `idx`.
        Works block-wise: only the block containing `idx` is updated.
        After the update, the row/col for `idx` become (near) zero.
        """
        b = idx // self.B          # which block
        j = idx % self.B           # column/row within the block

        # safety for padding (when d is not multiple of B)
        if b >= self.num_blocks:
            return

        block = self.F_inv[b]      # (B, B)
        alpha = block[j, j]
        if torch.isnan(alpha) or torch.isinf(alpha) or abs(alpha) < eps:
            # inverse is unstable for this index; skip downdate
            return

        col = block[:, j].clone()  # (B,)
        # rank-1 downdate: block <- block - (col col^T) / alpha
        block -= torch.ger(col, col) / (alpha + eps)

        # Zero out the eliminated row/col to avoid reuse
        block[j, :] = 0.0
        block[:, j] = 0.0

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]



# from sparseml.pytorch.sparsification.modifier_pruning_obs import EmpiricalBlockFisherInverse
class EmpiricalBlockFisherInverse:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
        switch_m = False,
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device
        self.switch_m = switch_m

        self.num_blocks = math.ceil(self.d / self.B)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory
    def fisher_diag(self) -> Tensor:
        """
        Returns the diagonal of the *Fisher* matrix 
        (≈ Hessian) by inverting each block of F_inv.
        """
        # F_inv has shape (num_blocks, B, B)
        # invert each block
        # using torch.linalg.inv over the last two dims:
        F_blocks = torch.linalg.inv(self.F_inv)           # -> (num_blocks, B, B)
        # grab the block‐diagonals, flatten, and trim padding
        return F_blocks.diagonal(dim1=1, dim2=2).flatten()[: self.d]
    
    def to(self, device: torch.device):
        """
        Moves the internal tensors of the object to the specified device.
        """
        self.F_inv = self.F_inv.to(device)
        self.dev = device
        return self # Return self to allow for chaining, e.g., model.to('cpu')

    def cpu(self):
        """
        Moves the internal tensors to the CPU. A convenient shortcut for .to(torch.device('cpu')).
        """
        return self.to(torch.device('cpu'))

    def sub(self, block):
        F_blocks = torch.linalg.inv(self.F_inv)
        F_blocks_input = torch.linalg.inv(block.F_inv)

        epsilon = 1e-8
        # Calculate the absolute ratio
        ratio = torch.abs(F_blocks / (F_blocks_input + epsilon))
        # Apply the condition element-wise
        result = torch.where(ratio > 1, F_blocks, F_blocks * ratio)

        F_blocks_input = torch.linalg.inv(result)
        self.F_inv = F_blocks_input
    
    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device )]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched F_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g)

        # scalar denominator for each batch: (batch)
        if self.switch_m:
            alpha =  (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        else:
            alpha =  (self.m + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        # (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze_(1)
        Finv_g /= alpha

        # update F_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)
        self.m += 1  
    
    def downdate_removed_index(self, idx: int, eps: float = 1e-12):
        """
        Rank-1 downdate of the inverse for eliminating weight `idx`.
        Works block-wise: only the block containing `idx` is updated.
        After the update, the row/col for `idx` become (near) zero.
        """
        b = idx // self.B          # which block
        j = idx % self.B           # column/row within the block

        # safety for padding (when d is not multiple of B)
        if b >= self.num_blocks:
            return

        block = self.F_inv[b]      # (B, B)
        alpha = block[j, j]
        if torch.isnan(alpha) or torch.isinf(alpha) or abs(alpha) < eps:
            # inverse is unstable for this index; skip downdate
            return

        col = block[:, j].clone()  # (B,)
        # rank-1 downdate: block <- block - (col col^T) / alpha
        block -= torch.ger(col, col) / (alpha + eps)

        # Zero out the eliminated row/col to avoid reuse
        block[j, :] = 0.0
        block[:, j] = 0.0

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]


class EmpiricalBlockFisherInverseLLM:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
        switch_m = False
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device
        self.switch_m = switch_m

        self.num_blocks = math.ceil(self.d / self.B)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, dtype=torch.float32, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory
    def fisher_diag(self) -> Tensor:
        """
        Returns the diagonal of the *Fisher* matrix 
        (≈ Hessian) by inverting each block of F_inv.
        """
        # F_inv has shape (num_blocks, B, B)
        # invert each block
        # using torch.linalg.inv over the last two dims:
        F_blocks = torch.linalg.inv(self.F_inv)           # -> (num_blocks, B, B)
        # grab the block‐diagonals, flatten, and trim padding
        return F_blocks.diagonal(dim1=1, dim2=2).flatten()[: self.d]
    
    def to(self, device: torch.device):
        """
        Moves the internal tensors of the object to the specified device.
        """
        self.F_inv = self.F_inv.to(device)
        self.dev = device
        return self # Return self to allow for chaining, e.g., model.to('cpu')

    def cpu(self):
        """
        Moves the internal tensors to the CPU. A convenient shortcut for .to(torch.device('cpu')).
        """
        return self.to(torch.device('cpu'))

    def sub(self, block):
        F_blocks = torch.linalg.inv(self.F_inv)
        F_blocks_input = torch.linalg.inv(block.F_inv)

        epsilon = 1e-8
        # Calculate the absolute ratio
        ratio = torch.abs(F_blocks / (F_blocks_input + epsilon))
        # Apply the condition element-wise
        result = torch.where(ratio > 1, F_blocks, F_blocks * ratio)

        F_blocks_input = torch.linalg.inv(result)
        self.F_inv = F_blocks_input
    
    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        g = g.detach().to(torch.float32)
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched F_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g)

        # scalar denominator for each batch: (batch)
        if self.switch_m:
            alpha =  (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        else:
            alpha =  (self.m + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        # (1.0 + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze_(1)
        Finv_g /= alpha

        # update F_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)

        diag = torch.diagonal(self.F_inv, 0, 1, 2)                   # (num_blocks, B)
        bad = (~torch.isfinite(diag)).any(dim=1)                    # True if any NaN/Inf
        if bad.any():
            eyeB = (1.0 / self.damp) * torch.eye(self.B, device=self.dev, dtype=self.F_inv.dtype)
            self.F_inv[bad] = eyeB    
            print("ADD GRAD BAD IS HERE")

        self.m += 1  
    
    def downdate_removed_index(self, idx: int, eps: float = 1e-12):
        """
        Rank-1 downdate of the inverse for eliminating weight `idx`.
        Works block-wise: only the block containing `idx` is updated.
        After the update, the row/col for `idx` become (near) zero.
        """
        b = idx // self.B          # which block
        j = idx % self.B           # column/row within the block

        # safety for padding (when d is not multiple of B)
        if b >= self.num_blocks:
            return

        block = self.F_inv[b]      # (B, B)
        alpha = block[j, j]
        if torch.isnan(alpha) or torch.isinf(alpha) or abs(alpha) < eps:
            # inverse is unstable for this index; skip downdate
            return

        col = block[:, j].clone()  # (B,)
        # rank-1 downdate: block <- block - (col col^T) / alpha
        block -= torch.ger(col, col) / (alpha + eps)

        # Zero out the eliminated row/col to avoid reuse
        block[j, :] = 0.0
        block[:, j] = 0.0

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]



from torch import nn
def prune_and_compensate_all_layers(
    model: nn.Module,
    prune_mask: torch.Tensor,  # shape: [12, 3072]
    # fisher_inv: EmpiricalBlockFisherInverse,
    fisher_invs: List[EmpiricalBlockFisherInverse],
    block_size: int = 50,
    damp: float = 1e-3,
    learning_rate: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    downdate = False
):
    """
    Prune and compensate all encoder MLP layers using OBS.

    Args:
        model: Vision Transformer model with encoder layers
        prune_mask: Boolean tensor of shape [12, 3072] 
        grad_samples: Tensor of shape [num_grads, 12, 3072, 768]
        block_size: Block size for Fisher inverse approximation
        damp: Damping factor
        device: Target device
    """
    # Import EmpiricalBlockFisherInverse from this file or use the one above
    num_layers = prune_mask.shape[0]
    # num_grads = grad_samples.shape[0]

    # TODO: uncomment this line to use the original Fisher inverse
    # for layer_idx in range(10,12):
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx + 1}/{num_layers}...")
        layer_name = f"encoder.layer.{layer_idx}.intermediate.dense.weight"
        param = dict(model.named_parameters())[layer_name]
        param.requires_grad = True
        param_data = param.data.detach().clone().to(device)
        param_shape = param_data.shape
        param_flat = param_data.view(-1)
        print_interval = 40000

        fisher_inv = fisher_invs[layer_idx]
        mask = prune_mask[layer_idx]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        # print the number of non-zero elements in the mask
        mask_flat = mask.unsqueeze(1).expand(-1, 768).contiguous().view(-1)
        prune_indices = mask_flat.nonzero(as_tuple=True)[0]
        # clone the parameter data to avoid modifying the original tensor
        new_weights = param_flat.clone()

        for idx in prune_indices:
            if idx%print_interval == 0:
                print(f"Processing index {idx} out of {len(prune_indices)}...")

            diag_val = fisher_inv.diag()[idx] + 1e-10
            e_i = torch.zeros_like(new_weights)
            e_i[idx] = 1.0
            delta_vec = fisher_inv.mul(e_i)
            if idx % print_interval == 0:
                print(f"[DEBUG] idx={idx}")
                print(f"[DEBUG] new_weights[idx] = {new_weights[idx]}")
                print(f"[DEBUG] diag_val = {diag_val}")
                print(f"[DEBUG] delta_vec norm = {delta_vec.norm()} \t shape: {delta_vec.shape}")
            if diag_val < 1e-12:
                print(f"Warning: diag_val is too small for index {idx}. Skipping compensation.")
                continue
            scale = -new_weights[idx] / diag_val
            delta_w = scale * delta_vec
            # sum of abs values of delta_w except index of [idx]
            if idx % print_interval == 0:
                exp_delta_w = delta_w.clone()
                exp_delta_w[idx] = 0.0
                print(f'*********** delta_w except idx: {torch.sum(torch.abs(exp_delta_w))}')
            new_weights += delta_w * learning_rate
            if idx % print_interval == 0:
                print(f'delta_w: {torch.sum(torch.abs(delta_w))}')

            new_weights[idx] = 0.0
            if downdate:
                fisher_inv.downdate_removed_index(idx)
        param.data.copy_(new_weights.view(param_shape))
        param.requires_grad = False

# New function: prune_only_all_layers
def prune_only_all_layers(
    model: nn.Module,
    prune_mask: torch.Tensor,  # shape: [num_layers, mlp_dim], True means prune that neuron
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Prune all encoder MLP layers by zeroing out the specified neurons, without compensation.

    Args:
        model: Vision Transformer model with encoder layers
        prune_mask: Boolean tensor of shape [num_layers, mlp_dim]; True entries
                    indicate neurons to prune (zero out entire row of weights).
        device: Target device for mask tensor.
    """
    num_layers = prune_mask.shape[0]
    # Iterate over layers
    for layer_idx in range(num_layers):
        # Parameter name for the MLP weight
        layer_name = f"encoder.layer.{layer_idx}.intermediate.dense.weight"
        param = dict(model.named_parameters())[layer_name]
        # Move mask for this layer to the parameter's device and reshape for broadcasting
        prune_mask_layer = prune_mask[layer_idx]
        if not isinstance(prune_mask_layer, torch.Tensor):
            prune_mask_layer = torch.tensor(prune_mask_layer, dtype=torch.bool, device=device)
        mask = prune_mask_layer.to(device).view(-1, 1).type_as(param.data)
        # Zero out each row where mask==True
        param.data.mul_(1.0 - mask)


from torch.utils.data import DataLoader, TensorDataset
def create_dummy_dataloader(batch_size=16, num_batches=10, num_classes=10, device='cpu'):
    # Total number of samples
    total_samples = batch_size * num_batches

    # Create random image tensors (simulating RGB images of size 224x224)
    inputs = torch.randn(total_samples, 3, 224, 224)

    # Create random labels (classification task)
    targets = torch.randint(0, num_classes, (total_samples,))

    # Wrap into a dataset and dataloader
    dataset = TensorDataset(inputs.to(device), targets.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class WrappedDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, transform):
                    self.base = base_dataset
                    self.transform = transform
                def __getitem__(self, idx):
                    sample = self.base[idx]
                    image = sample['image']
                    label = sample['label']
                    return self.transform(image), label
                def __len__(self):
                    return len(self.base)
                
class WrappedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
    def __iter__(self):
        for sample in self.base:
            image = sample['image']
            label = sample['label']
            yield self.transform(image), label


def calculate_grad_hess_single_layer(model, c, dataloader, hessian_coe, layer_to_prune = None):
    num_grads = c.num_grads
    device = next(model.model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    num_layers = len(model.layers)
    # --- MODIFICATION START ---
    # Define which layers to process based on the layer_to_prune argument
    if layer_to_prune is not None:
        # If a specific layer is provided, create a list with just that layer's index
        target_layers = [layer_to_prune]
        print(f"Scoring only layer: {layer_to_prune}")
    else:
        # Otherwise, create a range to process all layers
        target_layers = range(num_layers)
        print("Scoring all layers.")
    # --- MODIFICATION END ---
    
    layer_grad_sums = []
    fisher_invs = []
    layer_grads = [None] * num_layers  # Scratch space for per-batch gradients

    # Initialize data structures for ALL layers, as hooks rely on the full index range.
    # This is a simple approach; only the target layers will actually be populated.
    for i in range(num_layers):
        w = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        layer_grad_sums.append(torch.zeros_like(w, device=device))
        
        # Initialize Fisher inverse for this layer
        fisher_invs.append(
            EmpiricalBlockFisherInverse(
                num_grads=c.num_grads,
                fisher_block_size=c.fisher_block_size,
                num_weights=w.numel(),
                damp=1e-3,
                device=device,
            )
        )

    def make_hook(idx):
        def _hook(g):
            with torch.no_grad():
                layer_grads[idx] = g
                layer_grad_sums[idx] += g
        return _hook

    # Register hooks ONLY for the target layers
    handles = []
    # --- MODIFICATION START ---
    for i in target_layers: # Iterate over target_layers instead of all layers
        p = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        handles.append(p.register_hook(make_hook(i)))
    # --- MODIFICATION END ---

    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if counter > num_grads:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)

        logits = model.get_logits(pixel_values=inputs)[:, 0, :]
        loss = loss_fn(logits, targets)
        loss.backward()  # Only hooks for target_layers will fire here

        # Update Fisher inverse. This loop is fine as is, since it checks for g != None.
        # Gradients (g) will only exist for the target layers.
        for i, g in enumerate(layer_grads):
            if g is not None:
                g_flat = g.flatten()
                fisher_invs[i].add_grad(g_flat)

        # Reset scratch buffer
        layer_grads = [None] * num_layers

        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")

    for h in handles:
        h.remove()

    layer_gradients = [g_sum / num_grads for g_sum in layer_grad_sums]
    layer_hessian_diags = [
        fisher_inv.fisher_diag().reshape_as(grad)
        for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]

    raw_scores_per_layer = {}
    total_sum = 0.0

    # --- MODIFICATION START ---
    # Calculate scores ONLY for the target layers
    for i in target_layers: # Iterate over target_layers instead of all layers
        weights = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"].data
        grad = layer_gradients[i]
        hessian_diag = layer_hessian_diags[i]

        lambda_reg = 1e-2
        raw_scores = (abs(grad * weights) + hessian_coe * abs(hessian_diag * weights.pow(2))).sum(dim=1)
        
        raw_scores_per_layer[f"layer_{i}"] = raw_scores
        total_sum += raw_scores.sum()
    # --- MODIFICATION END ---

    eps = 1e-12
    neuron_scores = {}
    for name, raw in raw_scores_per_layer.items():
        neuron_scores[name] = raw / (raw.sum() + eps)

    all_scores = []
    layer_ranges = {}
    start_idx = 0

    # --- MODIFICATION START ---
    # Flatten scores ONLY for the target layers
    for i in target_layers: # Iterate over target_layers instead of all layers
        # Check if the key exists before accessing
        scores_key = f"layer_{i}"
        if scores_key in neuron_scores:
            scores = neuron_scores[scores_key]
            num_neurons = scores.shape[0]
            all_scores.append(scores)
            layer_ranges[i] = (start_idx, start_idx + num_neurons)
            start_idx += num_neurons
    # --- MODIFICATION END ---

    global_scores = torch.cat(all_scores) if all_scores else torch.tensor([])
    print("Global Scores:")
    print(global_scores)
    return global_scores, all_scores, layer_ranges

def calculate_grad_hess(model, c, dataloader, hessian_coe):
    num_grads = c.num_grads
    device = next(model.model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    num_layers = len(model.layers)
    layer_grad_sums = []
    fisher_invs = []
    layer_grads = [None] * num_layers  # Scratch space for per-batch gradients
    

    for i in range(num_layers):
        w = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        layer_grad_sums.append(torch.zeros_like(w, device=device))
        
        # Initialize Fisher inverse for this layer
        fisher_invs.append(
            EmpiricalBlockFisherInverse(
                num_grads=c.num_grads,
                fisher_block_size=c.fisher_block_size,
                num_weights=w.numel(),  # Total weights in this layer
                damp=1e-3,  # Damping factor for stability
                device=device,
            )
        )

    def make_hook(idx):
        def _hook(g):
            with torch.no_grad():
                layer_grads[idx] = g  # Store raw gradient for this batch
                layer_grad_sums[idx] += g  # Accumulate for mean gradient
        return _hook

    # Register hooks
    handles = []
    for i in range(num_layers):
        p = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        handles.append(p.register_hook(make_hook(i)))

    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if counter > num_grads:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)

        # Forward + backward
        logits = model.get_logits(pixel_values=inputs)[:, 0, :]  # CLS token
        loss = loss_fn(logits, targets)
        loss.backward()  # Hooks fire here

        # Update Fisher inverse for each layer
        for i, g in enumerate(layer_grads):
            if g is not None:
                g_flat = g.flatten()
                fisher_invs[i].add_grad(g_flat)  # Update Fisher inverse

        # Reset scratch buffer
        layer_grads = [None] * num_layers

        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")

    # Remove hooks
    for h in handles:
        h.remove()

    layer_gradients = [g_sum / num_grads for g_sum in layer_grad_sums]
    layer_hessian_diags = [
        fisher_inv.fisher_diag().reshape_as(grad)
        for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]

    raw_scores_per_layer = {}
    total_sum = 0.0

    for i in range(num_layers):
        weights = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"].data
        grad = layer_gradients[i]
        hessian_diag = layer_hessian_diags[i]


        lambda_reg = 1e-2 
        # Compute importance scores using Fisher diagonal
        raw_scores = (abs(grad * weights) + hessian_coe * abs(hessian_diag * weights.pow(2))).sum(dim=1)
        

        # now add a magnitude‐based bonus
        # option A: L1-based bonus
        # mag_bonus = weights.abs().sum(dim=1)  
        # option B: L2-based bonus (group-Lasso style)
        # mag_bonus = weights.pow(2).sum(dim=1).sqrt()

        # raw_scores = raw_scores + lambda_reg * mag_bonus
        raw_scores_per_layer[f"layer_{i}"] = raw_scores
        total_sum += raw_scores.sum()

    # Normalize globally
    # neuron_scores = {}
    # for i in range(num_layers):
    #     neuron_scores[f"layer_{i}"] = raw_scores_per_layer[f"layer_{i}"] / (total_sum + 1e-12)

    # Normalize layerwise
    eps = 1e-12
    neuron_scores = {}
    for name, raw in raw_scores_per_layer.items():   # raw is shape [out_features]
        neuron_scores[name] = raw / (raw.sum() + eps)

    # Flatten scores across layers
    all_scores = []
    layer_ranges = {}
    start_idx = 0

    for i in range(num_layers):
        scores = neuron_scores[f"layer_{i}"]
        num_neurons = scores.shape[0]
        all_scores.append(scores)
        layer_ranges[i] = (start_idx, start_idx + num_neurons)
        start_idx += num_neurons
    global_scores = torch.cat(all_scores)
    print("GLobal Scores:")
    print(global_scores)
    return global_scores, all_scores, layer_ranges



def calculate_grad_hess_rademache(model, c, dataloader, hessian_coe, l1 = None):
    """
    Returns three lists (one per param):
        gradients, diag_hessians, scores
    where
        score_i = g_i * w_i + h_i * w_i**2
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device   = next(model.model.parameters()).device
    num_grads = c.num_grads
    loss_fn  = nn.CrossEntropyLoss()
    num_layers = len(model.layers)

    params = [
        dict(model.model.named_parameters())
        [f"encoder.layer.{i}.intermediate.dense.weight"]
        for i in range(len(model.layers))
    ]

    grad_sums  = [torch.zeros_like(p, device=device) for p in params]
    hess_sums  = [torch.zeros_like(p, device=device) for p in params]

    # ------------ Hutchinson helpers ------------
    def _rademacher_like(p):
        return (torch.randint_like(p, high=2) * 2 - 1).type_as(p)

    def accumulate_diag_hessian(loss):
        # first‑order grads w.r.t params (creates graph ⇒ we can do HVPs)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        for _ in range(4):
            v   = [_rademacher_like(p) for p in params]            # probe
            hvp = torch.autograd.grad(grads, params, v, retain_graph=True)
            for h_acc, vi, hvpi in zip(hess_sums, v, hvp):
                h_acc += hvpi * vi                                 # v ⊙ (H v)
    # --------------------------------------------

    # main loop
    for step, (inputs, targets) in enumerate(dataloader, 1):
        if step > num_grads: break

        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)

        logits = model.get_logits(pixel_values=inputs)[:, 0, :]
        loss = loss_fn(logits, targets)

        # mean gradient
        loss.backward(retain_graph=True)           # keep graph for HVP
        for g_acc, p in zip(grad_sums, params):
            g_acc += p.grad.detach()

        # diagonal Hessian
        accumulate_diag_hessian(loss)

        if step % 50 == 0:
            print(f"[{step:>4}/{num_grads}] processed")

    layer_gradients = [g_sum / num_grads for g_sum in grad_sums]
    diag_hessians = [d / (num_grads * 4) for d in hess_sums]
    raw_scores_per_layer = {}
    total_sum = 0.0

    for i in range(num_layers):
        weights = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"].data
        grad = layer_gradients[i]
        hessian_diag = diag_hessians[i]

        # Compute importance scores using Fisher diagonal

        lambda_reg = l1 if l1 else 1e-2
        # Compute importance scores using Fisher diagonal
        raw_scores = (abs(grad * weights) + hessian_coe * abs(hessian_diag * weights.pow(2))).sum(dim=1)
        

        # now add a magnitude‐based bonus
        # option A: L1-based bonus
        mag_bonus = weights.abs().sum(dim=1)  
        # option B: L2-based bonus (group-Lasso style)
        # mag_bonus = weights.pow(2).sum(dim=1).sqrt()

        raw_scores = raw_scores + lambda_reg * mag_bonus
        raw_scores_per_layer[f"layer_{i}"] = raw_scores
        total_sum += raw_scores.sum()

    # Normalize globally
    # neuron_scores = {}
    # for i in range(num_layers):
    #     neuron_scores[f"layer_{i}"] = raw_scores_per_layer[f"layer_{i}"] / (total_sum + 1e-12)

    # Normalize layerwise
    eps = 1e-12
    neuron_scores = {}
    for name, raw in raw_scores_per_layer.items():   # raw is shape [out_features]
        neuron_scores[name] = raw / (raw.sum() + eps)

    # Flatten scores across layers
    all_scores = []
    layer_ranges = {}
    start_idx = 0

    for i in range(num_layers):
        scores = neuron_scores[f"layer_{i}"]
        num_neurons = scores.shape[0]
        all_scores.append(scores)
        layer_ranges[i] = (start_idx, start_idx + num_neurons)
        start_idx += num_neurons
    global_scores = torch.cat(all_scores)
    print("GLobal Scores:")
    print(global_scores)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    return global_scores, all_scores, layer_ranges

def get_VIT_dataloder(dataset_name, batch_size = 1, apply_transforms = True, custom_transforms = None, is_validation = False):
    eval_config = infer_dataset_config(dataset_name)
    #TODO: Not always train
    eval_config.dataset_split = "train"
    eval_config.is_train_mode = True
    if is_validation:
        eval_config.dataset_split = "validation"
        eval_config.is_train_mode = False
    dataset  = prepare_dataset(eval_config)
    
    if apply_transforms:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3‑channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    if custom_transforms is not None:
        print("in custom transforms")
        transform = custom_transforms

    if hasattr(dataset, 'transform'):
        dataset.transform = transform
    else:
        # fallback: wrap dataset, choose wrapper based on base type
        if isinstance(dataset, IterableDataset):
            print("in wrapped iterable dataset")
            dataset = WrappedIterableDataset(dataset,transform)
        else:
            print("in just wrapped dataset")
            dataset = WrappedDataset(dataset,transform)

    try:
        _ = len(dataset)
        has_len = True
    except TypeError:
        has_len = False

    print("has len is", has_len)
    sampler = SequentialSampler(dataset) if has_len else None

    print("after sampler")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,          
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        sampler=sampler
    )
    print("after create dataloader")
    return dataloader



def snip_prune_and_evaluate(
        opt: Model,
        pruning_config: PruningConfig,
        prune_frac = 0.02,
        second_order: bool = True,
        hessian_coefficient = 0.5,
        output = None,
        l1 = None,
        layer_to_prune = None,
    ):
    """
    Prune and evaluate the model

    Args:
        opt (Model): model to prune and evaluate
        pruning_config (PruningConfig): config for pruning
        focus_out (dict): output of get_midlayer_data for focus dataset
        cripple_out (dict): output of get_midlayer_data for cripple dataset
        iteration (int): iteration number for when activations are not recalculated

    Returns:
        output (RunDataItem): Eval data to add to RunDataHistory.
    """
    c = copy.deepcopy(pruning_config)
 
    cripple_dataloader = get_VIT_dataloder(c.cripple)
    retain_dataloader = get_VIT_dataloder(c.focus)
    num_grads = c.num_grads
    
    # calculate_grad_hess_single_layer(opt, c, cripple_dataloader, hessian_coefficient, layer_to_prune = layer_to_prune)
    # calculate_grad_hess_rademache(opt, c, cripple_dataloader, hessian_coefficient, l1)
    
    
    global_scores, all_scores, layer_ranges = calculate_grad_hess(opt, c, cripple_dataloader, hessian_coefficient)
    global_scores_retain, all_scores_retain, layer_ranges_retain = calculate_grad_hess(opt, c, retain_dataloader, hessian_coefficient)
    global_scores = global_scores/global_scores_retain
    
    N = global_scores.numel()
    forget_neurons = max(1, math.floor(N * prune_frac))
    print(f'forget neurons: {forget_neurons}')

        # Get global top-k indices
    scores, global_topk_indices = torch.topk(global_scores, k=forget_neurons)
    print("Top scores:")
    print(scores)
    print("Top Indices:")
    print(global_topk_indices)
    
    # Initialize output tensors
    max_neurons = max(s.shape[0] for s in all_scores)
    global_mask = torch.zeros(len(opt.layers), max_neurons, dtype=torch.bool)
    layer_masks = []
    
    # Create masks for each layer
    neurons_kept = 0
   
    for i in layer_ranges: 
        
        start, end = layer_ranges[i]
        num_neurons = end - start
        
        # Get indices that fall in this layer's range
        layer_indices = global_topk_indices[(global_topk_indices >= start) & 
                                        (global_topk_indices < end)] - start
        
        # Create layer mask
        layer_mask = torch.zeros(num_neurons, dtype=torch.bool)
        if len(layer_indices) > 0:
            layer_mask[layer_indices] = True
        layer_masks.append(layer_mask)
        
        # Update global mask for the correct layer 'i'
        # The other layers will remain False (or 0), which is the correct behavior.
        global_mask[i, :num_neurons] = layer_mask
        neurons_kept += len(layer_indices)
        
        print(f"Layer {i}: Kept {len(layer_indices)}/{num_neurons} neurons")

    global_mask = global_mask.int()
    print("\nGlobal Mask: ")
    for i in range(len(opt.layers)):
        # This will now correctly show the mask for the scored layer(s) 
        # and all zeros for the unscored layers.
        print(f"Layer {i}: {global_mask[i]}")

    global_mask = global_mask.detach().cpu().numpy()
   
   
    # for i in range(len(opt.layers)):
    #     start, end = layer_ranges[i]
    #     num_neurons = end - start
        
    #     # Get indices that fall in this layer's range
    #     layer_indices = global_topk_indices[(global_topk_indices >= start) & 
    #                                     (global_topk_indices < end)] - start
        
    #     # Create layer mask
    #     layer_mask = torch.zeros(num_neurons, dtype=torch.bool)
    #     layer_mask[layer_indices] = True
    #     layer_masks.append(layer_mask)
        
    #     # Update global mask
    #     global_mask[i, :num_neurons] = layer_mask
    #     neurons_kept += len(layer_indices)
        
    #     print(f"Layer {i}: Pruning {len(layer_indices)}/{num_neurons} neurons")
    # global_mask = global_mask.int()
    # print("Global Mask: ")
    # for i in range(len(opt.layers)):
    #     print(global_mask[i])
    # global_mask = global_mask.detach().cpu().numpy()

    print(f'prune mask 1 percentage: {np.sum(global_mask==1)/global_mask.size}') # .numel() is not a numpy function
    print(f'prune mask 0 percentage: {np.sum(global_mask==0)/global_mask.size}')
    directory = f"outputs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if output is None:
        torch.save(global_mask, f"{directory}/prune_mask_forget_frac_{prune_frac}_layer_valid.pt")
        print(f"Saved prune_mask to {directory}/prune_mask.pt")
    else:
         torch.save(global_mask, f"{directory}" +output)
         print(f"Saved prune_mask to {directory}"+output)
    return
    model = opt
    model.model.train()
    focus_dataloader = get_VIT_dataloder(c.focus)
    device = next(model.model.parameters()).device
    loss_fn=nn.CrossEntropyLoss()

    # register hooks to capture gradients
    layer_grads = [None] * len(model.layers)  # One for each encoder layer
    def make_hook(layer_idx):
        def hook(grad):
            layer_grads[layer_idx] = grad.detach().cpu()
        return hook

    # compute the gradients
    num_grads = c.num_grads

    fisher_invs = [
        EmpiricalBlockFisherInverse(
            num_grads         = num_grads,
            fisher_block_size = c.fisher_block_size,
            num_weights       = 3072 * 768,   # one MLP matrix per layer
            damp              = 1e-3,
            device            = device,
        )
        for _ in range(len(model.layers))
    ]
    counter = 0
    for inputs, targets in focus_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.model.zero_grad()
        
        # Register hooks
        handles = []
        for i in range(len(model.layers)):
            layer = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
            handles.append(layer.register_hook(make_hook(i)))

        # Backward pass
        with torch.set_grad_enabled(True):
            logits = model.get_logits(pixel_values=inputs)
            logits = logits[:, 0, :]  # take only CLS token for classification
            loss = loss_fn(logits, targets)
            # print("Acc:", (logits.argmax(dim=-1) == targets).float().mean().item())
            loss.backward()

        # Add the grads to fisher inverses
        if all(g is not None for g in layer_grads):
            # Stream each layer's gradient into its own inverse
            for i, g_i in enumerate(layer_grads):
                v = g_i.flatten().to(device)   # TODO: maybe this one v = g_i.detach().flatten().to(device, non_blocking=True)
                v = v / (v.norm() + 1e-12)
                fisher_invs[i].add_grad(v)
                del v  # free memory
            layer_grads = [None] * len(model.layers)

        # Remove hooks
        for h in handles:
            h.remove()

        counter += 1
        if counter >= num_grads:
            break

    torch.cuda.empty_cache()
    
    prune_and_compensate_all_layers(
        model.model,
        prune_mask = global_mask,
        fisher_invs = fisher_invs,
        block_size=c.fisher_block_size, damp=1e-3, device=device
    )
    print("c.datasets",c.datasets)
    with torch.no_grad():
        print(f'********* starting evalution after pruning+compensation *********')
        eval_out = evaluate_all(model, c.eval_sample_size, c.datasets,
                                dataset_tokens_to_skip=c.collection_sample_size)

    # exclude 'token_count' from eval_out to print
    print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    print(f'----- debug - model performance after pruning+compensation -> eval_out: {print_eval_out} ----- \n\n')


     # TODO: maybe eval works better

    return 


def run_snip_pruning(c: PruningConfig, forget_frac = None, second_order: bool = True, hessian_coefficient = 0.5, output = None, l1 = None, layer_to_prune = None):
    # Initialise Model and show details about model
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    # Prepare data logging
    # history = RunDataHistory(c.datasets)
    # wandb.init(
    #     project=c.wandb_project,
    #     entity=c.wandb_entity,
    #     name=c.wandb_run_name,
    #     )
    # wandb.config.update(c.to_dict(), allow_val_change=True)

    # Evaluate model before removal of any neurons
    assert c.run_pre_test==False, "run_pre_test is not supported in this version"
    print(f'********* starting evalution before pruning *********')
    # data = evaluate_all(opt, c.eval_sample_size,
    #     c.datasets, c.collection_sample_size)
    # history.add(data)
    # print(history.df.T)

    
    
    # print(f'START OF PRUNING STEP')
    # tmp_opt = copy.deepcopy(opt)
    snip_prune_and_evaluate(opt, c, forget_frac, second_order, hessian_coefficient=hessian_coefficient, output=output, layer_to_prune = layer_to_prune)
    # del tmp_opt

    # print(asb1)

    # Format history to print
    # print(history.history[-1])
    # print(history.df.T)
    # print(history.df.T.to_csv())

    return opt

def calculate_grad_hess_flags(model, c, dataloader, hessian_coe, normalize_layerwise=True, l1 = 0):
    num_grads = c.num_grads
    device = next(model.model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    num_layers = len(model.layers)
    layer_grad_sums = []
    fisher_invs = []
    layer_grads = [None] * num_layers  # Scratch space for per-batch gradients
    

    for i in range(num_layers):
        w = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        layer_grad_sums.append(torch.zeros_like(w, device=device))
        
        # Initialize Fisher inverse for this layer
        fisher_invs.append(
            EmpiricalBlockFisherInverse(
                num_grads=c.num_grads,
                fisher_block_size=c.fisher_block_size,
                num_weights=w.numel(),  # Total weights in this layer
                damp=1e-3,  # Damping factor for stability
                device=device,
            )
        )

    def make_hook(idx):
        def _hook(g):
            with torch.no_grad():
                layer_grads[idx] = g  # Store raw gradient for this batch
                layer_grad_sums[idx] += g  # Accumulate for mean gradient
        return _hook

    # Register hooks
    handles = []
    for i in range(num_layers):
        p = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        handles.append(p.register_hook(make_hook(i)))

    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if counter > num_grads:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)

        # Forward + backward
        logits = model.get_logits(pixel_values=inputs)[:, 0, :]  # CLS token
        loss = loss_fn(logits, targets)
        loss.backward()  # Hooks fire here

        # Update Fisher inverse for each layer
        for i, g in enumerate(layer_grads):
            if g is not None:
                g_flat = g.flatten()
                fisher_invs[i].add_grad(g_flat)  # Update Fisher inverse

        # Reset scratch buffer
        layer_grads = [None] * num_layers

        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")

    # Remove hooks
    for h in handles:
        h.remove()

    layer_gradients = [g_sum / num_grads for g_sum in layer_grad_sums]
    layer_hessian_diags = [
        fisher_inv.fisher_diag().reshape_as(grad)
        for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]

    raw_scores_per_layer = {}
    total_sum = 0.0

    for i in range(num_layers):
        weights = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"].data
        grad = layer_gradients[i]
        hessian_diag = layer_hessian_diags[i]


        lambda_reg = l1 
        raw_scores = (abs(grad * weights) + hessian_coe * abs(hessian_diag * weights.pow(2))).sum(dim=1)
        

        # now add a magnitude‐based bonus
        # L1-based bonus
        mag_bonus = weights.abs().sum(dim=1)  
        # L2-based bonus
        # mag_bonus = weights.pow(2).sum(dim=1).sqrt()

        raw_scores = raw_scores + lambda_reg * mag_bonus
        raw_scores_per_layer[f"layer_{i}"] = raw_scores
        total_sum += raw_scores.sum()


    # Normalize layerwise
    eps = 1e-12
    neuron_scores = {}
    if normalize_layerwise:
        for name, raw in raw_scores_per_layer.items():   # raw is shape [out_features]
            neuron_scores[name] = raw / (raw.sum() + eps)
    else:
         for name, raw in raw_scores_per_layer.items():
             neuron_scores[name] = raw

    # Flatten scores across layers
    all_scores = []
    layer_ranges = {}
    start_idx = 0

    for i in range(num_layers):
        scores = neuron_scores[f"layer_{i}"]
        num_neurons = scores.shape[0]
        all_scores.append(scores)
        layer_ranges[i] = (start_idx, start_idx + num_neurons)
        start_idx += num_neurons
    global_scores = torch.cat(all_scores)
    print("GLobal Scores:")
    print(global_scores)
    return global_scores, all_scores, layer_ranges



def calculate_grad_hess_rademache_flags(model, c, dataloader, hessian_coe, l1 = 0, 
                                    normalize_layerwise = True):
    """
    Returns three lists (one per param):
        gradients, diag_hessians, scores
    where
        score_i = g_i * w_i + h_i * w_i**2
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device   = next(model.model.parameters()).device
    num_grads = c.num_grads
    loss_fn  = nn.CrossEntropyLoss()
    num_layers = len(model.layers)


    params = [
        dict(model.model.named_parameters())
        [f"encoder.layer.{i}.intermediate.dense.weight"]
        for i in range(len(model.layers))
    ]


    grad_sums  = [torch.zeros_like(p, device=device) for p in params]
    hess_sums  = [torch.zeros_like(p, device=device) for p in params]

    # ------------ Hutchinson helpers ------------
    def _rademacher_like(p):
        return (torch.randint_like(p, high=2) * 2 - 1).type_as(p)

    def accumulate_diag_hessian(loss):
        # first‑order grads w.r.t params (creates graph ⇒ we can do HVPs)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        for _ in range(4):
            v   = [_rademacher_like(p) for p in params]            # probe
            hvp = torch.autograd.grad(grads, params, v, retain_graph=True)
            for h_acc, vi, hvpi in zip(hess_sums, v, hvp):
                h_acc += hvpi * vi                                 # v ⊙ (H v)
    # --------------------------------------------

    # main loop
    for step, (inputs, targets) in enumerate(dataloader, 1):
        if step > num_grads: break

        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)

        logits = model.get_logits(pixel_values=inputs)[:, 0, :]
        loss = loss_fn(logits, targets)

        # mean gradient
        loss.backward(retain_graph=True)           # keep graph for HVP
        for g_acc, p in zip(grad_sums, params):
            g_acc += p.grad.detach()

        # diagonal Hessian
        accumulate_diag_hessian(loss)

        if step % 50 == 0:
            print(f"[{step:>4}/{num_grads}] processed")

    layer_gradients = [g_sum / num_grads for g_sum in grad_sums]
    diag_hessians = [d / (num_grads * 4) for d in hess_sums]
    raw_scores_per_layer = {}
    total_sum = 0.0

    for i in range(num_layers):
        weights = dict(model.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"].data
        grad = layer_gradients[i]
        hessian_diag = diag_hessians[i]

        # Compute importance scores using Fisher diagonal

        lambda_reg = l1
        # Compute importance scores using Fisher diagonal
        raw_scores = (abs(grad * weights) + hessian_coe * abs(hessian_diag * weights.pow(2))).sum(dim=1)

        # now add a magnitude‐based bonus
        # option A: L1-based bonus
        mag_bonus = weights.abs().sum(dim=1)  
        # option B: L2-based bonus (group-Lasso style)
        # mag_bonus = weights.pow(2).sum(dim=1).sqrt()

        raw_scores = raw_scores + lambda_reg * mag_bonus
        raw_scores_per_layer[f"layer_{i}"] = raw_scores
        total_sum += raw_scores.sum()



    # Normalize layerwise
    eps = 1e-12
    neuron_scores = {}
    if normalize_layerwise:
        for name, raw in raw_scores_per_layer.items():   # raw is shape [out_features]
            neuron_scores[name] = raw / (raw.sum() + eps)
    else:
        for name, raw in raw_scores_per_layer.items():
            neuron_scores[name] = raw


    # Flatten scores across layers
    all_scores = []
    layer_ranges = {}
    start_idx = 0

    for i in range(num_layers):
        scores = neuron_scores[f"layer_{i}"]
        num_neurons = scores.shape[0]
        all_scores.append(scores)
        layer_ranges[i] = (start_idx, start_idx + num_neurons)
        start_idx += num_neurons
    global_scores = torch.cat(all_scores)
    print("GLobal Scores:")
    print(global_scores)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    return global_scores, all_scores, layer_ranges

def snip_prune_and_evaluate_flags(
        opt: Model,
        pruning_config: PruningConfig,
        prune_frac = 0.02,
        hessian_coefficient = 0.5,
        output = None,
        l1 = None,
        retain_dataloader = False,
        retain_coefficient = 1,
        normalize_layerwise = True,
        hessian_approximation = None
    ):
    """
    Prune and evaluate the model

    Args:
        opt (Model): model to prune and evaluate
        pruning_config (PruningConfig): config for pruning
        focus_out (dict): output of get_midlayer_data for focus dataset
        cripple_out (dict): output of get_midlayer_data for cripple dataset
        iteration (int): iteration number for when activations are not recalculated

    Returns:
        output (RunDataItem): Eval data to add to RunDataHistory.
    """
    c = copy.deepcopy(pruning_config)
    
    print(f'cripple: {c.cripple}')
    print(f'focus: {c.focus}')
    print(f'retain_dataloader: {retain_dataloader}')
    print(f'retain_coefficient: {retain_coefficient}')
    print(f'hessian_coefficient: {hessian_coefficient}')
    print(f'hessian_approximation: {hessian_approximation}')
    print(f'l1: {l1}')
    
    if hessian_approximation == "rademache":
        hessian_approximation = calculate_grad_hess_rademache_flags
    elif hessian_approximation =="fisher": 
        hessian_approximation = calculate_grad_hess_flags
    cripple_dataloader = get_VIT_dataloder(c.cripple)
    if retain_dataloader:
        retain_dataloader = get_VIT_dataloder(c.focus)
        global_scores, all_scores, layer_ranges = hessian_approximation(opt, c, cripple_dataloader, hessian_coefficient, normalize_layerwise=normalize_layerwise, l1=l1)
        global_scores_retain, all_scores_retain, layer_ranges_retain = hessian_approximation(opt, c, retain_dataloader, hessian_coefficient, normalize_layerwise=normalize_layerwise, l1=0)
        epsilon = 1e-12
        global_scores = global_scores / ((retain_coefficient * global_scores_retain) + epsilon)
    else:
        global_scores, all_scores, layer_ranges = hessian_approximation(opt, c, cripple_dataloader, hessian_coefficient, normalize_layerwise=normalize_layerwise, l1=l1)

    N = global_scores.numel()
    forget_neurons = max(1, math.floor(N * prune_frac))
    print(f'forget neurons: {forget_neurons}')

    # Get global top-k indices
    scores, global_topk_indices = torch.topk(global_scores, k=forget_neurons)
    print("Top scores:")
    print(scores)
    print("Top Indices:")
    print(global_topk_indices)
    
    # Initialize output tensors
    max_neurons = max(s.shape[0] for s in all_scores)
    global_mask = torch.zeros(len(opt.layers), max_neurons, dtype=torch.bool)
    layer_masks = []
    
    # Create masks for each layer
    neurons_kept = 0
   
    for i in layer_ranges: 
        
        start, end = layer_ranges[i]
        num_neurons = end - start
        
        # Get indices that fall in this layer's range
        layer_indices = global_topk_indices[(global_topk_indices >= start) & 
                                        (global_topk_indices < end)] - start
        
        # Create layer mask
        layer_mask = torch.zeros(num_neurons, dtype=torch.bool)
        if len(layer_indices) > 0:
            layer_mask[layer_indices] = True
        layer_masks.append(layer_mask)
        
        # Update global mask for the correct layer 'i'
        # The other layers will remain False (or 0), which is the correct behavior.
        global_mask[i, :num_neurons] = layer_mask
        neurons_kept += len(layer_indices)
        
        print(f"Layer {i}: Pruned {len(layer_indices)}/{num_neurons} neurons")

    global_mask = global_mask.int()
    print("\nGlobal Mask: ")
    for i in range(len(opt.layers)):
        print(f"Layer {i}: {global_mask[i]}")

    global_mask = global_mask.detach().cpu().numpy()
   
    print(f'Global mask size: {global_mask.size}')
    print(f'prune mask 1 percentage: {np.sum(global_mask==1)/global_mask.size}') # .numel() is not a numpy function
    print(f'prune mask 0 percentage: {np.sum(global_mask==0)/global_mask.size}')
    directory = f"outputs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if output is None:
        torch.save(global_mask, f"{directory}/prune_mask_forget_frac_{prune_frac}_layer_valid.pt")
        print(f"Saved prune_mask to {directory}/prune_mask.pt")
    else:
         torch.save(global_mask, f"{directory}" +output)
         print(f"Saved prune_mask to {directory}"+output)
    return

def run_snip_pruning_flags(c: PruningConfig, forget_frac = None, hessian_coefficient = 0.5, output = None, 
                       l1 = None, retain_dataloader = False,
                        retain_coefficient = 1,
                        normalize_layerwise = True,
                        hessian_approximation = None):
    # 2 different hessian approximations
    # no abs 
    # Initialise Model and show details about model
    print("after login")
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )
    print("after opt")

    assert c.run_pre_test==False, "run_pre_test is not supported in this version"
    # print(f'********* starting evalution before pruning *********')
    # data = evaluate_all(opt, c.eval_sample_size,
    #     c.datasets, c.collection_sample_size)
    # history.add(data)
    # print(history.df.T)

    
    snip_prune_and_evaluate_flags(opt, c, forget_frac, 
                                    l1=l1, hessian_coefficient=hessian_coefficient, output=output, 
                                    retain_dataloader=retain_dataloader, retain_coefficient = retain_coefficient, 
                                    normalize_layerwise= normalize_layerwise,
                                    hessian_approximation = hessian_approximation
                                )

    return opt


@torch.no_grad()
def prune_and_compensate_all_layers_CAP_new(
    model: torch.nn.Module,
    prune_mask: torch.Tensor,                         # [num_layers, out_features]; True => prune that neuron (entire row)
    fisher_invs: list[EmpiricalBlockFisherInverseCap],
    layer_params: list[torch.nn.Parameter] | None = None,
    eps: float = 1e-12,
    device: str | torch.device | None = None,
):
    """
    CAP compensation:
      For each masked neuron (row), iteratively remove the scalar weight that
      minimizes w_j^2 / diag(F^{-1})_jj, apply OBS update, then rank-1 downdate
      the inverse; repeat until the whole row is zero.
    """
    print(f'device is {device}')
    if device is None:
        device = next(model.parameters()).device
    num_layers = prune_mask.shape[0]

    # param cache if not provided
    if layer_params is None:
        layer_params = [
            dict(model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
            for i in range(num_layers)
        ]

    for layer_idx in range(num_layers):
        W_param = layer_params[layer_idx]
        W = W_param.data.view(-1)                           # flattened view
        out_features, in_features = W_param.shape           # [3072, 768] for ViT-B/16
        Finv = fisher_invs[layer_idx]
        diag_Finv = Finv.diag()                             # length 3072*768

        # rows to prune (mask==True)
        row_mask = prune_mask[layer_idx].to(device).bool()
        rows = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
        
        for r in rows:
            row_start = r * in_features
            row_slice = slice(row_start, row_start + in_features)

            # Iteratively zero-out all columns in this row
            # (greedy CAP: pick j with min w_j^2 / diag(F^-1)_jj at each step)
            # stop when all entries in the row are ~0
            # To be robust, we do exactly `in_features` iterations.
            for _ in range(in_features):
                w_row = W[row_slice]
                alive = (w_row != 0)
                if not torch.any(alive):
                    break

                diag_row = diag_Finv[row_slice]

                # avoid tiny / NaN denominators
                denom = diag_row.clamp_min(eps)

                # CAP selection
                scores = (w_row ** 2) / denom
                scores[~alive] = float("inf")
                j = int(torch.argmin(scores).item())
                idx = row_start + j

                # OBS update to zero weight idx
                djj = float(diag_Finv[idx].clamp_min(eps))
                wj = float(W[idx])
                if abs(wj) < 1e-40:
                    # already zero or numerically tiny; skip
                    continue

                # delta_w = -(w_j / (F^-1)_{jj}) * (F^-1 e_j)
                e_j = torch.zeros_like(W)
                e_j[idx] = 1.0
                Finv_ej = Finv.mul(e_j)                     # (F^-1) e_j
                scale = -wj / djj
                if r % 1000 == 0:
                    print(f'r is {r}')
                    print(scale)
                W.add_(scale * Finv_ej)                     # apply OBS

                if (r % 1000) == 0:
                    topk = torch.topk(Finv_ej.abs(), k=5)
                    print("ej top idx:", topk.indices.tolist())

                # explicitly zero the coordinate
                W[idx] = 0.0

                # CAP downdate (Sherman–Morrison on the inverse for removed index)
                Finv.downdate_removed_index(idx)

            # ensure numerical cleanup of the row
            W[row_slice] = 0.0

        # write back
        W_param.data.copy_(W.view_as(W_param))


# def apply_negrad_unlearning(c):
#     from munl.unlearning.neggradplus import (
#     NegGradPlus,
#     DefaultNegGradPlusUnlearningConfig,
#     )


#     opt = Model(
#         c.model_size,
#         limit=c.token_limit,
#         dtype=c.dtype,
#         svd_attn=c.svd_attn,
#         use_accelerator=c.use_accelerator,
#         model_device=c.model_device,
#         mask_fn=c.mask_fn,
#         )
#     torch_model = opt.model 
#     # Wrap the model to ensure it returns only logits

#     # torch_model = TakerAsModule(opt)   
#     # torch_model = opt.model   
#     cripple_dataloader = get_VIT_dataloder(c.cripple)
#     retain_dataloader = get_VIT_dataloder(c.focus)
#     num_grads = c.num_grads
#     from munl.utils import DictConfig
#     from omegaconf import OmegaConf, open_dict

#     cfg = OmegaConf.structured(DefaultNegGradPlusUnlearningConfig())
#     with open_dict(cfg):
#         cfg.weight_decay = 0.0 
#     cfg.num_epochs   = 5          # ⇦ override defaults if needed
#     cfg.alpha        = 0.99
#     cfg.batch_size   = 64
#     print(cfg)
#     # cfg.optimizer = DictConfig(cfg.optimizer)
#     # cfg.optimizer.learning_rate = 1e-3
#     # cfg.optimizer.learning_rate = 1e-3        # SGD by default

#     device = "cuda"            # or "cpu"
#     unlearner = NegGradPlus(cfg, device=device, writer=None)

#     scrubbed_model = unlearner.unlearn(
#         model=torch_model,
#         retain_loader=cripple_dataloader,
#         forget_loader=retain_dataloader,
#         val_loader = None
#     )
#     return scrubbed_model
from transformers.modeling_outputs import BaseModelOutputWithPooling

class CEEnsureLong(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, target):
        # TEMP sanity print so you can see it actually gets used once
        if not hasattr(self, "_printed"):
            print("CEEnsureLong called with",
                  "logits", tuple(logits.shape), logits.dtype,
                  "target", tuple(target.shape), target.dtype)
            self._printed = True

        # squeeze [B,1] -> [B]
        if target.ndim > 1 and target.size(-1) == 1:
            target = target.squeeze(-1)
        # one-hot / soft labels -> indices
        if target.ndim > 1 and target.size(-1) == logits.size(-1):
            target = target.argmax(dim=-1)
        return self.ce(logits, target.long())


class ViTLogitsWrapper(torch.nn.Module):
    """
    A wrapper for a Hugging Face ViT model to ensure it returns only logits.
    """
    def __init__(self, model, taker):
        super().__init__()
        self.model = model
        self.taker = taker

    def forward(self, pixel_values):
        # Call the original model via the taker instance
        outputs = self.taker.get_logits(pixel_values=pixel_values)
        logits = outputs[:, 0, :]
        
        return logits
    
def apply_negrad_unlearning(c):
    from munl.unlearning.neggradplus import (
        NegGradPlus,
        DefaultNegGradPlusUnlearningConfig,
    )
    from omegaconf import OmegaConf, open_dict

    print("In negrad unlearning")
    opt = Model(
        "google/vit-base-patch16-224",
        limit=1000,
        dtype="fp32",
        svd_attn=False,
        use_accelerator=True,
        model_device=None,
        mask_fn="step",
    )

    print("model has been loaded in")
    device = next(opt.model.parameters()).device
    print(f"device is{device}")
    opt.model.to(device)
    original_torch_model = opt.model

    model_for_unlearning = ViTLogitsWrapper(original_torch_model, opt)

    cripple_dataloader = get_VIT_dataloder(c.cripple)
    retain_dataloader = get_VIT_dataloder(c.focus)

    cfg = OmegaConf.structured(DefaultNegGradPlusUnlearningConfig())
    with open_dict(cfg):
        cfg.weight_decay = 0.0
    cfg.num_epochs = 5
    cfg.alpha = 0.99
    cfg.batch_size = 64
    print(cfg)
    print("before unlearniner")
    unlearner = NegGradPlus(cfg, device=device, writer=None)
    unlearner.criterion = CEEnsureLong() 
    print("before unlearniner")
    scrubbed_model = unlearner.unlearn(
        model=model_for_unlearning,  
        retain_loader=cripple_dataloader,
        forget_loader=retain_dataloader,
        val_loader=None
    )
    return scrubbed_model