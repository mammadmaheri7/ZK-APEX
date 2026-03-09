import copy
import torch
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict
from .model import Model
from .prune import get_VIT_dataloder, EmpiricalBlockFisherInverse
from .fine_tune import WrappedDataset, compute_personalized_fisher_invs, personalize_vit
from .model_saving import save_taker_payload, load_taker_payload
from .activations import get_midlayer_data
from .scoring import score_indices_by
from torchvision import transforms
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch.nn.functional as F
from taker.eval import evaluate_all
import os
from PIL import Image
import time
import gc



def _clear_all_hooks(root):
    for m in root.modules():
        for attr in ("_forward_pre_hooks","_forward_hooks","_backward_pre_hooks","_backward_hooks"):
            if hasattr(m, attr):
                getattr(m, attr).clear()

def rehydrate_taker(payload_path, device="cuda", add_hooks=False, dtype="fp32"):
    # 1) rebuild wrapper + load weights (your loader from earlier)
    m = load_taker_payload(payload_path, dtype=dtype, model_device=device)

    # 2) nuke any stale hooks that might have been left hanging
    _clear_all_hooks(m.model)

    # 3) if you WANT pruning/compensation hooks active, re-attach them now
    #    otherwise leave them off for a plain forward
    if add_hooks:
        # if your wrapper exposes this; otherwise call whatever re-adds hooks
        m.init_model(add_hooks=True)

    # 4) move + eval
    m.model.to(device).eval()

    # 5) quick sanity checks (these should be normal)
    cfg = m.model.config
    assert getattr(cfg, "hidden_act", "gelu") in {"gelu","relu","gelu_fast","gelu_new","silu"}, cfg.hidden_act
    assert getattr(cfg, "num_hidden_layers", 12) == 12, cfg.num_hidden_layers

    return m

def evaluate_personalization_set(opt):
    cache_root = "../cache_personalization"
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_val_cache = os.path.join(cache_root, "sketch_val")


    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])
    
    imagenet_val_dataset = load_from_disk(im_val_cache)
    val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
    val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )


    sketch_val_dataset = load_from_disk(sk_val_cache)
    val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
    val_loader_sketch = DataLoader(
        val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
    }
    with torch.no_grad():
        device = next(opt.model.parameters()).device
        opt.model.to(device)
        eval_results = evaluate(opt, val_loaders, device=next(opt.model.parameters()).device)
    return eval_results


@torch.no_grad()
def build_weight_masks_per_param(
    per_weight_scores: Dict[str, torch.Tensor],
    *,
    top_fraction: Optional[float] = None,
    top_k_per_param: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Per-layer selection: for each param (out,in), take the top fraction (or top-k)
    of **individual weights** as pruned. Returns {name: BoolTensor[out,in]}.
    """
    if (top_fraction is None) == (top_k_per_param is None):
        raise ValueError("Specify exactly one of top_fraction or top_k_per_param")

    masks: Dict[str, torch.Tensor] = {}
    for name, S in per_weight_scores.items():
        flat = S.detach().reshape(-1)
        n = flat.numel()
        if top_k_per_param is not None and name in top_k_per_param:
            k = max(0, min(int(top_k_per_param[name]), n))
        else:
            k = max(0, min(int(round((top_fraction or 0.0) * n)), n))

        mask_flat = torch.zeros(n, dtype=torch.bool, device="cpu")
        if k > 0:
            idx = torch.topk(flat.to("cpu"), k, largest=True, sorted=False).indices
            mask_flat[idx] = True

        masks[name] = mask_flat.view_as(S).to(S.device)
        pct = masks[name].float().mean().item() * 100
        print(f"[per-param] {name}: prune {pct:.2f}% of weights")
    return masks


@torch.no_grad()
def build_weight_masks_global_fraction(
    per_weight_scores: Dict[str, torch.Tensor],
    *,
    top_fraction: float,
    min_at_least_one: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Global selection across ALL layers: take the top fraction of **individual weights**
    (largest scores = pruned). Returns {name: BoolTensor[out,in]}.
    """
    if not (0.0 <= top_fraction <= 1.0):
        raise ValueError("top_fraction must be in [0, 1].")

    names = list(per_weight_scores.keys())
    flats = [per_weight_scores[n].detach().reshape(-1).to("cpu") for n in names]
    sizes = [f.numel() for f in flats]
    total = sum(sizes)

    if total == 0:
        return {n: torch.zeros_like(per_weight_scores[n], dtype=torch.bool) for n in names}

    k = int(round(top_fraction * total))
    if min_at_least_one and k == 0 and top_fraction > 0.0:
        print("in min at least one")
        k = 1
    k = max(0, min(k, total))

    if k == 0:
        return {n: torch.zeros_like(per_weight_scores[n], dtype=torch.bool) for n in names}

    concat = torch.cat(flats, dim=0)  # [total] on CPU
    sel = torch.topk(concat, k, largest=True, sorted=False).indices  # [k]

    # map back to per-layer masks
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes):
        mask_flat = torch.zeros(size, dtype=torch.bool)
        in_this = sel[(sel >= offset) & (sel < offset + size)] - offset
        if in_this.numel():
            mask_flat[in_this] = True
        S = per_weight_scores[name]
        masks[name] = mask_flat.view_as(S).to(S.device)
        pct = masks[name].float().mean().item() * 100
        print(f"[global] {name}: prune {pct:.2f}% of weights")
        offset += size

    return masks

# ---- You must provide this class from your codebase ----
# It should support: __init__(num_grads, fisher_block_size, num_weights, damp, device),
#   .add_grad(grad_flat: Tensor)   and   .fisher_diag() -> Tensor[num_weights]
# from your module import EmpiricalBlockFisherInverse

def _zero_rows_(lin_weight: torch.Tensor, lin_bias: Optional[torch.Tensor], row_mask: torch.Tensor) -> None:
    lin_weight.data[row_mask, :] = 0
    if lin_bias is not None:
        lin_bias.data[row_mask] = 0

def _default_forward_fn(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Tries common ViTForImageClassification forward patterns to produce logits.
    Override via forward_fn if your wrapper differs.
    """
    out = model(pixel_values=pixel_values)
    # huggingface returns an object with .logits; if it's a tuple, first is logits
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    # fallback: assume model returns logits directly
    return out

def _default_forward_fn(wrapper_model, x: torch.Tensor) -> torch.Tensor:
    # Use your wrapper's API, same as your current code
    # Produces logits shaped [B, num_classes]
    return wrapper_model.get_logits(pixel_values=x)[:, 0, :]

def _find_intermediate_weight_params(hf_model: nn.Module):
    """
    Find the ViT MLP intermediate weight params:
      ...encoder.layer.{i}.intermediate.dense.weight
    Returns a sorted list of (layer_index, full_name, Parameter).
    """
    hits = []
    for name, p in hf_model.named_parameters():
        if "encoder.layer." in name and name.endswith(".intermediate.dense.weight"):
            parts = name.split(".")
            li = parts.index("layer")
            idx = int(parts[li + 1])
            hits.append((idx, name, p))
    hits.sort(key=lambda t: t[0])
    return hits

def _layer_key_from_full_name(full_name: str) -> str:
    # Normalize keys to "encoder.layer.{i}"
    parts = full_name.split(".")
    if "encoder" in parts and "layer" in parts:
        try:
            li = parts.index("layer")
            return f"encoder.layer.{int(parts[li+1])}"
        except Exception:
            pass
    # Fallback: best-effort; last resort use the tail
    return ".".join(full_name.split(".")[-4:-2])


@torch.no_grad()  # we still capture grads via hooks; backward happens inside a grad-enabled context below
def rank_vit_intermediate_nodes_return_grad_hessian(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    EmpiricalBlockFisherInverse,  # your class
    num_grads: int = 512,
    fisher_block_size: int = 1024,
    switch_m: bool = False,
    damp: float = 1e-3,
    device: Optional[torch.device] = None,
    loss_fn: Optional[Callable] = None,
    selective_pruning: bool = False,  # kept for signature compatibility; not used here
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = (
        lambda W, dW, F: (dW.abs() * W.abs()) + ((F if F is not None else 0) * (W * W).abs())
    ),
    c=None,             # kept for signature compatibility; not used here
    weight_based: bool = True,  # irrelevant now; we always return per-weight scores
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, object]]:
    """
    Collects mean gradients and blockwise EmpiricalBlockFisherInverse per ViT encoder layer's
    intermediate dense weight, then computes *per-weight* scores using `scoring_fn`.

    Returns:
      {
        "encoder.layer.{i}": (per_weight_scores [out,in],
                              per_weight_mean_grad [out,in],
                              fisher_inverse_object)
      }
    """
    # Resolve devices & fns
    base = getattr(model, "model", model)  # support wrapper with .model
    if device is None:
        device = next(base.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if forward_fn is None:
        forward_fn = _default_forward_fn

    base.eval()  # disable dropout etc.

    # 1) Discover target params
    param_targets = _find_intermediate_weight_params(base)
    if not param_targets:
        raise RuntimeError("No encoder.layer.{i}.intermediate.dense.weight params found.")
    num_layers = len(param_targets)

    # 2) Allocate accumulators + per-layer Fisher inverse objects
    layer_grad_sums: List[torch.Tensor] = []
    fisher_invs: List[object] = []
    for _, _, p in param_targets:
        layer_grad_sums.append(torch.zeros_like(p, device=device))
        inv = EmpiricalBlockFisherInverse(
            num_grads=num_grads,
            fisher_block_size=fisher_block_size,
            num_weights=p.numel(),
            damp=damp,
            device=p.device,
        )
        # Optional branch control if your class supports it
        if hasattr(inv, "switch_m"):
            inv.switch_m = switch_m
        fisher_invs.append(inv)

    # 3) Hooks to capture grads each backward
    layer_grads: List[Optional[torch.Tensor]] = [None] * num_layers
    def make_hook(idx: int):
        def _hook(g: torch.Tensor):
            layer_grads[idx] = g
            layer_grad_sums[idx].add_(g)
        return _hook

    handles = []
    name_to_idx = {full_name: i for i, full_name, _ in param_targets}
    params = dict(base.named_parameters())
    for i, full_name, _ in param_targets:
        p = params[full_name]
        if not p.requires_grad:
            p.requires_grad_(True)
        handles.append(p.register_hook(make_hook(i)))

    # 4) Iterate batches (up to num_grads): forward+backward, update Fisher inverses
    seen = 0
    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if seen >= num_grads:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        base.zero_grad(set_to_none=True)
        # enable grads just for this pass
        with torch.enable_grad():
            logits = model.get_logits(pixel_values=inputs)[:, 0, :] 
            loss = loss_fn(logits, targets)
            loss.backward()

        # add grads to Fisher inverses (row-major flatten)
        for i, g in enumerate(layer_grads):
            if g is not None:
                fisher_invs[i].add_grad(g.reshape(-1))
        layer_grads = [None] * num_layers

        seen += 1
        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")

    for h in handles:
        h.remove()

    if seen == 0:
        raise ValueError("Dataloader yielded zero batches; cannot compute scores.")

    # 5) Mean per-weight gradients and Fisher-diagonal (shaped like W)
    layer_mean_grads: List[torch.Tensor] = [g_sum / float(seen) for g_sum in layer_grad_sums]
    layer_fisher_diags: List[torch.Tensor] = [
        inv.fisher_diag().reshape_as(g) for inv, g in zip(fisher_invs, layer_mean_grads)
    ]

    # 6) Per-weight scoring (no row reduction), and package results per encoder layer
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor, object]] = OrderedDict()
    for (i, full_name, p), dW, Fdiag, inv in zip(param_targets, layer_mean_grads, layer_fisher_diags, fisher_invs):
        W = p.data
        per_weight_scores = scoring_fn(W, dW, Fdiag)  # shape [out,in]
        layer_key = _layer_key_from_full_name(full_name)  # "encoder.layer.{i}"
        results[layer_key] = (per_weight_scores.detach(), dW.detach().cpu(), inv.cpu())

    return results

# --- helpers to convert scores to masks and apply (optional) ---

@torch.no_grad()
def select_top_rows(row_scores: torch.Tensor, *, top_fraction: Optional[float] = None, top_k: Optional[int] = None) -> torch.Tensor:
    n = row_scores.numel()
    if (top_fraction is None) == (top_k is None):
        raise ValueError("Specify exactly one of top_fraction or top_k")
    k = top_k if top_k is not None else max(1, int(round(top_fraction * n)))
    k = min(max(k, 0), n)
    mask = torch.zeros(n, dtype=torch.bool, device=row_scores.device)
    if k > 0:
        idx = torch.topk(row_scores, k, largest=True, sorted=False).indices
        mask[idx] = True
    return mask


@torch.no_grad()
def build_masks_global_fraction(
    row_scores: Dict[str, torch.Tensor],
    *,
    top_fraction: float,
    min_at_least_one: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Selects the top fraction of rows ACROSS ALL LAYERS (largest scores = pruned).
    Returns {param_name: BoolTensor[out_features]}.
    """
    if not (0.0 <= top_fraction <= 1.0):
        raise ValueError("top_fraction must be in [0, 1].")

    names = list(row_scores.keys())
    sizes = [row_scores[n].numel() for n in names]
    total = sum(sizes)
    if total == 0:
        return {n: torch.zeros_like(row_scores[n], dtype=torch.bool) for n in names}

    # how many rows globally
    k = int(round(top_fraction * total))
    if min_at_least_one and k == 0 and total > 0 and top_fraction > 0.0:
        k = 1
    k = max(0, min(k, total))

    # concat on CPU for simplicity/safety across devices
    concat = torch.cat([row_scores[n].detach().to("cpu") for n in names], dim=0)  # [total]
    if k == 0:
        return {n: torch.zeros(s, dtype=torch.bool, device=row_scores[n].device) for n, s in zip(names, sizes)}

    # take global top-k
    top_idx = torch.topk(concat, k, largest=True, sorted=False).indices  # [k] in [0,total)

    # boundaries to map global -> per-layer
    bounds = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), dim=0).tolist()), dtype=torch.long)  # [L+1]
    # which layer each selected index falls into
    layer_idx = torch.bucketize(top_idx, bounds[1:])  # [k] values in [0, L-1]

    # build per-layer masks
    masks: Dict[str, torch.Tensor] = {}
    for li, name in enumerate(names):
        dev = row_scores[name].device
        out = sizes[li]
        mask = torch.zeros(out, dtype=torch.bool, device="cpu")
        sel_in_layer = top_idx[layer_idx == li]
        if sel_in_layer.numel() > 0:
            local = sel_in_layer - bounds[li]
            mask[local] = True
        masks[name] = mask.to(dev)
        print(f"Prune mask in percentage of {name}")
        print(masks[name].float().mean() * 100)
    return masks


@torch.no_grad()
def build_masks_per_param(row_scores: Dict[str, torch.Tensor], *, top_fraction: Optional[float] = None, top_k_per_param: Optional[Dict[str, int]] = None) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for pname, scores in row_scores.items():
        if top_k_per_param and pname in top_k_per_param:
            masks[pname] = select_top_rows(scores, top_k=top_k_per_param[pname])
        else:
            masks[pname] = select_top_rows(scores, top_fraction=top_fraction)

        print(f"Prune mask in percentage of {pname}")
        print(masks[pname].float().mean() * 100)
    return masks

@torch.no_grad()
def apply_masks_by_param_name_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Zero rows for each parameter name in masks."""
    # Map param name -> (module weight tensor, optional bias)
    param_name_to_linear = {}
    module_map = {n: m for n, m in model.named_modules()}
    for full_name, mask in masks.items():
        # full_name ends with ".weight", we want the module to zero rows (and bias)
        module_name = full_name.rsplit(".weight", 1)[0]
        lin = module_map.get(module_name, None)
        if not isinstance(lin, nn.Linear):
            # Fallback: zero weight rows directly; bias unknown
            weight = dict(model.named_parameters())[full_name]
            _zero_rows_(weight, None, mask)
        else:
            _zero_rows_(lin.weight, lin.bias, mask)



def evaluate(model, val_loaders, device):
    model.model.eval()
    if isinstance(val_loaders, dict):
        results = {}
        for key, loader in val_loaders.items():
            total, correct, loss_sum = 0, 0, 0
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model.predictor(pixel_values=imgs).logits
                loss = F.cross_entropy(outputs, labels)
                loss_sum += loss.item()
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            results[key] = {"accuracy": correct / total if total > 0 else 0, "avg_loss": loss_sum / len(loader)}
        return results
    else:
        total, correct, loss_sum = 0, 0, 0
        for imgs, labels in val_loaders:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model.predictor(pixel_values=imgs).logits
            loss = F.cross_entropy(outputs, labels)
            loss_sum += loss.item()
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = loss_sum / len(val_loaders)
        accuracy = correct / total if total > 0 else 0
        return {"combined": {"accuracy": accuracy, "avg_loss": avg_loss}}


import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Iterable, Union
from collections import OrderedDict


# ---------- 2) Build Fisher inverses in column-major space (wrapper) ----------

################################################################################################################################################################


@torch.no_grad()
def apply_weight_masks_(model, weight_masks):
    base = _base_model(model)  # works with wrapper or raw HF model
    params = dict(base.named_parameters())
    for key, m in weight_masks.items():
        pname = key
        if pname not in params:
            # Treat key as "encoder.layer.i" and resolve to the real param
            pname = _resolve_layer_weight_name(base, key)
        p = params[pname]
        if m.shape != p.shape:
            raise ValueError(
                f"Mask for '{key}' has shape {tuple(m.shape)} but param '{pname}' is {tuple(p.shape)}"
            )
        p.data[m.to(device=p.device)] = 0.0

def score_pruned(c, path, weight_based = False, personalization_hps = None):
        masks = torch.load(path, map_location="cuda")
        opt = Model(
            c.model_size,
            limit=c.token_limit,
            dtype=c.dtype,
            svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator,
            model_device=c.model_device,
            mask_fn=c.mask_fn,
        )
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])
        
        cache_root = "../cache_personalization"
        im_tr_cache = os.path.join(cache_root, "imagenet_train")
        imagenet_train_dataset = load_from_disk(im_tr_cache)
        sk_tr_cache = os.path.join(cache_root, "sketch_train")
        sketch_train_dataset = load_from_disk(sk_tr_cache)
        im_val_cache = os.path.join(cache_root, "imagenet_val")
        sk_val_cache = os.path.join(cache_root, "sketch_val") 

        train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
        train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
        # combined_train_ds = ConcatDataset([train_ds_sk])
        train_loader_pers = DataLoader(
            combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )
        
        imagenet_val_dataset = load_from_disk(im_val_cache)
        sketch_val_dataset = load_from_disk(sk_val_cache)
        val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
        val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
        val_loader_sketch = DataLoader(
            val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_loaders = {
                "imagenet": val_loader_imagenet,
                "sketch":   val_loader_sketch,
        }
        personalize_vit_hps = personalization_hps if personalization_hps else {
            "num_epochs": 6,
            "base_lr": 5e-5,
            "layer_decay": 0.8,
            "freeze_layers": 3,
        }

        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
        personalized_model = personalize_vit(
            opt,
            train_loader_pers,
            # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
            val_loaders,
            num_epochs= personalize_vit_hps["num_epochs"],
            base_lr= personalize_vit_hps["base_lr"],
            layer_decay= personalize_vit_hps["layer_decay"],
            freeze_layers= personalize_vit_hps["freeze_layers"],
            total_steps=total_steps
        )
        tmp_opt = copy.deepcopy(opt)
        device = next(opt.model.parameters()).device
        tmp_opt.model.to(device)
        if not weight_based:
            apply_masks_by_param_name_(tmp_opt.model, masks)
        else:
            apply_weight_masks_(tmp_opt.model, masks)
        datasets = ['imagenet-1k-birds']
        eval_sample_size = 1e4
        with torch.no_grad():
            device = next(tmp_opt.model.parameters()).device
            tmp_opt.model.to(device)
            eval_results = evaluate_personalization_set(tmp_opt)
            for key, metrics in eval_results.items():
                print(f"[Personalize] prune only {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
            eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                        dataset_tokens_to_skip=None)
        
        print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
        print(f'----- debug - model performance after unlearning (on birds/birdsless) without compensation -> eval_out: {print_eval_out} ----- \n\n') 

def collate_fn(batch):
    imgs = []
    labels = []
    for sample in batch:
        # Convert flattened list back to tensor and reshape it to (3, 224, 224)
        img = torch.tensor(sample['image']).reshape(3, 224, 224)
        imgs.append(img)
        labels.append(sample['label'])
    return torch.stack(imgs), torch.tensor(labels)



@torch.no_grad()
def build_fisher_invs_rowmajor(
    model: nn.Module,
    target_param_names: List[str],
    EmpiricalBlockFisherInverse,          # <- your class
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_grads: int = 256,
    fisher_block_size: int = 64,
    damp: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,
    switch_m = False
) -> Dict[str, object]:
    """
    Build one EmpiricalBlockFisherInverse per parameter in **row-major** coord order.

    We hook each target parameter, collect g = dL/dW, flatten row-major, and call .add_grad(g_flat).
    Returns {param_full_name: EmpiricalBlockFisherInverse}.
    """
    print("in build_fisher_invs_rowmajor")
    if device is None:
        device = next(model.model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()


    params = dict(model.model.named_parameters())
    # allocate one inverse per target param
    fisher_invs: Dict[str, object] = OrderedDict()

    for pname in target_param_names:
        W = params[pname]
        d = W.numel()
        fisher_invs[pname] = EmpiricalBlockFisherInverse(
            num_grads=num_grads,
            fisher_block_size=fisher_block_size,
            num_weights=d,
            damp=damp,
            switch_m = switch_m,
            device=W.device,
        )
        # fisher_invs[pname].cpu()


    
    # scratch & hooks
    scratch: List[Optional[torch.Tensor]] = [None] * len(target_param_names)
    name2idx = {p: i for i, p in enumerate(target_param_names)}

    def make_hook(idx):
        def _hook(g):
            scratch[idx] = g
        return _hook
    
    handles = []
    for i, pname in enumerate(target_param_names):
        p = params[pname]
        handles.append(p.register_hook(make_hook(i)))

    seen = 0
    for inputs, labels in dataloader:
        if seen >= num_grads:
            break
        
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            logits = model.get_logits(pixel_values=inputs)[:, 0, :]  # CLS token
            loss = loss_fn(logits, labels)
            loss.backward()

        # push grads into row-major inverses
        for i, pname in enumerate(target_param_names):
            g = scratch[i]
            if g is None:
                continue
            # fisher_invs[pname].to(next(model.model.parameters()).device)
            fisher_invs[pname].add_grad(g.reshape(-1))  # row-major flat
            # fisher_invs[pname].cpu()
        scratch = [None] * len(target_param_names)
        seen += 1

    for h in handles:
        h.remove()

    return fisher_invs

@torch.no_grad()
def cap_compensate_from_weight_masks_batched_rowmajor(
    model: nn.Module,
    weight_masks: Dict[str, torch.Tensor],              # { "<param>.weight": Bool[out,in] }
    fisher_invs: Dict[str, object],                     # { "<param>.weight": EmpiricalBlockFisherInverse }
    *,
    eps: float = 1e-9,
) -> None:
    """
    Multi-constraint OBS compensation for **individual weights** (unstructured).
    Everything is done in ROW-MAJOR flattened coordinates (same as your inverse builder).

    For each parameter:
      - theta = W.flatten(row-major)
      - S = indices where weight_mask is True (row-major)
      - For each block b (size B; last block has B_eff):
          * Ainv_b = F_inv[b, :B_eff, :B_eff]
          * S_loc = local indices in this block
          * y = solve(Ainv_b[S,S], theta[S])
          * delta = - Ainv_b[:,S] @ y
          * theta_b += delta ; theta_b[S_loc] = 0
          * Ainv_b -= Ainv_b[:,S] @ solve(Ainv_b[S,S], Ainv_b[S,:])
      - Write theta back into W.view_as(weight)
    """
    params = dict(model.named_parameters())
    print("in cap_compensate_from_weight_masks_batched_rowmajor")
    for pname, mask2d in weight_masks.items():
        if pname not in params or pname not in fisher_invs:
            continue
        W = dict(model.named_parameters())[pname].data
        print(pname, "|W[mask]| sum =", W[mask2d].abs().sum().item())
        W = params[pname]                 # [out, in]
        d = W.numel()
        # fisher_invs[pname].to(next(model.parameters()).device)
        B = fisher_invs[pname].B
        num_blocks = fisher_invs[pname].num_blocks

        # flatten row-major
        theta = W.data.reshape(-1)
        print("theta before:")
        print(theta)
        # selection set S in row-major
        sel_flat = mask2d.to(torch.bool, copy=False).reshape(-1)
        S = sel_flat.nonzero(as_tuple=False).view(-1)
        if S.numel() == 0:
            continue

        # operate in the param's dtype/device
        Ainv_full = fisher_invs[pname].F_inv  # (num_blocks, B, B)
        assert Ainv_full.device == theta.device, \
            f"F_inv on {Ainv_full.device}, weights on {theta.device}; move them to match."

        # per-block multi-OBS
        for b in range(num_blocks):
            g_start = b * B
            g_end   = min((b + 1) * B, d)
            B_eff   = g_end - g_start
            if B_eff <= 0:
                continue

            # local selection inside this block
            in_block = (S >= g_start) & (S < g_end)
            if not torch.any(in_block):
                continue
            S_loc = (S[in_block] - g_start).to(torch.long)  # (k,)

            # block views
            Ainv_b = Ainv_full[b, :B_eff, :B_eff]           # (B_eff,B_eff) view
            theta_b = theta[g_start:g_end]                  # (B_eff,) view

            # submatrices
            A_ss  = Ainv_b.index_select(0, S_loc).index_select(1, S_loc).clone()
            A_ss.diagonal().add_(eps)
            theta_s = theta_b.index_select(0, S_loc) 
            A_alls = Ainv_b.index_select(1, S_loc)          # (B_eff,k)
            A_sall = Ainv_b.index_select(0, S_loc)          # (k,B_eff)


            # compensation Δ = -A[:,S] @ (A[S,S]^{-1} @ theta[S])
            y = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)  # (k,)
            delta = - A_alls @ y                                           # (B_eff,)
            theta_b.add_(delta)                                            # update all coords
            theta_b.index_fill_(0, S_loc, 0.0)                              # enforce sparsity

            # inverse downdate: Ainv <- Ainv - A[:,S] @ inv(A[S,S]) @ A[S,:]
            K = torch.linalg.solve(A_ss, A_sall)                            # (k,B_eff)
            Ainv_b.sub_(A_alls @ K)

        print("theta after:")
        print(theta)
        # fisher_invs[pname].cpu()
        # write back
        W.data.copy_(theta.view_as(W))



def _base_model(obj):
    # Works with your wrapper (obj.model) or raw HF model
    return getattr(obj, "model", obj)

def _resolve_layer_weight_name(base, layer_name: str, weight_suffix: str = "intermediate.dense.weight") -> str:
    """
    Returns the *exact* parameter name in base.named_parameters() for this layer.
    Matches by suffix to be robust to prefixes like 'vit.' or 'model.vit.'.
    """
    want_suffix = f"{layer_name}.{weight_suffix}"
    exact = []
    suffix_hits = []
    for n, _ in base.named_parameters():
        if n == want_suffix:
            exact.append(n)
        elif n.endswith(want_suffix):
            suffix_hits.append(n)
    if exact:
        return exact[0]
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    if not suffix_hits:
        raise KeyError(f"Could not find param ending with '{want_suffix}' in model.")
    raise KeyError(f"Ambiguous param for '{want_suffix}': {suffix_hits}")

def build_fisher_inv_rowmajor_for_layer(
    model: nn.Module,
    layer_name: str,                              # e.g. "encoder.layer.3"
    EmpiricalBlockFisherInverse,                  # your class
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_grads: int = 128,
    fisher_block_size: int = 256,                 # larger blocks help per-weight comp
    damp: float = 3e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,
    switch_m: bool = False,
) -> Tuple[str, object]:
    """
    Builds a *single* EmpiricalBlockFisherInverse for {layer_name}.intermediate.dense.weight
    in ROW-MAJOR coordinate order. Returns (param_name, fisher_inv).
    """
    base = _base_model(model)
    if device is None:
        device = next(base.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # find the exact parameter name once
    pname = _resolve_layer_weight_name(base, layer_name)
    params = dict(base.named_parameters())
    print(f"pname is {pname}")
    p = params[pname]
    if not isinstance(p, torch.nn.Parameter):
        raise TypeError(f"{pname} is not an nn.Parameter (got {type(p)})")
    if not p.requires_grad:
        p.requires_grad_(True)

    # allocate inverse (row-major)
    fisher_inv = EmpiricalBlockFisherInverse(
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        num_weights=p.numel(),
        damp=damp,
        device=p.device,
        switch_m = switch_m
    )

    # hook to capture this param's grad
    grad_buf = [None]
    def _hook(g):
        grad_buf[0] = g
    handle = p.register_hook(_hook)

    seen = 0
    for inputs, labels in dataloader:
        if seen >= num_grads:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        base.zero_grad(set_to_none=True)

        # Use your wrapper forward (same as your code):
        logits = model.get_logits(pixel_values=inputs)[:, 0, :]  # CLS token
        loss = loss_fn(logits, labels)
        loss.backward()

        g = grad_buf[0]
        if g is not None:
            fisher_inv.add_grad(g.reshape(-1))  # row-major flat
        grad_buf[0] = None
        seen += 1

    handle.remove()
    if seen == 0:
        raise ValueError("Dataloader yielded zero batches; cannot build Fisher inverse.")

    return pname, fisher_inv

@torch.no_grad()
def cap_compensate_weights_rowmajor_for_layer(
    model: nn.Module,
    layer_name: str,                         # e.g. "encoder.layer.3"
    weight_mask_2d: torch.Tensor,            # Bool [out, in] for THIS layer only
    fisher_inv,                               # EmpiricalBlockFisherInverse for this layer
    basis = None,
    *,
    eps: float = 1e-9,
) -> None:
    """
    Multi-constraint OBS compensation for *individual weights* of a single layer,
    done in ROW-MAJOR coordinate order. Then hard-zeros the selected weights.
    """
    base = _base_model(model)
    pname = _resolve_layer_weight_name(base, layer_name)
    params = dict(base.named_parameters())
    W = params[pname]                         # [out, in]
    d = W.numel()

    print(f"panme in compensation {pname}")
    # sanity: mask shape & device
    assert weight_mask_2d.shape == W.shape, f"mask shape {tuple(weight_mask_2d.shape)} != {tuple(W.shape)}"
    sel_flat = weight_mask_2d.to(torch.bool, copy=False).reshape(-1)
    S = sel_flat.nonzero(as_tuple=False).view(-1)
    if S.numel() == 0:
        return

    # work in row-major flat coords
    theta = W.data.reshape(-1)
    Ainv_full = fisher_inv.F_inv  # (num_blocks, B, B)
    if Ainv_full.device != theta.device:
        Ainv_full = Ainv_full.to(theta.device)

    B = fisher_inv.B
    num_blocks = fisher_inv.num_blocks

    for b in range(num_blocks):
        g_start = b * B
        g_end   = min((b + 1) * B, d)
        B_eff   = g_end - g_start
        if B_eff <= 0:
            continue

        in_block = (S >= g_start) & (S < g_end)
        if not torch.any(in_block):
            continue
        S_loc = (S[in_block] - g_start).to(torch.long)  # (k,)

        Ainv_b  = Ainv_full[b, :B_eff, :B_eff]          # (B_eff,B_eff) view
        theta_b = theta[g_start:g_end]                  # (B_eff,) view

        # sub-blocks
        A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc).clone()
        A_ss.diagonal().add_(eps)
        theta_s = theta_b.index_select(0, S_loc)        # (k,)
        A_alls  = Ainv_b.index_select(1, S_loc)         # (B_eff,k)
        A_sall  = Ainv_b.index_select(0, S_loc)         # (k,B_eff)

        # Δ = -A[:,S] @ (A[S,S]^{-1} @ theta[S])
        y     = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)
        delta = - A_alls @ y

        theta_b.add_(delta)
        theta_b.index_fill_(0, S_loc, 0.0)              # enforce sparsity exactly

        # Woodbury downdate: Ainv <- Ainv - A[:,S] @ inv(A[S,S]) @ A[S,:]
        K = torch.linalg.solve(A_ss, A_sall)            # (k,B_eff)
        Ainv_b.sub_(A_alls @ K)

    # write back shaped param
    W.data.copy_(theta.view_as(W))




    # sketch code
        # Pass in score, Hessian, grad and weights
        # outside plus personalized
        # Calculate difference of weights   
        # for every layer
            # Calculate score

def calculate_score_and_hessian_and_grad(c, hessian_coefficient = 0.5, retain_dataloader = False, retain_coefficient = 1, 
                             forget_frac = 0.02, switch_m = False, damp = 1e-3, num_grads = 50, fisher_block_size = 50):
    
    print(f"Hessian coefficient {hessian_coefficient}, retain dataloader {retain_dataloader}, forget frac: {forget_frac}, switch_m: {switch_m}, \
        fisher block size {fisher_block_size} ")

    def my_scoring(W, dW, F, hessian_coefficient = hessian_coefficient):
        # Pure Fisher (OBD-style) with tiny epsilon

        # eps = 1e-12
        return (abs(dW * W) + hessian_coefficient * abs(F * W.pow(2)))
    
    cripple_dataloader = get_VIT_dataloder(c.cripple)
    print(f"Device is {c.model_device}")
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
    )

    scores_dict = rank_vit_intermediate_nodes_return_grad_hessian(
        opt,
        dataloader=cripple_dataloader,
        EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        damp = damp,
        scoring_fn=my_scoring,
        switch_m= switch_m,
        c = c      # plug your scorer here
    )

    focus_dataloader = get_VIT_dataloder(c.focus)
    scores_dict_retain = rank_vit_intermediate_nodes_return_grad_hessian(
        opt,
        dataloader=focus_dataloader,
        EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        damp=damp,
        scoring_fn=my_scoring,
        switch_m= switch_m,
        c = c      # plug your scorer here
    )


    return scores_dict, _find_intermediate_weight_params(opt.model), scores_dict_retain

    
def per_weight_score_personalize_and_prune_layer(c, personalization_hps = None,  hessian_coefficient = 0.5, retain_dataloader = False, retain_coefficient = 1, 
                             forget_frac = 0.02,  print_personalization_scores = False, p_hps = None, fisher_block_size = 50,
                             damp_compensate = 1e-3, damp_mask = 1e-7, eps = 1e-9, num_grads = 256, print_pruned_only_scores = True, switch_m = False, use_basis = False,  signs = "ABS"):
    print(f"fisher_block_size: {fisher_block_size}, num_grads: {num_grads}, switch_m: {switch_m}")

    scores_dict, pretrained_weights, scores_dict_retain = calculate_score_and_hessian_and_grad(c, hessian_coefficient, retain_dataloader, retain_coefficient, 
                                                                           forget_frac, switch_m, damp = damp_mask, num_grads = num_grads, fisher_block_size = 50)
    
    # opt = Model(
    #     c.model_size,
    #     limit=c.token_limit,
    #     dtype=c.dtype,
    #     svd_attn=c.svd_attn,
    #     use_accelerator=c.use_accelerator,
    #     model_device=c.model_device,
    #     mask_fn=c.mask_fn,
    # )


    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
        transforms.ToTensor(),
    ])
    
    cache_root = "../cache_personalization"
    im_tr_cache = os.path.join(cache_root, "imagenet_train")
    imagenet_train_dataset = load_from_disk(im_tr_cache)
    sk_tr_cache = os.path.join(cache_root, "sketch_train")
    sketch_train_dataset = load_from_disk(sk_tr_cache)
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_val_cache = os.path.join(cache_root, "sketch_val") 

    train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
    train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)
    from torch.utils.data import ConcatDataset
    combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
    # combined_train_ds = ConcatDataset([train_ds_sk])
    train_loader_pers = DataLoader(
        combined_train_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    imagenet_val_dataset = load_from_disk(im_val_cache)
    sketch_val_dataset = load_from_disk(sk_val_cache)
    val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
    val_loader_imagenet = DataLoader(
        val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
    val_loader_sketch = DataLoader(
        val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
    }

    personalize_vit_hps = personalization_hps if personalization_hps else {
        "num_epochs": 6,
        "base_lr": 5e-5,
        "layer_decay": 0.8,
        "freeze_layers": 3,
    }

    total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
    # personalized_model = personalize_vit(
    #     opt,
    #     train_loader_pers,
    #     # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
    #     val_loaders,
    #     num_epochs= personalize_vit_hps["num_epochs"],
    #     base_lr= personalize_vit_hps["base_lr"],
    #     layer_decay= personalize_vit_hps["layer_decay"],
    #     freeze_layers= personalize_vit_hps["freeze_layers"],
    #     total_steps=total_steps
    # )


    # opt = personalized_model
    res = {}

    # print("saving model")
    # save_taker_payload(opt, "taker_30_epochs_8e-4_l4_0.8_decay_0_freeze_AGAIN.pt")
        
    print("Loading model")
    opt = rehydrate_taker("taker_30_epochs_8e-4_l4_0.8_decay_0_freeze_shuffled_train.pt", device="cuda", add_hooks=False, dtype="fp32")
    personalized_weights = _find_intermediate_weight_params(opt.model)

    if print_personalization_scores:
        eval_sample_size = 1e4
        datasets = ['imagenet-1k-birds']
        with torch.no_grad():
            device = next(opt.model.parameters()).device
            opt.model.to(device)
            eval_results = evaluate_personalization_set(opt)
            for key, metrics in eval_results.items():
                print(f"[Personalize] after personalization {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                res["after personalization " + key] = metrics['accuracy']
            eval_out = evaluate_all(opt, eval_sample_size, datasets,
                                    dataset_tokens_to_skip=None)
            
        # exclude 'token_count' from eval_out to print
        print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
        print(f'----- debug - model performance after personalizing -> eval_out: {print_eval_out} ----- \n\n') 
        res["personalization imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base'] / 100


     
    combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
    # combined_train_ds = ConcatDataset([sketch_train_dataset])
    fisher_loader = DataLoader(
        combined_train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    opt.remove_hooks()
    layers = [f"encoder.layer.{i}" for i in range(12)]
    TAU = 0.95
    new_scores = {}
    for idx, layer_name in enumerate(layers):
        W0 = pretrained_weights[idx][2]
        Wp = personalized_weights[idx][2]
        BA = Wp - W0
        _, layer_grad, layer_hessian = scores_dict[layer_name]
        _, layer_grad_retain, layer_hessian_retain = scores_dict_retain[layer_name]
        H = layer_hessian.fisher_diag().reshape_as(W0)
        H_retain = layer_hessian_retain.fisher_diag().reshape_as(W0).to(W0.device)

        BA    = BA.to(dtype=W0.dtype, device=W0.device)
        H     = H.to(dtype=W0.dtype, device=W0.device)
        grad = layer_grad.to(dtype=W0.dtype, device=W0.device)
        grad_retain = layer_grad_retain.to(dtype=W0.dtype, device=W0.device)


        if signs == "ABS":
            new_scores[layer_name] = abs(grad * W0 + grad * BA) + hessian_coefficient * abs(H * W0.pow(2) + H * BA.pow(2))
            if retain_dataloader:
                new_scores[layer_name] = new_scores[layer_name] / retain_coefficient *((abs(grad_retain * W0) + hessian_coefficient * abs(H_retain * W0.pow(2)) + 1e-12))
        
        elif signs == "All Neg":
            new_scores[layer_name] = - (grad * W0 + grad * BA) - hessian_coefficient * (H * W0.pow(2) + H * BA.pow(2))
            if retain_dataloader:
                new_scores[layer_name] = new_scores[layer_name] / retain_coefficient *(-(hessian_coefficient * (grad_retain * W0) - hessian_coefficient *  (H_retain * W0.pow(2)) + 1e-12))
        
        elif signs == "Second Order Neg":
            new_scores[layer_name] =  (grad * W0 + grad * BA) - hessian_coefficient * (H * W0.pow(2) + H * BA.pow(2))
            if retain_dataloader:
                new_scores[layer_name] = new_scores[layer_name] / retain_coefficient *((hessian_coefficient * (grad_retain * W0) - hessian_coefficient *  (H_retain * W0.pow(2)) + 1e-12))
        
        elif signs == "First Order Neg":
            new_scores[layer_name] =  -(grad * W0 + grad * BA) + hessian_coefficient * (H * W0.pow(2) + H * BA.pow(2))
            if retain_dataloader:
                new_scores[layer_name] = new_scores[layer_name] / retain_coefficient *(-(hessian_coefficient * (grad_retain * W0) + hessian_coefficient *  (H_retain * W0.pow(2)) + 1e-12))
        
        elif signs == "All positive":
            new_scores[layer_name] =  (grad * W0 + grad * BA) + hessian_coefficient * (H * W0.pow(2) + H * BA.pow(2))
            if retain_dataloader:
                new_scores[layer_name] = new_scores[layer_name] / retain_coefficient *((hessian_coefficient * (grad_retain * W0) + hessian_coefficient *  (H_retain * W0.pow(2)) + 1e-12))
       

    print("New scores:")
    print(new_scores)
    print(new_scores['encoder.layer.0'])
    masks = build_weight_masks_global_fraction(new_scores, top_fraction=forget_frac)
    
    print("mask is:")
    print(masks)

    if print_pruned_only_scores:
        tmp_opt = copy.deepcopy(opt)
        device = next(opt.model.parameters()).device
        tmp_opt.model.to(device)
        apply_weight_masks_(tmp_opt.model, masks)
        
        datasets = ['imagenet-1k-birds']
        eval_sample_size = 1e4
        with torch.no_grad():
            device = next(tmp_opt.model.parameters()).device
            tmp_opt.model.to(device)
            eval_results = evaluate_personalization_set(tmp_opt)
            for key, metrics in eval_results.items():
                print(f"[Personalize] prune only {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                res["prune only " + key] = metrics['accuracy']
            eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                        dataset_tokens_to_skip=None)
        # exclude 'token_count' from eval_out to print
        print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
        print(f'----- debug - model performance after unlearning (on birds/birdsless) without compensation -> eval_out: {print_eval_out} ----- \n\n') 
        res["prune only imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base'] / 100
        del tmp_opt
    # Have to develop the scores first 
    for idx, layer_name in enumerate(layers):
        # 1) build Fisher inverse for THIS layer only
        pname, F_inv = build_fisher_inv_rowmajor_for_layer(
            model=opt,                                   # your wrapper
            layer_name=layer_name,
            EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
            dataloader=fisher_loader,                    # calibration data
            num_grads=num_grads,                                # smaller can help off-diagonals
            fisher_block_size=fisher_block_size,                       # bigger than 64; tune to VRAM
            damp=damp_compensate,
            switch_m=switch_m                              # if your class supports the SM denom
        )

        
        mask_2d = masks[layer_name]                # Bool [out,in]

        # 3) compensate+prune this single layer
        cap_compensate_weights_rowmajor_for_layer(
            model=opt,
            layer_name=layer_name,
            weight_mask_2d=mask_2d,
            fisher_inv=F_inv,
            eps=1e-9,
        )
     

        # 4) free memory for this layer before the next one
        del F_inv
        torch.cuda.empty_cache()
        opt.remove_hooks()
    
    opt.remove_hooks()
    eval_sample_size = 1e4
    datasets = ['imagenet-1k-birds']
    with torch.no_grad():
        device = next(opt.model.parameters()).device
        opt.model.to(device)
        eval_results = evaluate_personalization_set(opt)
        for key, metrics in eval_results.items():
            print(f"[Personalize] prune and compensate {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
            res["prune and compensate " + key] = metrics['accuracy']
        eval_out = evaluate_all(opt, eval_sample_size, datasets,
                                dataset_tokens_to_skip=None)
        
    # exclude 'token_count' from eval_out to print
    print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    print(f'----- debug - model performance after unlearning and compensation (on birds/birdsless) -> eval_out: {print_eval_out} ----- \n\n') 
    res["prune and compensate imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base'] / 100

    print("after masking + compensating row major res is: ")
    print(res)

    return (res['prune and compensate imagenet'], res['prune and compensate sketch'], res["prune and compensate imagenet-1k-birds"])