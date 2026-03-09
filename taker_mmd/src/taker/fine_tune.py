import copy
import hashlib
import os
import logging
import warnings
import matplotlib.pyplot as plt
import datetime

from taker.eval import evaluate_all
from taker.prune import EmpiricalBlockFisherInverse, prune_and_compensate_all_layers, prune_only_all_layers, prune_and_compensate_all_layers_CAP_new, EmpiricalBlockFisherInverseCap
# Filter out specific warnings from torch._dynamo
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

# Standard logging configuration (may not be necessary, but keep it for general use)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
logging.getLogger('torch._logging').setLevel(logging.ERROR)

import random
from taker.model import Model
from taker.texts import infer_dataset_config, prepare_dataset
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler, ConcatDataset
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from datasets import load_from_disk
from PIL import Image
import optuna
import gc
import math
from typing import List, Optional, Tuple
import numpy as np

from datasets import Dataset
from itertools import islice

import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers import logging as hf_logging
import logging as py_logging
hf_logging.set_verbosity_error()
py_logging.getLogger("transformers.file_utils").setLevel(py_logging.WARNING)

def evaluate(model, val_loaders, device):
    model.model.eval()
    with torch.no_grad():
        if isinstance(val_loaders, dict):
            flag = True
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
def collate_fn(batch):
    imgs = []
    labels = []
    for sample in batch:
        # Convert flattened list back to tensor and reshape it to (3, 224, 224)
        img = torch.tensor(sample['image']).reshape(3, 224, 224)
        imgs.append(img)
        labels.append(sample['label'])
    return torch.stack(imgs), torch.tensor(labels)


def limited_iterable_split(iterable_dataset, total_items, test_ratio=0.5):
    """ Convert only a limited amount of the iterable dataset to list and split it. """
    dataset_list = list(islice(iterable_dataset, total_items))
    split_idx = int(len(dataset_list) * (1 - test_ratio))
    # shuffle the dataset_list if needed
    random.shuffle(dataset_list)
    train = Dataset.from_list(dataset_list[:split_idx])
    test = Dataset.from_list(dataset_list[split_idx:])
    return train, test


def limit_samples_per_class(dataset, per_class_count, selected_classes):
    """
    Iterate over the dataset and select exactly `per_class_count` samples 
    for each class in selected_classes.
    Returns a HuggingFace Dataset created by Dataset.from_list().
    """
    counts = {c: 0 for c in selected_classes}
    collected = []

    # Choose iteration strategy: for HuggingFace or PyTorch datasets use indexing, otherwise iterator
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        iterator = (dataset[i] for i in range(len(dataset)))
    else:
        iterator = iter(dataset)

    for sample in iterator:
        # extract label from dict or tuple/list
        if isinstance(sample, dict) and 'label' in sample:
            label = sample['label']
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            # assume sample == (data, label)
            label = sample[1]
        else:
            # fallback: try dict get or index access
            try:
                label = sample.get('label')
            except Exception:
                label = sample[1] if isinstance(sample, (tuple, list)) and len(sample) > 1 else None

        # normalize sample to dict for consistent Dataset.from_list
        if isinstance(sample, dict):
            sample_dict = sample
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            sample_dict = {"image": sample[0], "label": sample[1]}
        else:
            # unsupported sample type
            continue

        if label in counts and counts[label] < per_class_count:
            collected.append(sample_dict)
            counts[label] += 1
        if all(count >= per_class_count for count in counts.values()):
            break
    return Dataset.from_list(collected)


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
    def __getitem__(self, idx):
        sample = self.base[idx]
        img_data = sample.get('image', None)
        label = sample.get('label', None)

        # If img_data is a cached list or a Tensor, assume it's already preprocessed
        if isinstance(img_data, list) or isinstance(img_data, torch.Tensor):
            img_tensor = torch.tensor(img_data) if isinstance(img_data, list) else img_data
        else:
            # Raw PIL.Image or numpy array—apply the transform
            img_tensor = self.transform(img_data)

        return img_tensor, label
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



def get_filtered_dataset(dataset, selected_classes=None, streaming=None):
    # Filter the dataset by selected classes and remove negative labels.
    if selected_classes is not None:
        dataset = dataset.filter(lambda x: x['label'] in selected_classes)
    dataset = dataset.filter(lambda x: x['label'] != -1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
        transforms.ToTensor(),
    ])

    if streaming:
        # Wrap the dataset as an IterableDataset so that no __len__ is required.
        class LazyTransformIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform

            def __iter__(self):
                for sample in self.dataset:
                    img = sample["image"]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    sample["image"] = self.transform(img)
                    yield sample

        return LazyTransformIterableDataset(dataset, transform)
    else:
        # For non-streaming case, use WrappedDataset
        return WrappedDataset(dataset, transform)


def print_dataset_labels(dataset, max_samples=10):
    """Print the labels of the first `max_samples` samples in the dataset."""
    print(f"Printing labels for the first {max_samples} samples:")
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        if 'label' in sample:
            print(f"Sample {idx}: Label = {sample['label']}")
        else:
            print(f"Sample {idx}: No 'label' field found in sample.")


# ==================== Personalization function ====================
def personalize_vit(
    opt,
    train_loader,
    val_loaders,
    num_epochs: int = 3,
    base_lr: float = 5e-5,
    layer_decay: float = 0.8,
    freeze_layers: int = 8,
    device: str = None,
    total_steps = None,
):
    """
    Fine-tune the ViT model with layer-wise learning rate decay and selective layer fine-tuning.
    - freeze the first `freeze_layers` encoder layers
    - apply exponential LR decay per layer
    """
    print(f"[Personalize] Fine-tuning ViT with {num_epochs} epochs, base LR: {base_lr}, layer decay: {layer_decay}, freeze layers: {freeze_layers}")
    if device is None:
        device = next(opt.model.parameters()).device
    opt.model.to(device)
    # Freeze early layers
    for name, param in opt.model.named_parameters():
        # match encoder.layer.{i}
        if name.startswith("encoder.layer."):
            layer_id = int(name.split(".")[2])
            if layer_id < freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # keep head and other params trainable
            param.requires_grad = True

     # Compute total_steps if not provided
    if total_steps is None:
        try:
            total_steps = num_epochs * len(train_loader)
        except TypeError:
            raise ValueError("total_steps must be provided when train_loader has no length (e.g. an itertools.chain).")
        
    # Build parameter groups with layer-wise LR
    num_layers = opt.model.config.num_hidden_layers
    param_groups = []
    for name, param in opt.model.named_parameters():
        if not param.requires_grad:
            continue
        # default to head (highest LR)
        lr = base_lr
        if name.startswith("encoder.layer."):
            layer_id = int(name.split(".")[2])
            # deeper layers get higher LR
            lr = base_lr * (layer_decay ** (num_layers - 1 - layer_id))
        param_groups.append({"params": [param], "lr": lr})
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler: linear warmup + decay
    # total_steps = num_epochs * len(train_loader)
    # warmup_steps = max(100, total_steps // 10)  # Ensure minimum warmup steps
    warmup_steps = 3   # TODO: not harcoded

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Print initial learning rate
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    # Initial evaluation using the global evaluate()
    metrics = evaluate(opt, val_loaders, device)
    for split, m in metrics.items():
        print(f"[Personalize] Initial {split} Val Accuracy: {m['accuracy']:.4f}, Val Loss: {m['avg_loss']:.4f}")

    # Modify the training loop:
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Personalize] Epoch {epoch+1}/{num_epochs} - learning rate: {current_lr:.6f}")
        
        # Training
        opt.model.train()
        train_loss = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = opt.predictor(pixel_values=imgs).logits
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            # Add training progress monitoring
            # if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # avg_train_loss = train_loss / len(train_loader)
        avg_train_loss = train_loss
        print(f"[Personalize] Epoch {epoch+1}/{num_epochs} finished - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Per-epoch evaluation using the global evaluate()
        metrics = evaluate(opt, val_loaders, device)
        for split, m in metrics.items():
            print(f"[Personalize] Epoch {epoch+1}/{num_epochs} {split} Val Loss: {m['avg_loss']:.4f}, Val Accuracy: {m['accuracy']:.4f}")
    return opt



def compute_personalized_fisher_invs(model, dataloader=None, num_grads=512, block_size=50, damp=1e-3, loss_fn=nn.CrossEntropyLoss(), device=None, v_norm = True):
    model.model.train() #JUST ADDED

    fisher_invs = [
        EmpiricalBlockFisherInverse(
            num_grads         = num_grads,
            fisher_block_size = block_size,
            num_weights       = 3072 * 768,   # one MLP matrix per layer #TODO: not hardcoded
            damp              = damp,
            device            = device,
        )
        for _ in range(12)
    ]

    # counter = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.model.zero_grad()
        
        layer_grads = [None] * 12
        # Register hooks
        def make_hook(layer_idx):
            def hook(grad):
                layer_grads[layer_idx] = grad.detach().cpu()
            return hook
        
        handles = []
        for i in range(12):
            layer_name = f"encoder.layer.{i}.intermediate.dense.weight"
            layer = dict(model.model.named_parameters())[layer_name]
            # Ensure requires_grad is True before registering the hook
            if not layer.requires_grad:
                # print(f"Layer {layer_name} does not require gradient.  Setting requires_grad=True")
                layer.requires_grad_(True)  # Inplace operation to set requires_grad
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
                if v_norm:
                    v = v / (v.norm()+ 1e-12)
                fisher_invs[i].add_grad(v)
                del v  # free memory
            layer_grads = [None] * 12

        # Remove hooks
        for h in handles:
            h.remove()

        # counter += 1
        # if counter >= num_grads:
        #     break

    torch.cuda.empty_cache()
    return fisher_invs

def pick_misaligned_B(in_features: int, target: int = 96) -> int:
    """
    Return a B near `target` such that gcd(B, in_features) is small and
    in_features % B != 0, so blocks straddle rows.
    """
    import math
    candidates = []
    for delta in range(0, 33, 8):  # search target± up to 32 in steps of 8
        for sgn in (+1, -1):
            B = max(32, target + sgn * delta)
            if (in_features % B) != 0 and math.gcd(B, in_features) <= 8:
                candidates.append(B)
    return candidates[0] if candidates else (target - 8) 

@torch.no_grad()
def compute_personalized_fisher_invs_CAP_new(
    opt,
    dataloader,
    num_grads: int,
    damp: float = 1e-3,
    block_size: int = 50,
    v_norm: bool = True,
):
    """
    CAP uses the same block-diagonal EmpiricalBlockFisherInverse per layer,
    we just return (fisher_invs, layer_params) to avoid repeated lookups later.
    """
    print("in compute_personalized_fisher_invs_CAP_new")
    device = next(opt.model.parameters()).device
    opt.model.train()
    W0 = dict(opt.model.named_parameters())["encoder.layer.0.intermediate.dense.weight"]
    out_features, in_features = W0.shape
    B = pick_misaligned_B(in_features, target=96)  # e.g., 88, 104, 112, ...

    fisher_invs = [
        EmpiricalBlockFisherInverseCap(
            num_grads=num_grads,
            fisher_block_size=B,
            num_weights=W0.numel(),
            damp=damp,                        # consider 1e-2 .. 5e-2 for stability
            device=device,
        )
        for _ in range(opt.model.config.num_hidden_layers)
    ]

    # grab params once for convenience
    num_layers = opt.model.config.num_hidden_layers
    layer_params = [
        dict(opt.model.named_parameters())[f"encoder.layer.{i}.intermediate.dense.weight"]
        for i in range(num_layers)
    ]

    loss_fn = torch.nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt.model.zero_grad(set_to_none=True)

        # register hooks
        scratch = [None] * num_layers
        def make_hook(li):
            def _hook(g):
                scratch[li] = g.detach()
            return _hook
        handles = [p.register_hook(make_hook(i)) for i, p in enumerate(layer_params)]

        # fwd/back
        with torch.set_grad_enabled(True):
            logits = opt.get_logits(pixel_values=inputs)[:, 0, :]
            loss = loss_fn(logits, targets)
            loss.backward()

        # stream grads into the matching layer inverse
        for i, g in enumerate(scratch):
            if g is None:  # if layer was frozen
                continue
            v = g.flatten().to(device)
            if v_norm:
                v = v / (v.norm() + 1e-12)
            fisher_invs[i].add_grad(v)

        for h in handles:
            h.remove()

    torch.cuda.empty_cache()
    return fisher_invs, layer_params


# def compute_personalized_fisher_invs_CAP(
#     model,
#     dataloader,
#     num_grads: Optional[int] = None,
#     block_size_target: int = 64,
#     damp: float = 1e-7,
#     inv_device: str = "cpu",
#     loss_fn: nn.Module = nn.CrossEntropyLoss(),
# ):
#     """
#     CAP-style Fisher inverse collection (per-layer, per-row blocks).

#     For each target linear weight W ∈ R[out, in], we build an EmpiricalBlockFisherInverse
#     with block_size = in so that F_inv has shape (out, in, in).

#     Returns:
#         fisher_invs: List[EmpiricalBlockFisherInverse] aligned with layer order.
#         layer_params: List[(layer_name, Parameter)] in the same order (for reuse downstream).
#     """
#     model.model.train()
#     run_device = next(model.model.parameters()).device

#     # target layers (match your ViT MLP intermediate)
#     layer_params: List[Tuple[str, torch.nn.Parameter]] = [
#         (n, p) for n, p in model.model.named_parameters()
#         if n.endswith(".intermediate.dense.weight")
#     ]

#     # stable order by layer index
#     def _idx(n):
#         try: return int(n.split("encoder.layer.")[1].split(".")[0])
#         except: return 10**9
#     layer_params.sort(key=lambda kv: _idx(kv[0]))

#     fisher_invs: List[EmpiricalBlockFisherInverseCAP] = []
#     for _, p in layer_params:
#         out, in_features = p.shape
#         # choose B: nearest divisor of in_features >= block_size_target
#         divisors = [d for d in range(1, in_features + 1) if in_features % d == 0]
#         B = next((d for d in divisors if d >= block_size_target), divisors[-1])
#         # build inverse on inv_device
#         finv = EmpiricalBlockFisherInverseCAP(
#             num_grads=0,
#             fisher_block_size=B,
#             num_weights=p.numel(),         # out * in
#             damp=damp,
#             device=torch.device(inv_device),
#             dtype=torch.float32,
#         )
#         fisher_invs.append(finv)

#         # ensure grads will be created
#         if not p.requires_grad:
#             p.requires_grad_(True)

#     # stream gradients
#     seen = 0
#     limit = len(dataloader) if num_grads is None else min(num_grads, len(dataloader))
#     for batch in dataloader:
#         if isinstance(batch, (tuple, list)) and len(batch) >= 2:
#             x, y = batch[0].to(run_device), batch[1].to(run_device)
#         else:
#             x, y = batch["image"].to(run_device), batch["label"].to(run_device)

#         model.model.zero_grad(set_to_none=True)
#         logits = model.get_logits(pixel_values=x)
#         if logits.ndim == 3:
#             logits = logits[:, 0, :]
#         loss = loss_fn(logits, y)
#         loss.backward()

#         for (_, p), finv in zip(layer_params, fisher_invs):
#             if p.grad is None: continue
#             g = p.grad.detach().reshape(-1).to(finv.dev)
#             finv.add_grad(g)

#         seen += 1
#         if seen >= limit:
#             break

#     torch.cuda.empty_cache()
#     return fisher_invs, layer_params



def get_weight_delta(model, weights):
    state_a = model.state_dict()


    # (Optional) Check keys & shapes match
    assert state_a.keys() == weights.keys(), "Parameter layouts differ!"

    # ----- 2. Build a “delta” state_dict -----
    delta_state = {}
    for k in state_a.keys():
        delta_state[k] = weights[k] - state_a[k]

    return delta_state
    # Optional here
    torch.save(delta_state, "delta.pt") 

def calculate_score(personalized_model, weights_path, gradient, hessian):
        pretrained_state = torch.load(weights_path, map_location="cpu")
        current_state    = personalized_model.state_dict()
        scores = {}

        with torch.no_grad():                 # no autograd bookkeeping
            for name, w in current_state.items():
                if name not in pretrained_state:
                    continue                  # skip layers that don’t exist in the checkpoint

                w0    = pretrained_state[name]          # pretrained parameter
                delta     = w - w0                          # weight delta

                g     = gradient[name]                  # ∂L/∂w
                H     = hessian[name]                   # diagonal of ∂²L/∂w²

                # element‑wise score
                score_tensor = g * w + g * delta + H * delta * w + H * delta.pow(2)

                scores[name] = score_tensor             # keep full tensor (or .sum() for scalar)

        return scores


def prune_personalize_and_compensate(mask_dir, compensation_hyperparameters = None, personalization_split = 0.5, 
                                     print_birds = True, downdate = False, v_norm = True, use_old_fisher_loader = False, 
                                     CAP_compensation = False, print_prune_bird = False, personalization_hps = None, misc_flags = None):
    log = open(f'log_for_default.txt', 'w')
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # select personalization classes
    selected_classes = [0, 1, 2, 
                        3, 4, 5, 6,                            # fish
                        389, 390, 391, 392, 393, 394, 395, 396, 397,    # Marine            
    ] + list(range(924, 970))                                           # Food  
    
    # TODO: would be extended to otehr categories

    number_of_personalization_classes = 20
    selected_classes = selected_classes[:number_of_personalization_classes]
    print(f"Selected classes: {selected_classes}")

    # Set the alpha value for the split
    alpha = 0.5
    total_samples = 50  # Total number of samples for each class
    imagenet_samples = int(total_samples * alpha)
    sketch_samples = total_samples - imagenet_samples
    # Split per-class counts into train/val halves
    per_class_train_imagenet = imagenet_samples // 2
    per_class_val_imagenet   = imagenet_samples - per_class_train_imagenet
    per_class_train_sketch    = sketch_samples   // 2
    per_class_val_sketch      = sketch_samples   - per_class_train_sketch
    # assert that sketch_samples is less than or equal to 50 (25 for train and 25 for val)
    assert sketch_samples <= 25, f"Sketch samples {sketch_samples} exceed the limit of 50."

    # Personlization HPs
    personalize_vit_hps = personalization_hps if personalization_hps else {
        "num_epochs": 6,
        "base_lr": 5e-5,
        "layer_decay": 0.8,
        "freeze_layers": 3,
    }

   
    # TODO: define experiemnt and name and set the following directories based on that if needed

    # Load ImageNet train split
    imagenet_config = infer_dataset_config("imagenet-1k")
    imagenet_config.dataset_split = "train"
    imagenet_config.is_train_mode = True # TODO: set proper value
    imagenet_config.streaming = False
    imagenet_train = prepare_dataset(imagenet_config)

    # Load ImageNet val split
    imagenet_val_config = infer_dataset_config("imagenet-1k")
    imagenet_val_config.dataset_split = "validation"
    imagenet_val_config.is_train_mode = False # TODO: set proper value
    imagenet_val_config.streaming = False
    imagenet_test = prepare_dataset(imagenet_val_config)

    # Load Sketch-ImageNet train split
    sketch_config = infer_dataset_config("imagenet_sketch")
    sketch_config.dataset_split = "train"
    sketch_config.is_train_mode = False # TODO: set proper value
    sketch_config.streaming = False
    sketch_length = len(prepare_dataset(sketch_config))
    sketch = prepare_dataset(sketch_config)

    # -- cache setup --
    cache_root = "../cache_personalization"
    os.makedirs(cache_root, exist_ok=True)
    im_tr_cache = os.path.join(cache_root, "imagenet_train")
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_tr_cache = os.path.join(cache_root, "sketch_train")
    sk_val_cache = os.path.join(cache_root, "sketch_val")

    print("Preparing ImageNet train split...")
    if os.path.isdir(im_tr_cache):
        imagenet_train_dataset = load_from_disk(im_tr_cache)
    else:
        tmp = get_filtered_dataset(imagenet_train, selected_classes, streaming=imagenet_config.streaming)
        imagenet_train_dataset = limit_samples_per_class(tmp, per_class_train_imagenet, selected_classes)
        imagenet_train_dataset.save_to_disk(im_tr_cache)

    print("Preparing ImageNet val split...")
    if os.path.isdir(im_val_cache):
        imagenet_val_dataset = load_from_disk(im_val_cache)
    else:
        tmp = get_filtered_dataset(imagenet_test, selected_classes, streaming=imagenet_val_config.streaming)
        imagenet_val_dataset = limit_samples_per_class(tmp, per_class_val_imagenet, selected_classes)
        imagenet_val_dataset.save_to_disk(im_val_cache)

    print("Converting Sketch dataset to list for splitting...")
    # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=2000, test_ratio=0.5)
    # print(f'LEN OF THE SKETCH DATASET: {len(sketch)}')

    if os.path.isdir(sk_tr_cache) and os.path.isdir(sk_val_cache):
        print("NO NEED TO SPLITE SKETCH")
    else:
        sketch_train, sketch_val = limited_iterable_split(sketch, total_items=sketch_length, test_ratio=0.5)  # TODO: undo to len(sketch)
        # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=100, test_ratio=0.5)

    print("Preparing Sketch train split...")
    if os.path.isdir(sk_tr_cache):
        sketch_train_dataset = load_from_disk(sk_tr_cache)
    else:
        tmp = get_filtered_dataset(sketch_train, selected_classes, streaming=sketch_config.streaming)
        sketch_train_dataset = limit_samples_per_class(tmp, per_class_train_sketch, selected_classes)
        sketch_train_dataset.save_to_disk(sk_tr_cache)

    print("Preparing Sketch val split...")
    if os.path.isdir(sk_val_cache):
        sketch_val_dataset = load_from_disk(sk_val_cache)
    else:
        tmp = get_filtered_dataset(sketch_val, selected_classes, streaming=sketch_config.streaming)
        sketch_val_dataset = limit_samples_per_class(tmp, per_class_val_sketch, selected_classes)
        sketch_val_dataset.save_to_disk(sk_val_cache)
    # Add validation check here
    # validate_splits()

    # Then proceed with model creation and training
    opt = Model(
        "google/vit-base-patch16-224",
        limit=1000,
        dtype="fp32",
        svd_attn=False,
        use_accelerator=True,
        model_device=None,
        mask_fn="step",
    )

    # --- Personalization ---
    total_steps = None
    combined_train_ds = None
    # Prepare train/val DataLoaders for personalization
    # Define collate function (unchanged)
    # Decide based on streaming flag (here we use imagenet_config.streaming)
    if imagenet_config.streaming:
    # imagenet_config.streaming:
        # Create separate DataLoaders for each dataset:
        train_loader_imagenet = DataLoader(
            imagenet_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        train_loader_sketch = DataLoader(
            sketch_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Create individual validation loaders:
        val_loader_imagenet = DataLoader(
            imagenet_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_loader_sketch = DataLoader(
            sketch_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Keep the validation loaders in a dictionary
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch": val_loader_sketch
        }
        # Combine ImageNet and Sketch datasets for training into a single DataLoader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        train_loader_pers = DataLoader(
            combined_train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Recompute total_steps using the combined loader
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
    else:
        # Non-streaming: apply the same transforms via WrappedDataset, then DataLoader
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])

        # Wrap the already-limited HF Datasets to apply transforms
        train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
        train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)

        # Validation loaders (no shuffle)
        val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
        val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
        val_loader_sketch = DataLoader(
            val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Bundle them into the dict your training loop expects
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
        }

        # Combine train sets into one re-iterable loader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
        train_loader_pers = DataLoader(
            combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

        # And recompute total_steps for the scheduler
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)

    # --- Personalization Training & Evaluation ---

    # Now run personalization.
    # The personalize_vit function remains mostly unchanged except that it now expects a single val_loader in non-streaming mode,
    # and in streaming mode you can later call evaluate() on the model with the dictionary of loaders.
    print("MEMORY:")
    print(torch.cuda.memory_summary())
    personalized_model = personalize_vit(
        opt,
        train_loader_pers,
        # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
        val_loaders,
        num_epochs=personalize_vit_hps['num_epochs'],
        base_lr=personalize_vit_hps['base_lr'],
        layer_decay=personalize_vit_hps['layer_decay'],
        freeze_layers=personalize_vit_hps['freeze_layers'],
        total_steps=total_steps
    )

    print("[Debug] DONE PERSONALIZING")
    
    # Model(
    #     "./personalized_vit",     # ← load from here instead of the HF hub
    #     limit=1000,
    #     dtype="fp32",
    #     svd_attn=False,
    #     use_accelerator=True,
    #     model_device=None,
    #     mask_fn="step",
    # )
    
    opt = personalized_model

    # Optionally save the personalized model weights
    # personalized_model.model.save_pretrained("./personalized_vit")

    # Evaluate after personalization

    # print("EVALUATION AFTER PERSONALIZATION")
    # with torch.no_grad():
    #     if isinstance(val_loaders, dict):
    #         eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
    #         for key, metrics in eval_results.items():
    #             print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
    #             log.write(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
    #             log.flush()
    #     else:
    #         eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
    #         print(f"[Personalize] Combined Val Accuracy: {eval_results['combined']['accuracy']:.4f}, Avg Loss: {eval_results['combined']['avg_loss']:.4f}")



    # # Evaluate the personalized model on whole dataset (birds/birdsless)
    # # set the config for evaluation
    eval_sample_size = 1e5
    datasets = ['imagenet-1k-birds']
    collection_sample_size = None # TODO: set proper value
    # 
    # sh()

    print("\n\n ============ UNLEARNING ==========")
    # Apply the UNLEARNING (apply the previous trained mask to the model - the mask is result of running the pruning script - prune_vit.sh) + COMPENSATION

    # Load the mask
    mask_path = mask_dir  # Path to the mask file
    # print current directory
    print(f'Current directory: {os.getcwd()}')
    mask = torch.load(mask_path)
    print(f' --- debug - mask shape: {mask.shape} --- ')

    # Evaluate the model when applying the mask without compensation
    res = {}
    # Eval1: On personalization data
    if True:
        with torch.no_grad():
            print(f'********* starting evalution after unlearning (on birds) without compensation *********')
            tmp_opt = copy.deepcopy(opt)
            device = next(opt.model.parameters()).device
            tmp_opt.model.to(device)
            prune_only_all_layers(tmp_opt.model, prune_mask=mask, device=device)
            eval_results = evaluate(tmp_opt, val_loaders, device=next(tmp_opt.model.parameters()).device)
            for key, metrics in eval_results.items():
                print(f"[Personalize] prune only {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                res["prune only " + key] = metrics['accuracy']
        # # Eval2: On whole dataset (birds/birdless)
            if print_prune_bird:
                print(f'********* starting evalution after unlearning (on birds/birdsless) without compensation *********')
                with torch.no_grad():
                    eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                            dataset_tokens_to_skip=collection_sample_size)
                # exclude 'token_count' from eval_out to print
                print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
                res["prune only imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base']
                print(f'----- debug - model performance after unlearning (on birds/birdsless) without compensation -> eval_out: {print_eval_out} ----- \n\n') 
            del tmp_opt
    

    compensation_hps = compensation_hyperparameters if compensation_hyperparameters else {
        'block_size': 50,
        'damp': 1e-3,
        'learning_rate': 4,
    }


    print("\n\n ============ COMPENSATION ==========")
    print(compensation_hps)
    # compute the fisher inverses which is needed for compensation
    # Build a combined dataset for Fisher inverse computation
    if use_old_fisher_loader:
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        fisher_loader = DataLoader(
            combined_train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        fisher_loader = DataLoader(
            combined_train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn
        )

    if misc_flags == "sketch_only":
            print("in sketch only")
            fisher_loader = DataLoader(
                train_ds_sk, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
            )
    num_grads = len(fisher_loader)
    print("\n NUM GRANDS: ",num_grads)

    if CAP_compensation:
        print('In CAP personalized fisher invs')
        fisher_invs, layer_params = compute_personalized_fisher_invs_CAP_new(
            opt,
            fisher_loader,
            num_grads=num_grads,
            damp=compensation_hps['damp'],
        )
    else:
        print('In normal compute fisher invs')
        fisher_invs = compute_personalized_fisher_invs(
            opt,
            dataloader=fisher_loader,
            num_grads=num_grads,
            block_size=compensation_hps['block_size'],
            damp=compensation_hps['damp'],
            device=opt.model.device,
            v_norm=v_norm
        )

    # Apply the mask to the model and compensate
    if CAP_compensation:
        print('In cap prune and compensate')
        if isinstance(mask, np.ndarray):
            prune_mask = torch.from_numpy(mask)
        elif not isinstance(mask, torch.Tensor):
            prune_mask = torch.tensor(mask)

        prune_mask = prune_mask.bool()
        print(prune_mask)


        prune_and_compensate_all_layers_CAP_new(
            model = opt.model,
            prune_mask = prune_mask,
            fisher_invs = fisher_invs,
            layer_params=layer_params,
            device=opt.model.device,
        )
        # with torch.no_grad():
        #     # Evaluate the model after unlearning
        #     eval_results = evaluate(opt, val_loaders, device=next(opt.model.parameters()).device)
        #     for key, metrics in eval_results.items():
        #         print(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
        #         res["prune + compensation " + key] = metrics['accuracy']

        #     # Eval2: on whole dataset (birds/birdless)
        #     if print_birds:
        #         print(f'********* starting evalution after unlearning (on birs) and compensation (on personalization data) *********')
        #         with torch.no_grad():
        #             eval_out = evaluate_all(opt, eval_sample_size, datasets,
        #                                     dataset_tokens_to_skip=collection_sample_size)
        #         # exclude 'token_count' from eval_out to print
        #         print_eval_out = {k: v  for k, v in eval_out.items() if k != 'misc'}
        #         print(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
        #         res["prune + compensation imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base']
        #         print("res is: ")
        #         print(res)
    else:
        print('In normal prune and compensate')
        prune_and_compensate_all_layers(
            opt.model,
            prune_mask = mask,
            fisher_invs = fisher_invs,
            block_size=compensation_hps['block_size'], 
            damp=compensation_hps['damp'],
            learning_rate=compensation_hps['learning_rate'],
            device=opt.model.device,
            downdate = downdate
        )
    with torch.no_grad():
        # Evaluate the model after unlearning
        eval_results = evaluate(opt, val_loaders, device=next(opt.model.parameters()).device)
        for key, metrics in eval_results.items():
            print(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
            res["prune + compensation " + key] = metrics['accuracy']

        # Eval2: on whole dataset (birds/birdless)
        if print_birds:
            print(f'********* starting evalution after unlearning (on birs) and compensation (on personalization data) *********')
            with torch.no_grad():
                eval_out = evaluate_all(opt, eval_sample_size, datasets,
                                        dataset_tokens_to_skip=collection_sample_size)
            # exclude 'token_count' from eval_out to print
            print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
            print(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
            res["prune + compensation imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base']

    log.close()
    del opt
    print("res is: ")
    print(res)
    return (personalization_split * eval_results['imagenet']['accuracy'] + personalization_split * eval_results['sketch']['accuracy'],  0.01*print_eval_out['accuracy']['imagenet-1k-birds']['base'])



def compensation_hp_tuning(opt, combined_train_ds, fisher_loader, mask, val_loaders, lr = 4, block_size = 50, damp = 1e-3, personalization_split = 0.5):
            compensation_hps = {
                'damp': damp,
                'block_size': block_size,
                'learning_rate': lr,
            }
            
            # log.write(f"Hyperparameters: {str(compensation_hps)}")
            # log.flush()

            print("\n\n ============ COMPENSATION ==========")
            print(compensation_hps)
            tmp_opt = copy.deepcopy(opt)
            num_grads = len(combined_train_ds)
            print("\n NUM GRANDS: ",num_grads)
            fisher_invs = compute_personalized_fisher_invs(
                tmp_opt,
                dataloader=fisher_loader,
                num_grads=num_grads,
                block_size=compensation_hps['block_size'],
                damp=compensation_hps['damp'],
                device=tmp_opt.model.device
            )

            # Apply the mask to the model and compensate
            prune_and_compensate_all_layers(
                tmp_opt.model,
                prune_mask = mask,
                fisher_invs = fisher_invs,
                block_size=compensation_hps['block_size'], 
                damp=compensation_hps['damp'],
                learning_rate=compensation_hps['learning_rate'],
                device=opt.model.device,
            )

            # Evaluate the model after unlearning
            eval_results = evaluate(tmp_opt, val_loaders, device=next(opt.model.parameters()).device)
            for key, metrics in eval_results.items():
                print(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                # log.write(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
                # log.flush()

            del tmp_opt
            return personalization_split * eval_results['imagenet']['accuracy'] + personalization_split * eval_results['sketch']['accuracy']
            # Eval2: on whole dataset (birds/birdless)
            print(f'********* starting evalution after unlearning (on birs) and compensation (on personalization data) *********')
            with torch.no_grad():
                eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                        dataset_tokens_to_skip=collection_sample_size)
            # exclude 'token_count' from eval_out to print
            print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
            print(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
            log.write(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
            log.flush()



def make_objective(opt, combined_train_ds, fisher_loader, mask, val_loaders, personalization_split = 0.5):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 5.0)
        # damp = trial.suggest_float('damp', 1e-5, 1e-1, log=True)
        # block_size = trial.suggest_int('block_size', 8, 64, log=True)
        return compensation_hp_tuning(
            opt=opt,
            combined_train_ds=combined_train_ds,
            fisher_loader=fisher_loader,
            mask = mask,
            val_loaders=val_loaders,
            lr=lr,
            damp = 1e-3,
            block_size= 50,
            personalization_split = personalization_split

        )
    return objective

def prune_personalize_and_compensate_hp_tuning(mask_dir):
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print("in prune_personalize_and_compensate_hp_tuning")

    # select personalization classes
    selected_classes = [0, 1, 2, 
                        3, 4, 5, 6,                            # fish
                        389, 390, 391, 392, 393, 394, 395, 396, 397,    # Marine            
    ] + list(range(924, 970))                                           # Food  
    
    # TODO: would be extended to otehr categories

    number_of_personalization_classes = 20
    selected_classes = selected_classes[:number_of_personalization_classes]
    print(f"Selected classes: {selected_classes}")

    # Set the alpha value for the split
    alpha = 0.5
    total_samples = 50  # Total number of samples for each class
    imagenet_samples = int(total_samples * alpha)
    sketch_samples = total_samples - imagenet_samples
    # Split per-class counts into train/val halves
    per_class_train_imagenet = imagenet_samples // 2
    per_class_val_imagenet   = imagenet_samples - per_class_train_imagenet
    per_class_train_sketch    = sketch_samples   // 2
    per_class_val_sketch      = sketch_samples   - per_class_train_sketch
    # assert that sketch_samples is less than or equal to 50 (25 for train and 25 for val)
    assert sketch_samples <= 25, f"Sketch samples {sketch_samples} exceed the limit of 50."

    # Personlization HPs
    personalize_vit_hps = {
        "num_epochs": 6,
        "base_lr": 5e-5,
        "layer_decay": 0.8,
        "freeze_layers": 3,
    }

    # TODO: define experiemnt and name and set the following directories based on that if needed

    # Load ImageNet train split
    imagenet_config = infer_dataset_config("imagenet-1k")
    imagenet_config.dataset_split = "train"
    imagenet_config.is_train_mode = True # TODO: set proper value
    imagenet_config.streaming = False
    imagenet_train = prepare_dataset(imagenet_config)

    # Load ImageNet val split
    imagenet_val_config = infer_dataset_config("imagenet-1k")
    imagenet_val_config.dataset_split = "validation"
    imagenet_val_config.is_train_mode = False # TODO: set proper value
    imagenet_val_config.streaming = False
    imagenet_test = prepare_dataset(imagenet_val_config)

    # Load Sketch-ImageNet train split
    sketch_config = infer_dataset_config("imagenet_sketch")
    sketch_config.dataset_split = "train"
    sketch_config.is_train_mode = False # TODO: set proper value
    sketch_config.streaming = False
    sketch_length = len(prepare_dataset(sketch_config))
    sketch = prepare_dataset(sketch_config)

    # -- cache setup --
    cache_root = "../cache_personalization"
    os.makedirs(cache_root, exist_ok=True)
    im_tr_cache = os.path.join(cache_root, "imagenet_train")
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_tr_cache = os.path.join(cache_root, "sketch_train")
    sk_val_cache = os.path.join(cache_root, "sketch_val")

    print("Preparing ImageNet train split...")
    if os.path.isdir(im_tr_cache):
        imagenet_train_dataset = load_from_disk(im_tr_cache)
    else:
        tmp = get_filtered_dataset(imagenet_train, selected_classes, streaming=imagenet_config.streaming)
        imagenet_train_dataset = limit_samples_per_class(tmp, per_class_train_imagenet, selected_classes)
        imagenet_train_dataset.save_to_disk(im_tr_cache)

    print("Preparing ImageNet val split...")
    if os.path.isdir(im_val_cache):
        imagenet_val_dataset = load_from_disk(im_val_cache)
    else:
        tmp = get_filtered_dataset(imagenet_test, selected_classes, streaming=imagenet_val_config.streaming)
        imagenet_val_dataset = limit_samples_per_class(tmp, per_class_val_imagenet, selected_classes)
        imagenet_val_dataset.save_to_disk(im_val_cache)

    print("Converting Sketch dataset to list for splitting...")
    # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=2000, test_ratio=0.5)
    # print(f'LEN OF THE SKETCH DATASET: {len(sketch)}')

    if os.path.isdir(sk_tr_cache) and os.path.isdir(sk_val_cache):
        print("NO NEED TO SPLITE SKETCH")
    else:
        sketch_train, sketch_val = limited_iterable_split(sketch, total_items=sketch_length, test_ratio=0.5)  # TODO: undo to len(sketch)
        # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=100, test_ratio=0.5)

    print("Preparing Sketch train split...")
    if os.path.isdir(sk_tr_cache):
        sketch_train_dataset = load_from_disk(sk_tr_cache)
    else:
        tmp = get_filtered_dataset(sketch_train, selected_classes, streaming=sketch_config.streaming)
        sketch_train_dataset = limit_samples_per_class(tmp, per_class_train_sketch, selected_classes)
        sketch_train_dataset.save_to_disk(sk_tr_cache)

    print("Preparing Sketch val split...")
    if os.path.isdir(sk_val_cache):
        sketch_val_dataset = load_from_disk(sk_val_cache)
    else:
        tmp = get_filtered_dataset(sketch_val, selected_classes, streaming=sketch_config.streaming)
        sketch_val_dataset = limit_samples_per_class(tmp, per_class_val_sketch, selected_classes)
        sketch_val_dataset.save_to_disk(sk_val_cache)


    opt = Model(
        "google/vit-base-patch16-224",
        limit=1000,
        dtype="fp32",
        svd_attn=False,
        use_accelerator=True,
        model_device=None,
        mask_fn="step",
    )

    # --- Personalization ---
    total_steps = None

    # Prepare train/val DataLoaders for personalization
    # Define collate function (unchanged)
    # Decide based on streaming flag (here we use imagenet_config.streaming)
    if imagenet_config.streaming:
    # imagenet_config.streaming:
        # Create separate DataLoaders for each dataset:
        train_loader_imagenet = DataLoader(
            imagenet_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        train_loader_sketch = DataLoader(
            sketch_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Create individual validation loaders:
        val_loader_imagenet = DataLoader(
            imagenet_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_loader_sketch = DataLoader(
            sketch_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Keep the validation loaders in a dictionary
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch": val_loader_sketch
        }
        # Combine ImageNet and Sketch datasets for training into a single DataLoader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        train_loader_pers = DataLoader(
            combined_train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Recompute total_steps using the combined loader
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
    else:
        # Non-streaming: apply the same transforms via WrappedDataset, then DataLoader
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])

        # Wrap the already-limited HF Datasets to apply transforms
        train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
        train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)

        # Validation loaders (no shuffle)
        val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
        val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
        val_loader_sketch = DataLoader(
            val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Bundle them into the dict your training loop expects
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
        }

        # Combine train sets into one re-iterable loader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
        train_loader_pers = DataLoader(
            combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

        # And recompute total_steps for the scheduler
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)

    # --- Personalization Training & Evaluation ---

    # Now run personalization.
    # The personalize_vit function remains mostly unchanged except that it now expects a single val_loader in non-streaming mode,
    # and in streaming mode you can later call evaluate() on the model with the dictionary of loaders.
    personalized_model = personalize_vit(
        opt,
        train_loader_pers,
        # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
        val_loaders,
        num_epochs=personalize_vit_hps['num_epochs'],
        base_lr=personalize_vit_hps['base_lr'],
        layer_decay=personalize_vit_hps['layer_decay'],
        freeze_layers=personalize_vit_hps['freeze_layers'],
        total_steps=total_steps
    )
    opt = personalized_model

    # Optionally save the personalized model weights
    # personalized_model.model.save_pretrained("./personalized_vit")

    # Evaluate after personalization
    print("EVALUATION AFTER PERSONALIZATION")
    if isinstance(val_loaders, dict):
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        for key, metrics in eval_results.items():
             print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
    else:
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        print(f"[Personalize] Combined Val Accuracy: {eval_results['combined']['accuracy']:.4f}, Avg Loss: {eval_results['combined']['avg_loss']:.4f}")


    # # Evaluate the personalized model on whole dataset (birds/birdsless)
    # # set the config for evaluation
    eval_sample_size = 1e5
    datasets = ['imagenet-1k-birds']
    collection_sample_size = None # TODO: set proper value
    # with torch.no_grad():
    #     print(f'********* starting evalution after personalization on the bird/birdless dataset (original dataset) *********')
    #     eval_out = evaluate_all(opt, eval_sample_size, datasets,
    #                             dataset_tokens_to_skip=collection_sample_size)
    # # exclude 'token_count' from eval_out to print
    # print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    # print(f'----- debug - model performance after personalization on the bird/birdless dataset (original dataset) -> eval_out: {print_eval_out} ----- \n\n')
    # log.write(f'----- debug - model performance after personalization on the bird/birdless dataset (original dataset) -> eval_out: {print_eval_out} ----- \n\n')
    # log.flush()


    print("\n\n ============ UNLEARNING ==========")
    # Apply the UNLEARNING (apply the previous trained mask to the model - the mask is result of running the pruning script - prune_vit.sh) + COMPENSATION

    # Load the mask
    mask_path = mask_dir  # Path to the mask file
    # print current directory
    print(f'Current directory: {os.getcwd()}')
    mask = torch.load(mask_path)
    print(f' --- debug - mask shape: {mask.shape} --- ')

    # Evaluate the model when applying the mask without compensation
    # Eval1: On personalization data
    print(f'********* starting evalution after unlearning (on birds) without compensation *********')
    tmp_opt = copy.deepcopy(opt)
    device = next(opt.model.parameters()).device
    tmp_opt.model.to(device)
    prune_only_all_layers(tmp_opt.model, prune_mask=mask, device=device)
    eval_results = evaluate(tmp_opt, val_loaders, device=next(tmp_opt.model.parameters()).device)
    for key, metrics in eval_results.items():
        print(f"[Personalize without compensation] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")

    # # Eval2: On whole dataset (birds/birdless)
    print(f'********* starting evalution after unlearning (on birds/birdsless) without compensation *********')
    with torch.no_grad():
        eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                dataset_tokens_to_skip=collection_sample_size)
    # exclude 'token_count' from eval_out to print
    print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    print(f'----- debug - model performance after unlearning (on birds/birdsless) without compensation -> eval_out: {print_eval_out} ----- \n\n')
    del tmp_opt
    
        # Compensation HPs
    compensation_hps_options = {
        'block_size': [32, 64],
        'damp': [1e-4, 1e-3, 1e-2,],
        'learning_rate': [0.5, 1, 3, 8]
        # 
    }
    if False:
        for criteria in ['learning_rate', 'block_size', 'damp']:
            for val in compensation_hps_options[criteria]:
                compensation_hps = {
                    'block_size': 50,
                    'damp': 1e-3,
                    'learning_rate': 4,
                }
                
                compensation_hps[criteria] = val
                log.write(f"Hyperparameters: {str(compensation_hps)}")
                log.flush()

                print("\n\n ============ COMPENSATION ==========")
                print(compensation_hps)
                tmp_opt = copy.deepcopy(opt)
                # compute the fisher inverses which is needed for compensation
                # Build a combined dataset for Fisher inverse computation
                combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
                fisher_loader = DataLoader(
                    combined_train_ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=collate_fn
                )
                num_grads = len(combined_train_ds)
                print("\n NUM GRANDS: ",num_grads)
                fisher_invs = compute_personalized_fisher_invs(
                    tmp_opt,
                    dataloader=fisher_loader,
                    num_grads=num_grads,
                    block_size=compensation_hps['block_size'],
                    damp=compensation_hps['damp'],
                    device=tmp_opt.model.device
                )

                # Apply the mask to the model and compensate
                prune_and_compensate_all_layers(
                    tmp_opt.model,
                    prune_mask = mask,
                    fisher_invs = fisher_invs,
                    block_size=compensation_hps['block_size'], 
                    damp=compensation_hps['damp'],
                    learning_rate=compensation_hps['learning_rate'],
                    device=opt.model.device,
                )

                # Evaluate the model after unlearning
                eval_results = evaluate(tmp_opt, val_loaders, device=next(opt.model.parameters()).device)
                for key, metrics in eval_results.items():
                    print(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")


                # Eval2: on whole dataset (birds/birdless)
                print(f'********* starting evalution after unlearning (on birs) and compensation (on personalization data) *********')
                with torch.no_grad():
                    eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
                                            dataset_tokens_to_skip=collection_sample_size)
                # exclude 'token_count' from eval_out to print
                print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
                print(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
                del tmp_opt
    else:
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        fisher_loader = DataLoader(
            combined_train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        objective = make_objective(opt, combined_train_ds, fisher_loader, mask, val_loaders)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            storage="sqlite:///optuna_lr.db", load_if_exists=True   # optional persistence
        )
        study.optimize(objective, n_trials=30, timeout=None, gc_after_trial=True)

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
    
    return
def personalize_after_prune(model):
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # select personalization classes
    selected_classes = [0, 1, 2, 
                        3, 4, 5, 6,                            # fish
                        389, 390, 391, 392, 393, 394, 395, 396, 397,    # Marine            
    ] + list(range(924, 970))                                           # Food  
    
    # TODO: would be extended to otehr categories

    number_of_personalization_classes = 20
    selected_classes = selected_classes[:number_of_personalization_classes]
    print(f"Selected classes: {selected_classes}")

    # Set the alpha value for the split
    alpha = 0.5
    total_samples = 50  # Total number of samples for each class
    imagenet_samples = int(total_samples * alpha)
    sketch_samples = total_samples - imagenet_samples
    # Split per-class counts into train/val halves
    per_class_train_imagenet = imagenet_samples // 2
    per_class_val_imagenet   = imagenet_samples - per_class_train_imagenet
    per_class_train_sketch    = sketch_samples   // 2
    per_class_val_sketch      = sketch_samples   - per_class_train_sketch
    # assert that sketch_samples is less than or equal to 50 (25 for train and 25 for val)
    assert sketch_samples <= 25, f"Sketch samples {sketch_samples} exceed the limit of 50."

    # Personlization HPs
    personalize_vit_hps = {
        "num_epochs": 6,
        "base_lr": 5e-5,
        "layer_decay": 0.8,
        "freeze_layers": 3,
    }

    # TODO: define experiemnt and name and set the following directories based on that if needed

    # Load ImageNet train split
    imagenet_config = infer_dataset_config("imagenet-1k")
    imagenet_config.dataset_split = "train"
    imagenet_config.is_train_mode = True # TODO: set proper value
    imagenet_config.streaming = False
    imagenet_train = prepare_dataset(imagenet_config)

    # Load ImageNet val split
    imagenet_val_config = infer_dataset_config("imagenet-1k")
    imagenet_val_config.dataset_split = "validation"
    imagenet_val_config.is_train_mode = False # TODO: set proper value
    imagenet_val_config.streaming = False
    imagenet_test = prepare_dataset(imagenet_val_config)

    # Load Sketch-ImageNet train split
    sketch_config = infer_dataset_config("imagenet_sketch")
    sketch_config.dataset_split = "train"
    sketch_config.is_train_mode = False # TODO: set proper value
    sketch_config.streaming = False
    sketch_length = len(prepare_dataset(sketch_config))
    sketch = prepare_dataset(sketch_config)

    # -- cache setup --
    cache_root = "./cache_personalization"
    os.makedirs(cache_root, exist_ok=True)
    im_tr_cache = os.path.join(cache_root, "imagenet_train")
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_tr_cache = os.path.join(cache_root, "sketch_train")
    sk_val_cache = os.path.join(cache_root, "sketch_val")

    print("Preparing ImageNet train split...")
    if os.path.isdir(im_tr_cache):
        imagenet_train_dataset = load_from_disk(im_tr_cache)
    else:
        tmp = get_filtered_dataset(imagenet_train, selected_classes, streaming=imagenet_config.streaming)
        imagenet_train_dataset = limit_samples_per_class(tmp, per_class_train_imagenet, selected_classes)
        imagenet_train_dataset.save_to_disk(im_tr_cache)

    print("Preparing ImageNet val split...")
    if os.path.isdir(im_val_cache):
        imagenet_val_dataset = load_from_disk(im_val_cache)
    else:
        tmp = get_filtered_dataset(imagenet_test, selected_classes, streaming=imagenet_val_config.streaming)
        imagenet_val_dataset = limit_samples_per_class(tmp, per_class_val_imagenet, selected_classes)
        imagenet_val_dataset.save_to_disk(im_val_cache)

    print("Converting Sketch dataset to list for splitting...")
    # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=2000, test_ratio=0.5)
    # print(f'LEN OF THE SKETCH DATASET: {len(sketch)}')

    if os.path.isdir(sk_tr_cache) and os.path.isdir(sk_val_cache):
        print("NO NEED TO SPLITE SKETCH")
    else:
        sketch_train, sketch_val = limited_iterable_split(sketch, total_items=sketch_length, test_ratio=0.5)  # TODO: undo to len(sketch)
        # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=100, test_ratio=0.5)

    print("Preparing Sketch train split...")
    if os.path.isdir(sk_tr_cache):
        sketch_train_dataset = load_from_disk(sk_tr_cache)
    else:
        tmp = get_filtered_dataset(sketch_train, selected_classes, streaming=sketch_config.streaming)
        sketch_train_dataset = limit_samples_per_class(tmp, per_class_train_sketch, selected_classes)
        sketch_train_dataset.save_to_disk(sk_tr_cache)

    print("Preparing Sketch val split...")
    if os.path.isdir(sk_val_cache):
        sketch_val_dataset = load_from_disk(sk_val_cache)
    else:
        tmp = get_filtered_dataset(sketch_val, selected_classes, streaming=sketch_config.streaming)
        sketch_val_dataset = limit_samples_per_class(tmp, per_class_val_sketch, selected_classes)
        sketch_val_dataset.save_to_disk(sk_val_cache)

    # Add validation check here
    # validate_splits()

    # Then proceed with model creation and training
    opt = model

    # --- Personalization ---
    total_steps = None

    # Prepare train/val DataLoaders for personalization
    # Define collate function (unchanged)
    # Decide based on streaming flag (here we use imagenet_config.streaming)
    if imagenet_config.streaming:
        # Create separate DataLoaders for each dataset:
        train_loader_imagenet = DataLoader(
            imagenet_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        train_loader_sketch = DataLoader(
            sketch_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Create individual validation loaders:
        val_loader_imagenet = DataLoader(
            imagenet_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_loader_sketch = DataLoader(
            sketch_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Keep the validation loaders in a dictionary
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch": val_loader_sketch
        }
        # Combine ImageNet and Sketch datasets for training into a single DataLoader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        train_loader_pers = DataLoader(
            combined_train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Recompute total_steps using the combined loader
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
    else:
        # Non-streaming: apply the same transforms via WrappedDataset, then DataLoader
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])

        # Wrap the already-limited HF Datasets to apply transforms
        train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
        train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)

        # Validation loaders (no shuffle)
        val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
        val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
        val_loader_sketch = DataLoader(
            val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Bundle them into the dict your training loop expects
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
        }
        from torch.utils.data import ConcatDataset
        # Combine train sets into one re-iterable loader
        combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
        train_loader_pers = DataLoader(
            combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

        # And recompute total_steps for the scheduler
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)

    # --- Personalization Training & Evaluation ---


    # Now run personalization.
    # The personalize_vit function remains mostly unchanged except that it now expects a single val_loader in non-streaming mode,
    # and in streaming mode you can later call evaluate() on the model with the dictionary of loaders.
    personalized_model = personalize_vit(
        opt,
        train_loader_pers,
        # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
        val_loaders,
        num_epochs=personalize_vit_hps['num_epochs'],
        base_lr=personalize_vit_hps['base_lr'],
        layer_decay=personalize_vit_hps['layer_decay'],
        freeze_layers=personalize_vit_hps['freeze_layers'],
        total_steps=total_steps
    )
    opt = personalized_model

    # Optionally save the personalized model weights
    # personalized_model.model.save_pretrained("./personalized_vit")

    # Evaluate after personalization
    print("EVALUATION AFTER PERSONALIZATION")
    if isinstance(val_loaders, dict):
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        for key, metrics in eval_results.items():
             print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
    else:
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        print(f"[Personalize] Combined Val Accuracy: {eval_results['combined']['accuracy']:.4f}, Avg Loss: {eval_results['combined']['avg_loss']:.4f}")


    # # Evaluate the personalized model on whole dataset (birds/birdsless)
    # # set the config for evaluation
    eval_sample_size = 1e5
    datasets = ['imagenet-1k-birds']
    collection_sample_size = None # TODO: set proper value
    with torch.no_grad():
        print(f'********* starting evalution after personalization on the bird/birdless dataset (original dataset) *********')
        eval_out = evaluate_all(opt, eval_sample_size, datasets,
                                dataset_tokens_to_skip=collection_sample_size)
    # exclude 'token_count' from eval_out to print
    print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    print(f'----- debug - model performance after personalization on the bird/birdless dataset (original dataset) -> eval_out: {print_eval_out} ----- \n\n')
    
    return opt

# def personalize_then_negrad(c, cripple_loader, retain_loader):
#     # Set the random seed for reproducibility
#     random.seed(42)
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)

#     # select personalization classes
#     selected_classes = [0, 1, 2, 
#                         3, 4, 5, 6,                            # fish
#                         389, 390, 391, 392, 393, 394, 395, 396, 397,    # Marine            
#     ] + list(range(924, 970))                                           # Food  
    
#     # TODO: would be extended to otehr categories
#     number_of_personalization_classes = 20
#     selected_classes = selected_classes[:number_of_personalization_classes]
#     print(f"Selected classes: {selected_classes}")

#     # Set the alpha value for the split
#     alpha = 0.5
#     total_samples = 50  # Total number of samples for each class
#     imagenet_samples = int(total_samples * alpha)
#     sketch_samples = total_samples - imagenet_samples
#     # Split per-class counts into train/val halves
#     per_class_train_imagenet = imagenet_samples // 2
#     per_class_val_imagenet   = imagenet_samples - per_class_train_imagenet
#     per_class_train_sketch    = sketch_samples   // 2
#     per_class_val_sketch      = sketch_samples   - per_class_train_sketch
#     # assert that sketch_samples is less than or equal to 50 (25 for train and 25 for val)
#     assert sketch_samples <= 25, f"Sketch samples {sketch_samples} exceed the limit of 50."

#     # Personlization HPs
#     personalize_vit_hps = {
#         "num_epochs": 6,
#         "base_lr": 5e-5,
#         "layer_decay": 0.8,
#         "freeze_layers": 3,
#     }

#     # TODO: define experiemnt and name and set the following directories based on that if needed

#     # Load ImageNet train split
#     imagenet_config = infer_dataset_config("imagenet-1k")
#     imagenet_config.dataset_split = "train"
#     imagenet_config.is_train_mode = True # TODO: set proper value
#     imagenet_config.streaming = False
#     imagenet_train = prepare_dataset(imagenet_config)

#     # Load ImageNet val split
#     imagenet_val_config = infer_dataset_config("imagenet-1k")
#     imagenet_val_config.dataset_split = "validation"
#     imagenet_val_config.is_train_mode = False # TODO: set proper value
#     imagenet_val_config.streaming = False
#     imagenet_test = prepare_dataset(imagenet_val_config)

#     # Load Sketch-ImageNet train split
#     sketch_config = infer_dataset_config("imagenet_sketch")
#     sketch_config.dataset_split = "train"
#     sketch_config.is_train_mode = False # TODO: set proper value
#     sketch_config.streaming = False
#     sketch_length = len(prepare_dataset(sketch_config))
#     sketch = prepare_dataset(sketch_config)

#     # -- cache setup --
#     cache_root = "./cache_personalization"
#     os.makedirs(cache_root, exist_ok=True)
#     im_tr_cache = os.path.join(cache_root, "imagenet_train")
#     im_val_cache = os.path.join(cache_root, "imagenet_val")
#     sk_tr_cache = os.path.join(cache_root, "sketch_train")
#     sk_val_cache = os.path.join(cache_root, "sketch_val")

#     print("Preparing ImageNet train split...")
#     if os.path.isdir(im_tr_cache):
#         imagenet_train_dataset = load_from_disk(im_tr_cache)
#     else:
#         tmp = get_filtered_dataset(imagenet_train, selected_classes, streaming=imagenet_config.streaming)
#         imagenet_train_dataset = limit_samples_per_class(tmp, per_class_train_imagenet, selected_classes)
#         imagenet_train_dataset.save_to_disk(im_tr_cache)

#     print("Preparing ImageNet val split...")
#     if os.path.isdir(im_val_cache):
#         imagenet_val_dataset = load_from_disk(im_val_cache)
#     else:
#         tmp = get_filtered_dataset(imagenet_test, selected_classes, streaming=imagenet_val_config.streaming)
#         imagenet_val_dataset = limit_samples_per_class(tmp, per_class_val_imagenet, selected_classes)
#         imagenet_val_dataset.save_to_disk(im_val_cache)

#     print("Converting Sketch dataset to list for splitting...")
#     # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=2000, test_ratio=0.5)
#     # print(f'LEN OF THE SKETCH DATASET: {len(sketch)}')

#     if os.path.isdir(sk_tr_cache) and os.path.isdir(sk_val_cache):
#         print("NO NEED TO SPLITE SKETCH")
#     else:
#         sketch_train, sketch_val = limited_iterable_split(sketch, total_items=sketch_length, test_ratio=0.5)  # TODO: undo to len(sketch)
#         # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=100, test_ratio=0.5)

#     print("Preparing Sketch train split...")
#     if os.path.isdir(sk_tr_cache):
#         sketch_train_dataset = load_from_disk(sk_tr_cache)
#     else:
#         tmp = get_filtered_dataset(sketch_train, selected_classes, streaming=sketch_config.streaming)
#         sketch_train_dataset = limit_samples_per_class(tmp, per_class_train_sketch, selected_classes)
#         sketch_train_dataset.save_to_disk(sk_tr_cache)

#     print("Preparing Sketch val split...")
#     if os.path.isdir(sk_val_cache):
#         sketch_val_dataset = load_from_disk(sk_val_cache)
#     else:
#         tmp = get_filtered_dataset(sketch_val, selected_classes, streaming=sketch_config.streaming)
#         sketch_val_dataset = limit_samples_per_class(tmp, per_class_val_sketch, selected_classes)
#         sketch_val_dataset.save_to_disk(sk_val_cache)

#     # Add validation check here
#     # validate_splits()

#     opt = Model(
#         "google/vit-base-patch16-224",
#         limit=1000,
#         dtype="fp32",
#         svd_attn=False,
#         use_accelerator=True,
#         model_device=None,
#         mask_fn="step",
#     )

#     # --- Personalization ---
#     total_steps = None

#     # Prepare train/val DataLoaders for personalization
#     # Define collate function (unchanged)
#     # Decide based on streaming flag (here we use imagenet_config.streaming)
#     if imagenet_config.streaming:
#         # Create separate DataLoaders for each dataset:
#         train_loader_imagenet = DataLoader(
#             imagenet_train_dataset,
#             batch_size=32,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )
#         train_loader_sketch = DataLoader(
#             sketch_train_dataset,
#             batch_size=32,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )
#         # Create individual validation loaders:
#         val_loader_imagenet = DataLoader(
#             imagenet_val_dataset,
#             batch_size=32,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )
#         val_loader_sketch = DataLoader(
#             sketch_val_dataset,
#             batch_size=32,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )
#         # Keep the validation loaders in a dictionary
#         val_loaders = {
#             "imagenet": val_loader_imagenet,
#             "sketch": val_loader_sketch
#         }
#         # Combine ImageNet and Sketch datasets for training into a single DataLoader
#         from torch.utils.data import ConcatDataset
#         combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
#         train_loader_pers = DataLoader(
#             combined_train_ds,
#             batch_size=32,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )
#         # Recompute total_steps using the combined loader
#         total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
#     else:
#         # Non-streaming: apply the same transforms via WrappedDataset, then DataLoader
#         data_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
#             transforms.ToTensor(),
#         ])

#         # Wrap the already-limited HF Datasets to apply transforms
#         train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
#         train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)

#         # Validation loaders (no shuffle)
#         val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
#         val_loader_imagenet = DataLoader(
#             val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
#         )

#         val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
#         val_loader_sketch = DataLoader(
#             val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
#         )

#         # Bundle them into the dict your training loop expects
#         val_loaders = {
#             "imagenet": val_loader_imagenet,
#             "sketch":   val_loader_sketch,
#         }
#         from torch.utils.data import ConcatDataset
#         # Combine train sets into one re-iterable loader
#         combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
#         train_loader_pers = DataLoader(
#             combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
#         )

#         # And recompute total_steps for the scheduler
#         total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)

#     # --- Personalization Training & Evaluation ---


#     # Now run personalization.
#     # The personalize_vit function remains mostly unchanged except that it now expects a single val_loader in non-streaming mode,
#     # and in streaming mode you can later call evaluate() on the model with the dictionary of loaders.
#     personalized_model = personalize_vit(
#         opt,
#         train_loader_pers,
#         # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
#         val_loaders,
#         num_epochs=personalize_vit_hps['num_epochs'],
#         base_lr=personalize_vit_hps['base_lr'],
#         layer_decay=personalize_vit_hps['layer_decay'],
#         freeze_layers=personalize_vit_hps['freeze_layers'],
#         total_steps=total_steps
#     )
#     opt = personalized_model

#     # Optionally save the personalized model weights
#     # personalized_model.model.save_pretrained("./personalized_vit")

#     # Evaluate after personalization
#     print("EVALUATION AFTER PERSONALIZATION")
#     if isinstance(val_loaders, dict):
#         eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
#         for key, metrics in eval_results.items():
#              print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
#     else:
#         eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
#         print(f"[Personalize] Combined Val Accuracy: {eval_results['combined']['accuracy']:.4f}, Avg Loss: {eval_results['combined']['avg_loss']:.4f}")


#     # # Evaluate the personalized model on whole dataset (birds/birdsless)
#     # # set the config for evaluation
#     eval_sample_size = 1e5
#     datasets = ['imagenet-1k-birds']
#     collection_sample_size = None # TODO: set proper value
#     with torch.no_grad():
#         print(f'********* starting evalution after personalization on the bird/birdless dataset (original dataset) *********')
#         eval_out = evaluate_all(opt, eval_sample_size, datasets,
#                                 dataset_tokens_to_skip=collection_sample_size)
#     # exclude 'token_count' from eval_out to print
#     print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
#     print(f'----- debug - model performance after personalization on the bird/birdless dataset (original dataset) -> eval_out: {print_eval_out} ----- \n\n')

#     from munl.unlearning.neggradplus import (
#     NegGradPlus,
#     DefaultNegGradPlusUnlearningConfig,
#     )
#     from omegaconf import OmegaConf, open_dict

#     print("Starting negrad unlearning")
#     device = next(opt.model.parameters()).device
#     print(f"device is{device}")
#     opt.model.to(device)
#     original_torch_model = opt.model

#     model_for_unlearning = ViTLogitsWrapper(original_torch_model, opt)



#     cfg = OmegaConf.structured(DefaultNegGradPlusUnlearningConfig())
#     with open_dict(cfg):
#         cfg.weight_decay = 0.0
#     cfg.num_epochs = 5
#     cfg.alpha = 0.99
#     cfg.batch_size = 64
#     print(cfg)
#     print("before unlearniner")
#     unlearner = NegGradPlus(cfg, device=device, writer=None)
#     unlearner.criterion = CEEnsureLong() 
#     print("before unlearniner")
#     scrubbed_model = unlearner.unlearn(
#         model=model_for_unlearning,  
#         retain_loader=retain_loader,
#         forget_loader=cripple_loader,
#         val_loader=None
#     )
#     return scrubbed_model

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


if __name__ == "__main__":
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # select personalization classes
    selected_classes = [0, 1, 2, 
                        3, 4, 5, 6,                            # fish
                        389, 390, 391, 392, 393, 394, 395, 396, 397,    # Marine            
    ] + list(range(924, 970))                                           # Food  
    
    # TODO: would be extended to otehr categories

    number_of_personalization_classes = 20
    selected_classes = selected_classes[:number_of_personalization_classes]
    print(f"Selected classes: {selected_classes}")

    # Set the alpha value for the split
    alpha = 0.5
    total_samples = 50  # Total number of samples for each class
    imagenet_samples = int(total_samples * alpha)
    sketch_samples = total_samples - imagenet_samples
    # Split per-class counts into train/val halves
    per_class_train_imagenet = imagenet_samples // 2
    per_class_val_imagenet   = imagenet_samples - per_class_train_imagenet
    per_class_train_sketch    = sketch_samples   // 2
    per_class_val_sketch      = sketch_samples   - per_class_train_sketch
    # assert that sketch_samples is less than or equal to 50 (25 for train and 25 for val)
    assert sketch_samples <= 25, f"Sketch samples {sketch_samples} exceed the limit of 50."

    # Personlization HPs
    personalize_vit_hps = {
        "num_epochs": 6,
        "base_lr": 5e-5,
        "layer_decay": 0.8,
        "freeze_layers": 3,
    }

    # TODO: define experiemnt and name and set the following directories based on that if needed

    # Load ImageNet train split
    imagenet_config = infer_dataset_config("imagenet-1k")
    imagenet_config.dataset_split = "train"
    imagenet_config.is_train_mode = True # TODO: set proper value
    imagenet_config.streaming = False
    imagenet_train = prepare_dataset(imagenet_config)

    # Load ImageNet val split
    imagenet_val_config = infer_dataset_config("imagenet-1k")
    imagenet_val_config.dataset_split = "validation"
    imagenet_val_config.is_train_mode = False # TODO: set proper value
    imagenet_val_config.streaming = False
    imagenet_test = prepare_dataset(imagenet_val_config)

    # Load Sketch-ImageNet train split
    sketch_config = infer_dataset_config("imagenet_sketch")
    sketch_config.dataset_split = "train"
    sketch_config.is_train_mode = False # TODO: set proper value
    sketch_config.streaming = False
    sketch_length = len(prepare_dataset(sketch_config))
    sketch = prepare_dataset(sketch_config)

    # -- cache setup --
    cache_root = "./cache_personalization"
    os.makedirs(cache_root, exist_ok=True)
    im_tr_cache = os.path.join(cache_root, "imagenet_train")
    im_val_cache = os.path.join(cache_root, "imagenet_val")
    sk_tr_cache = os.path.join(cache_root, "sketch_train")
    sk_val_cache = os.path.join(cache_root, "sketch_val")

    print("Preparing ImageNet train split...")
    if os.path.isdir(im_tr_cache):
        imagenet_train_dataset = load_from_disk(im_tr_cache)
    else:
        tmp = get_filtered_dataset(imagenet_train, selected_classes, streaming=imagenet_config.streaming)
        imagenet_train_dataset = limit_samples_per_class(tmp, per_class_train_imagenet, selected_classes)
        imagenet_train_dataset.save_to_disk(im_tr_cache)

    print("Preparing ImageNet val split...")
    if os.path.isdir(im_val_cache):
        imagenet_val_dataset = load_from_disk(im_val_cache)
    else:
        tmp = get_filtered_dataset(imagenet_test, selected_classes, streaming=imagenet_val_config.streaming)
        imagenet_val_dataset = limit_samples_per_class(tmp, per_class_val_imagenet, selected_classes)
        imagenet_val_dataset.save_to_disk(im_val_cache)

    print("Converting Sketch dataset to list for splitting...")
    # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=2000, test_ratio=0.5)
    # print(f'LEN OF THE SKETCH DATASET: {len(sketch)}')

    if os.path.isdir(sk_tr_cache) and os.path.isdir(sk_val_cache):
        print("NO NEED TO SPLITE SKETCH")
    else:
        sketch_train, sketch_val = limited_iterable_split(sketch, total_items=sketch_length, test_ratio=0.5)  # TODO: undo to len(sketch)
        # sketch_train, sketch_val = limited_iterable_split(sketch, total_items=100, test_ratio=0.5)

    print("Preparing Sketch train split...")
    if os.path.isdir(sk_tr_cache):
        sketch_train_dataset = load_from_disk(sk_tr_cache)
    else:
        tmp = get_filtered_dataset(sketch_train, selected_classes, streaming=sketch_config.streaming)
        sketch_train_dataset = limit_samples_per_class(tmp, per_class_train_sketch, selected_classes)
        sketch_train_dataset.save_to_disk(sk_tr_cache)

    print("Preparing Sketch val split...")
    if os.path.isdir(sk_val_cache):
        sketch_val_dataset = load_from_disk(sk_val_cache)
    else:
        tmp = get_filtered_dataset(sketch_val, selected_classes, streaming=sketch_config.streaming)
        sketch_val_dataset = limit_samples_per_class(tmp, per_class_val_sketch, selected_classes)
        sketch_val_dataset.save_to_disk(sk_val_cache)

    # Add validation check here
    # validate_splits()

    # Then proceed with model creation and training
    opt = Model(
        "google/vit-base-patch16-224",
        limit=1000,
        dtype="fp32",
        svd_attn=False,
        use_accelerator=True,
        model_device=None,
        mask_fn="step",
    )

    # --- Personalization ---
    total_steps = None

    # Prepare train/val DataLoaders for personalization
    # Define collate function (unchanged)
    # Decide based on streaming flag (here we use imagenet_config.streaming)
    if imagenet_config.streaming:
        # Create separate DataLoaders for each dataset:
        train_loader_imagenet = DataLoader(
            imagenet_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        train_loader_sketch = DataLoader(
            sketch_train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Create individual validation loaders:
        val_loader_imagenet = DataLoader(
            imagenet_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_loader_sketch = DataLoader(
            sketch_val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Keep the validation loaders in a dictionary
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch": val_loader_sketch
        }
        # Combine ImageNet and Sketch datasets for training into a single DataLoader
        from torch.utils.data import ConcatDataset
        combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
        train_loader_pers = DataLoader(
            combined_train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # Recompute total_steps using the combined loader
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)
    else:
        # Non-streaming: apply the same transforms via WrappedDataset, then DataLoader
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.ToTensor(),
        ])

        # Wrap the already-limited HF Datasets to apply transforms
        train_ds_im = WrappedDataset(imagenet_train_dataset, data_transform)
        train_ds_sk = WrappedDataset(sketch_train_dataset, data_transform)

        # Validation loaders (no shuffle)
        val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
        val_loader_imagenet = DataLoader(
            val_ds_im, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
        val_loader_sketch = DataLoader(
            val_ds_sk, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Bundle them into the dict your training loop expects
        val_loaders = {
            "imagenet": val_loader_imagenet,
            "sketch":   val_loader_sketch,
        }

        # Combine train sets into one re-iterable loader
        combined_train_ds = ConcatDataset([train_ds_im, train_ds_sk])
        train_loader_pers = DataLoader(
            combined_train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

        # And recompute total_steps for the scheduler
        total_steps = personalize_vit_hps["num_epochs"] * len(train_loader_pers)

    # --- Personalization Training & Evaluation ---

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

    # Now run personalization.
    # The personalize_vit function remains mostly unchanged except that it now expects a single val_loader in non-streaming mode,
    # and in streaming mode you can later call evaluate() on the model with the dictionary of loaders.
    personalized_model = personalize_vit(
        opt,
        train_loader_pers,
        # Pass the appropriate val_loader: if streaming, we pass the dict; otherwise, a single loader.
        val_loaders,
        num_epochs=personalize_vit_hps['num_epochs'],
        base_lr=personalize_vit_hps['base_lr'],
        layer_decay=personalize_vit_hps['layer_decay'],
        freeze_layers=personalize_vit_hps['freeze_layers'],
        total_steps=total_steps
    )
    opt = personalized_model

    # Optionally save the personalized model weights
    # personalized_model.model.save_pretrained("./personalized_vit")

    # Evaluate after personalization
    print("EVALUATION AFTER PERSONALIZATION")
    if isinstance(val_loaders, dict):
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        for key, metrics in eval_results.items():
             print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")
    else:
        eval_results = evaluate(personalized_model, val_loaders, device=next(opt.model.parameters()).device)
        print(f"[Personalize] Combined Val Accuracy: {eval_results['combined']['accuracy']:.4f}, Avg Loss: {eval_results['combined']['avg_loss']:.4f}")


    # # Evaluate the personalized model on whole dataset (birds/birdsless)
    # # set the config for evaluation
    eval_sample_size = 1e5
    datasets = ['imagenet-1k-birds', 'imagenet-1k-birdless']
    collection_sample_size = None # TODO: set proper value
    # with torch.no_grad():
    #     print(f'********* starting evalution after personalization on the bird/birdless dataset (original dataset) *********')
    #     eval_out = evaluate_all(opt, eval_sample_size, datasets,
    #                             dataset_tokens_to_skip=collection_sample_size)
    # # exclude 'token_count' from eval_out to print
    # print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    # print(f'----- debug - model performance after personalization on the bird/birdless dataset (original dataset) -> eval_out: {print_eval_out} ----- \n\n')


    print("\n\n ============ UNLEARNING ==========")
    # Apply the UNLEARNING (apply the previous trained mask to the model - the mask is result of running the pruning script - prune_vit.sh) + COMPENSATION

    # Load the mask
    mask_path = "/vol/bitbucket/sc2124/selective_pruning_mmd/exp6-snip-pruning/outputs/prune_mask.pt"  # Path to the mask file
    # print current directory
    print(f'Current directory: {os.getcwd()}')
    mask = torch.load(mask_path)
    print(f' --- debug - mask shape: {mask.shape} --- ')

    # Evaluate the model when applying the mask without compensation
    # Eval1: On personalization data
    print(f'********* starting evalution after unlearning (on birds) without compensation *********')
    tmp_opt = copy.deepcopy(opt)
    device = next(opt.model.parameters()).device
    tmp_opt.model.to(device)
    prune_only_all_layers(tmp_opt.model, prune_mask=mask, device=device)
    eval_results = evaluate(tmp_opt, val_loaders, device=next(tmp_opt.model.parameters()).device)
    for key, metrics in eval_results.items():
        print(f"[Personalize] {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")

    # # Eval2: On whole dataset (birds/birdless)
    print(f'********* starting evalution after unlearning (on birds/birdsless) without compensation *********')
    # with torch.no_grad():
    #     eval_out = evaluate_all(tmp_opt, eval_sample_size, datasets,
    #                             dataset_tokens_to_skip=collection_sample_size)
    # # exclude 'token_count' from eval_out to print
    # print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    # print(f'----- debug - model performance after unlearning (on birds/birdsless) without compensation -> eval_out: {print_eval_out} ----- \n\n')
    del tmp_opt
    
    print("\n\n ============ COMPENSATION ==========")
    # compute the fisher inverses which is needed for compensation


    # Compensation HPs
    compensation_hps = {
        'block_size': 50,
        'damp': 1e-3,
        'learning_rate': 4,
    }
    # Build a combined dataset for Fisher inverse computation
    combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
    fisher_loader = DataLoader(
        combined_train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    num_grads = len(combined_train_ds)
    print("\n NUM GRANDS: ",num_grads)
    fisher_invs = compute_personalized_fisher_invs(
        opt,
        dataloader=fisher_loader,
        num_grads=num_grads,
        block_size=compensation_hps['block_size'],
        damp=compensation_hps['damp'],
        device=opt.model.device
    )
    
    # Apply the mask to the model and compensate
    prune_and_compensate_all_layers(
        opt.model,
        prune_mask = mask,
        fisher_invs = fisher_invs,
        block_size=compensation_hps['block_size'], 
        damp=compensation_hps['damp'],
        learning_rate=compensation_hps['learning_rate'],
        device=opt.model.device,
    )

    # Evaluate the model after unlearning
    eval_results = evaluate(opt, val_loaders, device=next(opt.model.parameters()).device)
    for key, metrics in eval_results.items():
        print(f"[Personalize]  - after applying pruning+compensation: {key} Val Accuracy: {metrics['accuracy']:.4f}, Avg Loss: {metrics['avg_loss']:.4f}")


    # Eval2: on whole dataset (birds/birdless)
    # print(f'********* starting evalution after unlearning (on birs) and compensation (on personalization data) *********')
    # with torch.no_grad():
    #     eval_out = evaluate_all(opt, eval_sample_size, datasets,
    #                             dataset_tokens_to_skip=collection_sample_size)
    # # exclude 'token_count' from eval_out to print
    # print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    # print(f'----- debug - model performance after unlearning (on birds) and compensation (on personalization data) -> eval_out: {print_eval_out} ----- \n\n')
    



