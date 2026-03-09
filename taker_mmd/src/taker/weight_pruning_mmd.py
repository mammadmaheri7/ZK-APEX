import copy
import torch
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
from .model import Model
from .prune import get_VIT_dataloder, EmpiricalBlockFisherInverse
from .fine_tune import WrappedDataset, compute_personalized_fisher_invs, personalize_vit
from .activations import get_midlayer_data
# from .benchmarks import apply_negrad_vit
from .scoring import score_indices_by
from .model_saving import save_taker_payload, load_taker_payload
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from datasets import load_dataset,load_from_disk, Image as HFImage
import torch.nn.functional as F
from taker.eval import evaluate_all
import os
import shutil
from PIL import Image as PILImage
import time
import gc
from pathlib import Path
import sys
from .texts import prepare_dataset, infer_dataset_config
import numpy as np


class LabelFilter(torch.utils.data.IterableDataset):
    """
    Include or exclude a set of labels from a streaming iterable dataset.
    - labels: iterable of class ids to include/exclude
    - mode: 'include' -> keep only these labels; 'exclude' -> drop these labels
    """
    def __init__(self, src_iterable, labels, label_key="label", mode="include"):
        assert mode in ("include", "exclude")
        self.src = src_iterable
        self.labels = {int(x) for x in labels}
        self.label_key = label_key
        self.mode = mode

    @staticmethod
    def _get_label(ex, label_key):
        if isinstance(ex, dict):
            return int(ex.get(label_key, ex.get("labels")))
        elif isinstance(ex, (tuple, list)) and len(ex) >= 2:
            return int(ex[1])  # (image, label)
        else:
            raise TypeError(f"Unsupported sample type: {type(ex)}")

    def __iter__(self):
        for ex in self.src:
            y = self._get_label(ex, self.label_key)
            in_set = y in self.labels
            if (self.mode == "include" and in_set) or (self.mode == "exclude" and not in_set):
                yield ex

class PerClassHeadFilter(torch.utils.data.IterableDataset):
    """
    Wraps a streaming HF IterableDataset and either:
      - 'exclude' the first N examples per class_id (mode='exclude'), or
      - 'keep' everything except the first N examples per class_id (mode='keep').

    trim_counts: dict[int, int]  # {class_id: N_to_exclude_from_the_head}
    label_key: name of the label field in each example (default 'label').
    """
    def __init__(self, src_iterable, trim_counts, label_key="label", mode="keep"):
        assert mode in ("keep", "exclude")
        self.src = src_iterable
        self.trim_counts = dict(trim_counts or {})
        self.label_key = label_key
        self.mode = mode

    def __iter__(self):
        seen = defaultdict(int)  # per-class counters
        for ex in self.src:
            if isinstance(ex, dict):
                y = int(ex.get(self.label_key, ex.get("labels")))
            elif isinstance(ex, (tuple, list)) and len(ex) >= 2:
                y = int(ex[1])   # (image, label)
            else:
                raise TypeError(f"Unsupported sample type: {type(ex)}")
            # How many from head should be excluded for this class?
            head_n = self.trim_counts.get(y, 0)
            cnt = seen[y]
            is_head = cnt < head_n
            seen[y] = cnt + 1

            if self.mode == "exclude":
                if is_head:
                    yield ex
            else:  # mode == "keep"
                if not is_head:
                    yield ex


def _clear_all_hooks(root):
    for m in root.modules():
        for attr in ("_forward_pre_hooks","_forward_hooks","_backward_pre_hooks","_backward_hooks"):
            if hasattr(m, attr):
                getattr(m, attr).clear()

def rehydrate_taker(payload_path, device="cuda", add_hooks=False, dtype="fp32"):
    # 1) rebuild wrapper + load weights (your loader from earlier)
    print("in rehydrate_taker")
    m = load_taker_payload(payload_path, dtype=dtype, model_device=device)
    print("loaded pauload")
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

from torch.utils.data import DataLoader, SequentialSampler, Subset
from datasets import Features, Value
from io import BytesIO

BIRD_EVAL_CACHE_ROOT = Path("/vol/bitbucket/sc2124/selective_pruning_mmd/exp18-refinetune-vit/cache/bird_eval")
_TO_TENSOR = transforms.ToTensor()


def _standardize_sample(sample, transform=None):
    """
    Convert a dataset sample into (image_tensor, label_int) on CPU.
    Supports dict-based and tuple/list-based samples.
    """
    if isinstance(sample, dict):
        image = sample.get("image")
        if image is None:
            image = sample.get("pixel_values")
        label = sample.get("label", sample.get("labels"))
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        image, label = sample[0], sample[1]
    else:
        raise TypeError(f"Unsupported sample type for caching: {type(sample)}")

    if isinstance(label, torch.Tensor):
        label = int(label.item())
    else:
        label = int(label)

    if isinstance(image, torch.Tensor):
        image_tensor = image.detach().cpu()
    elif transform is not None:
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().cpu()
        else:
            if isinstance(image, PILImage):
                pil_img = image
            elif isinstance(image, HFImage):
                pil_img = image
            else:
                pil_img = PILImage.fromarray(np.array(image))
            transformed = transform(pil_img)
            image_tensor = transformed.detach().cpu() if isinstance(transformed, torch.Tensor) else torch.as_tensor(transformed)
    elif isinstance(image, PILImage):
        image_tensor = _TO_TENSOR(image).detach()
    elif isinstance(image, HFImage):
        image_tensor = _TO_TENSOR(image).detach()
    else:
        image_tensor = torch.as_tensor(image)
        if image_tensor.ndim == 3 and image_tensor.shape[0] not in (1, 3):
            image_tensor = _TO_TENSOR(image)
        image_tensor = image_tensor.detach()
    return image_tensor.cpu(), label


class CachedSampleDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset backed by samples serialized to disk.
    Each sample is stored as an individual .pt file under cache_dir.
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.pt"
        if not index_path.exists():
            raise FileNotFoundError(f"Cached dataset missing index at {index_path}")
        meta = torch.load(index_path, map_location="cpu")
        self.files = meta.get("files", [])
        self.meta = meta.get("meta", {})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.cache_dir / self.files[idx]
        data = torch.load(sample_path, map_location="cpu")
        image = data["image"]
        label = int(data["label"])
        return image, label


def _cache_iterable_split(iterable_ds, cache_root, split_name, meta, transform=None):
    """
    Materialize an iterable dataset split to disk to avoid repeated filtering.
    """
    cache_root = Path(cache_root)
    split_dir = cache_root / split_name
    index_path = split_dir / "index.pt"

    if index_path.exists():
        stored_meta = torch.load(index_path, map_location="cpu")
        if stored_meta.get("meta") == meta:
            print(f"[bird eval] Using cached split '{split_name}' from {split_dir}")
            return CachedSampleDataset(split_dir)
        print(f"[bird eval] Cache metadata mismatch for '{split_name}', rebuilding...")
        raise ValueError('BAD NEWS')
        return
        # shutil.rmtree(split_dir, ignore_errors=True)
    print("in cache index path: ")
    print(index_path)
    print(index_path.exists())
    raise ValueError('In else')
    split_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for idx, sample in enumerate(iterable_ds):
        if idx % 100 == 0:
            print(f"handling index {idx}")
            sys.stdout.flush()
        file_name = f"{idx:08d}.pt"
        out_path = split_dir / file_name
        if not out_path.exists():
            image_tensor, label = _standardize_sample(sample)
            out_path = split_dir / file_name
            torch.save(
                    {"image": image_tensor, "label": label},
                    split_dir / file_name,
            )
        files.append(file_name)

    torch.save({"files": files, "meta": meta}, index_path)
    print(f"[bird eval] Cached split '{split_name}' with {len(files)} samples at {split_dir}")
    return CachedSampleDataset(split_dir)

def _extract_label_sequence(dataset):
    """
    Try to obtain a list of labels for an indexable dataset without invoking
    the potentially expensive transform pipeline. Falls back to indexing if
    needed.
    """
    base_dataset = getattr(dataset, "base", None)

    if base_dataset is not None:
        try:
            labels = base_dataset["label"]
            return [int(lbl) for lbl in labels]
        except Exception:
            pass

    try:
        n = len(dataset)
    except TypeError:
        raise TypeError("Dataset does not support len(); cannot build label subsets.")

    labels = []
    for idx in range(n):
        if base_dataset is not None:
            sample = base_dataset[idx]
        else:
            sample = dataset[idx]

        if isinstance(sample, dict):
            label = sample.get("label", sample.get("labels"))
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            label = sample[1]
        else:
            raise TypeError(f"Unsupported sample type while extracting labels: {type(sample)}")

        labels.append(int(label))

    return labels


def build_bird_eval_loaders(dataset, bird_labels, n_per_label, batch_size, transform=None):
    """
    Pre-compute static dataloaders for:
      - the dataset with the first `n_per_label` bird samples removed per label,
      - the first `n_per_label` samples for each bird label,
      - all remaining bird samples after the first `n_per_label` occurrences.

    If `transform` is provided, any cached splits will persist tensors that already
    have that transform applied.

    Returns an OrderedDict mapping split names to DataLoader instances.
    """
    if not bird_labels:
        print("[bird eval] No bird labels supplied; skipping pre-computed bird splits.")
        return OrderedDict()

    n_per_label = max(int(n_per_label), 0)
    if n_per_label == 0:
        print("[bird eval] Requested first_n_per_label=0; skipping bird subset loaders.")
        return OrderedDict()

    bird_label_set = {int(lbl) for lbl in bird_labels}

    try:
        dataset_len = len(dataset)
        is_indexable = hasattr(dataset, "__getitem__")
    except TypeError:
        dataset_len = None
        is_indexable = False

    if dataset_len == 0:
        print("[bird eval] Dataset is empty; skipping pre-computed bird splits.")
        return OrderedDict()

    if not is_indexable:
        print("[bird eval] Dataset appears streaming; building iterable filters.")
        trim_counts = {int(lbl): n_per_label for lbl in bird_labels}

        loaders = OrderedDict()

        dataset_without_head = PerClassHeadFilter(
            dataset,
            trim_counts,
            label_key="label",
            mode="keep",
        )
        loaders["dataset_without_first_n_bird_images"] = DataLoader(
            dataset_without_head,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        birds_only_for_head = LabelFilter(
            dataset,
            labels=bird_labels,
            label_key="label",
            mode="include",
        )
        birds_head = PerClassHeadFilter(
            birds_only_for_head,
            trim_counts,
            label_key="label",
            mode="exclude",
        )
        cache_root = BIRD_EVAL_CACHE_ROOT
        cache_root.mkdir(parents=True, exist_ok=True)
        head_split_name = f"birds_first_{n_per_label}_per_label"
        transform_repr = repr(transform) if transform is not None else None
        head_meta = {
            "split": head_split_name,
            "n_per_label": n_per_label,
            "bird_labels": sorted(bird_label_set),
        }
        print(f"head_meta is {head_meta}")
        sys.stdout.flush()
        cached_head = _cache_iterable_split(
            birds_head,
            cache_root,
            head_split_name,
            meta=head_meta,
        )
        loaders[head_split_name] = DataLoader(
            cached_head,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        birds_only_for_tail = LabelFilter(
            dataset,
            labels=bird_labels,
            label_key="label",
            mode="include",
        )
        birds_tail = PerClassHeadFilter(
            birds_only_for_tail,
            trim_counts,
            label_key="label",
            mode="keep",
        )
        tail_split_name = f"birds_after_first_{n_per_label}_per_label"
        tail_meta = {
            "split": tail_split_name,
            "n_per_label": n_per_label,
            "bird_labels": sorted(bird_label_set),
        }

        print(f"tail_meta is {tail_meta}")
        cached_tail = _cache_iterable_split(
            birds_tail,
            cache_root,
            tail_split_name,
            meta=tail_meta,
        )
        loaders[tail_split_name] = DataLoader(
            cached_tail,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
        return loaders

    try:
        label_sequence = _extract_label_sequence(dataset)
    except TypeError:
        print("[bird eval] Unable to extract labels without iteration; skipping bird splits.")
        return OrderedDict()

    bird_label_set = {int(lbl) for lbl in bird_labels}
    label_counts = defaultdict(int)

    first_n_indices = []
    bird_tail_indices = []
    dataset_without_head_indices = []

    for idx, label in enumerate(label_sequence):
        label = int(label)
        if label in bird_label_set:
            cnt = label_counts[label]
            if cnt < n_per_label:
                first_n_indices.append(idx)
            else:
                bird_tail_indices.append(idx)
                dataset_without_head_indices.append(idx)
            label_counts[label] += 1
        else:
            dataset_without_head_indices.append(idx)

    loaders = OrderedDict()

    def _make_loader(indices, description):
        if not indices:
            print(f"[bird eval] No samples for split '{description}', skipping.")
            return None
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

    without_loader = _make_loader(
        dataset_without_head_indices,
        "dataset_without_first_n_bird_images",
    )
    if without_loader is not None:
        loaders["dataset_without_first_n_bird_images"] = without_loader

    head_split_name = f"birds_first_{n_per_label}_per_label"
    head_loader = _make_loader(first_n_indices, head_split_name)
    if head_loader is not None:
        loaders[head_split_name] = head_loader

    tail_split_name = f"birds_after_first_{n_per_label}_per_label"
    tail_loader = _make_loader(bird_tail_indices, tail_split_name)
    if tail_loader is not None:
        loaders[tail_split_name] = tail_loader

    return loaders



def retrain_without_forget(c):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # retain_dataloader = get_VIT_dataloder("imagenet-1k-birdless")
    custom_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),                 # match TFDS 3-ch decode
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4,4/3),
                                    interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3),                 # [-1,1]
    ])
    
    

    classes_to_trim = {
        "7": "Cock",
        "8": "Hen",
        "9": "Ostrich (Struthio camelus)",
        "10": "Brambling (Fringilla montifringilla)",
        "11": "Goldfinch (Carduelis carduelis)",
        "12": "House Finch (Carpodacus mexicanus)",
        "13": "Junco (Snowbird)",
        "14": "Indigo Bunting (Passerina cyanea)",
        "15": "Robin (American Robin, Turdus migratorius)",
        "16": "Bulbul",
        "17": "Jay",
        "18": "Magpie",
        "19": "Chickadee",
        "20": "Water Ouzel (Dipper)",
        "21": "Kite",
        "22": "Bald Eagle (American Eagle, Haliaeetus leucocephalus)",
        "23": "Vulture",
        "24": "Great Grey Owl (Strix nebulosa)",
        "83": "Black Grouse",
        "84": "Ptarmigan",
        "85": "Ruffed Grouse (Partridge, Bonasa umbellus)",
        "86": "Prairie Chicken (Prairie Grouse, Prairie Fowl)",
        "87": "Peacock",
        "88": "Quail",
        "89": "Partridge",
        "90": "Lorikeet",
        "91": "Coucal",
        "92": "Bee Eater",
        "93": "Hornbill",
        "94": "Hummingbird",
        "95": "Jacamar",
        "96": "Toucan",
        "97": "Drake",
        "98": "Red-breasted Merganser (Mergus serrator)",
        "99": "Goose",
        "100": "Black Swan (Cygnus atratus)",
        "127": "White Stork (Ciconia ciconia)",
        "128": "Black Stork (Ciconia nigra)",
        "129": "Spoonbill",
        "130": "Flamingo",
        "131": "Little Blue Heron (Egretta caerulea)",
        "132": "American Egret (Great White Heron, Egretta albus)",
        "133": "Bittern",
        "134": "Crane",
        "135": "Limpkin (Aramus pictus)",
        "136": "European Gallinule (Porphyrio porphyrio)",
        "137": "American Coot (Marsh Hen, Mud Hen, Water Hen, Fulica americana)",
        "138": "Bustard",
        "139": "Ruddy Turnstone (Arenaria interpres)",
        "140": "Red-backed Sandpiper (Dunlin, Erolia alpina)",
        "141": "Redshank (Tringa totanus)",
        "142": "Dowitcher",
        "143": "Oystercatcher (Oyster Catcher)",
        "144": "Pelican",
        "145": "King Penguin (Aptenodytes patagonica)",
        "146": "Albatross (Mollymawk)"}
    
    sys.stdout.flush()
    trim_counts = { int(c): 600 for c in classes_to_trim.keys() }
    print(trim_counts)
    batch_size = 64
    train_loader = get_VIT_dataloder("imagenet-1k", batch_size = batch_size, custom_transforms=custom_transforms)
    dataset = train_loader.dataset

    bird_labels = sorted(int(lbl) for lbl in trim_counts.keys())

    validation_loader = get_VIT_dataloder("imagenet-1k", is_validation = True, batch_size = 64, custom_transforms=custom_transforms)
    birds_only_val = LabelFilter(
            validation_loader.dataset,
            labels=bird_labels,
            label_key="label",
            mode="include",
        )
    
    birds_only_val_loader = DataLoader(
            birds_only_val,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

    bird_eval_first_n = int(getattr(c, "bird_eval_top_n", 600))
    print(f"Configuring bird subset evaluation with first_n_per_label={bird_eval_first_n}")
    bird_subset_eval_loaders = build_bird_eval_loaders(dataset, bird_labels, bird_eval_first_n, batch_size)
    if bird_subset_eval_loaders:
        print(f"Prepared bird eval splits: {list(bird_subset_eval_loaders.keys())}")
    bird_eval_max_batches = getattr(c, "bird_eval_max_batches", None)

    TRAIN_SPLIT_KEY = "dataset_without_first_n_bird_images"
    evaluation_splits = OrderedDict()

    primary_train_loader = bird_subset_eval_loaders.get(TRAIN_SPLIT_KEY)
    if primary_train_loader is None:
        print(f"[bird eval] Split '{TRAIN_SPLIT_KEY}' missing; using base train loader instead.")
        primary_train_loader = train_loader
    evaluation_splits[TRAIN_SPLIT_KEY] = {
        "name": TRAIN_SPLIT_KEY,
        "loader": primary_train_loader,
        "eval_fn": evaluate_validation,
        "eval_kwargs": {},
    }

    evaluation_splits["validation"] = {
        "name": "validation",
        "loader": validation_loader,
        "eval_fn": evaluate_validation,
        "eval_kwargs": {},
    }

    evaluation_splits["birds_only_val_loader"] = {
        "name": "birds_only_val_loader",
        "loader": birds_only_val_loader,
        "eval_fn": evaluate_validation,
        "eval_kwargs": {},
    }

    eval_kwargs = {}
    if bird_eval_max_batches is not None:
        eval_kwargs["max_batches"] = bird_eval_max_batches

    for split_name, loader in bird_subset_eval_loaders.items():
        if split_name == TRAIN_SPLIT_KEY:
            continue
        evaluation_splits[split_name] = {
            "name": split_name,
            "loader": loader,
            "eval_fn": evaluate_validation,
            "eval_kwargs": eval_kwargs.copy(),
        }

    run_train(evaluation_splits)

def evaluate_validation(val_loader, model, device, scaler=None, max_batches=None, skip_batches = 0):
    sys.stdout.flush()
    was_training = model.training
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    # use AMP if you use it for train (it speeds up eval too)
    use_amp = (scaler is not None) and getattr(scaler, "is_enabled", lambda: False)()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i > skip_batches:
                if max_batches is not None and i >= max_batches+skip_batches:
                    break
                # support either (images, labels) or dicts
                if isinstance(batch, dict):
                    images, labels = batch["pixel_values"], batch["labels"]
                else:
                    images, labels = batch

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # autocast is safe in eval
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(pixel_values=images)
                    # sum-reduction so we can average later over samples
                    val_loss = F.cross_entropy(out.logits, labels, reduction="sum")

                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.numel()
                loss_sum += val_loss.item()

    if was_training:
        model.train()
    avg_loss = loss_sum / max(total, 1)
    avg_acc  = correct / max(total, 1)
    return avg_loss, avg_acc


from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler


def run_train(evaluation_splits):
    train_split_key = "dataset_without_first_n_bird_images"
    train_entry = evaluation_splits.get(train_split_key)
    if train_entry is None:
        if not evaluation_splits:
            raise ValueError("No evaluation splits provided; unable to determine training loader.")
        # fall back to the first available split
        train_split_key, train_entry = next(iter(evaluation_splits.items()))
        print(f"[train] '{train_split_key}' chosen as training split (default fallback).")

    train_loader = train_entry.get("loader")
    if train_loader is None:
        raise ValueError(f"Training split '{train_split_key}' does not provide a DataLoader.")
    
    NUM_CLASSES = 1000
    bs_per_gpu = 64                  # example; adjust for your GPU
    accum = 8                        # 64 * 8 = 512 global batch on 1 GPU; scale accordingly
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_every_updates = 1000       # e.g., like vit_jax checkpoint_every
    out_dir = "./vit_b16_in1k_ft_bigger_forget_set_1000_logging"

    cfg = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    cfg.num_labels = NUM_CLASSES
    cfg.hidden_dropout_prob = 0.0
    cfg.attention_probs_dropout_prob = 0.0

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=cfg,
        ignore_mismatched_sizes=True,  # re-init classification head
    ).to(device)   
    base_lr = 3e-2           # 0.03
    warmup_steps = 500
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    optim = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0)

    total_updates  = 20_000
    cosine_updates = total_updates - warmup_steps

    warmup = LinearLR(optim, start_factor=1.0/warmup_steps, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optim, T_max=cosine_updates, eta_min=0.0)
    sched  = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler = GradScaler()  # set to None if you don't want AMP
    model.train()

    update_step = 0
    micro_step = 0
    print("Any trainable params?:", any(p.requires_grad for p in model.parameters()))
    print("Num trainable params:", sum(p.requires_grad for p in model.parameters()))

    # If everything is frozen (0), unfreeze:
    if not any(p.requires_grad for p in model.parameters()):
        model.requires_grad_(True)  # unfreeze whole model

    # Ensure we are NOT in a global no-grad mode
    import torch
    torch.set_grad_enabled(True)


    import time

    # evaluation_splits = list(evaluation_splits or [])

    log_every_updates = 300
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    tic = time.time()

    def _run_epoch_evaluations(epoch_idx):
        if not evaluation_splits:
            return
        print(f"[epoch {epoch_idx}] running evaluation splits...")
        sys.stdout.flush()
        for split_name, split in evaluation_splits.items():
            if split_name == train_split_key:
                continue
            name = split.get("name", split_name)
            print("Evaluating ")
            loader = split.get("loader")
            if loader is None:
                continue
            eval_fn = split.get("eval_fn", evaluate_validation)
            eval_kwargs = dict(split.get("eval_kwargs", {}))
            eval_kwargs.setdefault("skip_batches", 0)
            print(f"[epoch {epoch_idx}] eval -> {name}")
            result = eval_fn(
                loader,
                model,
                device,
                scaler=scaler,
                **eval_kwargs,
            )
            if isinstance(result, tuple) and len(result) == 2:
                avg_loss, avg_acc = result
                summary = {
                    "avg_loss": avg_loss,
                    "acc": avg_acc,
                }
                if "max_batches" in eval_kwargs:
                    summary["max_batches"] = eval_kwargs["max_batches"]
                if "skip_batches" in eval_kwargs:
                    summary["skip_batches"] = eval_kwargs["skip_batches"]
                print(summary)
            else:
                print(result)
            sys.stdout.flush()

    epoch = 0
    while update_step < total_updates:
        epoch += 1
        print(f"=== Starting epoch {epoch} ===")
        sys.stdout.flush()
        for images, labels in train_loader:
            if update_step >= total_updates:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                out = model(pixel_values=images)
                raw_loss = F.cross_entropy(out.logits, labels)
                loss = raw_loss / accum

            with torch.no_grad():
                preds = out.logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_count += labels.numel()
                running_loss += raw_loss.item() * labels.size(0)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            if micro_step % accum == 0:
                if scaler:
                    scaler.unscale_(optim)

                clip_grad_norm_(model.parameters(), max_norm=1.0)
                sched.step()

                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

                optim.zero_grad(set_to_none=True)

                update_step += 1
                if (update_step % log_every_updates == 0) or (update_step == total_updates):
                    lr = optim.param_groups[0]["lr"]
                    dt = time.time() - tic
                    ips = running_count / max(dt, 1e-6)
                    avg_loss = running_loss / max(running_count, 1)
                    avg_acc = 100.0 * running_correct / max(running_count, 1)
                    print(
                        f"[upd {update_step:6d}/{total_updates}] "
                        f"loss {avg_loss:.4f} | acc {avg_acc:.2f}% | lr {lr:.5g} | {ips:.1f} img/s"
                    )

                    _run_epoch_evaluations(epoch)
                    sys.stdout.flush()
                    running_loss = running_correct = running_count = 0
                    tic = time.time()

                if (update_step % save_every_updates == 0) or (update_step == total_updates):
                    save_checkpoint_ckpt(
                        out_dir=out_dir,
                        model=model,
                        optimizer=optim,
                        scheduler=sched,
                        scaler=scaler,
                        update_step=update_step,
                        keep_last=15,
                    )

        _run_epoch_evaluations(epoch)
        if update_step >= total_updates:
            break
    save_checkpoint_ckpt(
        out_dir=out_dir,
        model=model,
        optimizer=optim,
        scheduler=sched,
        scaler=scaler,
        update_step=update_step,
        keep_last=15,
    )

import os, shutil, glob, torch
from pathlib import Path



def load_last_checkpoint(out_dir, model, optimizer, scheduler, scaler, checkpoint_int = 1):
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"))
    if not ckpts:
        return 0
    last = ckpts[-checkpoint_int]
    # reload model (HF)
    model_to_load = model.module if hasattr(model, "module") else model
    loaded = ViTForImageClassification.from_pretrained(last, ignore_mismatched_sizes=False)
    model_to_load.load_state_dict(loaded.state_dict())

    # reload trainer state
    state = torch.load(last / "trainer_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"): scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"): scaler.load_state_dict(state["scaler"])
    return int(state.get("update_step", 0))

def is_main_process():
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True

def save_checkpoint_ckpt(
    out_dir: str,
    model,
    optimizer,
    scheduler,
    scaler,                 # can be None
    update_step: int,
    keep_last: int = 5,
):
    if not is_main_process():
        return

    ckpt_dir = Path(out_dir) / f"checkpoint-{update_step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Save HF-style weights + config (nice for later loading) ---
    # If using DDP, model may be wrapped; unwrap for saving
    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(ckpt_dir, safe_serialization=True)  # saves model.safetensors + config.json

    # --- Save training state (optimizer/scheduler/scaler/step) ---
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "update_step": update_step,
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")

    # --- Prune old checkpoints (keep only last `keep_last`) ---
    all_ckpts = sorted(Path(out_dir).glob("checkpoint-*"))
    # if len(all_ckpts) > keep_last:
    #     for p in all_ckpts[:-keep_last]:
            # shutil.rmtree(p, ignore_errors=True)


import math, sys, torch, torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification, AutoImageProcessor, ViTConfig

def run_retraining(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_ID = "google/vit-base-patch16-224-in21k"
    NUM_CLASSES = 1000  # <-- set this

    # If your dataset isn't already 384 + [-1,1], adjust its transform:
    #   transforms.Resize((384,384), InterpolationMode.BICUBIC)
    #   transforms.ToTensor(); transforms.Normalize([0.5]*3, [0.5]*3)
    # (HF processor shown here just for reference)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # NOTE: We do NOT rely on processor to make tensors since you already have a DataLoader.

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,  # swap head safely
    )
    # Zero-init head to match the paper’s note
    with torch.no_grad():
        model.classifier.weight.zero_()
        if model.classifier.bias is not None:
            model.classifier.bias.zero_()

    model.to(device)

    # --- Paper-like optimizer/schedule ---
    BASE_LR = 0.01        # try {0.003, 0.01, 0.03}; 0.06 if dataset/scale tolerates it
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0    # paper uses no WD for fine-tuning
    GRAD_CLIP = 1.0
    EFFECTIVE_BS = 512    # paper batch size
    bs = getattr(dataloader, 'batch_size', 1)
    accum_steps = max(1, EFFECTIVE_BS // bs)

    optim = SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(dataloader)
    target_steps = 20_000  # paper’s ImageNet fine-tuning horizon
    epochs = math.ceil(target_steps / max(1, steps_per_epoch))

    # Cosine schedule stepped *per iteration*
    scheduler = CosineAnnealingLR(optim, T_max=target_steps, eta_min=0.0)

    criterion = nn.CrossEntropyLoss()

    # (Optional) Polyak/EMA like the paper’s high-accuracy runs (Table 2 used EMA 0.9999)
    use_ema = False
    ema_decay = 0.9999
    ema_shadow = {n: p.detach().clone() for n, p in model.named_parameters()} if use_ema else None

    import os, torch, math

    CKPT_DIR = "ckpts/vit_b16_ft"; os.makedirs(CKPT_DIR, exist_ok=True)
    best_val = float("inf")  # or use -inf for "best accuracy"
    step = 0                 # keep your existing step counter

    def save_ckpt(tag, extra=None):
        payload = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "accum_steps": accum_steps,
            "model_id": MODEL_ID,
            "num_labels": NUM_CLASSES,
        }
        if use_ema:
            # keep on CPU to shrink GPU memory usage during save
            payload["ema_shadow"] = {k: v.cpu() for k, v in ema_shadow.items()}
        if extra: payload.update(extra)
        tmp = os.path.join(CKPT_DIR, f"{tag}.pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(CKPT_DIR, f"{tag}.pt"))
    
    def ema_update():
        if not use_ema: return
        with torch.no_grad():
            for (n, p) in model.named_parameters():
                ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

    step = 0
    model.train()
    for epoch in range(epochs):
        for it, (images, labels) in enumerate(dataloader):
            # Ensure images are (B,3,384,384) and normalized to [-1,1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(pixel_values=images).logits  # HF ViT expects normalized tensors already
            loss = criterion(logits, labels) / accum_steps
            loss.backward()

            if (it + 1) % accum_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optim.step()
                optim.zero_grad(set_to_none=True)
                ema_update()

                step += 1
                scheduler.step()

                if step % 100 == 0:
                    with torch.no_grad():
                        pred = logits.argmax(dim=-1)
                        acc = (pred == labels).float().mean()
                    print(f"ep {epoch} step {step}/{target_steps} | loss {loss.item()*accum_steps:.4f} | acc {acc.item():.3f} | lr {scheduler.get_last_lr()[0]:.5f}")
                    sys.stdout.flush()

            if step >= target_steps:
                break
        
        
        if step >= target_steps:
            break

    # If you used EMA, you can swap weights for evaluation:
    if use_ema:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(ema_shadow[n])





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
            val_ds_im, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )


    sketch_val_dataset = load_from_disk(sk_val_cache)
    val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
    val_loader_sketch = DataLoader(
        val_ds_sk, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
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
import torch
from typing import Dict

@torch.no_grad()
def build_weight_masks_global_fraction_normalized(
    per_weight_scores: Dict[str, torch.Tensor],
    *,
    top_fraction: float,
    min_at_least_one: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Global selection across ALL layers after *layerwise min-max normalization*.
    For each layer ℓ:
        s_ℓ_norm = (s_ℓ - min(s_ℓ)) / (max(s_ℓ) - min(s_ℓ))
    Then select the top fraction globally over the concatenated s_ℓ_norm.

    Returns {name: BoolTensor[out,in]} where True = prune this weight.
    """
    if not (0.0 <= top_fraction <= 1.0):
        raise ValueError("top_fraction must be in [0, 1].")

    names = list(per_weight_scores.keys())
    if len(names) == 0:
        return {}

    # 1) Build normalized (0..1) flat score vectors per layer on CPU
    flats_norm = []
    sizes = []
    for name in names:
        S = per_weight_scores[name].detach()
        s = S.reshape(-1).to("cpu", dtype=torch.float32)

        s_min = torch.min(s)
        s_max = torch.max(s)
        rng = (s_max - s_min)

        if rng.abs() < 1e-12:
            # All scores equal in this layer -> make them all zeros after normalization
            s_norm = torch.zeros_like(s)
        else:
            s_norm = (s - s_min) / rng

        # Guard against non-finites (shouldn't occur, but be safe)
        s_norm[~torch.isfinite(s_norm)] = 0.0

        flats_norm.append(s_norm)
        sizes.append(s_norm.numel())

    total = int(sum(sizes))
    if total == 0:
        return {n: torch.zeros_like(per_weight_scores[n], dtype=torch.bool) for n in names}

    # 2) Decide how many weights to prune globally
    k = int(round(top_fraction * total))
    if min_at_least_one and k == 0 and top_fraction > 0.0:
        k = 1
    k = max(0, min(k, total))

    if k == 0:
        return {n: torch.zeros_like(per_weight_scores[n], dtype=torch.bool) for n in names}

    # 3) Concatenate normalized scores and take global Top-K
    concat_norm = torch.cat(flats_norm, dim=0)  # [total] on CPU
    sel = torch.topk(concat_norm, k, largest=True, sorted=False).indices  # [k]

    # 4) Map global indices back to per-layer masks
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes):
        mask_flat = torch.zeros(size, dtype=torch.bool)
        in_this = sel[(sel >= offset) & (sel < offset + size)] - offset
        if in_this.numel():
            mask_flat[in_this] = True

        S = per_weight_scores[name]
        masks[name] = mask_flat.view_as(S).to(S.device)

        pct = masks[name].float().mean().item() * 100.0
        print(f"[global-normalized] {name}: prune {pct:.2f}% of weights")

        offset += size

    return masks

# ---- You must provide this class from your codebase ----
# It should support: __init__(num_grads, fisher_block_size, num_weights, damp, device),
#   .add_grad(grad_flat: Tensor)   and   .fisher_diag() -> Tensor[num_weights]
# from your module import EmpiricalBlockFisherInverse

def _find_intermediate_weight_params(model: nn.Module) -> List[Tuple[int, str, torch.nn.Parameter]]:
    """
    Returns a list of (layer_index, full_param_name, param) for
    'vit.encoder.layer.{i}.intermediate.dense.weight' ordered by i.
    """
    hits = []
    for name, p in model.named_parameters():
        if name.endswith("encoder.layer.0.intermediate.dense.weight"):
            root = name.split("encoder.layer.0.intermediate.dense.weight")[0]
            # ensures we're on the 'vit.' path; but we'll just use suffix matching
        if ".intermediate.dense.weight" in name and "encoder.layer." in name:
            # Extract the {i}
            try:
                # name like: vit.encoder.layer.7.intermediate.dense.weight
                parts = name.split(".")
                # find "layer" and take the next index
                li = parts.index("layer")
                idx = int(parts[li + 1])
                hits.append((idx, name, p))
            except Exception:
                pass
    # sort by layer index
    hits.sort(key=lambda t: t[0])
    return hits

def _reduce_rows(scores: torch.Tensor) -> torch.Tensor:
    """Sum over columns -> per-row score."""
    return scores.sum(dim=1)

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

def rank_vit_intermediate_nodes_hooked_layer_wise(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    EmpiricalBlockFisherInverse,  # pass your class here
    num_grads: int = 512,
    fisher_block_size: int = 1024,
    switch_m=False,
    damp: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[Callable] = None,
    selective_pruning=False,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = lambda W, dW, F: (dW.abs() * W.abs()) + ((F if F is not None else 0) * (W * W).abs()),
    c=None,
    weight_based=True,
    sequential: bool = False,   # <<< NEW
) -> Dict[str, torch.Tensor]:

    if device is None:
        device = next(model.model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if forward_fn is None:
        forward_fn = _default_forward_fn
    model.model.eval()

    # 1) Discover target parameters (ordered by layer index)
    param_targets = _find_intermediate_weight_params(model.model)
    if not param_targets:
        raise RuntimeError("No encoder.layer.{i}.intermediate.dense.weight params found.")

    # Optional: precompute midlayer scores once (if used)
    midlayer_scores = None
    if selective_pruning:
        midlayer_scores = get_midlayaer_scoring(c)


    # ---------------- existing all-at-once path below (unchanged) ----------------
    print(f'device in rank_vit: {device}')
    num_layers = len(param_targets)
    print(f'num layers: {num_layers}')

    layer_grad_sums: List[torch.Tensor] = []
    fisher_invs = []
    for i, full_name, p in param_targets:
        w = p
        layer_grad_sums.append(torch.zeros_like(w, device=device))
        fisher_invs.append(
            EmpiricalBlockFisherInverse(
                num_grads=num_grads,
                fisher_block_size=fisher_block_size,
                num_weights=w.numel(),
                damp=damp,
                device=device,
                switch_m=switch_m,
            )
        )

    layer_grads: List[Optional[torch.Tensor]] = [None] * num_layers
    def make_hook(idx: int):
        def _hook(g: torch.Tensor):
            layer_grads[idx] = g
            layer_grad_sums[idx].add_(g)
        return _hook

    handles = []
    for i, full_name, p in param_targets:
        handles.append(p.register_hook(make_hook(i)))

    seen = 0
    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if counter > num_grads:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.model.zero_grad(set_to_none=True)
        logits = model.get_logits(pixel_values=inputs)[:, 0, :]
        loss = loss_fn(logits, targets)
        loss.backward()

        for i, g in enumerate(layer_grads):
            if g is not None:
                if counter > num_grads:
                    break
                fisher_invs[i].add_grad(g.flatten())
        layer_grads = [None] * num_layers

        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")
            sys.stdout.flush()
        seen += 1

    for h in handles:
        h.remove()
    if seen == 0:
        raise ValueError("Dataloader yielded zero batches; cannot compute scores.")

    layer_gradients = [g_sum / float(seen) for g_sum in layer_grad_sums]
    layer_fisher_diags = [
        fisher_inv.fisher_diag().reshape_as(grad) for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]

    from collections import OrderedDict
    row_scores: Dict[str, torch.Tensor] = OrderedDict()
    if selective_pruning:
        midlayer_scores = get_midlayaer_scoring(c)
        for (i, full_name, p), dW, F, m in zip(param_targets, layer_gradients, layer_fisher_diags, midlayer_scores):
            W = p.data
            per_weight = scoring_fn(W, dW, F, m)
            row_scores[full_name] = per_weight
    else:
        for (i, full_name, p), dW, F in zip(param_targets, layer_gradients, layer_fisher_diags):
            W = p.data
            per_weight = scoring_fn(W, dW, F, None)
            row_scores[full_name] = per_weight
    return row_scores

def rank_vit_intermediate_nodes_hooked(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    EmpiricalBlockFisherInverse,  # pass your class here
    num_grads: int = 512,
    fisher_block_size: int = 1024,
    switch_m = False,
    damp: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[Callable] = None,
    selective_pruning = False,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = lambda W, dW, F: (dW.abs() * W.abs()) + ((F if F is not None else 0) * (W * W).abs()),
    c = None,
    weight_based = False,
    cutoff = 10_000
) -> Dict[str, torch.Tensor]:
    """
    Reproduces your hook-based gradient mean + block-Fisher diag collection, then
    computes a per-row score for each ViT intermediate dense layer.

    Returns:
        {param_full_name: row_scores} with row_scores.shape == [out_features]
    """
    if device is None:
        device = next(model.model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if forward_fn is None:
        forward_fn = _default_forward_fn
    print(f'device in rank_vit: {device}')
    model.model.eval()  # no dropout etc, but we need grads

    # 1) Discover target parameters (ordered by layer index)
    param_targets = _find_intermediate_weight_params(model.model)
    if not param_targets:
        raise RuntimeError("No encoder.layer.{i}.intermediate.dense.weight params found.")

    num_layers = len(param_targets)
    print(f'num layers: {num_layers}')

    # 2) Allocate accumulators + per-layer Fisher inverse objects
    layer_grad_sums: List[torch.Tensor] = []
    fisher_invs = []
    for i, full_name, p in param_targets:
        w = p  # parameter tensor
        layer_grad_sums.append(torch.zeros_like(w, device=device))
        fisher_invs.append(
            EmpiricalBlockFisherInverse(
                num_grads=num_grads,
                fisher_block_size=fisher_block_size,
                num_weights=w.numel(),
                damp=damp,
                device=device,
                switch_m = switch_m
            )
        )

    # 3) Register hooks to capture per-batch grads at each weight tensor
    layer_grads: List[Optional[torch.Tensor]] = [None] * num_layers
    def make_hook(idx: int):
        def _hook(g: torch.Tensor):
            # store raw grad for this batch AND accumulate for mean
            layer_grads[idx] = g
            layer_grad_sums[idx].add_(g)
        return _hook

    handles = []
    name_to_idx = {full_name: i for i, full_name, _ in param_targets}
    for i, full_name, p in param_targets:
        handles.append(p.register_hook(make_hook(i)))

    # 4) Iterate batches (up to num_grads), run forward/backward, update Fisher inverses
    seen = 0
    for counter, (inputs, targets) in enumerate(dataloader, start=1):
        if counter > cutoff:
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
                if counter > num_grads:
                    break
                g = g.detach()
                g_flat = g.flatten()
                fisher_invs[i].add_grad(g_flat)  # Update Fisher inverse

        # Reset scratch buffer
        layer_grads = [None] * num_layers

        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")
            sys.stdout.flush()
        seen+=1
    
    for h in handles:
        h.remove()

    if seen == 0:
        raise ValueError("Dataloader yielded zero batches; cannot compute scores.")

    # 5) Mean gradients and Fisher-diagonal per weight
    layer_gradients = [g_sum / float(seen) for g_sum in layer_grad_sums]
    layer_fisher_diags = [
        fisher_inv.fisher_diag().reshape_as(grad) for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]


    # 6) Per-weight -> per-row scoring
    row_scores: Dict[str, torch.Tensor] = OrderedDict()
    
    if selective_pruning:
        midlayer_scores = get_midlayaer_scoring(c)
        for (i, full_name, p), dW, F, midlayer_score in zip(param_targets, layer_gradients, layer_fisher_diags, midlayer_scores):
            W = p.data
            per_weight = scoring_fn(W, dW, F, midlayer_score)  # same shape as W
            row_scores[full_name] = midlayer_score
            # _reduce_rows(per_weight).detach()
    else: 
        for (i, full_name, p), dW, F in zip(param_targets, layer_gradients, layer_fisher_diags):
            W = p.data
            per_weight = scoring_fn(W, dW, F, None)  # same shape as W
            if not weight_based:
                row_scores[full_name] = _reduce_rows(per_weight).detach()
            else:
                row_scores[full_name] = per_weight
    return row_scores

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
        # avg_loss = loss_sum / len(val_loaders)
        accuracy = correct / total if total > 0 else 0
        return {"combined": {"accuracy": accuracy}}





############################################################################
#                           Compensation code                              #
#############################################################################

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

@torch.no_grad()
def _module_name_from_param(pname: str) -> str:
    assert pname.endswith(".weight")
    return pname[:-7]

def _layer_weight_by_name(model: nn.Module, full_param_name: str) -> nn.Parameter:
    params = dict(model.model.named_parameters())
    if full_param_name not in params:
        raise KeyError(f"Parameter {full_param_name} not found in model.")
    return params[full_param_name]

def _named_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return {n: m for n, m in model.model.named_modules()}

def _extract_block(Ainv_full: torch.Tensor, b: int, B_eff: int) -> torch.Tensor:
    # Ainv_full: (num_blocks, B, B). Return a *view* of the (B_eff x B_eff) top-left submatrix
    return Ainv_full[b, :B_eff, :B_eff]

def _flatten_row_mask(row_mask: torch.Tensor, in_features: int) -> torch.Tensor:
    """
    row_mask: [out] boolean -> flat mask over [out*in] in row-major order.
    """
    return row_mask.repeat_interleave(in_features)

def _indices_in_block(global_start: int, global_end: int, sel_flat_mask: torch.Tensor) -> torch.Tensor:
    """
    Return local indices (0..B_eff-1) within a flattened interval [global_start, global_end)
    where sel_flat_mask is True.
    """
    block_mask = sel_flat_mask[global_start:global_end]  # [B_eff]
    return torch.nonzero(block_mask, as_tuple=False).view(-1)  # local indices

@torch.no_grad()
def cap_compensate_from_saved_masks_batched_with_fisher(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],                         # { "<module>.weight": bool[out_features] }
    fisher_invs: Dict[str, "EmpiricalBlockFisherInverse"],  # same keys as masks
    *,
    zero_bias: bool = True,
    eps: float = 1e-9,
) -> None:
    """
    Apply CAP-like multi-constraint OBS compensation using your EmpiricalBlockFisherInverse,
    eliminating ALL rows indicated by `masks` in one batched step per layer & per block.

    Requirements:
      • fisher_invs[pname].d == weight.numel()
      • fisher_invs[pname].B is the block size used to form F_inv
      • fisher_invs[pname].F_inv is the inverse Fisher blocks (num_blocks x B x B)

    This function:
      1) Flattens layer weights
      2) For each block, builds S = {all positions belonging to masked rows in that block}
      3) Δθ_block = -Ainv[:,S] @ solve(Ainv[S,S], θ[S])
      4) θ[S] = 0; write back to layer weights
      5) (Optional but done here) Ainv <- Ainv - Ainv[:,S] @ solve(Ainv[S,S], Ainv[S,:])
         so the inverse reflects eliminated params (useful if you prune more later)
    """
    name_to_module = _named_modules(model)

    for p_name, row_mask in masks.items():
        print(f"p_name is {p_name}")
        # retrieve weight param & sanity-check shapes
        W = _layer_weight_by_name(model, p_name)
        if W.grad is not None:
            # not strictly necessary, but keeps grads clean
            W.grad = None

        out_features, in_features = W.shape
        flat = W.data.reshape(-1)  # θ
        d = flat.numel()

        # fisher inverse object for this layer
        if p_name not in fisher_invs:
            raise KeyError(f"No fisher inverse provided for {p_name}")
        Fobj = fisher_invs[p_name]
        B = Fobj.B
        num_blocks = Fobj.num_blocks

        if Fobj.d != d:
            raise ValueError(f"Fisher inverse d={Fobj.d} does not match weight size {d} for {p_name}")

        # Build flattened selection mask for ALL weights in masked rows
        # row_mask: [out]; we need flat mask over [out*in] marking all columns in those rows
        sel_flat = _flatten_row_mask(row_mask.to(dtype=torch.bool, device=flat.device), in_features)

        # Work block-by-block (vectorized inside each block)
        # We'll *also* downdate F_inv to reflect removal (so you can run more rounds if desired).
        for b in range(num_blocks):
            g_start = b * B
            g_end   = min((b + 1) * B, d)
            B_eff   = g_end - g_start
            if B_eff <= 0:
                continue

            # Local inverse block (view) and local theta segment (view)
            Ainv_b = _extract_block(Fobj.F_inv, b, B_eff)  # (B_eff, B_eff) view
            theta_b = flat[g_start:g_end]                   # (B_eff,) view

            # Indices to eliminate in THIS block (local indexing 0..B_eff-1)
            S_loc = _indices_in_block(g_start, g_end, sel_flat)
            if S_loc.numel() == 0:
                continue

            # Build submatrices / subvectors
            # A_ss is (k,k), A_sall is (k,B_eff), A_alls is (B_eff,k)
            # We'll use solves instead of explicit inverses for stability.
            A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc).clone()
            # regularize (tiny) to avoid singular solves
            A_ss.diagonal().add_(eps)

            theta_s = theta_b.index_select(0, S_loc)                 # (k,)
            A_alls  = Ainv_b.index_select(1, S_loc)                  # (B_eff, k)
            A_sall  = Ainv_b.index_select(0, S_loc)                  # (k, B_eff)

            # ---- Compensation: Δθ_block = - Ainv[:,S] @ solve(Ainv[S,S], θ[S]) ----
            y = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)   # (k,)
            delta = - A_alls @ y                                            # (B_eff,)

            if b % 100 == 0:
                print(f"delta is {delta}")
            theta_b.add_(delta)
            theta_b.index_fill_(0, S_loc, 0.0)                               # exact zero

            # ---- Downdate inverse (Woodbury for removing S) ----
            # Ainv <- Ainv - Ainv[:,S] @ solve(Ainv[S,S], Ainv[S,:])
            K = torch.linalg.solve(A_ss, A_sall)                             # (k, B_eff)
            Ainv_b.sub_(A_alls @ K)                                          # in-place

        # write back reshaped weights
        W.data.copy_(flat.view_as(W))

        # zero bias rows if requested
        if zero_bias:
            mod_name = _module_name_from_param(p_name)
            mod = name_to_module.get(mod_name, None)
            if isinstance(mod, nn.Linear) and (mod.bias is not None):
                mod.bias.data[row_mask] = 0.0


################################################################################################################################################################

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Iterable, Union
from collections import OrderedDict

# ---------- 1) Permutation utilities ----------

@torch.no_grad()
def build_colmajor_permutation(out_features: int, in_features: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      perm  : LongTensor[d]  mapping row-major index -> col-major index
      pinv  : LongTensor[d]  inverse permutation (col-major -> row-major)
    Row-major k = r*in + c ;  Col-major k' = c*out + r
    """
    device = device or torch.device("cpu")
    r = torch.arange(out_features, device=device)
    c = torch.arange(in_features, device=device)
    R, C = torch.meshgrid(r, c, indexing="ij")  # R: [out,in], C: [out,in]
    k_row = (R * in_features + C).reshape(-1)   # [d]
    k_col = (C * out_features + R).reshape(-1)  # [d]
    # We want perm such that: k_col = perm[k_row]  ==> perm[k_row] = k_col
    perm = torch.empty_like(k_row)
    perm[k_row] = k_col
    # inverse
    pinv = torch.empty_like(perm)
    pinv[perm] = torch.arange(perm.numel(), device=device)
    return perm, pinv

def _param_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return {n: m for n, m in model.model.named_modules()}

def _param_name_to_module_name(pname: str) -> str:
    assert pname.endswith(".weight")
    return pname[:-7]

# ---------- 2) Build Fisher inverses in column-major space (wrapper) ----------

@torch.no_grad()
def build_fisher_invs_colmajor(
    model: nn.Module,
    target_param_names: List[str],
    EmpiricalBlockFisherInverse,          # <-- your class (pass the symbol)
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_grads: int = 256,
    fisher_block_size: int = 64,
    damp: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,
    forward_fn: Optional[callable] = None,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Builds one EmpiricalBlockFisherInverse per targeted parameter, but **in column-major coordinate order**.
    We do this by permuting each collected grad g_flat(row-major) -> g_flat_col = g_flat[perm].

    Returns:
      fisher_invs : {pname: EmpiricalBlockFisherInverse}  (F_inv lives in col-major space)
      perms       : {pname: LongTensor[d]}                (row-major -> col-major)
      pinvs       : {pname: LongTensor[d]}                (col-major -> row-major)
    """
    model.model.eval()
    if device is None:
        device = next(model.model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()


    # discover tensors and allocate fisher
    params = dict(model.model.named_parameters())
    fisher_invs: Dict[str, object] = OrderedDict()
    perms: Dict[str, torch.Tensor] = {}
    pinvs: Dict[str, torch.Tensor] = {}

    for pname in target_param_names:
        W = params[pname]
        out, in_ = W.shape
        d = out * in_
        perm, pinv = build_colmajor_permutation(out, in_, device=W.device)
        perms[pname], pinvs[pname] = perm, pinv
        fisher_invs[pname] = EmpiricalBlockFisherInverse(
            num_grads=num_grads,
            fisher_block_size=fisher_block_size,
            num_weights=d,
            damp=damp,
            device=W.device,
        )

    # register hooks to capture grads
    name_to_index = {pname: i for i, pname in enumerate(target_param_names)}
    scratch: List[Optional[torch.Tensor]] = [None] * len(target_param_names)

    def make_hook(idx: int):
        def _hook(g: torch.Tensor):
            scratch[idx] = g
        return _hook

    handles = []
    for i, pname in enumerate(target_param_names):
        p = params[pname]
        if not isinstance(p, torch.nn.Parameter):
            raise TypeError(f"{pname} is not an nn.Parameter (got {type(p)})")
        if not p.requires_grad:
            p.requires_grad_(True) 
        handles.append(p.register_hook(make_hook(i)))

    # iterate batches
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

        # push grads into col-major fisher inverses
        for i, pname in enumerate(target_param_names):
            g = scratch[i]
            if g is None:
                continue
            g_rm = g.reshape(-1)
            g_cm = g_rm.index_select(0, perms[pname])   # permute to col-major order
            fisher_invs[pname].add_grad(g_cm)           # your class updates in this order
        scratch = [None] * len(target_param_names)
        seen += 1

    for h in handles:
        h.remove()

    return fisher_invs, perms, pinvs


@torch.no_grad()
def build_fisher_invs_colmajor_for_layer(
    model: nn.Module,
    target_param_names: List[str],
    EmpiricalBlockFisherInverse,          # <-- your class (pass the symbol)
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_grads: int = 256,
    fisher_block_size: int = 64,
    damp: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,
    layer_name = None,
    index = None,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Builds one EmpiricalBlockFisherInverse per targeted parameter, but **in column-major coordinate order**.
    We do this by permuting each collected grad g_flat(row-major) -> g_flat_col = g_flat[perm].

    Returns:
      fisher_invs : {pname: EmpiricalBlockFisherInverse}  (F_inv lives in col-major space)
      perms       : {pname: LongTensor[d]}                (row-major -> col-major)
      pinvs       : {pname: LongTensor[d]}                (col-major -> row-major)
    """
    model.model.eval()
    if device is None:
        device = next(model.model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()


    # discover tensors and allocate fisher
    params = dict(model.model.named_parameters())
    # fisher_invs: Dict[str, object] = OrderedDict()
    # perms: Dict[str, torch.Tensor] = {}
    # pinvs: Dict[str, torch.Tensor] = {}
    pname = layer_name
    
    W = params[pname]
    out, in_ = W.shape
    d = out * in_
    perm, pinv = build_colmajor_permutation(out, in_, device=W.device)
    # perms[pname], pinvs[pname] = perm, pinv
    fisher_invs= EmpiricalBlockFisherInverse(
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        num_weights=d,
        damp=damp,
        device=W.device,
    )

    # register hooks to capture grads
    scratch: List[Optional[torch.Tensor]] = [None] * len(target_param_names)

    def make_hook(idx: int):
        def _hook(g: torch.Tensor):
            scratch[idx] = g
        return _hook

    handles = []
   
    p = params[pname]
    if not isinstance(p, torch.nn.Parameter):
        raise TypeError(f"{pname} is not an nn.Parameter (got {type(p)})")
    if not p.requires_grad:
        p.requires_grad_(True) 
    handles.append(p.register_hook(make_hook(index)))

    # iterate batches
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

        # push grads into col-major fisher inverses
        
        g = scratch[index]
        if g is None:
            continue
        g_rm = g.reshape(-1)
        g_cm = g_rm.index_select(0, perm)   # permute to col-major order
        fisher_invs.add_grad(g_cm)           # your class updates in this order
        scratch = [None] * len(target_param_names)
        seen += 1

    for h in handles:
        h.remove()

    return fisher_invs, perm, pinv


import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable, Optional, Union


# ---------- 3) Batched CAP/OBS compensation in column-major space ----------
def project_orthogonal(delta_w, grad):
    """
    Projects the weight update 'delta_w' onto the space orthogonal to the gradient 'grad'.

    Args:
        delta_w (torch.Tensor): The proposed weight update tensor.
        grad (torch.Tensor): The gradient tensor.

    Returns:
        torch.Tensor: The projected weight update tensor.
    """
    # Ensure inputs are tensors and have the same shape
    assert isinstance(delta_w, torch.Tensor) and isinstance(grad, torch.Tensor)
    # assert delta_w.shape == grad.shape, "Input tensors must have the same shape."

    # --- Step 1: Calculate the dot products ---
    # (delta_w . g)
    # dot_product = torch.sum(delta_w * grad)
    
    # (g . g) or ||g||^2
    # grad_norm_sq = torch.sum(grad * grad)

    # Avoid division by zero if the gradient is zero
    # if grad_norm_sq == 0:
    #     return delta_w

    # --- Step 2: Calculate the projection coefficient ---
    # (delta_w . g) / ||g||^2
    # coeff = dot_product / grad_norm_sq
    grad  = delta_w @ grad
    # --- Step 3: Subtract the parallel component ---
    # delta_w - coeff * g
    projected_delta_w = delta_w - grad

    return projected_delta_w

@torch.no_grad()
def cap_compensate_from_masks_batched_colmajor(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],                    # { "<module>.weight": bool[out] }
    fisher_invs_col: Dict[str, object],                # { "<module>.weight": your EmpiricalBlockFisherInverse }
    perms: Dict[str, torch.Tensor],                    # { "<module>.weight": perm (row->col) }
    pinvs: Dict[str, torch.Tensor],                    # { "<module>.weight": pinv (col->row) }
    *,
    eps: float = 1e-9,
    zero_bias: bool = True,
    forget_grads = None
) -> None:
    """
    Multi-constraint OBS per block (Woodbury), executed in **column-major** coordinates,
    then unpermuted back to the model's parameter layout.
    """
    params = dict(model.model.named_parameters())
    modules = _param_modules(model)
    for pname, row_mask in masks.items():
        if pname not in params or pname not in fisher_invs_col:
            continue
        W = params[pname]                       # [out, in]
        out, in_ = W.shape
        d = out * in_

        perm  = perms[pname].to(W.device)
        pinv  = pinvs[pname].to(W.device)
        Fobj  = fisher_invs_col[pname]

        # flatten weights in row-major then permute into col-major
        theta_rm = W.data.reshape(-1)
        theta_cloned = theta_rm.clone()
        theta_cm = theta_rm.index_select(0, perm).clone()   

        # build selection set in col-major coordinates
        sel_rows = row_mask.to(W.device)
        
        # If neuron (not individual weight this scales it up)                    
        sel_flat_rm = sel_rows.repeat_interleave(in_)       
        
        S_rm_idx = torch.nonzero(sel_flat_rm, as_tuple=False).view(-1)
        if S_rm_idx.numel() == 0:
            # nothing to do for this layer
            continue
        S_cm_idx = perm.index_select(0, S_rm_idx)           

        # 3) per-block batched OBS
        B = Fobj.B
        num_blocks = Fobj.num_blocks
        Ainv_full = Fobj.F_inv                              
        for b in range(num_blocks):
            g_start = b * B
            g_end   = min((b + 1) * B, d)
            B_eff   = g_end - g_start
            if B_eff <= 0:
                continue

            # local selection inside this block (col-major)
            mask_block = (S_cm_idx >= g_start) & (S_cm_idx < g_end)
            S_loc = (S_cm_idx[mask_block] - g_start).to(torch.long)  # [k]
            if S_loc.numel() == 0:
                continue

            Ainv_b = Ainv_full[b, :B_eff, :B_eff]            # view (B_eff,B_eff)
            theta_b = theta_cm[g_start:g_end]                # view (B_eff,)

            # Submatrices
            A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc).clone()
            A_ss.diagonal().add_(eps)
            theta_s = theta_b.index_select(0, S_loc)                 # (k,)
            A_alls  = Ainv_b.index_select(1, S_loc)                  # (B_eff, k)
            A_sall  = Ainv_b.index_select(0, S_loc)                  # (k, B_eff)

            # delta
            y = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)   # (k,)
            delta = - A_alls @ y

            if b % 1000 == 0:
                print(f"The `mask_block` mask is: {mask_block}")
                print(f"The resulting `S_loc` tensor is: {S_loc}")
                count_from_sum = mask_block.sum()
                print(f"Count from sum(): {count_from_sum.item()}")
                print(f"delta is {delta}")
                sum_of_sq = (delta ** 2).sum()
                print(f"The sum of squares  for delta is is: {sum_of_sq.item()}")


            if forget_grads is not None:
                Q_cpu_list = forget_grads.get(pname, None)
                if Q_cpu_list is not None:
                    Q_cpu = Q_cpu_list[g_start:g_end]                     # CPU (B_eff, r)
                    Q = Q_cpu.to(delta.device, dtype=delta.dtype)  # small move to GPU 
                    delta = project_orthogonal(delta, Q)
            
            # if gf_flat_cm_block is not None:
            #     g_cm_full = gf_flat_cm_block[pname]                 # col-major flat grad for this param
            #     gf_flat_cm_block_tensor = g_cm_full[g_start:g_end] # <-- this is what you asked for
            #     den = gf_flat_cm_block_tensor.dot(gf_flat_cm_block_tensor).clamp_min(1e-12)
            #     delta -= (delta.dot(gf_flat_cm_block_tensor) / den) * gf_flat_cm_block_tensor

            theta_b.add_(delta)
            theta_b.index_fill_(0, S_loc, 0.0)

            # Ainv <- Ainv - Ainv[:,S] @ solve(Ainv[S,S], Ainv[S,:])
            K = torch.linalg.solve(A_ss, A_sall)                             # (k,B_eff)
            Ainv_b.sub_(A_alls @ K)

        # 4) unpermute back to row-major and write weights
        theta_rm_updated = theta_cm.index_select(0, pinv)
        theta_cloned
        mse = F.mse_loss(theta_cloned, theta_rm_updated)
        print(f"The Mean Squared Difference is: {mse.item()}")
        W.data.copy_(theta_rm_updated.view_as(W))

        # 5) zero biases on the pruned rows (optional)
        if zero_bias:
            mname = _param_name_to_module_name(pname)
            mod = modules.get(mname, None)
            if isinstance(mod, nn.Linear) and (mod.bias is not None):
                mod.bias.data[row_mask] = 0.0




@torch.no_grad()
def cap_compensate_from_masks_batched_colmajor_for_layer(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],                    # { "<module>.weight": bool[out] }
    fisher_invs,                # { "<module>.weight": your EmpiricalBlockFisherInverse }
    perm,                    # { "<module>.weight": perm (row->col) }
    pinv ,                    # { "<module>.weight": pinv (col->row) }
    *,
    eps: float = 1e-9,
    zero_bias: bool = False,
    forget_grads = None,
    layer_name = None,
    basis = None
) -> None:
    """
    Multi-constraint OBS per block (Woodbury), executed in **column-major** coordinates,
    then unpermuted back to the model's parameter layout.
    """

    
    params = dict(model.model.named_parameters())
    modules = _param_modules(model)
    pname = layer_name
    row_mask = masks[pname]


    W = params[pname]                       # [out, in]
    out, in_ = W.shape
    d = out * in_

    perm  = perm.to(W.device)
    pinv  = pinv.to(W.device)
    Fobj  = fisher_invs

    # flatten weights in row-major then permute into col-major
    theta_rm = W.data.reshape(-1)
    theta_cm = theta_rm.index_select(0, perm).clone()   

    # build selection set in col-major coordinates
    sel_rows = row_mask.to(W.device)
    
    # If neuron (not individual weight this scales it up)                    
    sel_flat_rm = sel_rows.repeat_interleave(in_)       
    
    S_rm_idx = torch.nonzero(sel_flat_rm, as_tuple=False).view(-1)
    if S_rm_idx.numel() == 0:
        # nothing to do for this layer
        return
    S_cm_idx = perm.index_select(0, S_rm_idx)           

    # 3) per-block batched OBS
    B = Fobj.B
    num_blocks = Fobj.num_blocks
    Ainv_full = Fobj.F_inv                              
    for b in range(num_blocks):
        
        g_start = b * B
        g_end   = min((b + 1) * B, d)
        B_eff   = g_end - g_start
        if B_eff <= 0:
            continue
        S_cm = torch.arange(g_start, g_end, device=pinv.device)
        S_rm = pinv.index_select(0, S_cm)
        
        U  = basis.index_select(0, S_rm)  # [Bblk, r]

        R = basis[:, S_rm] 
        G_block = R.T @ R 
        U = B.index_select(0, S_rm)
        # local selection inside this block (col-major)
        mask_block = (S_cm_idx >= g_start) & (S_cm_idx < g_end)
        S_loc = (S_cm_idx[mask_block] - g_start).to(torch.long)  # [k]
        if S_loc.numel() == 0:
            continue

        Ainv_b = Ainv_full[b, :B_eff, :B_eff]            # view (B_eff,B_eff)
        theta_b = theta_cm[g_start:g_end]                # view (B_eff,)

        # Submatrices
        A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc).clone()
        A_ss.diagonal().add_(eps)
        theta_s = theta_b.index_select(0, S_loc)                 # (k,)
        A_alls  = Ainv_b.index_select(1, S_loc)                  # (B_eff, k)
        A_sall  = Ainv_b.index_select(0, S_loc)                  # (k, B_eff)

        # delta
        y = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)   # (k,)
        delta = - A_alls @ y

        if b % 1000 == 0:
            print(f"The `mask_block` mask is: {mask_block}")
            print(f"The resulting `S_loc` tensor is: {S_loc}")
            count_from_sum = mask_block.sum()
            print(f"Count from sum(): {count_from_sum.item()}")
            print(f"delta is {delta}")
            sum_of_sq = (delta ** 2).sum()
            print(f"The sum of squares  for delta is is: {sum_of_sq.item()}")


        if basis is not None:
            Q_cpu = basis[g_start:g_end]                     # CPU (B_eff, r)
            Q_cpu = Q_cpu @ Q_cpu.T
            if b % 1000 == 0:
                print(f"shape of q_cpu = {Q_cpu.shape}")
            Q = Q_cpu.to(delta.device, dtype=delta.dtype)  # small move to GPU 
            delta = project_orthogonal(delta, Q)
        
        # if gf_flat_cm_block is not None:
        #     g_cm_full = gf_flat_cm_block[pname]                 # col-major flat grad for this param
        #     gf_flat_cm_block_tensor = g_cm_full[g_start:g_end] # <-- this is what you asked for
        #     den = gf_flat_cm_block_tensor.dot(gf_flat_cm_block_tensor).clamp_min(1e-12)
        #     delta -= (delta.dot(gf_flat_cm_block_tensor) / den) * gf_flat_cm_block_tensor

        theta_b.add_(delta)
        theta_b.index_fill_(0, S_loc, 0.0)

        # Ainv <- Ainv - Ainv[:,S] @ solve(Ainv[S,S], Ainv[S,:])
        K = torch.linalg.solve(A_ss, A_sall)                             # (k,B_eff)
        Ainv_b.sub_(A_alls @ K)

    # 4) unpermute back to row-major and write weights
    theta_rm_updated = theta_cm.index_select(0, pinv)
    W.data.copy_(theta_rm_updated.view_as(W))

    # 5) zero biases on the pruned rows (optional)
    if zero_bias:
        mname = _param_name_to_module_name(pname)
        mod = modules.get(mname, None)
        if isinstance(mod, nn.Linear) and (mod.bias is not None):
            mod.bias.data[row_mask] = 0.0

################################################################################################################################################################



def get_midlayaer_scoring(c, ):
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
    )

    cripple_out = get_midlayer_data(opt, c.cripple,
                        c.collection_sample_size, c.attn_mode)
    focus_out = get_midlayer_data(opt, c.focus,
                        c.collection_sample_size, c.attn_mode)
    act_subset = c.scoring_normalization
    ff_focus_data   = focus_out.mlp[act_subset]
    ff_cripple_data = cripple_out.mlp[act_subset]
    ff_scoring_fn = score_indices_by(c.ff_scoring)
    ff_eps     =  c.ff_eps

    ff_scores = ff_scoring_fn(opt, ff_focus_data, ff_cripple_data, ff_eps)

    
    print("printing ff_scores")
    print(ff_scores)
    print(len(ff_scores))
    print(len(ff_scores[0]))
    return ff_scores

def count_zero_weights_in_encoder(model) -> dict:
    """
    Calculates the number of weights that are exactly 0.0 in the
    intermediate dense layer of each encoder block for a given ViT model.

    Args:
        model_name: The name of the Vision Transformer model on the Hugging Face Hub.

    Returns:
        A dictionary where keys are the layer indices and values are the
        counts of zero-valued weights for that layer's intermediate dense layer.
    """
    # Initialize a dictionary to store the results
    zero_weight_counts = {}

    try:

        # Set the model to evaluation mode
        # This is good practice as it disables dropout, etc.
        model.eval()

        # No need to calculate gradients for this task
        with torch.no_grad():
            # Iterate through each layer in the encoder
            for i, layer in enumerate(model.encoder.layer):
                # Access the specific weight tensor for the intermediate dense layer
                weights = layer.intermediate.dense.weight

                # Count the number of elements that are exactly 0.0
                # (weights == 0.0) creates a boolean tensor
                # .sum() counts the number of True values
                # .item() extracts the scalar value from the resulting tensor
                num_zeros = (weights == 0.0).sum().item()

                # Store the result in our dictionary
                zero_weight_counts[f"encoder.layer.{i}"] = num_zeros

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

    return zero_weight_counts

@torch.no_grad()
def apply_weight_masks_(model, weight_masks):
    params = dict(model.named_parameters())
    for name, m in weight_masks.items():
        params[name].data[m] = 0.0

def run_weight_based_masking(c, hessian_coefficient = 0.5, retain_dataloader = False, retain_coefficient = 1, 
                             forget_frac = 0.02, switch_m = False, selective_pruning = False, sub_retain = False, 
                            eval_pruned = False, weight_based = True, damp = 1e-3, num_grads = 480, fisher_block_size = 50,
                            normalize_by_layer = False, signs = "ABS"):
    print("RUNNING run_weight_based_masking --------------=============")
    print("C is:")
    print(c)

    print(f"""Hessian coefficient {hessian_coefficient}, retain dataloader {retain_dataloader}, forget frac: {forget_frac}, 
          switch_m: {switch_m}, selective pruning: {selective_pruning}, sub retain: {sub_retain}, damp: {damp},
          signs {signs}""")


    def my_scoring(W, dW, F, midlayer_scores, hessian_coefficient = hessian_coefficient):
        # Pure Fisher (OBD-style) with tiny epsilon
        if selective_pruning:
            return midlayer_scores
        else: 
            if signs == "ABS":
                return (abs(dW * W) +  hessian_coefficient * abs(F * W.pow(2)))
            elif signs == "All Neg":
                return (-(dW * W) +  hessian_coefficient * -(F * W.pow(2)))
            elif signs == "Second Order Neg":
                return ((dW * W) +  hessian_coefficient * -(F * W.pow(2)))
            elif signs == "First Order Neg":
                return (-(dW * W) +  hessian_coefficient * (F * W.pow(2)))
            elif signs == "All positive":
                return ((dW * W) +  hessian_coefficient * (F * W.pow(2)))
            

    retain_bool = retain_dataloader

    custom_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),                 # match TFDS 3-ch decode
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4,4/3),
                                    interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3),                 # [-1,1]
    ])

    bird_labels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
    bird_eval_first_n = 600
    train_loader = get_VIT_dataloder("imagenet-1k", batch_size = 1, custom_transforms=custom_transforms)
    dataset = train_loader.dataset
    bird_subset_eval_loaders = build_bird_eval_loaders(dataset, bird_labels, bird_eval_first_n, 1, transform=custom_transforms)
    cripple_dataloader = bird_subset_eval_loaders.get("birds_first_600_per_label")
    

    
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
    )

    # if layer_wise_rank:
    #     row_scores = rank_vit_intermediate_nodes_hooked_layer_wise(
    #         opt,
    #         dataloader=cripple_dataloader,
    #         EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
    #         num_grads=num_grads,
    #         fisher_block_size=fisher_block_size,
    #         damp=damp,
    #         scoring_fn=my_scoring,
    #         switch_m= switch_m,
    #         weight_based= weight_based,
    #         selective_pruning = selective_pruning,
    #         c = c,
    #         sequential = True
    #     )
    # else:
    print("I AM HERE (COMPUTING ROW SCORES FOR FORGET)")
    sys.stdout.flush()
    row_scores = rank_vit_intermediate_nodes_hooked(
        opt,
        dataloader=cripple_dataloader,
        EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        damp=damp,
        scoring_fn=my_scoring,
        switch_m= switch_m,
        weight_based= weight_based,
        selective_pruning = selective_pruning,
        cutoff = 10_000,
        c = c,      # plug your scorer here

    )

    print("Generating masks:")
    
    
    if retain_dataloader:
        print("I AM HERE - COMPUTING ROW SCORES for RETAIN")
        sys.stdout.flush()
        retain_dl = bird_subset_eval_loaders.get("dataset_without_first_n_bird_images")
        row_scores_retain = rank_vit_intermediate_nodes_hooked(
            opt,
            dataloader=retain_dl,
            EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
            num_grads=num_grads,
            fisher_block_size=fisher_block_size,
            weight_based= weight_based,
            damp=damp,
            scoring_fn=my_scoring, 
            switch_m= switch_m,       # plug your scorer here
            cutoff = 10_000
        )
        new_row_scores: Dict[str, torch.Tensor] = OrderedDict()
        for k, v in row_scores.items():
            if sub_retain:
                new_row_scores[k] = v - (retain_coefficient * row_scores_retain[k])
            else:
                new_row_scores[k] = v / (retain_coefficient * row_scores_retain[k] + 1e-12)
        row_scores = new_row_scores

    if not weight_based:
        masks = build_masks_global_fraction(row_scores, top_fraction=forget_frac) 
    else:
        if normalize_by_layer:
            masks = build_weight_masks_global_fraction_normalized(row_scores, top_fraction=forget_frac)
        else:
            masks = build_weight_masks_global_fraction(row_scores, top_fraction=forget_frac)   
    # masks = build_masks_per_param(row_scores, top_fraction=forget_frac)

    print("Masks:")
    print(masks)
    os.makedirs("./outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    res = {}
    pt_path = f"./outputs/vit_intermediate_row_masks__retaindl_{retain_bool}_retaincoefficient_{retain_coefficient}_-{stamp}_forgetfrac_{forget_frac}.pt"
    torch.save({k: v.detach().cpu() for k, v in masks.items()}, pt_path)
    validation_loader = get_VIT_dataloder("imagenet-1k", is_validation = True, batch_size = 64, custom_transforms=custom_transforms)
    
    bird_subset_eval_loaders = build_bird_eval_loaders(dataset, bird_labels, bird_eval_first_n, 64, )
    forget_dataloader_64_bs = bird_subset_eval_loaders.get("birds_first_600_per_label")

    if eval_pruned: 
        tmp_opt = copy.deepcopy(opt)
        device = next(opt.model.parameters()).device
        tmp_opt.model.to(device)
        if not weight_based:
            apply_masks_by_param_name_(tmp_opt.model, masks)
        else:
            apply_weight_masks_(tmp_opt.model, masks)
        with torch.no_grad():
            forget_res = evaluate(tmp_opt, forget_dataloader_64_bs, device)
            forget_acc  = forget_res["combined"]["accuracy"]
            print(f" ----- debug - model performance after pruning  -> forget_acc: {forget_acc}")

            validation_res = evaluate(opt, validation_loader, device)
            validation_acc  = validation_res["combined"]["accuracy"]
            print(f" ----- debug - model performance after pruning  -> validation_acc: {validation_acc}")
              
    
        return(pt_path, (forget_acc, validation_acc))
    return pt_path

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
def _rowmask_to_colmajor_indices(row_mask: torch.Tensor, in_features: int, perm: torch.Tensor) -> torch.Tensor:
    """
    row_mask: [out] boolean – which output neurons are pruned
    in_features: int
    perm: [out*in] LongTensor mapping row-major -> col-major
    returns: 1-D LongTensor of *col-major* flat indices to eliminate
    """
    sel_flat_rm = row_mask.to(torch.bool).repeat_interleave(in_features)         # [out*in] row-major
    S_rm = sel_flat_rm.nonzero(as_tuple=False).view(-1)
    if S_rm.numel() == 0:
        return S_rm
    return perm.index_select(0, S_rm)                                            # [k] col-major indices

def _block_windows(d: int, B: int) -> List[Tuple[int, int, int]]:
    """
    returns list of (g_start, g_end, B_eff) over flattened length d, block size B
    """
    wins = []
    start = 0
    while start < d:
        end = min(start + B, d)
        wins.append((start, end, end - start))
        start = end
    return wins

@torch.no_grad()
def compare_fisher_on_comp_indices(
    fisher_invs_run1: Dict[str, object],   # {pname: EmpiricalBlockFisherInverse}
    fisher_invs_run2: Dict[str, object],
    *,
    model,                                 # for shapes (named_parameters)
    masks: Dict[str, torch.Tensor],        # {pname: BoolTensor[out]}
    perms: Dict[str, torch.Tensor],        # {pname: LongTensor[d]} row->col
    compare_in_fisher_space: bool = True,  # if False, compare inverse blocks directly
    metrics: Tuple[str, ...] = ("fro", "fro_rel", "spec"),  # which norms to report
    eps: float = 1e-12,
) -> Dict[str, dict]:
    """
    Compares two Fisher estimates *at exactly the indices used in compensation (S)*.

    Returns a dict keyed by param name with:
      - 'blocks': list of per-block dicts { 'k': |S|, 'fro': ..., 'fro_rel': ..., 'spec': ... }
      - 'aggregate': dict with sums/maxes across blocks for each metric
    """
    params = dict(model.named_parameters())
    out = OrderedDict()

    for pname, row_mask in masks.items():
        if pname not in fisher_invs_run1 or pname not in fisher_invs_run2:
            continue
        W = params[pname]
        out_features, in_features = W.shape
        d = out_features * in_features

        F1 = fisher_invs_run1[pname]
        F2 = fisher_invs_run2[pname]
        B = F1.B
        assert F2.B == B, f"Block size mismatch for {pname}: {F1.B} vs {F2.B}"

        perm = perms[pname].to("cpu")
        S_cm = _rowmask_to_colmajor_indices(row_mask, in_features, perm)  # [k] (col-major)
        blocks = []
        agg = {m: 0.0 for m in metrics}
        agg['blocks_with_S'] = 0

        if S_cm.numel() == 0:
            out[pname] = {'blocks': [], 'aggregate': agg}
            continue

        windows = _block_windows(d, B)
        Ainv_full_1 = F1.F_inv.detach().to("cpu").float()    # (num_blocks,B,B)
        Ainv_full_2 = F2.F_inv.detach().to("cpu").float()

        # iterate blocks and compute metrics on the SxS submatrix
        offset = 0
        for b, (g_start, g_end, B_eff) in enumerate(windows):
            # local indices S within this window
            in_block = (S_cm >= g_start) & (S_cm < g_end)
            if not torch.any(in_block):
                blocks.append({'k': 0})
                continue
            S_loc = (S_cm[in_block] - g_start).to(torch.long)  # [k]
            k = int(S_loc.numel())

            # take Ainv block views (trim to B_eff)
            Ainv1 = Ainv_full_1[b, :B_eff, :B_eff]
            Ainv2 = Ainv_full_2[b, :B_eff, :B_eff]

            if compare_in_fisher_space:
                # invert block to Fisher space (CPU, small B_eff)
                A1 = torch.linalg.inv(Ainv1)   # Fisher block
                A2 = torch.linalg.inv(Ainv2)
                M1 = A1.index_select(0, S_loc).index_select(1, S_loc)  # SxS
                M2 = A2.index_select(0, S_loc).index_select(1, S_loc)
            else:
                # compare inverse directly on the selected rows/cols
                M1 = Ainv1.index_select(0, S_loc).index_select(1, S_loc)
                M2 = Ainv2.index_select(0, S_loc).index_select(1, S_loc)

            D = M1 - M2  # SxS
            block_metrics = {'k': k}

            if "fro" in metrics:
                block_metrics['fro'] = torch.linalg.norm(D, ord='fro').item()
            if "fro_rel" in metrics:
                denom = torch.linalg.norm(M1, ord='fro').item() + eps
                block_metrics['fro_rel'] = block_metrics.get('fro', torch.linalg.norm(D, ord='fro').item()) / denom
            if "spec" in metrics:
                # spectral norm via svdvals on CPU
                sv = torch.linalg.svdvals(D)
                block_metrics['spec'] = (sv[0].item() if sv.numel() else 0.0)

            # aggregate (sum fro, max spec, etc.)
            for m in metrics:
                if m == "spec":
                    agg[m] = max(agg.get(m, 0.0), block_metrics[m])
                else:
                    agg[m] += block_metrics[m]
            agg['blocks_with_S'] += 1

            blocks.append(block_metrics)

        out[pname] = {'blocks': blocks, 'aggregate': agg}

    return out


def get_pruned_grads(c, masks):
    print("In get pruned grads")
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
    )

    device = next(opt.model.parameters()).device
    forget_loader = get_VIT_dataloder(c.cripple)
    with torch.no_grad():
        apply_masks_by_param_name_(opt.model, masks)
    
    opt.model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss()
    for counter, (inputs, targets) in enumerate(forget_loader, start=1):
        if counter > 2:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward + backward
        logits = opt.get_logits(pixel_values=inputs)[:, 0, :]  # CLS token
        loss = loss_fn(logits, targets)
        loss.backward()

    target_prefix = "encoder.layer."
    target_suffix = ".intermediate.dense.weight"
    grads_dict = {
        name: param.grad.detach().clone().cpu()

        for name, param in opt.model.named_parameters()
        if param.grad is not None and name.startswith(target_prefix) and name.endswith(target_suffix)
    }

    print(grads_dict)
    return grads_dict




    
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
            if b % 1000 == 0:
                print(f"The `in_block` mask is: {in_block}")
                print(f"The resulting `S_loc` tensor is: {S_loc}")
                count_from_sum = in_block.sum()
                print(f"Count from sum(): {count_from_sum.item()}")
                print(f"delta is {delta}")
                sum_of_sq = (delta ** 2).sum()
                print(f"The sum of squares for delta is: {sum_of_sq.item()}")
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

def get_pruned_grads_total(c, opt, layer_name):
    layer_name = layer_name + ".intermediate.dense.weight"
    device = next(opt.model.parameters()).device
    opt.model.eval()  # deterministic dropout etc.

    # 1) Identify the target parameter first
    param = dict(opt.model.named_parameters())[layer_name]

    # 2) Freeze everything, unfreeze only target
    prev_requires = [p.requires_grad for p in opt.model.parameters()]
    for p in opt.model.parameters():
        p.requires_grad_(False)
    param.requires_grad_(True)

    loss_fn = nn.CrossEntropyLoss()
    grads_accumulator = defaultdict(list)


    try:
        forget_loader = get_VIT_dataloder(c.cripple)
        for counter, (inputs, targets) in enumerate(forget_loader, start=1):
            if counter > 500:
                break

            inputs, targets = inputs.to(device), targets.to(device)


            logits = opt.get_logits(pixel_values=inputs)[:, 0, :]
            loss = loss_fn(logits, targets)

            # 3) Gradient only w.r.t. `param`
            g, = torch.autograd.grad(
                loss, param,
                retain_graph=False,
                create_graph=False,
                allow_unused=False  # set True if param may be unused
            )

            grads_accumulator[layer_name].append(g.detach().flatten().to('cpu'))

            # Free per-batch temporaries promptly
            del loss, logits, g, inputs, targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # 4) Restore requires_grad flags
        for p, r in zip(opt.model.parameters(), prev_requires):
            p.requires_grad_(r)

    return {name: torch.stack(lst, dim=1) for name, lst in grads_accumulator.items()}, device




def try_get_pruned_grads(c, path, tau = 1.0):
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
    
    with torch.no_grad():
        apply_masks_by_param_name_(opt.model, masks)
    
    bases = {}
    for i in range(12):
        print(f"getting grad {i}")
        layer_name = f"encoder.layer.{i}.intermediate.dense.weight"
        grads, device = get_pruned_grads_total(c, opt, layer_name)
        
        val = grads[layer_name]
        val.to(device)
        print(f"val shape: {val.shape}")
        U, S, Vh = torch.linalg.svd(val, full_matrices=False)
        if tau < 1.0:
            total_var = (S**2).sum()
            var_cum = (S**2).cumsum(0)
            k_idx = (var_cum >= tau * total_var).nonzero(as_tuple=False)[0].item()
            k = k_idx + 1
            U = U[:, :k]
        val.cpu()

        print(f"U shape: {U.shape}")
        U = U.to(torch.float16).cpu()  
        bases[layer_name] = U
        print("Bases:")
        print(bases)
        del grads, S, Vh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return bases



import torch

def _unwrap(m):
    # If you use a wrapper with `.model` (e.g., taker), unwrap it; else return as-is
    return getattr(m, "model", m)

@torch.no_grad()
def add_weight_delta(model1, model2, model3, *, scale=1.0, skip_if_shape_mismatch=True, skip_keys=("classifier",)):
    """
    Compute delta = (model2 - model1) over parameters and add it to model3 in-place:
        θ3 ← θ3 + scale * (θ2 - θ1)

    Args:
      scale: multiply the delta (e.g., 0.5 for halfway interpolation, >1 for extrapolation)
      skip_if_shape_mismatch: skip any param whose shapes differ across models
      skip_keys: tuple of substrings; if a param name contains any, it will be skipped
                 (useful when num_labels differs -> classifier shapes differ)

    Returns:
      skipped: list of (name, reason) for transparency
    """
    print('in delta')
    a, b, c = map(_unwrap, (model1, model2, model3))
    p1 = dict(a.named_parameters())
    p2 = dict(b.named_parameters())
    p3 = dict(c.named_parameters())

    skipped = []
    for name, target in p3.items():
        if any(k in name for k in (skip_keys or ())):
            skipped.append((name, "excluded_by_skip_keys"))
            continue

        w1 = p1.get(name)
        w2 = p2.get(name)
        if w1 is None or w2 is None:
            skipped.append((name, "missing_in_m1_or_m2"))
            continue

        if (w1.shape != w2.shape) or (w1.shape != target.shape):
            if skip_if_shape_mismatch:
                skipped.append((name, f"shape_mismatch {w1.shape} vs {w2.shape} vs {target.shape}"))
                continue
            else:
                raise ValueError(f"Shape mismatch for {name}: {w1.shape} vs {w2.shape} vs {target.shape}")

        # Do math on target's device/dtype to avoid casts on assignment
        dev, dt = target.device, target.dtype
        delta = (w2.detach().to(device=dev, dtype=dt) - w1.detach().to(device=dev, dtype=dt)) * scale
        target.data.add_(delta)

    return skipped

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
    fisher_m_multiplier = 1
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
    p = params[pname]
    if not isinstance(p, torch.nn.Parameter):
        raise TypeError(f"{pname} is not an nn.Parameter (got {type(p)})")
    if not p.requires_grad:
        p.requires_grad_(True)

    # allocate inverse (row-major)
    fisher_inv = EmpiricalBlockFisherInverse(
        num_grads=num_grads*fisher_m_multiplier,
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
    compensation_lr = 1
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
    max_k = 0
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
        # if 200 > theta_s.shape[0] and theta_s.shape[0] > 180:
        #     print(f"Number of things pruned weights in block is: {theta_s.shape[0]}")
        #     torch.save(S_loc.detach().cpu(), "/vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning/mmd/S_loc.pt")
        #     torch.save(Ainv_b.detach().cpu(), "/vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning/mmd/Ainv_b.pt")
        #     torch.save(theta_s.detach().cpu(), "/vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning/mmd/theta_s.pt")
        #     torch.save(A_ss.detach().cpu(), "/vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning/mmd/A_ss.pt")
        #     torch.save(theta_b.detach().cpu(), "/vol/bitbucket/sc2124/selective_pruning_mmd/exp14-weight-pruning/mmd/theta_b.pt")
        #     print("SAVED THEM")
        #     raise Exception("Saved them")
        # Δ = -A[:,S] @ (A[S,S]^{-1} @ theta[S])
        y     = torch.linalg.solve(A_ss, theta_s.unsqueeze(1)).squeeze(1)
        delta = - A_alls @ y
        delta = compensation_lr * delta
        if basis is not None:
            basis_block = basis[:, g_start:g_end]
            basis_block_sq = basis_block.T @ basis_block
            basis_block_sq_cpu = basis_block_sq.to(delta.device, dtype=delta.dtype) 
            delta = project_orthogonal(delta, basis_block_sq_cpu)
            del basis_block_sq_cpu


        theta_b.add_(delta)
        theta_b.index_fill_(0, S_loc, 0.0)              # enforce sparsity exactly

        # Woodbury downdate: Ainv <- Ainv - A[:,S] @ inv(A[S,S]) @ A[S,:]
        # Not required for single shot
        K = torch.linalg.solve(A_ss, A_sall)            # (k,B_eff)
        Ainv_b.sub_(A_alls @ K)

    # write back shaped param
    W.data.copy_(theta.view_as(W))

def per_weight_prune(c, path, personalization_hps = None, print_personalization_scores = False, fisher_block_size = 64,
                     damp = 1e-3, eps = 1e-9, num_grads = 256, print_pruned_only_scores = True, switch_m = False, use_basis = False,
                     just_personalize = False, compensation_lr = 1, fisher_m_multiplier = 1, personalize_model = False):
    print(f"Path is {path}")
    print(f"fisher_block_size: {fisher_block_size}, num_grads: {num_grads}, switch_m: {switch_m}, damp: {damp}")

    masks = torch.load(path, map_location="cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, HFImage.Image) else img),
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
        combined_train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )
    
    imagenet_val_dataset = load_from_disk(im_val_cache)
    sketch_val_dataset = load_from_disk(sk_val_cache)
    val_ds_im = WrappedDataset(imagenet_val_dataset, data_transform)
    val_loader_imagenet = DataLoader(
        val_ds_im, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    val_ds_sk = WrappedDataset(sketch_val_dataset, data_transform)
    val_loader_sketch = DataLoader(
        val_ds_sk, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
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
    
    
    if personalize_model:
        opt = Model(
            c.model_size,
            limit=c.token_limit,
            dtype=c.dtype,
            svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator,
            model_device=c.model_device,
            mask_fn=c.mask_fn,
        )
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
        opt = personalized_model
        print("saving model")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_taker_payload(opt, f"taker_{stamp}.pt")
    else:
        print("Loading model")
        opt = rehydrate_taker("taker_30_epochs_8e-4_l4_0.8_decay_0_freeze_shuffled_train.pt", device="cuda", add_hooks=False, dtype="fp32")

    res = {}

    custom_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),                 # match TFDS 3-ch decode
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4,4/3),
                                    interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3),                 # [-1,1]
    ])

    bird_labels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
    bird_eval_first_n = 600
    train_loader = get_VIT_dataloder("imagenet-1k", batch_size = 1, custom_transforms=custom_transforms)
    dataset = train_loader.dataset
    
    bird_subset_eval_loaders = build_bird_eval_loaders(dataset, bird_labels, bird_eval_first_n, 64,)
    forget_dataloader_64_bs = bird_subset_eval_loaders.get("birds_first_600_per_label")

    if print_personalization_scores:
        with torch.no_grad():
            # device = next(opt.model.parameters()).device
            # opt.model.to(device)

            # MOHAMAD: uncomment following files to get accuracy of the personalziation before any unlearning components
            eval_results = evaluate_personalization_set(opt)
            for key, metrics in eval_results.items():
                print(f" ----- debug - model performance after personalization -> {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                res["personalize " + key] = metrics['accuracy']

            forget_res = evaluate(opt, forget_dataloader_64_bs, device)
            forget_acc  = forget_res["combined"]["accuracy"]
            res["personalize forget"] =  forget_acc
            print(f" ----- debug - model performance after personalization  -> forget_acc: {forget_acc}")
        
        

    
    if print_pruned_only_scores:
        tmp_opt = copy.deepcopy(opt)
        tmp_opt.model.to(device)
        apply_weight_masks_(tmp_opt.model, masks)
        
        with torch.no_grad():
            # MOHAMAD: I have changed the model passed to the following two evaluation functions to tmp_opt (Previously it was opt)
            tmp_opt.model.to(device)
            # eval_results = evaluate_personalization_set(opt)
            eval_results = evaluate_personalization_set(tmp_opt)
            for key, metrics in eval_results.items():
                print(f" ----- debug - model performance after personalization and pruning -> {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
                res["prune " + key] = metrics['accuracy']

            # forget_res = evaluate(opt, forget_dataloader_64_bs, device)
            forget_res = evaluate(tmp_opt, forget_dataloader_64_bs, device)
            forget_acc  = forget_res["combined"]["accuracy"]
            res["prune forget"] =  forget_acc
            print(f" ----- debug - model performance after personalization and pruning  -> forget_acc: {forget_acc}")
        del tmp_opt
    
    combined_train_ds = ConcatDataset([imagenet_train_dataset, sketch_train_dataset])
    # combined_train_ds = ConcatDataset([sketch_train_dataset])
    fisher_loader = DataLoader(
        combined_train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    num_samples = len(combined_train_ds)
    print(f"num_samples is {num_samples}")
    print(f"length of fisher loader = {len(fisher_loader)}")
    opt.remove_hooks()
    
    layers = [f"encoder.layer.{i}" for i in range(12)]
    TAU = 0.95
    tmp_opt = None

    for layer_name in layers:
        if use_basis:
            print("Now using the orthogonal")
            grads, device = get_pruned_grads_total(c, tmp_opt, layer_name)
            val = grads[layer_name+".intermediate.dense.weight"]
            print(f"val shape: {val.shape}")
            U, S, Vh = torch.linalg.svd(val, full_matrices=False)
            if TAU < 1.0:
                total_var = (S**2).sum()
                var_cum = (S**2).cumsum(0)
                k_idx = (var_cum >= TAU * total_var).nonzero(as_tuple=False)[0].item()
                k = k_idx + 1
                U = U[:, :k]
            val = val.cpu()
            print(f"U shape: {U.shape}")
            U = U.T.cpu()  
            basis = U
            del grads, S, Vh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else: 
            basis = None
        # 1) build Fisher inverse for THIS layer only
        pname, F_inv = build_fisher_inv_rowmajor_for_layer(
            model=opt,                                   # your wrapper
            layer_name=layer_name,
            EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
            dataloader=fisher_loader,                    # calibration data
            num_grads=num_samples,                                # smaller can help off-diagonals
            fisher_block_size=fisher_block_size,                       # bigger than 64; tune to VRAM
            damp=damp,
            switch_m=switch_m,                              # if your class supports the SM denom
            fisher_m_multiplier = fisher_m_multiplier
        )

        # 2) fetch the *per-weight* mask for this layer
        mask_2d = masks.get(pname, None)                # Bool [out,in]
        print(f"pname is: {pname}")

        # 3) compensate+prune this single layer
        cap_compensate_weights_rowmajor_for_layer(
            model=opt,
            layer_name=layer_name,
            weight_mask_2d=mask_2d,
            fisher_inv=F_inv,
            eps=1e-9,
            basis = basis,
            compensation_lr = compensation_lr
        )
        print(f"layer_name is: {layer_name}")
        # 4) free memory for this layer before the next one
        del F_inv
        torch.cuda.empty_cache()
        opt.remove_hooks()


    opt.model.eval()
    with torch.no_grad():
        opt.model.to(device)
        eval_results = evaluate_personalization_set(opt)
        for key, metrics in eval_results.items():
            print(f" ----- debug - model performance after personalization, prune, and compensate -> {key} Val Accuracy: {metrics['accuracy']:.6f}, Avg Loss: {metrics['avg_loss']:.4f}")
            res["compensate " + key] = metrics['accuracy']

        forget_res = evaluate(opt, forget_dataloader_64_bs, device)
        res["compensate forget"]  = forget_res["combined"]["accuracy"]
        # MOHAMAD
        # print(f" ----- debug - model performance after personalization, prune, and compensate  -> forget_acc: {forget_acc}")
        facc = res["compensate forget"]
        print(f" ----- debug - model performance after personalization, prune, and compensate  -> forget_acc: {facc}")
        
        print(res)
        sys.stdout.flush()
        # total = 0
        # correct = 0
        # for image, label in val_loader:
        #     images = image.to(device, non_blocking=True)
        #     labels = label.to(device, non_blocking=True)


        #     logits = opt.get_logits(pixel_values=images)
        #     logits = logits[:, 0, :]  # take only CLS token for classification

        #     if labels.ndim > 1:
        #         labels = labels.squeeze(-1)           # e.g., [B,1] -> [B]
        #     labels = labels.long()    
            
        #     preds  = logits.argmax(dim=-1)
        #     correct += (preds == labels).sum().item()
        #     total   += labels.numel()  
            
        #     total +=1
        #     sys.stdout.flush()
        #     if total == 480:
        #         break

        # print(f"correct is {correct}")
        # sys.stdout.flush()
        
    # exclude 'token_count' from eval_out to print
    # print_eval_out = {k: v for k, v in eval_out.items() if k != 'misc'}
    # print(f'----- debug - model performance after unlearning and compensation (on birds/birdsless) -> eval_out: {print_eval_out} ----- \n\n') 
    # res["prune and compensate imagenet-1k-birds"] = print_eval_out['accuracy']['imagenet-1k-birds']['base'] / 100

    # print("after masking + compensating row major res is: ")
    # print(res)

    # imagenet_gain = (res['prune and compensate imagenet'] - res['prune only imagenet']) / (0.8538461538461538 - res['prune only imagenet'])
    # sketch_gain = (res['prune and compensate sketch'] - res['prune only sketch']) / (0.7653846153846153 - res['prune only sketch'])
    # forget_gain = (res['prune and compensate imagenet-1k-birds'] - res['prune only imagenet-1k-birds']) / (0.9197994987468671 - res['prune only imagenet-1k-birds'])
    # res['prune and compensate imagenet-1k-birds'] = correct/total

    # sketch_gain_model_is_sketch_only = (res['prune and compensate sketch'] - res['prune only sketch']) / ( 0.7692307692307693- res['prune only sketch'])
    # print(f"Imagenet gain: {imagenet_gain}, sketch gain: {sketch_gain}, forget gain: {forget_gain}")
    print(res)
    return (res['compensate forget'], res['compensate sketch'], res['compensate imagenet'])
