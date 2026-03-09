from .weight_pruning import _cache_iterable_split, save_checkpoint_ckpt, is_main_process, LabelFilter, PerClassHeadFilter, build_bird_eval_loaders, evaluate_validation
from .prune import get_VIT_dataloder
import sys
from PIL import Image as PILImage
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification, AutoImageProcessor, ViTConfig
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F


def retrain_vit(c):
    print("In retrain without forget, this is for imagenet-1k, previous one was imagenet-1k-birds")
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
    validation_loader = get_VIT_dataloder("imagenet-1k", is_validation = True, batch_size = 64, custom_transforms=custom_transforms)

    

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
    

    print("in retrain without forget")
    sys.stdout.flush()
    trim_counts = { int(c): 600 for c in classes_to_trim.keys() }
    print(trim_counts)
    batch_size = 64
    train_loader = get_VIT_dataloder("imagenet-1k", batch_size = batch_size, custom_transforms=custom_transforms)
    dataset = train_loader.dataset

    bird_labels = sorted(int(lbl) for lbl in trim_counts.keys())
    print(f"Bird label ids ({len(bird_labels)}): {bird_labels}")
    sys.stdout.flush()

    bird_eval_first_n = int(getattr(c, "bird_eval_top_n", 600))
    print(f"Configuring bird subset evaluation with first_n_per_label={bird_eval_first_n}")
    sys.stdout.flush()
    bird_subset_eval_loaders = build_bird_eval_loaders(dataset, bird_labels, bird_eval_first_n, batch_size, transform=custom_transforms)
    if bird_subset_eval_loaders:
        print(f"Prepared bird eval splits: {list(bird_subset_eval_loaders.keys())}")
        sys.stdout.flush()
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

    run_train(train_loader, evaluation_splits)



def run_train(train_loader, evaluation_splits):
    train_split_key = "dataset_without_first_n_bird_images"

    if train_loader is None:
        raise ValueError(f"Training split '{train_split_key}' does not provide a DataLoader.")
    
    NUM_CLASSES = 1000
    bs_per_gpu = 64                  # example; adjust for your GPU
    accum = 8                        # 64 * 8 = 512 global batch on 1 GPU; scale accordingly
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_every_updates = 300       # e.g., like vit_jax checkpoint_every
    out_dir = "./vit_b16_in1k_ft_full_retrain_2"

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

    log_every_updates = 1000
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





