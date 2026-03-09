import copy
import torch
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict
from .model import Model
from .prune import get_VIT_dataloder, EmpiricalBlockFisherInverse, EmpiricalBlockFisherInverseLLM
from .fine_tune import WrappedDataset, compute_personalized_fisher_invs, personalize_vit
from .activations import get_midlayer_data
from .scoring import score_indices_by
from .model_saving import save_taker_payload, load_taker_payload
from torchvision import transforms
from datasets import load_from_disk,load_dataset, concatenate_datasets, interleave_datasets, Dataset
import datasets
from torch.cuda.amp import autocast

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from taker.eval import evaluate_all
import os
from PIL import Image
import time
import gc
import math
import sys
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json

from accelerate import Accelerator

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig, get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 768
BATCH_SIZE = 6
NUM_WORKERS = 0  # bump cautiously on your cluster
LR = 2e-5
EPOCHS = 3

SEED = 42
MAX_PER_LANG = 20_000      # cap per language (set None to disable)
TRAIN_RATIO = 0.98 

def _solve_in_fp32(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # A: [..., n, n], b: [..., n] or [..., n, k]
    with torch.cuda.amp.autocast(enabled=False):
        A32 = A.float().contiguous()
        b_in = b.unsqueeze(-1) if b.dim() == A.dim() - 1 else b
        b32 = b_in.float().contiguous()
        y32 = torch.linalg.solve(A32, b32)
        y32 = y32.squeeze(-1) if b.dim() == A.dim() - 1 else y32
    return y32



def save_codeparrot_loaders(
    langs: List[str],
    tokenizer,
    *,
    per_lang_train: int = 10000,
    per_lang_val: int   = 1000,
    max_len: int = 1024,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 1234,
    shuffle_buffer: int = 10_000
):
    if isinstance(langs, str):
        langs = [langs]
    # 7500 and 750
    """
    Streaming + deterministic:
    - server-side filter to languages (codeparrot/github-code)
    - shuffle(seed, buffer) -> take(train+val)
    - split via take/skip
    - tokenize on-the-fly (handles code/content/text)
    - interleave train streams deterministically
    """
    assert tokenizer.pad_token_id is not None, "Set tokenizer.pad_token = tokenizer.eos_token for OPT."
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    TEXT_KEYS = ("code", "content", "text")

    def _has_text(ex):
        # keep only samples that have a non-empty text field
        for k in TEXT_KEYS:
            v = ex.get(k, None)
            if isinstance(v, (str, bytes)) and len(v) > 0:
                return True
        return False
    def _iter_stream(stream):                            # NEW: tiny adapter for from_generator
        for x in stream:
            yield x

    def _extract_text(ex) -> str:
        for k in TEXT_KEYS:
            v = ex.get(k, None)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, bytes) and v:
                try:
                    return v.decode("utf-8", errors="ignore")
                except Exception:
                    pass
        return ""  # should not happen if we filtered, but safe default

    def _tok_one(ex):
        text = _extract_text(ex)
        out = tokenizer(text, truncation=True, max_length=max_len)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    lang_train_streams = []
    lang_val_streams: Dict[str, 'datasets.IterableDataset'] = {}
    total_needed = per_lang_train + per_lang_val
    collator = lambda batch: causal_lm_collate_clean(batch, pad_id=pad_id)
    out_dir = "snap_codeparrot"
    
    lang_train_maps = []                                 # NEW: map-style per-lang trains
    lang_val_maps: Dict[str, Dataset] = {}               # NEW: map-style per-lang vals 
    for lang in langs:
        # Stream this language only
        ds_lang = load_dataset(
            "codeparrot/github-code-clean",
            lang+"-all",           # server-side language filter
            split="train",
            streaming=True,
            trust_remote_code=True,     # ok for codeparrot scripts
        )

        # Filter out rows without usable text
        ds_lang = ds_lang.filter(_has_text)

        # Deterministic shuffle (buffered), then cap total samples
        ds_lang = ds_lang.shuffle(seed=seed, buffer_size=shuffle_buffer).take(total_needed)

        # Train / val split by counts
        ds_train_raw = ds_lang.take(per_lang_train)
        ds_val_raw   = ds_lang.skip(per_lang_train).take(per_lang_val)

        # Tokenize on the fly
        ds_train_tok = ds_train_raw.map(_tok_one)
        ds_val_tok   = ds_val_raw.map(_tok_one)

        # ---- MINIMAL NEW PART: materialize to map-style + save ----
        train_map = Dataset.from_generator(_iter_stream, gen_kwargs={"stream": ds_train_tok})  # NEW
        val_map   = Dataset.from_generator(_iter_stream, gen_kwargs={"stream": ds_val_tok})    # NEW

        train_map.save_to_disk(f"{out_dir}/train_{lang}")  # NEW (optional but usually what you want)
        val_map.save_to_disk(f"{out_dir}/val_{lang}")      # NEW

        lang_train_maps.append(train_map)                  # NEW
        lang_val_maps[lang] = val_map                      # NEW
        # -----------------------------------------------------------

    langs_str = "".join(langs)
    # If you also want a single combined train snapshot:
    train_map_all = concatenate_datasets(lang_train_maps).shuffle(seed=seed)   # NEW
    train_map_all.save_to_disk(f"{out_dir}/train_all_"+langs_str)                         # NEW



def load_datasets_val_only(langs, pad_id = None, batch_size = 4):
    # langs_str = "".join(langs)
    # train_ds = load_from_disk("snap_codeparrot/train_all_"+langs_str)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = lambda batch: causal_lm_collate_clean(batch, pad_id=pad_id)
    val_loaders = {
        lang: DataLoader(
            load_from_disk(f"snap_codeparrot/val_{lang}"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, batch_size),
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collator,
        )
        for lang in langs
    }

    return val_loaders

def load_individual_train(langs, pad_id = None, batch_size = 1):
    if not os.path.exists("snap_codeparrot/train_"+langs):
        save_codeparrot_loaders(langs, AutoTokenizer.from_pretrained("facebook/opt-125m"))
    train_ds = load_from_disk("snap_codeparrot/train_"+langs)
    print(f"The length of the train dataset of {langs} is {len(train_ds)}")

        # val_ds_py = load_from_disk("snap_codeparrot/val_python")  # example
    # pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = lambda batch: causal_lm_collate_clean(batch, pad_id=pad_id)

    # MOHAMAD: changed this to prevent encountering error
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     prefetch_factor=2,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     collate_fn=collator,               # your collator can keep making labels
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # single-process loading
        prefetch_factor=None,   # must be None if num_workers == 0
        persistent_workers=False,  # no worker processes to persist
        pin_memory=True,
        collate_fn=collator,    # keep your custom collator
    )

    return train_loader


def load_datasets(langs, pad_id = None, batch_size = 1):
    langs_str = "".join(langs)
    train_ds = load_from_disk("snap_codeparrot/train_all_"+langs_str)

        # val_ds_py = load_from_disk("snap_codeparrot/val_python")  # example
    # pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = lambda batch: causal_lm_collate_clean(batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collator,               # your collator can keep making labels
    )

    val_loaders = {
        lang: DataLoader(
            load_from_disk(f"snap_codeparrot/val_{lang}"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collator,
        )
        for lang in langs
    }

    return train_loader, val_loaders

def build_codeparrot_loaders_streaming(
    langs: List[str],
    tokenizer,
    *,
    per_lang_train: int = 7500,
    per_lang_val: int   = 7500,
    max_len: int = 1024,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 1234,
    shuffle_buffer: int = 10_000
):
    # 7500 and 750
    """
    Streaming + deterministic:
    - server-side filter to languages (codeparrot/github-code)
    - shuffle(seed, buffer) -> take(train+val)
    - split via take/skip
    - tokenize on-the-fly (handles code/content/text)
    - interleave train streams deterministically
    """
    assert tokenizer.pad_token_id is not None, "Set tokenizer.pad_token = tokenizer.eos_token for OPT."
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    TEXT_KEYS = ("code", "content", "text")

    def _has_text(ex):
        # keep only samples that have a non-empty text field
        for k in TEXT_KEYS:
            v = ex.get(k, None)
            if isinstance(v, (str, bytes)) and len(v) > 0:
                return True
        return False

    def _extract_text(ex) -> str:
        for k in TEXT_KEYS:
            v = ex.get(k, None)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, bytes) and v:
                try:
                    return v.decode("utf-8", errors="ignore")
                except Exception:
                    pass
        return ""  # should not happen if we filtered, but safe default

    def _tok_one(ex):
        text = _extract_text(ex)
        out = tokenizer(text, truncation=True, max_length=max_len)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    lang_train_streams = []
    lang_val_streams: Dict[str, 'datasets.IterableDataset'] = {}
    total_needed = per_lang_train + per_lang_val
    collator = lambda batch: causal_lm_collate_clean(batch, pad_id=pad_id)

    for lang in langs:
        # Stream this language only
        ds_lang = load_dataset(
            "codeparrot/github-code-clean",
            lang+"-all",           # server-side language filter
            split="train",
            streaming=True,
            trust_remote_code=True,     # ok for codeparrot scripts
        )

        # Filter out rows without usable text
        ds_lang = ds_lang.filter(_has_text)

        # Deterministic shuffle (buffered), then cap total samples
        ds_lang = ds_lang.shuffle(seed=seed, buffer_size=shuffle_buffer).take(total_needed)

        # Train / val split by counts
        ds_train_raw = ds_lang.take(per_lang_train)
        ds_val_raw   = ds_lang.skip(per_lang_train).take(per_lang_val)

        # Tokenize on the fly
        ds_train_tok = ds_train_raw.map(_tok_one)
        ds_val_tok   = ds_val_raw.map(_tok_one)

        lang_train_streams.append(ds_train_tok)
        lang_val_streams[lang] = ds_val_tok

    # Interleave per-language train streams deterministically (equal probs)
    train_stream = interleave_datasets(
        lang_train_streams,
        probabilities=None,
        seed=seed,
        stopping_strategy="first_exhausted",
    )

    # DataLoaders (IterableDataset: DataLoader(shuffle=...) is ignored)
    train_loader = DataLoader(
        train_stream,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loaders = {
        lang: DataLoader(
            lang_val_streams[lang],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        for lang in langs
    }

    return train_loader, val_loaders

def load_finetuned_for_eval_lora(save_dir: str, device: str = "cuda",
                            base_id: str | None = None, trainable: bool = False):
    is_adapter = os.path.exists(os.path.join(save_dir, "adapter_config.json"))

    if is_adapter:
        # Try to read base model from adapter_config.json if not provided
        if base_id is None:
            with open(os.path.join(save_dir, "adapter_config.json")) as f:
                base_id = json.load(f).get("base_model_name_or_path", "facebook/opt-125m")

        # Tokenizer: use adapter dir if present, else base model
        tok_src = save_dir if os.path.exists(os.path.join(save_dir, "tokenizer_config.json")) else base_id
        # tokenizer = AutoTokenizer.from_pretrained(tok_src)
        orig_base_id = "facebook/opt-125m"                    # or your local base path

        #        add tokenizer to merged_dir once
        tokenizer = AutoTokenizer.from_pretrained(orig_base_id, local_files_only=False)

        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, save_dir, is_trainable=trainable)
    else:
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "config"):
        model.config.use_cache = True

    model.to(device)
    model.train() if trainable else model.eval()
    return model, tokenizer


def train_opt_on_code_streaming_lora(
    model,
    train_loader,
    num_epochs=1,
    lr=2e-4,                                # LoRA likes a higher LR
    max_steps_per_epoch=None,
    tokenizer=None,
    val_loaders=None,
    warmup_ratio=0.03,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=("q_proj","k_proj","v_proj","out_proj","fc1","fc2"),
    save_adapters_only=True,                # save just LoRA adapters by default
    load_adapters = False
):
    # --- Wrap with LoRA (only once) ---
    device = next(model.parameters()).device
    if not load_adapters:
        if getattr(model, "peft_config", None) is None:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            if hasattr(model, "config"):
                model.config.use_cache = False   # good practice for training
            model = get_peft_model(model, lora_cfg)

    else:
        model, tokenizer = load_finetuned_for_eval_lora("models/opt125-lora-epoch-19-max_steps-None-20250914-220156-langs-Rust", base_id = "model-combined-c++-c-scala-java", 
                                             trainable = True)
        if hasattr(model, "config"):
            model.config.use_cache = False
    
    if False:
        merged = model.merge_and_unload()                 # apply LoRA deltas into base weights
        merged.save_pretrained("model/model-combined-c++-c-scala-java-personalized-rust", safe_serialization=True)
        return
    
    print(f"model device is {device}")
    # --- Optimizer & scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    base_lr = lr  # e.g., 1e-3 for LoRA
    for g in opt.param_groups:
        g["lr"] = base_lr
        g["initial_lr"] = base_lr
    steps_per_epoch = (max_steps_per_epoch or len(train_loader))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps,

    )
    print("in lora")
    langs = "".join(val_loaders.keys()) if val_loaders else ""
    for epoch in range(num_epochs):
        model.train()
        step, running = 0, 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn      = batch["attention_mask"].to(device, non_blocking=True)

            # ensure labels exist and ignore pads
            labels = batch.get("labels", None)
            if labels is None:
                labels = input_ids.clone()
            labels = labels.to(device, non_blocking=True).clone()
            labels[attn == 0] = -100

            out  = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            step += 1
            running += loss.item()
            if step % 500 == 0:
                print(f"Step {step}")
                print(f"Avg train loss {running / max(1, step):.4f}")
                print(f"LR {sched.get_last_lr()[0]:.2e}")
                if step % 1000 == 0:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    save_dir = f"models/opt125-lora-epoch-{epoch}-step-{step}-{stamp}"
                    if save_adapters_only:
                        model.save_pretrained(save_dir)          # saves LoRA adapters
                        if tokenizer: tokenizer.save_pretrained(save_dir)
                    else:
                        # your old saver (will save full weights)
                        save_finetuned(model, tokenizer=tokenizer, save_dir=save_dir, optimizer=opt)
                sys.stdout.flush()

            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break

        # ---- validation ----
        if val_loaders:
            for lang, dl in val_loaders.items():
                m = evaluate_code_lm(model, dl)
                model.train()
                print(lang, m)

        # ---- epoch end save ----
        avg = running / max(1, step)
        print(f"[Epoch {epoch+1}] avg train loss: {avg:.4f}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = f"models/opt125-lora-epoch-{epoch}-max_steps-{max_steps_per_epoch}-{stamp}-langs-{langs}"
        if save_adapters_only:
            model.save_pretrained(save_dir)
            if tokenizer: tokenizer.save_pretrained(save_dir)
        else:
            save_finetuned(model, tokenizer=tokenizer, save_dir=save_dir, optimizer=opt)
        print("saved model")

    return model


def train_opt_on_code_streaming(model, train_loader, num_epochs=10, lr=5e-5, max_steps_per_epoch=None, tokenizer = None, val_loaders = None, warmup_ratio=0.075): 
    model.train() 
    print(f"model device is {model.device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    steps_per_epoch = (max_steps_per_epoch or len(train_loader))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    langs = []
    for k in val_loaders.keys():
        langs.append(k)
    
    langs = "".join(langs)
    for epoch in range(num_epochs):
        step, running = 0, 0.0
        for batch in train_loader:

            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attn      = batch["attention_mask"].to(model.device, non_blocking=True)
            labels    = batch["labels"].to(model.device, non_blocking=True)

            
            # -------- AMP fp16 forward/backward --------
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            # -------------------------------------------

            step += 1
            running += loss.item()
            if step % 1000 == 0:
                print(f"Step {step}")
                print(f"Avg train loss {running / max(1, step) }")
                if step % 5000 == 0:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    save_finetuned(model, save_dir=f"models/opt125-epoch-{epoch}-step-{step}-{stamp}", optimizer= opt)
                sys.stdout.flush()
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
        for lang, dl in val_loaders.items():
            m = evaluate_code_lm(model, dl)
            model.train() 
            print(lang, m)
        
        avg = running / max(1, step) 
        print(f"[Epoch {epoch+1}] avg train loss: {avg:.4f}") 
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_finetuned(model, save_dir=f"models/opt125-epoch-{epoch}-max_steps-{max_steps_per_epoch}-{stamp}-langs-{langs}", optimizer= opt)
        print("saved model")
    
@torch.no_grad()
def evaluate_code_lm(model, loader, num_grads = 60) -> Dict[str, float]:
    model.eval()
    total_loss_tokens = 0.0
    total_tokens = 0
    total_correct = 0
    seen = 0
    with torch.inference_mode():  
        for batch in loader:
            if seen >= num_grads:
                break
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)  # shape [B, T], with -100 on pads

            # Forward for loss
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            # Count active tokens
            active = (labels != -100).sum().item()
            total_loss_tokens += loss.item() * active
            total_tokens += active

            # Next-token accuracy (manual, since we want top-1 over next token)
            # logits: [B, T, V]. Shift so that positions 0..T-2 predict 1..T-1
            logits = out.logits  # [B, T, V]
            # Create shifted labels (targets at t: labels[:, t], predicted by logits[:, t-1])
            # We only compare where labels != -100 AND t > 0
            B, T, V = logits.shape
            preds = logits[:, :-1, :].argmax(dim=-1)             # [B, T-1]
            tgt   = labels[:, 1:]                                # [B, T-1]
            mask  = (tgt != -100)                                # ignore padding and non-targets

            correct = (preds[mask] == tgt[mask]).sum().item()
            total_correct += correct
            seen += 1
    avg_loss = total_loss_tokens / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    acc = total_correct / max(1, total_tokens)
    return {"avg_loss": avg_loss, "perplexity": ppl, "top1_acc": acc}
 
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

# ---- You must provide this class from your codebase ----
# It should support: __init__(num_grads, fisher_block_size, num_weights, damp, device),
#   .add_grad(grad_flat: Tensor)   and   .fisher_diag() -> Tensor[num_weights]
# from your module import EmpiricalBlockFisherInverse

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

def _find_intermediate_weight_params(model: nn.Module) -> List[Tuple[int, str, torch.nn.Parameter]]:
    """
    Returns a list of (layer_index, full_param_name, param) for
    'vit.encoder.layer.{i}.intermediate.dense.weight' ordered by i.
    """

    hits = []
    for name, p in model.named_parameters():
        if (".fc1.weight" in name or ".fc2.weight" in name) and "decoder.layers" in name:
            # Extract the {i}
            try:
                # name like: vit.encoder.layer.7.intermediate.dense.weight
                parts = name.split(".")
                # find "layer" and take the next index
                li = parts.index("layers")
                idx = int(parts[li + 1])
                hits.append((idx, name, p))
            except Exception:
                pass
    # sort by layer index
    hits.sort(key=lambda t: t[0])
    return hits


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
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    scoring_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = lambda W, dW, F: (dW.abs() * W.abs()) + ((F if F is not None else 0) * (W * W).abs()),
    c = None,
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

    model.model.eval()  # no dropout etc, but we need grads

    # 1) Discover target parameters (ordered by layer index)
    
    param_targets = _find_intermediate_weight_params(model.model)
    print(f"param targets are {param_targets}")
    if not param_targets:
        raise RuntimeError("No encoder.layer.{i}.intermediate.dense.weight params found.")

    def _forward_lm(hf_model, batch):
        # accepts dict or (input_ids, labels, attention_mask)
        if isinstance(batch, dict):
            input_ids      = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels         = batch.get("labels", None)
        else:
            input_ids, labels, attention_mask = batch
        out = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss
    
    target_params = [p for _, _, p in param_targets]  #build_codeparrot_loaders_streaming preserve order
    # accumulators aligned 1:1 with param_targets
    layer_grad_sums = [torch.zeros_like(p, device=p.device) for _, _, p in param_targets]

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
    
    seen = 0
    model.model.eval()
    # (optional but recommended for speed during scoring)
    model.model.config.use_cache = False

    for counter, batch in enumerate(dataloader, start=1):
        if counter > num_grads:
            break

        # move batch to device
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        else:
            batch = tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)

        # no need to zero p.grad; autograd.grad doesn't touch .grad
        with torch.set_grad_enabled(True):
            loss = _forward_lm(model, batch)
            grads = torch.autograd.grad(
                loss,
                target_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,   # short sequences may skip some params
            )

        for i, g in enumerate(grads):
            if g is None:
                continue
            g = g.float()
            layer_grad_sums[i].add_(g)
            fisher_invs[i].add_grad(g.flatten())

        seen += 1
        if counter % 50 == 0:
            print(f"[{counter:>4}/{num_grads}] processed")

    if seen == 0:
        raise ValueError("Dataloader yielded zero batches; cannot compute scores.")

    # 5) Mean gradients and Fisher-diagonal per weight
    layer_gradients = [g_sum / float(seen) for g_sum in layer_grad_sums]
    layer_fisher_diags = [
        fisher_inv.fisher_diag().reshape_as(grad) for fisher_inv, grad in zip(fisher_invs, layer_gradients)
    ]

    # 6) Per-weight -> per-row scoring
    row_scores: Dict[str, torch.Tensor] = OrderedDict()
    

    for (i, full_name, p), dW, F in zip(param_targets, layer_gradients, layer_fisher_diags):
        W = p.data
        per_weight = scoring_fn(W, dW, F, None)  # same shape as W
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
        avg_loss = loss_sum / len(val_loaders)
        accuracy = correct / total if total > 0 else 0
        return {"combined": {"accuracy": accuracy, "avg_loss": avg_loss}}





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

################################################################################################################################################################

# ---------- 1) Permutation utilities ----------

# ---------- 2) Build Fisher inverses in column-major space (wrapper) ----------

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

################################################################################################################################################################


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
        params['model.'+name].data[m] = 0.0

def run_weight_based_masking_llm(c, hessian_coefficient = 0.5, retain_dataloader = False, retain_coefficient = 1, 
                             forget_frac = 0.02, switch_m = False, selective_pruning = False, sub_retain = False, l1 = 0,
                            eval_pruned = False, damp = 1e-3, layer_normalize = False, num_grads = 480,
                            fisher_block_size = 576, signs = "Second Order Neg"):
    
    # NUM GRADS DEFINES FORGET SET
    
    print("C is:")
    print(c)

    model = load_finetuned_for_eval("model-combined-c++-c-scala-java")
    print("loaded model")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    print(f"Hessian coefficient {hessian_coefficient}, retain dataloader {retain_dataloader}, forget frac: {forget_frac}, switch_m: {switch_m}, selective pruning: {selective_pruning}, sub retain: {sub_retain}, l1: {l1}")

    langs = ["C++", "Scala"]
    pad_id = model.config.pad_token_id
    
    scala_loader= load_individual_train("Scala", pad_id = pad_id, batch_size = 1)
    
    #TODO: COMBINE THESE DATALOADERS TO PASS INTO RETAIN CASE
    # cpp_loader = load_individual_train("C++", pad_id = pad_id, batch_size = 1)
    # java_loader = load_individual_train("Java", pad_id = pad_id, batch_size = 1)
    # c_lang_loader = load_individual_train("C", pad_id = pad_id, batch_size = 1)

    loaders = [
        load_individual_train("C", pad_id=pad_id, batch_size=1),
        load_individual_train("C++", pad_id=pad_id, batch_size=1),
        load_individual_train("Java", pad_id=pad_id, batch_size=1),
    ]

    mixed_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in loaders])

    mixed_loader = torch.utils.data.DataLoader(
        mixed_dataset,
        batch_size=1,
        shuffle=True,              
        num_workers=loaders[0].num_workers,
        collate_fn=loaders[0].collate_fn,
        pin_memory=loaders[0].pin_memory,
    )

    
    def my_scoring(W, dW, F, midlayer_scores, hessian_coefficient = hessian_coefficient):
        # Pure Fisher (OBD-style) with tiny epsilon
        if selective_pruning:
            return midlayer_scores
        else: 
            if signs == "Second Order Neg":
                return ((dW * W) +  hessian_coefficient * -(F * W.pow(2)))
            elif signs == "First Order Neg":
                return (-(dW * W) +  hessian_coefficient * (F * W.pow(2)))
            elif signs == "Abs Hessian Neg":
                return (abs(dW * W) -  hessian_coefficient * abs(F * W.pow(2)))
    # get_VIT_dataloder(c.cripple)
    row_scores = rank_vit_intermediate_nodes_hooked(
        model,
        dataloader=scala_loader,
        EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        damp=damp,
        scoring_fn=my_scoring,
        switch_m= switch_m,
        c = c      # plug your scorer here
    )
    print(row_scores)

    
    retain_bool = retain_dataloader
    #TODO: THE DATALOADER SHOULD BE A MIX OF THE DIFFERENT ONES
    if retain_dataloader:
        row_scores_retain = rank_vit_intermediate_nodes_hooked(
            model,
            dataloader=mixed_loader,
            EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverse,
            num_grads=num_grads*3,
            fisher_block_size=fisher_block_size,
            damp=damp,
            scoring_fn=my_scoring,
            switch_m= switch_m,
            c = c      # plug your scorer here
        )
        new_row_scores: Dict[str, torch.Tensor] = OrderedDict()
        for k, v in row_scores.items():
            if sub_retain:
                new_row_scores[k] = v - (retain_coefficient * row_scores_retain[k])
            else:
                new_row_scores[k] = v / (retain_coefficient * row_scores_retain[k] + 1e-12)
        row_scores = new_row_scores

    if layer_normalize:
        masks = build_weight_masks_global_fraction_normalized(row_scores, top_fraction=  forget_frac)
    else:
        masks = build_weight_masks_global_fraction(row_scores, top_fraction=forget_frac)   
    # masks = build_masks_per_param(row_scores, top_fraction=forget_frac)

    print("Masks:")
    print(masks)
    os.makedirs("./outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    
    pt_path = f"../outputs/llm_intermediate_row_masks__retaindl_{retain_bool}_retaincoefficient_{retain_coefficient}_-{stamp}_forgetfrac_{forget_frac}.pt"
    torch.save({k: v.detach().cpu() for k, v in masks.items()}, pt_path)
    
    

    if eval_pruned:
        val_loaders = load_datasets_val_only(["Scala", "C", "C++", "Java"], pad_id = pad_id, batch_size = 1)
        print('Just Masking')
        tmp_opt = copy.deepcopy(model)

        top_1 = 0

        for lang, dl in val_loaders.items():
            m = evaluate_code_lm(tmp_opt, dl, num_grads=num_grads)
            top_1 += m["top1_acc"]
            print(lang, m)
        top_1 = top_1 / 4
        apply_weight_masks_(tmp_opt, masks)
        scala_results = evaluate_code_lm(tmp_opt, scala_loader, num_grads=num_grads)
        print("Scala results", scala_results)
        del tmp_opt
        # train_opt_on_code_streaming(model, train_loader, max_steps_per_epoch = 5, tokenizer = tokenizer)
        sys.stdout.flush()
        pt_path, (scala_results["top1_acc"], top_1)


    return pt_path

def collate_fn(batch):
    imgs = []
    labels = []
    for sample in batch:
        # Convert flattened list back to tensor and reshape it to (3, 224, 224)
        img = torch.tensor(sample['image']).reshape(3, 224, 224)
        imgs.append(img)
        labels.append(sample['label'])
    return torch.stack(imgs), torch.tensor(labels)


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

from collections import defaultdict
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



["C", "C++", "C#", "Java", "Scala"]


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
    model: nn.Module,                              # <- pass the CausalLM (AutoModelForCausalLM), NOT model.model
    layer_name: str,                               # e.g. "decoder.layers.3"
    EmpiricalBlockFisherInverse,                   # your class
    dataloader: Iterable,                          # yields dict or (input_ids, labels, attention_mask)
    *,
    num_grads: int = 128,
    fisher_block_size: int = 256,
    damp: float = 3e-3,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn: Optional[nn.Module] = None,           # (unused for CausalLM; kept for signature compat)
    switch_m: bool = False,
    suf: str = None
) -> Tuple[str, object]:
    """
    Builds a single EmpiricalBlockFisherInverse for OPT's {layer_name}.fc1.weight (row-major).
    Returns (param_name, fisher_inv).
    """
    base = model  # use the full CausalLM; it accepts labels and returns loss
    if device is None:
        device = DEVICE

    # 1) Find the exact parameter name for OPT FFN fc1.weight
    #    Expect names like "...decoder.layers.{i}.fc1.weight"
    # suffix = f"{layer_name}.fc1.weight"
    suffix = f"{layer_name}"+suf
    pname, p = None, None
    for n, param in base.named_parameters():
        if n.endswith(suffix) and ".decoder.layers." in n:
            pname, p = n, param
            break
    if p is None:
        raise RuntimeError(f"Could not find parameter for OPT layer '{layer_name}' (expected '*.{suffix}').")

    if not isinstance(p, torch.nn.Parameter):
        raise TypeError(f"{pname} is not an nn.Parameter (got {type(p)})")
    if not p.requires_grad:
        p.requires_grad_(True)

    # 2) Allocate inverse (row-major)
    fisher_inv = EmpiricalBlockFisherInverse(
        num_grads=num_grads,
        fisher_block_size=fisher_block_size,
        num_weights=p.numel(),
        damp=damp,
        device=p.device,
        switch_m=switch_m,
    )

    # 3) Iterate batches; use autograd.grad to get grad wrt this single param
    base.eval()
    # cheaper forward for scoring (don’t cache KV)
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    seen = 0
    for batch in dataloader:
        if seen >= num_grads:
            break

        # Move batch to device; support dict OR (input_ids, labels, attention_mask)
        if isinstance(batch, dict):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels         = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
        else:
            input_ids, labels, attention_mask = batch
            input_ids      = input_ids.to(device, non_blocking=True)
            labels         = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None

        # Forward -> scalar loss via CausalLM
        with torch.set_grad_enabled(True):
            out = base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            # Get grad ONLY for this param (no hooks, no .backward, minimal memory)
            g, = torch.autograd.grad(
                loss, (p,),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        if g is not None:
            fisher_inv.add_grad(g.reshape(-1))  # row-major flat
            seen += 1
        else:
            print("MAYDAY")

    print(f"final seen value: {seen}")
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
    suf = None
) -> None:
    def _solve_spd(A, b):
        # fp32, symmetric, jittered solve with fallback
        A32 = A.to(torch.float32).clone()
        b32 = b.to(torch.float32).clone()

        # symmetrize to kill small asymmetries from estimation noise
        A32 = 0.5 * (A32 + A32.T)

        # scale-aware jitter
        jitter = (1e-6 * A32.diagonal().abs().mean()).clamp_min(1e-12)
        A32.diagonal().add_(jitter)

        L, info = torch.linalg.cholesky_ex(A32)
        if int(info) != 0:                    # still not SPD -> bump more and retry
            A32.diagonal().add_(10 * jitter)
            L = torch.linalg.cholesky(A32)

        return torch.cholesky_solve(b32.unsqueeze(1), L).squeeze(1)
    """
    Multi-constraint OBS compensation for *individual weights* of a single layer,
    done in ROW-MAJOR coordinate order. Then hard-zeros the selected weights.
    """
    base = model
    suffix = f"{layer_name}" + suf
    for n, p in base.named_parameters():
        if n.endswith(suffix) and ".decoder.layers." in n:
            pname, W = n, p
            break
    if W is None:
        raise RuntimeError(f"Could not find parameter '*.{suffix}' in OPT model.")
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

    compute_dtype = torch.float32
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

        # ########### ADDED THIS TO FOR ISSUES
        theta_b32 = theta_b.to(torch.float32)
        A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc)   # fp32
        theta_s32 = theta_b32.index_select(0, S_loc)
        A_alls = Ainv_b.index_select(1, S_loc)                          # fp32
        A_sall = Ainv_b.index_select(0, S_loc)     
                           
        ####################

        # GEMINI
        A_ss_view = Ainv_b.index_select(0, S_loc).index_select(1, S_loc)
        A_alls    = Ainv_b.index_select(1, S_loc)
        
        # 2. Prepare for the solve operation in float32 for better precision.
        #    - Clone to prevent modifying the original Ainv_b.
        #    - Cast to float32.
        #    - CRITICAL: Add the damping term for numerical stability.
        A_ss_damped = A_ss_view.clone().to(torch.float32)
        A_ss_damped.diagonal().add_(eps)
        theta_s = theta_b.index_select(0, S_loc).to(torch.float32)
        
        # GEMINI
        y = torch.linalg.solve(A_ss_damped, theta_s.unsqueeze(1)).squeeze(1)
        delta = -(A_alls.to(torch.float32) @ y)
        
        # A_ss   = Ainv_b.index_select(0, S_loc).index_select(1, S_loc)
        # A_alls = Ainv_b.index_select(1, S_loc)
        # A_sall = Ainv_b.index_select(0, S_loc)

        # theta_b32 = theta_b.to(torch.float32)
        # theta_s32 = theta_b32.index_select(0, S_loc)

        # y32       = _solve_spd(A_ss, theta_s32)             # fp32
        # delta_fp32   = -(A_alls.to(torch.float32) @ y32)       # fp32
        # theta_b.add_(delta_fp32.to(theta_b.dtype))
        # theta_b.index_fill_(0, S_loc, 0.0)
        


        # y32    = torch.linalg.solve(A_ss, theta_s32.unsqueeze(1)).squeeze(1)
        # delta_fp32  = -(A_alls @ y32) 

        # if b % 5000 == 0:
        #     print(f"delta is {delta_fp32}")
 


        # # --- Δ = -A[:,S] @ (A[S,S]^{-1} @ theta[S]) ---
        # try:
        #     print(f"A_ss is {A_ss}")
        #     y_fp32     = _solve_in_fp32(A_ss, theta_s)                      # fp32
        #     delta_fp32 = -(A_alls.to(compute_dtype) @ y_fp32)               # fp32 mm
        # except RuntimeError:
        #     bump = (1e-5 * A_ss.diagonal().abs().mean()).clamp_min(1e-8)
        #     A_ss.diagonal().add_(bump)
        #     y_fp32     = _solve_in_fp32(A_ss, theta_s)                      # fp32
        #     delta_fp32 = -(A_alls.to(compute_dtype) @ y_fp32)
            

        theta_b.add_(delta.to(theta_b.dtype))                      # cast back
        theta_b.index_fill_(0, S_loc, 0.0)

        # --- Woodbury downdate: Ainv <- Ainv - A[:,S] @ inv(A[S,S]) @ A[S,:] ---
        K_fp32     = _solve_in_fp32(A_ss, A_sall)                       # fp32 (k,B_eff)
        prod_fp32  = A_alls.to(compute_dtype) @ K_fp32                  # fp32 (B_eff,B_eff)
        Ainv_b.sub_(prod_fp32.to(Ainv_b.dtype))  

    # write back shaped param
    W.data.copy_(theta.view_as(W))

def save_finetuned(model, save_dir: str, optimizer=None, scaler=None, step: int=None):
    os.makedirs(save_dir, exist_ok=True)

    # Make sure we’re not storing KV cache-disabled accidentally for inference
    model.config.use_cache = True

    # (Optional) save training state so you can resume
    state = {}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if step is not None:
        state["step"] = step

    if state:
        torch.save(state, os.path.join(save_dir, "trainer_state.pt"))

    print(f"Saved fine-tuned model + tokenizer to: {save_dir}")



def personalize_and_save_model_lora(LANGS, model = None, tokenizer = None, max_steps_per_epoch = None, num_epochs = 10):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # model, tokenizer = load_finetuned_for_eval("models/opt125m-code-1-step")
    model = model.to(DEVICE)
    model.config.use_cache = False
    pad_id = model.config.pad_token_id  
    
    # print("Before loader")
    # time_before = time.time()
    train_loader, val_loaders = load_datasets(["Rust"], pad_id, batch_size=8)
    # print("after personalization:")
    # for lang, dl in val_loaders.items():
    #     m = evaluate_code_lm(model, dl, num_grads=500)
    #     print(lang, m)

    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.bfloat16)
    model = model.to(DEVICE)
    # print("original model:")
    # for lang, dl in val_loaders.items():
    #     m = evaluate_code_lm(model, dl)
    #     print(lang, m)

    # model, tokenizer = load_finetuned_for_eval_lora("models/opt125-lora-epoch-0-max_steps-None-20250914-195611-langs-Rust", base_id = "model-combined-c++-c-scala-java", 
    #                                          trainable = True)

    # = build_codeparrot_loaders_streaming(LANGS, tokenizer = tokenizer, max_len=MAX_LEN, batch_size = 8)

    train_opt_on_code_streaming_lora(model, train_loader, max_steps_per_epoch = max_steps_per_epoch, tokenizer = tokenizer, val_loaders = val_loaders, num_epochs = 20, load_adapters= False)

    for lang, dl in val_loaders.items():
        m = evaluate_code_lm(model, dl)
        print(lang, m)


def personalize_and_save_model(LANGS, model = None, tokenizer = None, max_steps_per_epoch = None, num_epochs = 10):
    if model is None:
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # model, tokenizer = load_finetuned_for_eval("models/opt125m-code-1-step")
    model = model.to(DEVICE)
    model.config.use_cache = False
    pad_id = model.config.pad_token_id  
    
    # print("Before loader")
    # time_before = time.time()
    train_loader, val_loaders = load_datasets(LANGS, pad_id, batch_size=8)

    # = build_codeparrot_loaders_streaming(LANGS, tokenizer = tokenizer, max_len=MAX_LEN, batch_size = 8)

    # train_opt_on_code_streaming(model, train_loader, max_steps_per_epoch = max_steps_per_epoch, tokenizer = tokenizer, val_loaders = val_loaders, num_epochs = num_epochs)

    for lang, dl in val_loaders.items():
        m = evaluate_code_lm(model, dl)
        print(lang, m)

def causal_lm_collate_clean(batch, pad_id: int, label_pad_id: int = -100):
    # Only keep the fields we need; ignore everything else (e.g. 'code', 'repo_name', ...)
    ids = []
    attn = []
    for ex in batch:
        if "input_ids" not in ex:
            continue
        ii = ex["input_ids"]
        am = ex.get("attention_mask", [1] * len(ii))
        # convert to tensors
        ids.append(torch.tensor(ii, dtype=torch.long))
        attn.append(torch.tensor(am, dtype=torch.long))

    # pad
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(attn, batch_first=True, padding_value=0)

    # labels = input_ids with pads masked to -100
    labels = input_ids.clone()
    labels[input_ids == pad_id] = label_pad_id

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_finetuned_for_eval(save_dir: str, device: str = "cuda"):
    # tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)  # or torch.float16/bfloat16 if you want
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(save_dir)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = getattr(cfg, "pad_token_id", None) or getattr(cfg, "eos_token_id", None)
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    
    model.config.use_cache = True     # speed up generation/eval
    model.to(device)
    model.eval()
    # return model, tokenizer
    return model

def per_weight_prune_llm(c, path, langs, model = None, tokenizer = None, personalization_hps = None, 
                        print_personalization_scores = False, fisher_block_size = 64,
                     damp = 1e-3, eps = 1e-9, num_grads = 256, print_pruned_only_scores = True, switch_m = False, use_basis = False,
                     just_personalize = False, fisher_m_multiplier=1.0, before_masking_evaluation = True,
                     ):
    
    
    print(f"Path is {path}")
    print(f"fisher_block_size: {fisher_block_size}, num_grads: {num_grads}, switch_m: {switch_m}, damp {damp}")
    masks = torch.load(path, map_location="cuda")
    # print(f"Masks are {masks}")
    if model is None:
        # throw an error 
        raise ValueError("The 'model' parameter must be provided. Ensure the model is initialized and passed to the function.")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        print("loaded model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.pad_token_id

    # model, tokenizer = load_finetuned_for_eval("models/opt125m-code-1-step")
    model = model.to(DEVICE)

    # LANGS = ["Rust", "C++", "Scala"] 

    pad_id = model.config.pad_token_id
    val_loaders = load_datasets_val_only(langs, pad_id = pad_id, batch_size = 1)
    # print("Just Model:")

    
    scala_loader= load_individual_train("Scala", pad_id = pad_id, batch_size = 1)
    # c_loader = load_individual_train("C++", pad_id = pad_id, batch_size = 1)

    if before_masking_evaluation:
        print("===========================")
        print('[Debug] Scores for before masking (after personalization)')
        
        for lang, dl in val_loaders.items():
            m = evaluate_code_lm(model, dl, num_grads=500)
            print(lang, "results before masking (after personalization):", m)
        
        scala_results = evaluate_code_lm(model, scala_loader, num_grads=480)
        print("Scala (forget set) results before maksin (after persoanlziation)", scala_results)
        print("=========================== \n\n")
        # train_opt_on_code_streaming(model, train_loader, max_steps_per_epoch = 5, tokenizer = tokenizer)
        sys.stdout.flush()

    if print_pruned_only_scores:
        print("===========================")
        print('[Debug] Scores for only Masking')
        tmp_opt = copy.deepcopy(model)
        
        apply_weight_masks_(tmp_opt, masks)

        for lang, dl in val_loaders.items():
            m = evaluate_code_lm(tmp_opt, dl, num_grads=500)
            print(lang, "results after only masking:", m)
        
        scala_results = evaluate_code_lm(tmp_opt, scala_loader, num_grads=480)
        print("Scala (forget set) results just masking", scala_results)
        del tmp_opt
        print("=========================== \n\n")
        # train_opt_on_code_streaming(model, train_loader, max_steps_per_epoch = 5, tokenizer = tokenizer)
        sys.stdout.flush()


    #######################################################################################################
    # save_finetuned(model, tokenizer, "models/opt125m-code-1-step")

    res = {}
    layers = [f"decoder.layers.{i}" for i in range(12)]
    fisher_loader= load_individual_train("Rust", pad_id = pad_id, batch_size = 1)
    print(f"Length of fisher loader is {len(fisher_loader)}")
    for layer_name in layers:
        for suf in [".fc1.weight", ".fc2.weight"]:
            # 1) build Fisher inverse for THIS layer only
            pname, F_inv = build_fisher_inv_rowmajor_for_layer(
                model=model,                                   # your wrapper
                layer_name=layer_name,
                EmpiricalBlockFisherInverse=EmpiricalBlockFisherInverseLLM,
                dataloader=fisher_loader,                    # calibration data
                num_grads=num_grads*fisher_m_multiplier,                                # smaller can help off-diagonals
                fisher_block_size=fisher_block_size,                       # bigger than 64; tune to VRAM
                damp=damp,
                switch_m=switch_m,                              # if your class supports the SM denom
                suf = suf
            )
            # 2) fetch the *per-weight* mask for this layer
            print("layer_name+suf is", layer_name+suf)
            mask_2d = masks.get(layer_name+suf, None)                # Bool [out,in]
            print("mask 2d is", mask_2d)
            # 3) compensate+prune this single layer
            cap_compensate_weights_rowmajor_for_layer(
                model=model,
                layer_name=layer_name,
                weight_mask_2d=mask_2d,
                fisher_inv=F_inv,
                eps=1e-9,
                suf = suf
            )
            print(f"layer_name is: {layer_name}")
            sys.stdout.flush()
            # 4) free memory for this layer before the next one
            del F_inv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    

    scala_results = evaluate_code_lm(model, scala_loader, num_grads=480)

    res = [None, None]
    res[1]  = scala_results["top1_acc"]
    print("Scala (forget set results) results after compensating", scala_results)

    
    val_loaders = load_datasets_val_only(["Rust"], pad_id = pad_id, batch_size = 1)
    print("after comp")
    for lang, dl in val_loaders.items():
        m = evaluate_code_lm(model, dl, num_grads=500)
        print(lang, m)

        if lang == 'Rust':
            res[0] = m["top1_acc"]
    
    tuple_res = tuple(res)

    print(f"[Debug] final scores = {tuple_res} after compensation")
    return tuple_res
