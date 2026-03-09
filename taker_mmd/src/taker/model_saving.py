# ---------- SAVE ----------
import torch, json, os

def save_taker_payload(taker_model, path="taker_payload.pt", extra_keys=("prune_mask","fisher_invs","cap_state")):
    # 1) capture the HF state dict
    sd = taker_model.model.state_dict()

    # 2) capture minimal args to rebuild the wrapper
    meta = {
        "base_repo": getattr(taker_model, "model_repo", "google/vit-base-patch16-224"),
        "limit": getattr(taker_model, "limit", None),
        "dtype": getattr(taker_model, "dtype", None),
        "use_accelerator": getattr(taker_model, "use_accelerator", True),
        "mask_fn": getattr(taker_model, "mask_fn", None),
        "svd_attn": getattr(taker_model, "svd_attn", False),
        "model_device": None,  # pick on load
    }

    # 3) add any extra picklable tensors you care about
    extras = {}
    for k in extra_keys:
        if hasattr(taker_model, k):
            try:
                extras[k] = getattr(taker_model, k)
            except Exception:
                pass

    payload = {"meta": meta, "state_dict": sd, "extras": extras}
    torch.save(payload, path)
    return path


# ---------- LOAD ----------
def load_taker_payload(path="taker_payload.pt", **overrides):
    import torch
    from taker.model import Model
    print("in load_taker_payload")
    payload = torch.load(path, map_location="cpu")
    meta = payload["meta"]

    print("before model")
    model = Model(
        overrides.get("base_repo", meta["base_repo"]),
        limit=overrides.get("limit", meta["limit"]),
        dtype=overrides.get("dtype", meta["dtype"]),
        use_accelerator=overrides.get("use_accelerator", meta["use_accelerator"]),
        mask_fn=overrides.get("mask_fn", meta["mask_fn"]),
        svd_attn=overrides.get("svd_attn", meta["svd_attn"]),
        model_device=overrides.get("model_device", meta["model_device"]),
    )
    print("after model")
    incompat = model.model.load_state_dict(payload["state_dict"], strict=False)
    # print(incompat)  # inspect if you changed classifier head

    # restore extras
    for k, v in payload.get("extras", {}).items():
        setattr(model, k, v)

    return model
