import torch
from taker.model import Model
from taker.prune import prune_only_all_layers

def test_prune_only_all_layers():
    # Step 1: Create the model like in fine_tune.py
    model = Model(
        "google/vit-base-patch16-224",
        limit=1000,
        dtype="fp32",
        svd_attn=False,
        use_accelerator=True,
        model_device=None,
        mask_fn="step",
    )

    # Step 2: Load the mask
    mask_path = "./exp1-is-effective/outputs/prune_mask.pt"
    prune_mask = torch.load(mask_path)

    # Step 3: Call the function prune_only_all_layers
    device = next(model.model.parameters()).device
    prune_only_all_layers(model.model, prune_mask, device=device)

    # Step 4: Check that the model weights are zeroed out where the mask is True
    num_layers = prune_mask.shape[0]
    mlp_dim = prune_mask.shape[1]
    inconsistencies = []

    for layer_idx in range(num_layers):
        layer_name = f"encoder.layer.{layer_idx}.intermediate.dense.weight"
        param = dict(model.model.named_parameters())[layer_name]
        param_data = param.data

        # Get the mask for this layer
        mask_layer = prune_mask[layer_idx]
        for neuron_idx in range(mlp_dim):
            if mask_layer[neuron_idx]:
                # Check that the entire row is zero
                if not torch.all(param_data[neuron_idx, :] == 0):
                    print(f'INCONSISTENCY at layer_idx"{layer_idx} - neuron_idx:{neuron_idx}')
                    inconsistencies.append((layer_idx, neuron_idx))

    if inconsistencies:
        print("Inconsistencies found at layers/neurons (row not all zeroed):")
        print(inconsistencies)
    else:
        print("All pruned weights are zero as expected.")

if __name__ == "__main__":
    test_prune_only_all_layers()