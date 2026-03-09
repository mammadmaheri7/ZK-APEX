import torch
import torch.nn as nn
from .model import Model
import sys
import time

# ==========================================
# 1. Define the Adapter Module
# ==========================================
class AdaptMLP(nn.Module):
    def __init__(self, dim, adapter_dim=64, scale=0.1):
        super().__init__()
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.scale = nn.Parameter(torch.tensor(scale))
        
        # Initialize up_proj to near-zero so training starts with neutral effect
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.scale * self.up_proj(self.act(self.down_proj(x)))

# ==========================================
# 2. Controller to Manage Hooks per Layer
# ==========================================
class AdaptFormerLayerHandler:
    """
    Manages state for a single layer. 
    It captures the input at the start of the MLP and injects 
    the adapter output at the end of the MLP.
    """
    def __init__(self, adapter_module):
        self.adapter = adapter_module
        self.cached_input = None

    def pre_hook(self, module, input):
        # Cache the input (x_in). Input is a tuple, we want the tensor.
        self.cached_input = input[0]

    def post_hook(self, module, input, output):
        # output is the result of the original MLP
        # We compute: Output_new = Output_old + Adapter(Cached_Input)
        if self.cached_input is None:
            raise ValueError("Pre-hook was not called before post-hook!")
        
        adapter_out = self.adapter(self.cached_input)
        
        # Clear cache to save memory
        self.cached_input = None
        
        return output + adapter_out

# ==========================================
# 3. Main Injection Script
# ==========================================
def inject_adaptformer(opt, adapter_dim=64):
    print(f"Injecting AdaptFormer (dim={adapter_dim}) into {opt.cfg.n_layers} layers...")
    
    # 1. Freeze the entire base model
    for param in opt.model.parameters():
        param.requires_grad = False
        
    adapter_layers = nn.ModuleList()
    
    ''
    # 2. Iterate over layers and attach hooks
    for layer_idx, layer in enumerate(opt.layers):
        # Identify the standard input/output modules using your Model class logic
        # 'pre_mlp' -> typically the LayerNorm before MLP
        # 'post_mlp' -> typically the Linear Output Projection of MLP
        _, pre_module = opt.get_module_for_hook_point(layer, "pre_mlp")
        _, post_module = opt.get_module_for_hook_point(layer, "post_mlp")
        
        # Create the adapter and handler
        adapter = AdaptMLP(dim=opt.cfg.d_model, adapter_dim=adapter_dim)
        handler = AdaptFormerLayerHandler(adapter)
        
        # Move adapter to correct device/dtype
        adapter.to(opt.device, dtype=opt.dtype)
        adapter_layers.append(adapter)
        
        # Register Native PyTorch Hooks
        # Note: We use native hooks because we need state passing (cached_input)
        # which is harder to do with the static 'collect' hooks in the config.
        pre_module.register_forward_pre_hook(handler.pre_hook)
        post_module.register_forward_hook(handler.post_hook)
        
    print(f"Successfully injected adapters. Trainable params: {sum(p.numel() for p in adapter_layers.parameters() if p.requires_grad)}")
    return adapter_layers


import torch
import torch.nn as nn
from tqdm import tqdm

def fine_tune_vit(opt, train_loader, val_loader, epochs=5, lr=1e-3, save_path="best_adapter.pt"):
    """
    Fine-tunes the injected AdaptFormer modules using provided DataLoaders.
    
    Args:
        opt: The Model instance (with AdaptFormer already injected).
        train_loader: torch.utils.data.DataLoader for training data.
        val_loader: torch.utils.data.DataLoader for validation data.
        epochs: Number of training epochs.
        lr: Learning rate.
        save_path: Filename to save the best adapter weights.
    """

    print(f"In fine_tune_vit")
    sys.stdout.flush()
    
    adapters = inject_adaptformer(opt) 
    
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    print(f"\nStarting training for {epochs} epochs on device: {opt.device}")
    sys.stdout.flush()
    
    for epoch in range(epochs):
        opt.model.train() 
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_pbar:
            # Handle batch unpacking: assuming (images, labels) or dictionary
            if isinstance(batch, dict):
                pixel_values = batch["pixel_values"].to(opt.device)
                labels = batch["labels"].to(opt.device)
            else:
                pixel_values, labels = batch
                pixel_values = pixel_values.to(opt.device)
                labels = labels.to(opt.device)
            
            # Forward pass
            outputs = opt.predictor(pixel_values).logits
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * pixel_values.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item()})
            
        train_loss_avg = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # --- Validation Phase ---
        opt.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                if isinstance(batch, dict):
                    pixel_values = batch["pixel_values"].to(opt.device)
                    labels = batch["labels"].to(opt.device)
                else:
                    pixel_values, labels = batch
                    pixel_values = pixel_values.to(opt.device)
                    labels = labels.to(opt.device)
                
                outputs = opt.predictor(pixel_values).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * pixel_values.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss_avg = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}, Learning Rate: {lr}: Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")
        save_path = f'best_adapter_{epoch}-of-{epochs}-epochs_{lr}-lr_{val_acc:.2f}-valacc.pt'
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> New best validation accuracy! Saving adapters to {save_path}")
            torch.save(adapters.state_dict(), save_path)
            
    print("Training Complete.")
    return adapters