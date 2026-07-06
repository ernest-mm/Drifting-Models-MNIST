import torch
import torchvision
import os


def load_state_dict_any(path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)

    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                return state[key]

    return state

def save_image_grid(tensor_batch, fp, nrow=8):
    """
    Takes a raw batch of generated or data image tensors, denormalizes them 
    from [-1, 1] back to [0, 1], and serializes them as a neat image grid sheet.
    
    Args:
        tensor_batch: Tensor of shape [B, 1, 28, 28] 
        fp: File path string or Path object where the grid image will be saved.
        nrow: Number of images displayed in each row of the grid.
    """
    # 1. Create directory if it doesn't exist yet
    dir_name = os.path.dirname(fp)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    # 2. Denormalize the pixels from [-1, 1] to [0, 1]
    # The dataset transforms used: x_norm = (x - 0.5) / 0.5 -> x = x_norm * 0.5 + 0.5
    denorm_batch = tensor_batch.detach().cpu() * 0.5 + 0.5
    # Clamp pixel values to safeguard against slight floating-point overflows/underflows
    denorm_batch = torch.clamp(denorm_batch, 0.0, 1.0)
    
    # 3. Compile the grid structure and save to disk
    torchvision.utils.save_image(denorm_batch, fp, nrow=nrow)


class CheckpointManager:
    """
    Handles saving and reloading model training states securely to prevent 
    progress loss and facilitate downstream single-step (1-NFE) inference tasks.
    """
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, loss, filename="drifting_generator_v1.pt"):
        """Saves weights along with critical training metadata."""
        path = os.path.join(self.checkpoint_dir, filename)
        
        # Unwrap model if it's wrapped in DistributedDataParallel or DataParallel
        model_to_save = model.module if hasattr(model, 'module') else model
        
        state = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(state, path)
        print(f"[*] Checkpoint successfully saved to: {path}")

    def load(self, model, optimizer=None, filename="drifting_generator_v1.pt", device="cpu"):
        """Loads weights and returns the historical epoch metadata."""
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            print(f"[!] No checkpoint found at {path}. Starting training from scratch.")
            return 0
            
        state = torch.load(path, map_location=device)
        model_to_load = model.module if hasattr(model, 'module') else model

        if isinstance(state, dict) and 'model_state_dict' in state:
            model_to_load.load_state_dict(state['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            epoch = state.get('epoch', 0)
        else:
            model_to_load.load_state_dict(state)
            epoch = 0

        print(f"[*] Checkpoint loaded from: {path} (Resuming from Epoch {epoch})")
        return epoch