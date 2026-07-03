import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=128, num_workers=4):
    """
    Downloads and instantiates the MNIST handwritten digit dataset wrapped in 
    highly optimized standard PyTorch DataLoaders (Section 4 & Section A.9).
    
    Bypasses the "Sample Queue" infrastructure from Appendix A.8, as MNIST
    fits comfortably within memory, allowing us to leverage native multi-process
    DataLoaders without disk overhead.

    Args:
        batch_size: Number of positive samples to draw per training step.
        num_workers: Number of CPU subprocesses to use for data loading.
        
    Returns:
        train_loader: DataLoader instance providing batches of (images, labels).
        test_loader: DataLoader instance providing verification images/labels.
    """
    
    # Define pixel-space transformations.
    # Convert PIL Images to Float Tensors and normalize to [-1, 1] range.
    # Normalizing around zero aligns pixel distributions structurally with 
    # the standard Gaussian noise variables input into the generator.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 1. Fetch training and testing partitions
    train_dataset = datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 2. Build training DataLoader
    # pin_memory speeds up tensor transfers from CPU RAM straight to host GPU VRAM
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True # Keeps batch sizes perfectly uniform for clean distance calculations
    )
    
    # 3. Build test/evaluation DataLoader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, test_loader