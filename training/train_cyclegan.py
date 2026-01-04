"""
CycleGAN Training Script for IR to Pseudo-RGB Generation

Component 4 of the GATE-IR pipeline.
Train a CycleGAN to translate thermal images to pseudo-RGB images.

Usage:
    python train_cyclegan.py --ir_dir /path/to/thermal --rgb_dir /path/to/rgb

Example with FLIR dataset:
    python train_cyclegan.py \
        --ir_dir ./data/FLIR/thermal \
        --rgb_dir ./data/COCO/images \
        --epochs 200 \
        --batch_size 4 \
        --img_size 256
"""

import os
import argparse
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

from cyclegan import (
    CycleGAN,
    Generator,
    create_cyclegan_optimizers,
    convert_ir_to_pseudo_rgb,
    CycleGANTrainer
)


# ==============================================================================
# Dataset Classes
# ==============================================================================

class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation.
    
    Loads images from two domains (IR and RGB) independently.
    """
    
    def __init__(
        self,
        ir_dir: str,
        rgb_dir: str,
        transform_ir: Optional[transforms.Compose] = None,
        transform_rgb: Optional[transforms.Compose] = None,
        img_size: int = 256,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            ir_dir: Directory containing thermal images
            rgb_dir: Directory containing RGB images
            transform_ir: Transforms for IR images
            transform_rgb: Transforms for RGB images
            img_size: Target image size
            max_samples: Maximum number of samples per domain
        """
        self.ir_dir = ir_dir
        self.rgb_dir = rgb_dir
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        
        self.ir_paths = []
        self.rgb_paths = []
        
        for ext in image_extensions:
            self.ir_paths.extend(glob.glob(os.path.join(ir_dir, '**', ext), recursive=True))
            self.rgb_paths.extend(glob.glob(os.path.join(rgb_dir, '**', ext), recursive=True))
        
        # Limit samples if specified
        if max_samples is not None:
            self.ir_paths = self.ir_paths[:max_samples]
            self.rgb_paths = self.rgb_paths[:max_samples]
        
        # Default transforms
        if transform_ir is None:
            self.transform_ir = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Map to [-1, 1]
            ])
        else:
            self.transform_ir = transform_ir
        
        if transform_rgb is None:
            self.transform_rgb = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform_rgb = transform_rgb
        
        print(f"Found {len(self.ir_paths)} IR images and {len(self.rgb_paths)} RGB images")
    
    def __len__(self) -> int:
        return max(len(self.ir_paths), len(self.rgb_paths))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cycle through if one domain is smaller
        ir_idx = idx % len(self.ir_paths)
        rgb_idx = idx % len(self.rgb_paths)
        
        # Load IR image
        ir_path = self.ir_paths[ir_idx]
        ir_img = Image.open(ir_path).convert('L')  # Grayscale
        ir_tensor = self.transform_ir(ir_img)
        
        # Load RGB image
        rgb_path = self.rgb_paths[rgb_idx]
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.transform_rgb(rgb_img)
        
        return ir_tensor, rgb_tensor


class DummyDataset(Dataset):
    """Dummy dataset for testing without actual data."""
    
    def __init__(self, length: int = 100, img_size: int = 256):
        self.length = length
        self.img_size = img_size
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ir = torch.randn(1, self.img_size, self.img_size)
        rgb = torch.randn(3, self.img_size, self.img_size)
        return ir, rgb


# ==============================================================================
# Training Functions
# ==============================================================================

def train_epoch(
    cyclegan: CycleGAN,
    dataloader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    cyclegan.train()
    epoch_losses = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, (real_IR, real_RGB) in enumerate(pbar):
        real_IR = real_IR.to(device)
        real_RGB = real_RGB.to(device)
        
        # Train generators
        optimizer_G.zero_grad()
        g_losses = cyclegan.train_generators(real_IR, real_RGB)
        g_losses['loss_G'].backward()
        optimizer_G.step()
        
        # Train discriminators
        optimizer_D.zero_grad()
        d_losses = cyclegan.train_discriminators(
            real_IR, real_RGB,
            g_losses['fake_IR'], g_losses['fake_RGB']
        )
        d_losses['loss_D'].backward()
        optimizer_D.step()
        
        # Accumulate losses
        batch_losses = {
            'G': g_losses['loss_G'].item(),
            'D': d_losses['loss_D'].item(),
            'cycle': g_losses['loss_cycle'].item()
        }
        
        for k, v in batch_losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({k: f"{v:.3f}" for k, v in batch_losses.items()})
    
    # Average losses
    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in epoch_losses.items()}


def save_samples(
    cyclegan: CycleGAN,
    ir_sample: torch.Tensor,
    rgb_sample: torch.Tensor,
    output_dir: str,
    epoch: int
):
    """Save sample translations for visualization."""
    cyclegan.eval()
    
    with torch.no_grad():
        fake_RGB = cyclegan.G_IR2RGB(ir_sample)
        fake_IR = cyclegan.G_RGB2IR(rgb_sample)
    
    # Denormalize from [-1, 1] to [0, 1]
    def denorm(x):
        return (x + 1) / 2
    
    # Save images
    from torchvision.utils import save_image
    
    save_image(
        denorm(ir_sample.expand(-1, 3, -1, -1)),
        os.path.join(output_dir, f'epoch_{epoch}_real_IR.png')
    )
    save_image(
        denorm(fake_RGB),
        os.path.join(output_dir, f'epoch_{epoch}_fake_RGB.png')
    )
    save_image(
        denorm(rgb_sample),
        os.path.join(output_dir, f'epoch_{epoch}_real_RGB.png')
    )
    save_image(
        denorm(fake_IR.expand(-1, 3, -1, -1)),
        os.path.join(output_dir, f'epoch_{epoch}_fake_IR.png')
    )


# ==============================================================================
# Main Training Script
# ==============================================================================

def main(args):
    """Main training function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Create dataset
    if args.use_dummy_data:
        print("Using dummy data for testing...")
        dataset = DummyDataset(length=args.dummy_samples, img_size=args.img_size)
    else:
        dataset = UnpairedImageDataset(
            ir_dir=args.ir_dir,
            rgb_dir=args.rgb_dir,
            img_size=args.img_size,
            max_samples=args.max_samples
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create model
    print("Creating CycleGAN model...")
    cyclegan = CycleGAN(
        ir_channels=1,
        rgb_channels=3,
        ngf=args.ngf,
        ndf=args.ndf,
        n_residual_blocks=args.n_residual_blocks,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in cyclegan.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizers
    optimizer_G, optimizer_D = create_cyclegan_optimizers(
        cyclegan,
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )
    
    # Learning rate schedulers
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - args.decay_epoch) / (args.epochs - args.decay_epoch + 1)
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        cyclegan.G_IR2RGB.load_state_dict(checkpoint['G_IR2RGB_state'])
        cyclegan.G_RGB2IR.load_state_dict(checkpoint['G_RGB2IR_state'])
        cyclegan.D_RGB.load_state_dict(checkpoint['D_RGB_state'])
        cyclegan.D_IR.load_state_dict(checkpoint['D_IR_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Get sample batch for visualization
    sample_batch = next(iter(dataloader))
    sample_IR = sample_batch[0][:4].to(device)
    sample_RGB = sample_batch[1][:4].to(device)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        losses = train_epoch(
            cyclegan, dataloader,
            optimizer_G, optimizer_D,
            device, epoch + 1
        )
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  G: {losses['G']:.4f}, D: {losses['D']:.4f}, Cycle: {losses['cycle']:.4f}")
        print(f"  LR: {scheduler_G.get_last_lr()[0]:.6f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save samples
        if (epoch + 1) % args.sample_interval == 0:
            save_samples(cyclegan, sample_IR, sample_RGB, args.sample_dir, epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'G_IR2RGB_state': cyclegan.G_IR2RGB.state_dict(),
                'G_RGB2IR_state': cyclegan.G_RGB2IR.state_dict(),
                'D_RGB_state': cyclegan.D_RGB.state_dict(),
                'D_IR_state': cyclegan.D_IR.state_dict(),
                'optimizer_G_state': optimizer_G.state_dict(),
                'optimizer_D_state': optimizer_D.state_dict(),
                'losses': losses
            }
            path = os.path.join(args.checkpoint_dir, f'cyclegan_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, path)
            print(f"  Saved checkpoint: {path}")
            
            # Also save generator separately for easy inference
            gen_path = os.path.join(args.checkpoint_dir, 'G_IR2RGB_latest.pth')
            torch.save(cyclegan.G_IR2RGB.state_dict(), gen_path)
    
    # Save final models
    print("\nTraining complete!")
    final_path = os.path.join(args.checkpoint_dir, 'cyclegan_final.pth')
    torch.save({
        'G_IR2RGB_state': cyclegan.G_IR2RGB.state_dict(),
        'G_RGB2IR_state': cyclegan.G_RGB2IR.state_dict(),
    }, final_path)
    print(f"Saved final model: {final_path}")


# ==============================================================================
# CLI Arguments
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CycleGAN for IR to Pseudo-RGB translation'
    )
    
    # Data arguments
    parser.add_argument('--ir_dir', type=str, default='./data/thermal',
                        help='Directory containing thermal/IR images')
    parser.add_argument('--rgb_dir', type=str, default='./data/rgb',
                        help='Directory containing RGB images')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per domain')
    
    # Model arguments
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator filters')
    parser.add_argument('--n_residual_blocks', type=int, default=9,
                        help='Number of residual blocks in generator')
    
    # Loss arguments
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='Identity loss weight (relative to cycle)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='Epoch to start LR decay')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cyclegan',
                        help='Directory for saving checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples/cyclegan',
                        help='Directory for saving sample images')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epochs between checkpoints')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Epochs between sample generation')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Testing arguments
    parser.add_argument('--use_dummy_data', action='store_true',
                        help='Use dummy data for testing')
    parser.add_argument('--dummy_samples', type=int, default=100,
                        help='Number of dummy samples')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
