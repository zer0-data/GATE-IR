"""
CycleGAN: Unpaired Image-to-Image Translation for IR to Pseudo-RGB

Component 4 of the GATE-IR pipeline.
Generates pseudo-RGB images from thermal IR images using unpaired
image-to-image translation for use in knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import os


# ==============================================================================
# Generator Architecture (ResNet-based)
# ==============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two convolutions and instance normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based Generator for CycleGAN.
    
    Architecture:
        Encoder → Residual Blocks → Decoder
    
    Uses reflection padding and instance normalization for
    better visual quality in image translation tasks.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        ngf: int = 64,
        n_residual_blocks: int = 9,
        use_dropout: bool = False
    ):
        """
        Initialize Generator.
        
        Args:
            in_channels: Number of input channels (1 for thermal)
            out_channels: Number of output channels (3 for RGB)
            ngf: Number of generator filters in first layer
            n_residual_blocks: Number of residual blocks
            use_dropout: Whether to use dropout in residual blocks
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution block
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        residual = []
        for _ in range(n_residual_blocks):
            residual.append(ResidualBlock(ngf * mult))
        
        # Upsampling
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2),
                    3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        ]
        
        self.encoder = nn.Sequential(*encoder)
        self.residual = nn.Sequential(*residual)
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate translated image.
        
        Args:
            x: Input image (B, in_channels, H, W)
        
        Returns:
            Translated image (B, out_channels, H, W)
        """
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x


# ==============================================================================
# Discriminator Architecture (PatchGAN)
# ==============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    Classifies 70x70 overlapping patches as real or fake.
    Outputs a 2D feature map where each location corresponds
    to a patch in the input image.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3
    ):
        """
        Initialize Discriminator.
        
        Args:
            in_channels: Number of input channels
            ndf: Number of discriminator filters in first layer
            n_layers: Number of convolutional layers
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate input image.
        
        Args:
            x: Input image
        
        Returns:
            Patch-level predictions (real/fake probabilities)
        """
        return self.model(x)


# ==============================================================================
# Image Buffer for Training Stability
# ==============================================================================

class ImageBuffer:
    """
    Buffer for storing generated images for discriminator training.
    
    Returns a mix of recent and buffered images to stabilize training.
    """
    
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.buffer: List[torch.Tensor] = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """Add images to buffer and return mix of new and buffered."""
        result = []
        
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image.clone())
                result.append(image)
            else:
                if torch.rand(1).item() > 0.5:
                    # Return random buffered image
                    idx = torch.randint(0, self.buffer_size, (1,)).item()
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = image.clone()
                    result.append(tmp)
                else:
                    result.append(image)
        
        return torch.cat(result, dim=0)


# ==============================================================================
# CycleGAN Model
# ==============================================================================

class CycleGAN(nn.Module):
    """
    Complete CycleGAN for unpaired IR to RGB translation.
    
    Components:
        - G_IR2RGB: Thermal → Pseudo-RGB generator
        - G_RGB2IR: RGB → Pseudo-Thermal generator
        - D_RGB: RGB domain discriminator
        - D_IR: Thermal domain discriminator
    
    Losses:
        - Adversarial loss (LSGAN)
        - Cycle consistency loss
        - Identity loss (optional)
    
    Example:
        >>> cyclegan = CycleGAN()
        >>> ir_image = torch.rand(4, 1, 256, 256)
        >>> rgb_image = torch.rand(4, 3, 256, 256)
        >>> losses = cyclegan.train_step(ir_image, rgb_image, optimizer_G, optimizer_D)
    """
    
    def __init__(
        self,
        ir_channels: int = 1,
        rgb_channels: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_residual_blocks: int = 9,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        use_identity_loss: bool = True
    ):
        """
        Initialize CycleGAN.
        
        Args:
            ir_channels: Number of IR (thermal) channels
            rgb_channels: Number of RGB channels
            ngf: Number of generator filters
            ndf: Number of discriminator filters
            n_residual_blocks: Number of residual blocks in generators
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss (relative to cycle)
            use_identity_loss: Whether to use identity loss
        """
        super().__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.use_identity_loss = use_identity_loss
        
        # Generators
        self.G_IR2RGB = Generator(ir_channels, rgb_channels, ngf, n_residual_blocks)
        self.G_RGB2IR = Generator(rgb_channels, ir_channels, ngf, n_residual_blocks)
        
        # Discriminators
        self.D_RGB = Discriminator(rgb_channels, ndf)
        self.D_IR = Discriminator(ir_channels, ndf)
        
        # Image buffers for training stability
        self.fake_RGB_buffer = ImageBuffer()
        self.fake_IR_buffer = ImageBuffer()
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def forward(self, ir_image: torch.Tensor) -> torch.Tensor:
        """
        Generate pseudo-RGB from IR (inference mode).
        
        Args:
            ir_image: Thermal image (B, 1, H, W)
        
        Returns:
            Pseudo-RGB image (B, 3, H, W)
        """
        return self.G_IR2RGB(ir_image)
    
    def generate_pseudo_rgb(self, ir_image: torch.Tensor) -> torch.Tensor:
        """Alias for forward() - generate pseudo-RGB from IR."""
        return self.forward(ir_image)
    
    def train_generators(
        self,
        real_IR: torch.Tensor,
        real_RGB: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator losses.
        
        Args:
            real_IR: Real thermal images
            real_RGB: Real RGB images
        
        Returns:
            Dictionary of losses
        """
        device = real_IR.device
        batch_size = real_IR.size(0)
        
        # Create target tensors
        real_label = torch.ones(batch_size, 1, device=device)
        
        # Identity loss (optional)
        loss_identity = torch.tensor(0.0, device=device)
        if self.use_identity_loss:
            # G_IR2RGB should be identity if input is RGB
            same_RGB = self.G_IR2RGB(real_RGB[:, :1])  # Take first channel
            loss_id_RGB = self.criterion_identity(same_RGB, real_RGB)
            
            # G_RGB2IR should be identity if input is IR
            fake_3ch_IR = real_IR.expand(-1, 3, -1, -1)  # Expand to 3 channels
            same_IR = self.G_RGB2IR(fake_3ch_IR)
            loss_id_IR = self.criterion_identity(same_IR, real_IR)
            
            loss_identity = (loss_id_RGB + loss_id_IR) * self.lambda_identity * self.lambda_cycle
        
        # GAN loss
        fake_RGB = self.G_IR2RGB(real_IR)
        pred_fake_RGB = self.D_RGB(fake_RGB)
        loss_GAN_IR2RGB = self.criterion_GAN(
            pred_fake_RGB.mean(dim=[1, 2, 3]),
            real_label.squeeze()
        )
        
        fake_IR = self.G_RGB2IR(real_RGB)
        pred_fake_IR = self.D_IR(fake_IR)
        loss_GAN_RGB2IR = self.criterion_GAN(
            pred_fake_IR.mean(dim=[1, 2, 3]),
            real_label.squeeze()
        )
        
        # Cycle consistency loss
        recovered_IR = self.G_RGB2IR(fake_RGB)
        loss_cycle_IR = self.criterion_cycle(recovered_IR, real_IR)
        
        recovered_RGB = self.G_IR2RGB(fake_IR)
        loss_cycle_RGB = self.criterion_cycle(recovered_RGB, real_RGB)
        
        loss_cycle = (loss_cycle_IR + loss_cycle_RGB) * self.lambda_cycle
        
        # Total generator loss
        loss_G = loss_GAN_IR2RGB + loss_GAN_RGB2IR + loss_cycle + loss_identity
        
        return {
            'loss_G': loss_G,
            'loss_GAN_IR2RGB': loss_GAN_IR2RGB,
            'loss_GAN_RGB2IR': loss_GAN_RGB2IR,
            'loss_cycle': loss_cycle,
            'loss_identity': loss_identity,
            'fake_RGB': fake_RGB.detach(),
            'fake_IR': fake_IR.detach()
        }
    
    def train_discriminators(
        self,
        real_IR: torch.Tensor,
        real_RGB: torch.Tensor,
        fake_IR: torch.Tensor,
        fake_RGB: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator losses.
        
        Args:
            real_IR: Real thermal images
            real_RGB: Real RGB images
            fake_IR: Generated thermal images
            fake_RGB: Generated RGB images
        
        Returns:
            Dictionary of losses
        """
        device = real_IR.device
        batch_size = real_IR.size(0)
        
        # Target tensors
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)
        
        # Use image buffer for stability
        fake_RGB = self.fake_RGB_buffer.push_and_pop(fake_RGB)
        fake_IR = self.fake_IR_buffer.push_and_pop(fake_IR)
        
        # D_RGB loss
        pred_real_RGB = self.D_RGB(real_RGB)
        loss_D_RGB_real = self.criterion_GAN(
            pred_real_RGB.mean(dim=[1, 2, 3]),
            real_label.squeeze()
        )
        
        pred_fake_RGB = self.D_RGB(fake_RGB.detach())
        loss_D_RGB_fake = self.criterion_GAN(
            pred_fake_RGB.mean(dim=[1, 2, 3]),
            fake_label.squeeze()
        )
        
        loss_D_RGB = (loss_D_RGB_real + loss_D_RGB_fake) * 0.5
        
        # D_IR loss
        pred_real_IR = self.D_IR(real_IR)
        loss_D_IR_real = self.criterion_GAN(
            pred_real_IR.mean(dim=[1, 2, 3]),
            real_label.squeeze()
        )
        
        pred_fake_IR = self.D_IR(fake_IR.detach())
        loss_D_IR_fake = self.criterion_GAN(
            pred_fake_IR.mean(dim=[1, 2, 3]),
            fake_label.squeeze()
        )
        
        loss_D_IR = (loss_D_IR_real + loss_D_IR_fake) * 0.5
        
        # Total discriminator loss
        loss_D = loss_D_RGB + loss_D_IR
        
        return {
            'loss_D': loss_D,
            'loss_D_RGB': loss_D_RGB,
            'loss_D_IR': loss_D_IR
        }
    
    def get_generator_params(self):
        """Get parameters for generator optimizer."""
        return list(self.G_IR2RGB.parameters()) + list(self.G_RGB2IR.parameters())
    
    def get_discriminator_params(self):
        """Get parameters for discriminator optimizer."""
        return list(self.D_RGB.parameters()) + list(self.D_IR.parameters())


# ==============================================================================
# Training Utilities
# ==============================================================================

def create_cyclegan_optimizers(
    cyclegan: CycleGAN,
    lr: float = 0.0002,
    betas: Tuple[float, float] = (0.5, 0.999)
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Create optimizers for CycleGAN training.
    
    Args:
        cyclegan: CycleGAN model
        lr: Learning rate
        betas: Adam beta parameters
    
    Returns:
        Tuple of (generator_optimizer, discriminator_optimizer)
    """
    optimizer_G = torch.optim.Adam(
        cyclegan.get_generator_params(),
        lr=lr, betas=betas
    )
    optimizer_D = torch.optim.Adam(
        cyclegan.get_discriminator_params(),
        lr=lr, betas=betas
    )
    return optimizer_G, optimizer_D


def convert_ir_to_pseudo_rgb(
    ir_image: torch.Tensor,
    generator: Generator,
    normalize_output: bool = True
) -> torch.Tensor:
    """
    Convert thermal IR image to pseudo-RGB using trained generator.
    
    Args:
        ir_image: Thermal image (B, 1, H, W), normalized to [-1, 1] or [0, 1]
        generator: Trained G_IR2RGB generator
        normalize_output: Scale output from [-1, 1] to [0, 1]
    
    Returns:
        Pseudo-RGB image (B, 3, H, W)
    """
    generator.eval()
    
    with torch.no_grad():
        # Ensure input is in [-1, 1] range for CycleGAN
        if ir_image.min() >= 0:
            ir_image = ir_image * 2 - 1
        
        pseudo_rgb = generator(ir_image)
        
        if normalize_output:
            pseudo_rgb = (pseudo_rgb + 1) / 2
    
    return pseudo_rgb


# ==============================================================================
# Training Script
# ==============================================================================

class CycleGANTrainer:
    """
    Complete training pipeline for CycleGAN.
    
    Handles data loading, training loop, checkpointing, and logging.
    """
    
    def __init__(
        self,
        cyclegan: CycleGAN,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda'),
        checkpoint_dir: str = './checkpoints'
    ):
        self.cyclegan = cyclegan.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_step(
        self,
        real_IR: torch.Tensor,
        real_RGB: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            real_IR: Batch of thermal images
            real_RGB: Batch of RGB images
        
        Returns:
            Dictionary of loss values
        """
        real_IR = real_IR.to(self.device)
        real_RGB = real_RGB.to(self.device)
        
        # Train generators
        self.optimizer_G.zero_grad()
        g_losses = self.cyclegan.train_generators(real_IR, real_RGB)
        g_losses['loss_G'].backward()
        self.optimizer_G.step()
        
        # Train discriminators
        self.optimizer_D.zero_grad()
        d_losses = self.cyclegan.train_discriminators(
            real_IR, real_RGB,
            g_losses['fake_IR'], g_losses['fake_RGB']
        )
        d_losses['loss_D'].backward()
        self.optimizer_D.step()
        
        # Combine losses for logging
        losses = {
            'G_total': g_losses['loss_G'].item(),
            'G_GAN_IR2RGB': g_losses['loss_GAN_IR2RGB'].item(),
            'G_GAN_RGB2IR': g_losses['loss_GAN_RGB2IR'].item(),
            'G_cycle': g_losses['loss_cycle'].item(),
            'G_identity': g_losses['loss_identity'].item(),
            'D_total': d_losses['loss_D'].item(),
            'D_RGB': d_losses['loss_D_RGB'].item(),
            'D_IR': d_losses['loss_D_IR'].item()
        }
        
        return losses
    
    def save_checkpoint(self, epoch: int, losses: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'G_IR2RGB_state': self.cyclegan.G_IR2RGB.state_dict(),
            'G_RGB2IR_state': self.cyclegan.G_RGB2IR.state_dict(),
            'D_RGB_state': self.cyclegan.D_RGB.state_dict(),
            'D_IR_state': self.cyclegan.D_IR.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
            'losses': losses
        }
        
        path = os.path.join(self.checkpoint_dir, f'cyclegan_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
        # Save generator separately for easy inference
        gen_path = os.path.join(self.checkpoint_dir, 'G_IR2RGB_latest.pth')
        torch.save(self.cyclegan.G_IR2RGB.state_dict(), gen_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.cyclegan.G_IR2RGB.load_state_dict(checkpoint['G_IR2RGB_state'])
        self.cyclegan.G_RGB2IR.load_state_dict(checkpoint['G_RGB2IR_state'])
        self.cyclegan.D_RGB.load_state_dict(checkpoint['D_RGB_state'])
        self.cyclegan.D_IR.load_state_dict(checkpoint['D_IR_state'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])
        
        return checkpoint['epoch']


# ==============================================================================
# Test Code
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CycleGAN Module Test")
    print("=" * 60)
    
    # Create model
    cyclegan = CycleGAN(
        ir_channels=1,
        rgb_channels=3,
        ngf=64,
        ndf=64,
        n_residual_blocks=6  # Smaller for testing
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in cyclegan.parameters())
    g_params = sum(p.numel() for p in cyclegan.get_generator_params())
    d_params = sum(p.numel() for p in cyclegan.get_discriminator_params())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create test data
    batch_size = 2
    ir_images = torch.rand(batch_size, 1, 256, 256) * 2 - 1  # [-1, 1]
    rgb_images = torch.rand(batch_size, 3, 256, 256) * 2 - 1
    
    print(f"\nIR input shape: {ir_images.shape}")
    print(f"RGB input shape: {rgb_images.shape}")
    
    # Test forward pass
    with torch.no_grad():
        pseudo_rgb = cyclegan.generate_pseudo_rgb(ir_images)
    
    print(f"Pseudo-RGB output shape: {pseudo_rgb.shape}")
    print(f"Pseudo-RGB range: [{pseudo_rgb.min():.3f}, {pseudo_rgb.max():.3f}]")
    
    # Test training step
    print("\n--- Training Step Test ---")
    optimizer_G, optimizer_D = create_cyclegan_optimizers(cyclegan, lr=0.0002)
    
    cyclegan.train()
    g_losses = cyclegan.train_generators(ir_images, rgb_images)
    d_losses = cyclegan.train_discriminators(
        ir_images, rgb_images,
        g_losses['fake_IR'], g_losses['fake_RGB']
    )
    
    print(f"Generator loss: {g_losses['loss_G'].item():.4f}")
    print(f"  - GAN IR→RGB: {g_losses['loss_GAN_IR2RGB'].item():.4f}")
    print(f"  - GAN RGB→IR: {g_losses['loss_GAN_RGB2IR'].item():.4f}")
    print(f"  - Cycle: {g_losses['loss_cycle'].item():.4f}")
    print(f"Discriminator loss: {d_losses['loss_D'].item():.4f}")
    
    # Test conversion utility
    print("\n--- Conversion Utility Test ---")
    generator = cyclegan.G_IR2RGB
    converted = convert_ir_to_pseudo_rgb(ir_images, generator)
    print(f"Converted shape: {converted.shape}")
    print(f"Converted range: [{converted.min():.3f}, {converted.max():.3f}]")
