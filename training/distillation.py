"""
Teacher-Student Knowledge Distillation for Cross-Modal Object Detection

Component 5 of the GATE-IR pipeline.
Implements knowledge distillation from a YOLOv8-Large teacher (fed pseudo-RGB)
to a YOLOv8-Small student (fed raw thermal), using feature mimic loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
import os
from tqdm import tqdm


# ==============================================================================
# Feature Alignment Modules
# ==============================================================================

class ChannelAdapter(nn.Module):
    """
    1x1 Convolution adapter for matching channel dimensions.
    
    Used when teacher and student feature maps have different
    channel counts but same spatial dimensions.
    """
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        use_bn: bool = True
    ):
        """
        Initialize ChannelAdapter.
        
        Args:
            student_channels: Number of student feature channels
            teacher_channels: Number of teacher feature channels
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(student_channels, teacher_channels, 1, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(teacher_channels))
        
        self.adapter = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class SpatialAdapter(nn.Module):
    """
    Adapter for matching both spatial and channel dimensions.
    
    Uses interpolation for spatial alignment and 1x1 conv for channels.
    """
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        mode: str = 'bilinear'
    ):
        super().__init__()
        
        self.channel_adapter = ChannelAdapter(student_channels, teacher_channels)
        self.mode = mode
    
    def forward(
        self,
        student_feat: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        # Spatial alignment
        if student_feat.shape[2:] != target_size:
            student_feat = F.interpolate(
                student_feat,
                size=target_size,
                mode=self.mode,
                align_corners=False if self.mode != 'nearest' else None
            )
        
        # Channel alignment
        return self.channel_adapter(student_feat)


# ==============================================================================
# Distillation Loss Functions
# ==============================================================================

class FeatureMimicLoss(nn.Module):
    """
    Feature Mimicking Loss for Knowledge Distillation.
    
    Computes MSE between aligned student and teacher feature maps
    at specified layers (typically P3, P4).
    
    Supports optional attention-based weighting to focus on
    important regions (e.g., object locations).
    """
    
    def __init__(
        self,
        student_channels: Dict[str, int],
        teacher_channels: Dict[str, int],
        layers: List[str] = ['P3', 'P4'],
        loss_type: str = 'mse',
        normalize_features: bool = True,
        use_attention: bool = False
    ):
        """
        Initialize FeatureMimicLoss.
        
        Args:
            student_channels: Dict mapping layer name to student channel count
            teacher_channels: Dict mapping layer name to teacher channel count
            layers: Which layers to compute loss on
            loss_type: 'mse', 'l1', or 'smooth_l1'
            normalize_features: Whether to normalize features before comparison
            use_attention: Weight loss by spatial attention maps
        """
        super().__init__()
        
        self.layers = layers
        self.normalize_features = normalize_features
        self.use_attention = use_attention
        
        # Create adapters for each layer
        self.adapters = nn.ModuleDict()
        for layer in layers:
            if student_channels[layer] != teacher_channels[layer]:
                self.adapters[layer] = ChannelAdapter(
                    student_channels[layer],
                    teacher_channels[layer]
                )
            else:
                self.adapters[layer] = nn.Identity()
        
        # Loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _normalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize feature maps along channel dimension."""
        return F.normalize(feat, p=2, dim=1)
    
    def _compute_attention(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map from features."""
        # Mean across channels, then normalize
        attention = feat.abs().mean(dim=1, keepdim=True)
        attention = attention / (attention.max() + 1e-8)
        return attention
    
    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute feature mimic loss.
        
        Args:
            student_features: Dict of student feature maps
            teacher_features: Dict of teacher feature maps
        
        Returns:
            total_loss: Summed loss across all layers
            layer_losses: Dict of per-layer losses
        """
        total_loss = torch.tensor(0.0, device=next(iter(student_features.values())).device)
        layer_losses = {}
        
        for layer in self.layers:
            student_feat = student_features[layer]
            teacher_feat = teacher_features[layer]
            
            # Adapt student features to match teacher
            student_feat = self.adapters[layer](student_feat)
            
            # Handle spatial size mismatch
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                student_feat = F.interpolate(
                    student_feat,
                    size=teacher_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Normalize if requested
            if self.normalize_features:
                student_feat = self._normalize(student_feat)
                teacher_feat = self._normalize(teacher_feat)
            
            # Compute loss
            loss = self.criterion(student_feat, teacher_feat.detach())
            
            # Apply attention weighting if requested
            if self.use_attention:
                attention = self._compute_attention(teacher_feat.detach())
                loss = loss * attention
            
            # Average over all dimensions
            layer_loss = loss.mean()
            layer_losses[layer] = layer_loss
            total_loss = total_loss + layer_loss
        
        return total_loss, layer_losses


class ResponseDistillationLoss(nn.Module):
    """
    Response-based distillation loss.
    
    Matches the soft predictions (logits) between teacher and student
    using KL divergence with temperature scaling.
    """
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute response distillation loss.
        
        Args:
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits
        
        Returns:
            KL divergence loss scaled by temperature
        """
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        loss = self.kl_div(student_soft, teacher_soft.detach())
        loss = loss * (self.temperature ** 2)
        
        return loss


# ==============================================================================
# Distillation Trainer
# ==============================================================================

class DistillationTrainer:
    """
    Complete Teacher-Student Distillation Training Pipeline.
    
    Trains a student network (YOLOv8-Small for thermal) to mimic
    a teacher network (YOLOv8-Large on pseudo-RGB) using:
    - Standard detection loss on student
    - Feature mimic loss between intermediate features
    
    Data Flow:
        Thermal Batch → CycleGAN → Pseudo-RGB → Teacher → Teacher Features
        Thermal Batch → Student → Student Features
        
        Loss = YOLO_Loss(Student) + α * FeatureMimic(Student, Teacher)
    
    Example:
        >>> trainer = DistillationTrainer(
        ...     student=student_model,
        ...     teacher=teacher_model,
        ...     cyclegan_generator=G_IR2RGB,
        ...     device='cuda'
        ... )
        >>> for thermal_batch, targets in dataloader:
        ...     losses = trainer.train_step(thermal_batch, targets)
    """
    
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        cyclegan_generator: nn.Module,
        yolo_loss_fn: Callable,
        student_channels: Dict[str, int],
        teacher_channels: Dict[str, int],
        distill_layers: List[str] = ['P3', 'P4'],
        alpha: float = 1.0,
        feature_loss_type: str = 'mse',
        device: torch.device = torch.device('cuda'),
        lr: float = 0.001,
        weight_decay: float = 0.0005
    ):
        """
        Initialize DistillationTrainer.
        
        Args:
            student: Student model (YOLOv8-Small Thermal)
            teacher: Teacher model (YOLOv8-Large, pretrained)
            cyclegan_generator: Trained G_IR2RGB generator
            yolo_loss_fn: YOLO detection loss function
            student_channels: Channel counts for student layers
            teacher_channels: Channel counts for teacher layers
            distill_layers: Which layers to distill from
            alpha: Weight for feature mimic loss
            feature_loss_type: Type of feature loss
            device: Training device
            lr: Learning rate
            weight_decay: Optimizer weight decay
        """
        self.device = device
        self.alpha = alpha
        
        # Models
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.cyclegan_generator = cyclegan_generator.to(device)
        
        # Freeze teacher and CycleGAN generator
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.cyclegan_generator.eval()
        for param in self.cyclegan_generator.parameters():
            param.requires_grad = False
        
        # Loss functions
        self.yolo_loss_fn = yolo_loss_fn
        self.feature_loss = FeatureMimicLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            layers=distill_layers,
            loss_type=feature_loss_type
        ).to(device)
        
        # Optimizer (student + feature adapters)
        params = list(self.student.parameters()) + list(self.feature_loss.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Adjust based on total epochs
            eta_min=lr * 0.01
        )
    
    def _generate_pseudo_rgb(self, thermal: torch.Tensor) -> torch.Tensor:
        """Generate pseudo-RGB from thermal using CycleGAN."""
        with torch.no_grad():
            # Normalize to [-1, 1] for CycleGAN if needed
            if thermal.min() >= 0:
                thermal_normalized = thermal * 2 - 1
            else:
                thermal_normalized = thermal
            
            pseudo_rgb = self.cyclegan_generator(thermal_normalized)
            
            # Convert back to [0, 1]
            pseudo_rgb = (pseudo_rgb + 1) / 2
            
        return pseudo_rgb
    
    def train_step(
        self,
        thermal_batch: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            thermal_batch: Batch of thermal images (B, 1, H, W)
            targets: Optional ground truth targets for YOLO loss
        
        Returns:
            Dictionary of loss values
        """
        thermal_batch = thermal_batch.to(self.device)
        if targets is not None:
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
        
        self.student.train()
        
        # Generate pseudo-RGB for teacher
        pseudo_rgb = self._generate_pseudo_rgb(thermal_batch)
        
        # Get teacher features (frozen)
        with torch.no_grad():
            teacher_output = self.teacher(pseudo_rgb, return_features=True)
            teacher_features = teacher_output['fused_features']
        
        # Get student features
        student_output = self.student(thermal_batch, return_features=True)
        student_features = student_output['fused_features']
        student_predictions = student_output['predictions']
        
        # Compute YOLO detection loss (if targets provided)
        if targets is not None and self.yolo_loss_fn is not None:
            yolo_loss = self.yolo_loss_fn(student_predictions, targets)
        else:
            # For unsupervised distillation, can use pseudo-labels from teacher
            yolo_loss = torch.tensor(0.0, device=self.device)
        
        # Compute feature mimic loss
        feature_loss, layer_losses = self.feature_loss(
            student_features,
            teacher_features
        )
        
        # Total loss
        total_loss = yolo_loss + self.alpha * feature_loss
        
        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Build loss dict
        losses = {
            'total_loss': total_loss.item(),
            'yolo_loss': yolo_loss.item() if isinstance(yolo_loss, torch.Tensor) else yolo_loss,
            'feature_loss': feature_loss.item(),
        }
        losses.update({f'feature_{k}': v.item() for k, v in layer_losses.items()})
        
        return losses
    
    def validate_step(
        self,
        thermal_batch: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Perform validation step (no gradients).
        
        Args:
            thermal_batch: Batch of thermal images
            targets: Optional ground truth targets
        
        Returns:
            Dictionary of loss values
        """
        self.student.eval()
        
        with torch.no_grad():
            thermal_batch = thermal_batch.to(self.device)
            
            # Generate pseudo-RGB
            pseudo_rgb = self._generate_pseudo_rgb(thermal_batch)
            
            # Get features
            teacher_output = self.teacher(pseudo_rgb, return_features=True)
            student_output = self.student(thermal_batch, return_features=True)
            
            # Compute losses
            feature_loss, layer_losses = self.feature_loss(
                student_output['fused_features'],
                teacher_output['fused_features']
            )
        
        return {
            'val_feature_loss': feature_loss.item(),
            **{f'val_feature_{k}': v.item() for k, v in layer_losses.items()}
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        path: str,
        losses: Optional[Dict] = None
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state': self.student.state_dict(),
            'feature_loss_state': self.feature_loss.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'alpha': self.alpha,
            'losses': losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint['student_state'])
        self.feature_loss.load_state_dict(checkpoint['feature_loss_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        return checkpoint['epoch']


# ==============================================================================
# Training Loop
# ==============================================================================

def train_distillation(
    trainer: DistillationTrainer,
    train_dataloader,
    val_dataloader=None,
    epochs: int = 100,
    log_interval: int = 10,
    save_interval: int = 10,
    checkpoint_dir: str = './checkpoints'
):
    """
    Complete training loop for distillation.
    
    Args:
        trainer: DistillationTrainer instance
        train_dataloader: Training data loader (yields thermal images, targets)
        val_dataloader: Optional validation data loader
        epochs: Number of training epochs
        log_interval: Steps between logging
        save_interval: Epochs between checkpoints
        checkpoint_dir: Directory for saving checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        trainer.student.train()
        epoch_losses = {}
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(pbar):
            # Unpack batch - adapt based on your dataloader format
            if isinstance(batch, (list, tuple)):
                thermal_batch = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                thermal_batch = batch
                targets = None
            
            # Training step
            losses = trainer.train_step(thermal_batch, targets)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            # Update progress bar
            if (step + 1) % log_interval == 0:
                avg_losses = {k: v / (step + 1) for k, v in epoch_losses.items()}
                pbar.set_postfix({k: f"{v:.4f}" for k, v in avg_losses.items()})
        
        # Epoch summary
        avg_losses = {k: v / len(train_dataloader) for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} Summary:")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        
        # Validation
        if val_dataloader is not None:
            trainer.student.eval()
            val_losses = {}
            
            for batch in val_dataloader:
                if isinstance(batch, (list, tuple)):
                    thermal_batch = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    thermal_batch = batch
                    targets = None
                
                step_losses = trainer.validate_step(thermal_batch, targets)
                
                for k, v in step_losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v
            
            avg_val_losses = {k: v / len(val_dataloader) for k, v in val_losses.items()}
            print("Validation:")
            for k, v in avg_val_losses.items():
                print(f"  {k}: {v:.4f}")
        
        # Update learning rate
        trainer.scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(checkpoint_dir, f'distill_epoch_{epoch+1}.pth')
            trainer.save_checkpoint(epoch + 1, path, avg_losses)
            print(f"Saved checkpoint: {path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'distill_final.pth')
    trainer.save_checkpoint(epochs, final_path, avg_losses)
    print(f"Training complete! Final model saved: {final_path}")


# ==============================================================================
# YOLO Loss Functions
# ==============================================================================

# Try to import official Ultralytics loss components
try:
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class UltralyticsYOLOLoss(nn.Module):
    """
    Wrapper for official Ultralytics YOLOv8 detection loss.
    
    THIS IS THE ONLY RECOMMENDED LOSS FOR FULL DETECTION TRAINING.
    
    This wrapper handles the compatibility between our custom YOLOv8Thermal
    model and the Ultralytics v8DetectionLoss which expects specific attributes:
    - model.stride: Tensor of strides for each detection scale
    - model.nc: Number of classes
    - model.reg_max: DFL regression maximum
    - model.args: Namespace with box, cls, dfl weights
    
    The loss includes:
    - Box regression with CIoU loss (after DFL decoding)
    - Classification with BCE + Focal Loss
    - DFL (Distribution Focal Loss) for precise localization
    - Task-Aligned Assigner for target matching
    """
    
    def __init__(self, model, num_classes: int = 80):
        """
        Initialize with model for proper loss computation.
        
        Args:
            model: YOLOv8Thermal model instance (must have stride, nc, reg_max, args)
            num_classes: Number of detection classes
        """
        super().__init__()
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is REQUIRED for proper YOLO loss computation.\n"
                "The YOLOv8 DFL output format requires proper decoding before CIoU.\n"
                "Install with: pip install ultralytics"
            )
        
        self.num_classes = num_classes
        self.model = model
        
        # Verify model has required attributes
        required_attrs = ['stride', 'nc', 'reg_max', 'args']
        missing = [attr for attr in required_attrs if not hasattr(model, attr)]
        if missing:
            raise AttributeError(
                f"Model is missing required attributes for Ultralytics loss: {missing}\n"
                f"Ensure YOLOv8Thermal model has: stride, nc, reg_max, args"
            )
        
        # Create a wrapper that v8DetectionLoss expects
        # v8DetectionLoss accesses model.model[-1] for detection head info
        class DetectWrapper:
            """Mimics Ultralytics Detect module interface."""
            def __init__(self, parent_model):
                self.nc = parent_model.nc
                self.reg_max = parent_model.reg_max
                self.no = parent_model.reg_max * 4 + parent_model.nc  # Output channels per anchor
                self.stride = parent_model.stride
        
        class ModelWrapper:
            """Mimics Ultralytics Model interface for v8DetectionLoss."""
            def __init__(self, parent_model):
                self.stride = parent_model.stride
                self.nc = parent_model.nc
                self.reg_max = parent_model.reg_max
                self.args = parent_model.args
                # v8DetectionLoss accesses model.model[-1]
                self.model = [DetectWrapper(parent_model)]
            
            def __getitem__(self, idx):
                return self.model[idx]
        
        # Create the wrapper and initialize Ultralytics loss
        self.model_wrapper = ModelWrapper(model)
        self.loss_fn = v8DetectionLoss(self.model_wrapper)
    
    def forward(self, predictions, targets):
        """
        Compute official YOLO loss with proper DFL decoding.
        
        Args:
            predictions: List of tensors from model._to_ultralytics_format()
                         or raw dict (will be converted)
            targets: Ground truth batch dict with 'batch_idx', 'cls', 'bboxes' keys
        
        Returns:
            Tuple of (total_loss, loss_components) from Ultralytics loss
        """
        # Convert dict predictions to Ultralytics format if needed
        if isinstance(predictions, dict):
            if hasattr(self.model, '_to_ultralytics_format'):
                predictions = self.model._to_ultralytics_format(predictions.get('predictions', predictions))
            else:
                raise ValueError(
                    "Predictions are in dict format but model doesn't have "
                    "_to_ultralytics_format method. Use model(x, ultralytics_format=True)"
                )
        
        return self.loss_fn(predictions, targets)


class DistillationOnlyLoss(nn.Module):
    """
    Minimal loss for feature-only distillation (no ground truth boxes).
    
    Use this when:
    - You only want feature mimic loss (no detection supervision)
    - Ultralytics is not available
    - Targets are not provided
    
    This loss only computes objectness BCE as a regularizer.
    The main training signal should come from FeatureMimicLoss.
    
    WARNING: This will NOT train a working detector on its own!
    It must be combined with FeatureMimicLoss from a teacher model.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute minimal objectness loss for regularization.
        
        This is NOT a complete detection loss - it only provides
        a small regularization signal. The main training comes from
        feature distillation.
        """
        device = next(iter(next(iter(predictions.values())).values())).device
        total_loss = torch.tensor(0.0, device=device)
        
        for scale_name, preds in predictions.items():
            if 'obj' in preds:
                obj_pred = preds['obj']
                obj_target = torch.zeros_like(obj_pred)
                total_loss = total_loss + self.bce(obj_pred, obj_target) * 0.1
        
        return total_loss


def create_yolo_loss(
    num_classes: int,
    model=None,
    require_ultralytics: bool = True
) -> nn.Module:
    """
    Factory function to create YOLO loss.
    
    IMPORTANT: For proper detection training, Ultralytics is REQUIRED.
    The YOLOv8 architecture outputs DFL distributions that must be
    decoded before CIoU loss can be computed.
    
    Args:
        num_classes: Number of detection classes
        model: Ultralytics YOLO model instance (REQUIRED for detection)
        require_ultralytics: If True, raise error when Ultralytics unavailable
    
    Returns:
        Loss module
    
    Raises:
        ImportError: If require_ultralytics=True and Ultralytics not available
    """
    if ULTRALYTICS_AVAILABLE and model is not None:
        try:
            print("✓ Using official Ultralytics v8DetectionLoss (recommended)")
            return UltralyticsYOLOLoss(model, num_classes)
        except Exception as e:
            print(f"Warning: Could not create Ultralytics loss: {e}")
    
    if require_ultralytics:
        raise ImportError(
            "Ultralytics is REQUIRED for proper YOLOv8 detection loss.\n"
            "\n"
            "The YOLOv8 detection head outputs DFL (Distribution Focal Loss)\n"
            "logits with shape (B, 64, H, W) representing probability\n"
            "distributions over 16 bins for each of 4 box coordinates.\n"
            "\n"
            "These MUST be decoded using dist2bbox() before CIoU can be computed.\n"
            "A custom implementation without this decoding will produce invalid\n"
            "gradients and the model will not learn to draw bounding boxes.\n"
            "\n"
            "Solutions:\n"
            "1. pip install ultralytics  (recommended)\n"
            "2. Use DistillationOnlyLoss with feature mimic (no box supervision)\n"
            "3. Set require_ultralytics=False for distillation-only mode\n"
        )
    
    print("⚠ WARNING: Using DistillationOnlyLoss - this will NOT train detection!")
    print("  The model will only learn from feature distillation.")
    print("  For proper detection training, install ultralytics.")
    return DistillationOnlyLoss(num_classes=num_classes)


# ==============================================================================
# Test Code
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Distillation Module Test")
    print("=" * 60)
    
    # Import model classes
    import sys
    sys.path.append('..')
    
    # Create mock models for testing
    class MockModel(nn.Module):
        def __init__(self, in_ch, channels):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, channels['P3'], 3, padding=1)
            self.channels = channels
        
        def forward(self, x, return_features=False):
            feat = self.conv(x)
            features = {
                'P3': F.adaptive_avg_pool2d(feat, 80),
                'P4': F.adaptive_avg_pool2d(feat, 40)
            }
            predictions = {'P3': {'cls': feat, 'reg': feat, 'obj': feat}}
            
            if return_features:
                return {'predictions': predictions, 'fused_features': features}
            return {'predictions': predictions}
    
    class MockGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, 1)
        
        def forward(self, x):
            return self.conv(x).tanh()
    
    # Setup
    student_channels = {'P3': 64, 'P4': 64}
    teacher_channels = {'P3': 128, 'P4': 128}
    
    student = MockModel(1, student_channels)
    teacher = MockModel(3, teacher_channels)
    generator = MockGenerator()
    yolo_loss = DistillationOnlyLoss(num_classes=3)  # Feature-only distillation
    
    print("Creating DistillationTrainer...")
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        cyclegan_generator=generator,
        yolo_loss_fn=yolo_loss,
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        distill_layers=['P3', 'P4'],
        alpha=1.0,
        device=torch.device('cpu')
    )
    
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Feature adapters: {sum(p.numel() for p in trainer.feature_loss.parameters()):,}")
    
    # Test training step
    print("\n--- Training Step Test ---")
    thermal_batch = torch.rand(2, 1, 256, 256)
    
    losses = trainer.train_step(thermal_batch, None)
    
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    
    # Test validation step
    print("\n--- Validation Step Test ---")
    val_losses = trainer.validate_step(thermal_batch, None)
    
    print("Validation Losses:")
    for k, v in val_losses.items():
        print(f"  {k}: {v:.4f}")
    
    # Test FeatureMimicLoss directly
    print("\n--- FeatureMimicLoss Test ---")
    feature_loss = FeatureMimicLoss(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        layers=['P3', 'P4']
    )
    
    student_feats = {
        'P3': torch.rand(2, 64, 80, 80),
        'P4': torch.rand(2, 64, 40, 40)
    }
    teacher_feats = {
        'P3': torch.rand(2, 128, 80, 80),
        'P4': torch.rand(2, 128, 40, 40)
    }
    
    loss, layer_losses = feature_loss(student_feats, teacher_feats)
    print(f"Total feature loss: {loss.item():.4f}")
    for k, v in layer_losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✓ All distillation tests passed!")
