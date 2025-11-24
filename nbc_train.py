"""
NBC (Neural Beamforming Convolution) Training Script
Train from scratch on multi-channel reverberant noisy speech separation dataset
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import soundfile as sf

# Import NBC model (assumes the model code is in nbc_model.py)
from nbc_model import NBC


class MultichannelSpeechDataset(Dataset):
    """
    Dataset for multi-channel speech separation
    
    Directory structure expected:
    dataset_root/
        mix/  - 8-channel mixtures (tr000000_mix.wav, cv000000_mix.wav, tt000000_mix.wav)
        s1/   - Speaker 1 reference (tr000000_s1.wav, ...)
        s2/   - Speaker 2 reference (tr000000_s2.wav, ...)
    """
    
    def __init__(
        self, 
        dataset_root: str,
        split: str = "tr",  # "tr", "cv", or "tt"
        segment_length: float = 4.0,  # seconds
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_channels: int = 8,
        n_speakers: int = 2,
    ):
        super().__init__()
        
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_channels = n_channels
        self.n_speakers = n_speakers
        self.segment_samples = int(segment_length * sample_rate)
        
        # Find all mixture files for this split
        self.mix_files = sorted(list((self.dataset_root / "mix").glob(f"{split}*.wav")))
        
        if len(self.mix_files) == 0:
            raise ValueError(f"No files found for split '{split}' in {self.dataset_root / 'mix'}")
        
        print(f"Loaded {len(self.mix_files)} files for split '{split}'")
    
    def __len__(self) -> int:
        return len(self.mix_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                'mix_stft': [n_freq, n_frames, n_channels*2] (real + imag)
                'target_stft': [n_freq, n_frames, n_speakers*2] (real + imag)
                'mix_audio': [n_channels, segment_samples]
                's1_audio': [segment_samples]
                's2_audio': [segment_samples]
        """
        # Get file paths
        mix_path = self.mix_files[idx]
        filename_base = mix_path.stem.replace("_mix", "")
        s1_path = self.dataset_root / "s1" / f"{filename_base}_s1.wav"
        s2_path = self.dataset_root / "s2" / f"{filename_base}_s2.wav"
        
        # Load audio
        mix_audio, sr = sf.read(mix_path)  # [T, n_channels]
        s1_audio, _ = sf.read(s1_path)     # [T]
        s2_audio, _ = sf.read(s2_path)     # [T]
        
        # Transpose mix to [n_channels, T]
        mix_audio = mix_audio.T
        
        # Random crop to segment_length (for training only)
        if self.split == "tr" and mix_audio.shape[1] > self.segment_samples:
            start = np.random.randint(0, mix_audio.shape[1] - self.segment_samples)
            mix_audio = mix_audio[:, start:start + self.segment_samples]
            s1_audio = s1_audio[start:start + self.segment_samples]
            s2_audio = s2_audio[start:start + self.segment_samples]
        else:
            # For validation/test, pad if necessary
            if mix_audio.shape[1] < self.segment_samples:
                pad_len = self.segment_samples - mix_audio.shape[1]
                mix_audio = np.pad(mix_audio, ((0, 0), (0, pad_len)), mode='constant')
                s1_audio = np.pad(s1_audio, (0, pad_len), mode='constant')
                s2_audio = np.pad(s2_audio, (0, pad_len), mode='constant')
            else:
                mix_audio = mix_audio[:, :self.segment_samples]
                s1_audio = s1_audio[:self.segment_samples]
                s2_audio = s2_audio[:self.segment_samples]
        
        # Convert to torch tensors
        mix_audio = torch.from_numpy(mix_audio.astype(np.float32))
        s1_audio = torch.from_numpy(s1_audio.astype(np.float32))
        s2_audio = torch.from_numpy(s2_audio.astype(np.float32))
        
        # Compute STFT
        mix_stft = self._compute_stft_multichannel(mix_audio)  # [n_freq, n_frames, n_channels*2]
        
        # For target, use first microphone channel as reference
        # In practice, you might want to use a beamformed reference
        s1_stft = self._compute_stft(s1_audio)  # [n_freq, n_frames, 2]
        s2_stft = self._compute_stft(s2_audio)  # [n_freq, n_frames, 2]
        
        # Concatenate speakers: [n_freq, n_frames, n_speakers*2]
        target_stft = torch.cat([s1_stft, s2_stft], dim=-1)
        
        return {
            'mix_stft': mix_stft,
            'target_stft': target_stft,
            'mix_audio': mix_audio,
            's1_audio': s1_audio,
            's2_audio': s2_audio,
            'filename': filename_base,
        }
    
    def _compute_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT for mono audio
        
        Args:
            audio: [T]
        
        Returns:
            stft: [n_freq, n_frames, 2] (real, imag)
        """
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True
        )  # [n_freq, n_frames]
        
        # Convert to real representation [n_freq, n_frames, 2]
        stft_real = torch.stack([stft.real, stft.imag], dim=-1)
        return stft_real
    
    def _compute_stft_multichannel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT for multi-channel audio
        
        Args:
            audio: [n_channels, T]
        
        Returns:
            stft: [n_freq, n_frames, n_channels*2]
        """
        n_channels = audio.shape[0]
        stft_list = []
        
        for ch in range(n_channels):
            stft_ch = self._compute_stft(audio[ch])  # [n_freq, n_frames, 2]
            stft_list.append(stft_ch)
        
        # Concatenate channels: [n_freq, n_frames, n_channels*2]
        stft_multichannel = torch.cat(stft_list, dim=-1)
        return stft_multichannel


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss
    
    SI-SDR is a popular metric for speech separation, invariant to scaling.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, time] or [batch, freq, time, 2]
            target: same shape as pred
        
        Returns:
            loss: scalar (negative SI-SDR)
        """
        # Flatten to [batch, -1]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        
        # Zero-mean normalization
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        # SI-SDR calculation
        alpha = (pred * target).sum(dim=1, keepdim=True) / (target ** 2).sum(dim=1, keepdim=True).clamp(min=self.epsilon)
        target_scaled = alpha * target
        
        noise = pred - target_scaled
        
        si_sdr = 10 * torch.log10(
            (target_scaled ** 2).sum(dim=1).clamp(min=self.epsilon) / 
            (noise ** 2).sum(dim=1).clamp(min=self.epsilon)
        )
        
        # Return negative SI-SDR as loss (we want to maximize SI-SDR)
        return -si_sdr.mean()


class STFTLoss(nn.Module):
    """
    Multi-resolution STFT Loss (spectral convergence + magnitude loss)
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_stft: [batch, freq, time, features]
            target_stft: [batch, freq, time, features]
        
        Returns:
            loss: scalar
        """
        return self.mse(pred_stft, target_stft)


class CombinedLoss(nn.Module):
    """
    Combined loss: STFT + SI-SDR (in time domain)
    """
    
    def __init__(
        self,
        stft_weight: float = 1.0,
        sisdr_weight: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        super().__init__()
        self.stft_loss = STFTLoss()
        self.sisdr_loss = SISDRLoss()
        self.stft_weight = stft_weight
        self.sisdr_weight = sisdr_weight
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def forward(
        self, 
        pred_stft: torch.Tensor, 
        target_stft: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # STFT loss
        loss_stft = self.stft_loss(pred_stft, target_stft)
        
        # SI-SDR loss (if time-domain signals provided)
        if pred_audio is not None and target_audio is not None:
            loss_sisdr = self.sisdr_loss(pred_audio, target_audio)
        else:
            loss_sisdr = torch.tensor(0.0, device=pred_stft.device)
        
        # Combined loss
        total_loss = self.stft_weight * loss_stft + self.sisdr_weight * loss_sisdr
        
        loss_dict = {
            'total': total_loss.item(),
            'stft': loss_stft.item(),
            'sisdr': loss_sisdr.item(),
        }
        
        return total_loss, loss_dict


def istft_reconstruction(stft: torch.Tensor, n_fft: int = 512, hop_length: int = 256) -> torch.Tensor:
    """
    Inverse STFT to reconstruct time-domain signal
    
    Args:
        stft: [batch, freq, time, 2] (real, imag)
    
    Returns:
        audio: [batch, time]
    """
    # Convert to complex
    stft_complex = torch.complex(stft[..., 0], stft[..., 1])  # [batch, freq, time]
    
    # Inverse STFT
    audio = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=stft.device),
    )
    
    return audio


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> Tuple[float, int]:
    """Train for one epoch"""
    
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        mix_stft = batch['mix_stft'].to(device)  # [B, F, T, C*2]
        target_stft = batch['target_stft'].to(device)  # [B, F, T, S*2]
        
        # Forward pass
        optimizer.zero_grad()
        pred_stft = model(mix_stft)  # [B, F, T, S*2]
        
        # Compute loss
        loss, loss_dict = criterion(pred_stft, target_stft)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Logging
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(STFT: {loss_dict['stft']:.4f}, SI-SDR: {loss_dict['sisdr']:.4f})")
            
            # TensorBoard logging
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/stft_loss', loss_dict['stft'], global_step)
            writer.add_scalar('train/sisdr_loss', loss_dict['sisdr'], global_step)
        
        global_step += 1
    
    return epoch_loss / len(dataloader), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    n_fft: int = 512,
    hop_length: int = 256,
) -> Tuple[float, float]:
    """Validate the model"""
    
    model.eval()
    val_loss = 0.0
    val_sisdr = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        mix_stft = batch['mix_stft'].to(device)
        target_stft = batch['target_stft'].to(device)
        s1_audio = batch['s1_audio'].to(device)
        s2_audio = batch['s2_audio'].to(device)
        
        # Forward pass
        pred_stft = model(mix_stft)
        
        # Compute loss
        loss, loss_dict = criterion(pred_stft, target_stft)
        val_loss += loss.item()
        
        # Compute SI-SDR in time domain
        # Reconstruct separated sources
        pred_s1_stft = pred_stft[..., :2]  # First speaker
        pred_s2_stft = pred_stft[..., 2:4]  # Second speaker
        
        pred_s1_audio = istft_reconstruction(pred_s1_stft, n_fft, hop_length)
        pred_s2_audio = istft_reconstruction(pred_s2_stft, n_fft, hop_length)
        
        # Match lengths
        min_len = min(pred_s1_audio.shape[1], s1_audio.shape[1])
        pred_s1_audio = pred_s1_audio[:, :min_len]
        pred_s2_audio = pred_s2_audio[:, :min_len]
        target_s1 = s1_audio[:, :min_len]
        target_s2 = s2_audio[:, :min_len]
        
        # SI-SDR for both speakers
        sisdr_loss = SISDRLoss()
        sisdr_s1 = -sisdr_loss(pred_s1_audio, target_s1).item()
        sisdr_s2 = -sisdr_loss(pred_s2_audio, target_s2).item()
        val_sisdr += (sisdr_s1 + sisdr_s2) / 2
    
    avg_val_loss = val_loss / len(dataloader)
    avg_val_sisdr = val_sisdr / len(dataloader)
    
    print(f"\n{'='*60}")
    print(f"Validation Epoch [{epoch}]")
    print(f"Avg Loss: {avg_val_loss:.4f}")
    print(f"Avg SI-SDR: {avg_val_sisdr:.2f} dB")
    print(f"{'='*60}\n")
    
    # TensorBoard logging
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    writer.add_scalar('val/sisdr', avg_val_sisdr, epoch)
    
    return avg_val_loss, avg_val_sisdr


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    val_sisdr: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_sisdr': val_sisdr,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MultichannelSpeechDataset(
        dataset_root=args.dataset_root,
        split="tr",
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_channels=args.n_channels,
        n_speakers=args.n_speakers,
    )
    
    val_dataset = MultichannelSpeechDataset(
        dataset_root=args.dataset_root,
        split="cv",
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_channels=args.n_channels,
        n_speakers=args.n_speakers,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print("Creating model...")
    model = NBC(
        dim_input=args.n_channels * 2,  # Real + Imag
        dim_output=args.n_speakers * 2,  # Real + Imag
        n_layers=args.n_layers,
        encoder_kernel_size=args.encoder_kernel_size,
        n_heads=args.n_heads,
        hidden_size=args.hidden_size,
        norm_first=args.norm_first,
        ffn_size=args.ffn_size,
        inner_conv_kernel_size=args.inner_conv_kernel_size,
        inner_conv_groups=args.inner_conv_groups,
        inner_conv_bias=args.inner_conv_bias,
        inner_conv_layers=args.inner_conv_layers,
        inner_conv_mid_norm=args.inner_conv_mid_norm,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize SI-SDR
        factor=0.5,
        patience=3,
        verbose=True,
    )
    
    # Create loss
    criterion = CombinedLoss(
        stft_weight=args.stft_weight,
        sisdr_weight=args.sisdr_weight,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_sisdr = -float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, global_step
        )
        
        # Validate
        val_loss, val_sisdr = validate(
            model, val_loader, criterion, device, epoch, writer, args.n_fft, args.hop_length
        )
        
        # Learning rate scheduling
        scheduler.step(val_sisdr)
        
        # Save checkpoint
        is_best = val_sisdr > best_val_sisdr
        if is_best:
            best_val_sisdr = val_sisdr
        
        if epoch % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, global_step, val_loss, val_sisdr,
                args.checkpoint_dir, is_best
            )
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Best SI-SDR so far: {best_val_sisdr:.2f} dB\n")
    
    print("Training complete!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NBC model for speech separation')
    
    # Dataset args
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to dataset root (containing mix/, s1/, s2/)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--segment_length', type=float, default=4.0,
                        help='Training segment length in seconds')
    parser.add_argument('--n_channels', type=int, default=8,
                        help='Number of microphone channels')
    parser.add_argument('--n_speakers', type=int, default=2,
                        help='Number of speakers')
    
    # STFT args
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for STFT')
    
    # Model args
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of NBC layers')
    parser.add_argument('--encoder_kernel_size', type=int, default=4,
                        help='Encoder convolution kernel size')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=192,
                        help='Hidden dimension size')
    parser.add_argument('--norm_first', action='store_true', default=True,
                        help='Apply layer norm before attention')
    parser.add_argument('--ffn_size', type=int, default=384,
                        help='Feed-forward network size')
    parser.add_argument('--inner_conv_kernel_size', type=int, default=3,
                        help='Inner convolution kernel size')
    parser.add_argument('--inner_conv_groups', type=int, default=8,
                        help='Inner convolution groups')
    parser.add_argument('--inner_conv_bias', action='store_true', default=True,
                        help='Use bias in inner convolution')
    parser.add_argument('--inner_conv_layers', type=int, default=3,
                        help='Number of inner convolution layers')
    parser.add_argument('--inner_conv_mid_norm', type=str, default='GN',
                        choices=['GN', None], help='Middle normalization type')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--stft_weight', type=float, default=1.0,
                        help='Weight for STFT loss')
    parser.add_argument('--sisdr_weight', type=float, default=0.1,
                        help='Weight for SI-SDR loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Checkpoint args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)