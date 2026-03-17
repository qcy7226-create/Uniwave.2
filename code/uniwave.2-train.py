# Downsampling training code

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score, f1_score
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Unified configuration parameters (key modification 1/3)
CONFIG = {
    # Hardware configuration
    'seed': 726,
    'num_workers': 6,
    # Data parameters
    'seq_length': 2000,
    'train_batch_size': 256,  # ↑ Increase batch size to speed up training
    'val_batch_size': 512,
    # Model parameters
    'num_classes': 9,
    # Training parameters (key adjustments)
    'weight_decay': 5e-4,  # ↑ Enhance regularization to prevent overfitting
    'epochs': 100,  # ↓ Shorten total epochs
    'cycle_epochs': 65,  # ↑ Extend cosine period
    'warmup_epochs': 5,  # ↓ Shorten warmup
    'early_stop_patience': 14,  # ↑ Increase patience to wait for later optimization
    'accumulation_steps': 2,  # ↑ Gradient accumulation for stable training
    'label_smoothing': 0.15,  # ↑ Enhance regularization
    'dropout_rate': 0.4,  # ↓ Reduce to prevent underfitting
    'grad_clip': 5.0,  # ↑ Relax gradient clipping
    # Learning rate (key adjustments)
    'max_lr': 1e-3,       # ↓ Lower peak to prevent oscillation
    'div_factor': 15.0,   # ↑ Smoother initial phase
    'init_lr': 1e-3 /10 ,
    # New evaluation parameters
    'eval_metrics': {
        'show_confusion_matrix': True,  # Show confusion matrix
        'plot_roc_curve': True,  # Plot ROC curve
        'average_type': 'macro',  # AUC averaging method
        'class_names': ['Class0', 'Class1','Class2', 'Class3','Class4', 'Class5','Class6', 'Class7','Class8']  # Class names
    },
    # New word embedding parameters (key modification)
    'embedding': {
        'enable': True,         # Whether to enable word embedding
        'dim': 128,             # Embedding dimension (high-dimensional space size)
        'kernel_size': 3        # 1D convolution kernel size (pointwise expansion)
    },
}


# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(CONFIG['seed'])


# %% Data leakage check (adapted for single-channel data)
def check_data_leak(h5_path):
    """Sample-level data leakage check"""
    with h5py.File(h5_path, 'r') as h5:
        train_hashes = set()
        # Collect training set hashes
        for cls in h5['train']:
            data = h5[f'train/{cls}/data']
            for i in range(data.shape[0]):
                sample_hash = hash(data[i].tobytes())
                if sample_hash in train_hashes:
                    print(f"Duplicate within training set: {cls}[{i}]")
                    return True
                train_hashes.add(sample_hash)

        # Check validation set
        for cls in h5['val']:
            data = h5[f'val/{cls}/data']
            for i in range(data.shape[0]):
                if hash(data[i].tobytes()) in train_hashes:
                    print(f"Data leakage: {cls}[{i}]")
                    return True
    return False


# %% Data loading class (adapted to your HDF5 structure)
class BioDataset(Dataset):
    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode = mode
        self.is_train = (mode == 'train')
        with h5py.File(h5_path, 'r') as h5:
            self.classes = sorted([c for c in h5[mode].keys() if c.startswith('class_')])
            self.samples = []
            for cls in self.classes:
                num_samples = h5[f'{mode}/{cls}/data'].shape[0]
                self.samples.extend([(cls, i) for i in range(num_samples)])
            print(f"Successfully loaded {len(self.samples)} samples in {mode} mode, number of classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, sample_idx = self.samples[idx]
        with h5py.File(self.h5_path, 'r') as h5:
            data = h5[f'{self.mode}/{cls}/data'][sample_idx]
            data = torch.FloatTensor(data).unsqueeze(0)

            return data, torch.tensor(int(cls.split('_')[1]))


# %% Model definition positional encoding
class WaveEncoder(nn.Module):
    def __init__(self, seq_len=2000, latent_dim=20):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.base_freq = nn.Parameter(torch.rand(1, latent_dim, 1) * 0.02)
        self.base_phase = nn.Parameter(torch.randn(1, latent_dim, 1) * 0.1)

        self.attention = nn.Sequential(
            nn.Conv1d(2 * latent_dim, 16, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=15, padding=7),
            nn.Sigmoid()
        )

    def forward(self, y):
        batch_size = y.size(0)
        device = y.device
        dtype = y.dtype

        freq = self.base_freq.to(device).expand(batch_size, -1, -1)
        phase = self.base_phase.to(device).expand(batch_size, -1, -1)

        pos = torch.linspace(0.0, 1.0, steps=self.seq_len, device=device, dtype=dtype).view(1, 1, self.seq_len)

        pos_encoding = torch.sin(2 * torch.pi * freq * pos + phase)

        if pos_encoding.shape[1] != y.shape[1]:
            pos_encoding = pos_encoding.expand(batch_size, y.shape[1], self.seq_len)

        combined = torch.cat([y, pos_encoding], dim=1)

        attn_weights = self.attention(combined)  # (batch, 1, seq_len)
        fused_feat = y * attn_weights + pos_encoding * (1 - attn_weights)

        return fused_feat


class InceptionModule(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 1),
            nn.BatchNorm1d(base_channels),
            nn.GELU()
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base_channels, base_channels, k,
                          padding=(k + (k-1)*(dilation-1) - 1) // 2, dilation=dilation, groups=base_channels),
                nn.BatchNorm1d(base_channels),
                nn.GELU()
            ) for k, dilation in zip([3, 21], [1, 3])
        ])
        self.channel_red = nn.Conv1d(base_channels * 3, base_channels * 2, 1)
        self.res_conv = nn.Conv1d(in_channels, base_channels * 2, 1) if in_channels != base_channels * 2 else None

    def forward(self, x):
        residual = x
        if self.res_conv is not None:
            residual = self.res_conv(residual)

        x = self.bottleneck(x)
        branches = [conv(x) for conv in self.convs]
        branches.append(nn.AdaptiveAvgPool1d(x.size(-1))(x))
        out = self.channel_red(torch.cat(branches, dim=1))
        return out + residual


class InceptionTime(nn.Module):
    def __init__(self, num_classes=9, seq_len=2000, embedding_dim=128):
        super().__init__()
        self.config = CONFIG
        self.embedding_dim = embedding_dim if CONFIG['embedding']['enable'] else 1

        # -------------------- New word embedding layer --------------------
        # 1D convolution for word embedding (expand 1-dim input to high dimension)
        kernel_size = CONFIG['embedding']['kernel_size']
        padding = (kernel_size - 1) // 2
        self.embedding = nn.Conv1d(
            in_channels=1,
            out_channels=self.embedding_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.wave_encoder = WaveEncoder(seq_len=seq_len, latent_dim=self.embedding_dim)
        self.input_norm = nn.BatchNorm1d(
            num_features=self.embedding_dim,
            eps=1e-3,  # Reduce epsilon to accommodate normalized data
            momentum=0.1,  # Use longer history statistics
            affine=True  # Keep learnable parameters γ and β
        )

        # Inception stack (multi-scale feature extraction, new intermediate feature collection)
        self.inception_blocks = nn.ModuleList([
            self._make_inception(self.embedding_dim, 32, pool=True),  # Initial block (input 2 channels, output 32 channels)
            self._make_inception(64, 32, pool=False),  # Intermediate block (no downsampling)
            self._make_inception(64, 32, pool=False),  # Intermediate block (no downsampling)
            self._make_inception(64, 32, pool=False)   # Final block (no downsampling)
        ])

        # Global pooling and classification head (original task)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),  # Input 128 (final Inception block output channels)
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # GRU temporal modeling module (new)
        self.gru = nn.GRU(
            input_size=64,  # Input dimension (channel number of Inception intermediate features)
            hidden_size=64,  # Hidden layer dimension
            num_layers=2,  # Number of layers
            batch_first=True,  # Input format: (batch, seq_len, features)
            bidirectional=True,  # Bidirectional GRU (capture context dependencies)
            dropout=0.2 if 2 > 1 else 0  # Dropout between layers (prevent overfitting)
        )
        self.gru_fusion = nn.Sequential(
            nn.Linear(64 * 2, 128),  # Bidirectional GRU output dimension 64*2
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def _make_inception(self, in_c, base_c, pool=False):
        """Build Inception block (with optional downsampling, keep original logic)"""
        layers = [InceptionModule(in_c, base_c)]
        if pool:  # Downsampling (reduce sequence length)
            layers.append(nn.MaxPool1d(3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # -------------------- Original Inception flow --------------------
        x = self.embedding(x)
        x = self.wave_encoder(x)
        x = self.input_norm(x)

        # Collect intermediate features (new)
        inception_feats = []
        for block in self.inception_blocks:
            x = block(x)  # Multi-scale feature extraction
            inception_feats.append(x)  # Save intermediate features (shape: (batch, channels, seq_len))

        # -------------------- Original classification head --------------------
        x_global = self.adaptive_pool(x).squeeze(-1)  # Global pooling for original task features (batch_size, 64)

        # -------------------- New GRU temporal modeling --------------------
        gru_input = inception_feats[-1]  # Take output of last block
        gru_input = gru_input.transpose(1, 2)

        # GRU forward pass (extract temporal features)
        gru_out, _ = self.gru(gru_input)
        gru_feat = gru_out[:, -1, :]
        gru_feat = self.gru_fusion(gru_feat)  # (batch_size, 64) (after downsampling)
        # Feature fusion (original global features + GRU temporal features)
        fused_feat = torch.cat([x_global, gru_feat], dim=1)

        # Classification prediction
        logits = self.classifier(fused_feat)  # Final classification result
        return logits


# %% Trainer class
# Replace BioTrainer class
class BioTrainer:
    def __init__(self, model, train_loader, val_loader, hard_val_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hard_val_loader = hard_val_loader  # Store challenging validation set

        # Optimizer configuration
        params = [
            {'params': [p for n, p in model.named_parameters() if 'norm' not in n],
             'weight_decay': CONFIG['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if 'norm' in n]}
        ]
        self.optimizer = optim.AdamW(params, lr=CONFIG['init_lr'])  # Use init_lr as initial lr
        steps_per_epoch = len(train_loader) // CONFIG['accumulation_steps']
        # Three-phase learning rate scheduler
        self.scheduler = self.ThreePhaseLRScheduler(
            optimizer=self.optimizer,
            num_epochs=CONFIG['epochs'],
            steps_per_epoch=steps_per_epoch,
            max_lr=CONFIG['max_lr'],
            div_factor=CONFIG['div_factor']
        )
        # Training tools
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        self.best_f1 = 0.0
        self.early_stop_counter = 0

    class ThreePhaseLRScheduler:
        def __init__(self, optimizer, num_epochs, steps_per_epoch,
                     max_lr=3e-4, div_factor=10.0):
            self.optimizer = optimizer
            self.steps_per_epoch = max(1, steps_per_epoch)
            self.total_steps = num_epochs * self.steps_per_epoch
            self.step_count = 0
            # Phase division adjustment
            self.phase1_ratio = 0.15  # Phase 1: warmup
            self.phase2_ratio = 0.70  # Phase 2: cosine annealing
            self.phase1_steps = int(self.phase1_ratio * self.total_steps)
            self.phase2_steps = int(self.phase2_ratio * self.total_steps)
            self.phase3_steps = self.total_steps - self.phase1_steps - self.phase2_steps
            # Learning rate bounds adjustment
            self.init_lr = max_lr / div_factor
            self.max_lr = max_lr
            self.final_lr = max(self.max_lr / 100, 2e-5)  # Lower bound set to 2e-5
            self.max_lr_phase3 = self.max_lr * 0.1  # Starting LR for phase 3
            self.last_val_loss = float('inf')
            self.current_lr = self.init_lr
            self.adjusted_phase2_steps = self.phase2_steps

        def step(self, val_loss=None):
            if val_loss is not None:
                self._update_phase2_progress(val_loss)
            self._calculate_lr()
            # Apply learning rate to optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(max(self.current_lr, self.final_lr), self.max_lr)
            self.step_count += 1

        def _update_phase2_progress(self, val_loss):
            phase = self._current_phase()
            if phase != 2: return
            # Dynamically adjust phase duration: allowed loss increase adjusted from 0.5% to 1%
            allowed_increase = max(0.01 * self.last_val_loss, 0.05)
            if val_loss > (self.last_val_loss + allowed_increase):
                # Slowly adjust phase duration
                self.adjusted_phase2_steps = max(
                    int(self.adjusted_phase2_steps * 0.95),
                    self.phase2_steps // 5
                )
            self.last_val_loss = val_loss

        def _current_phase(self):
            if self.step_count < self.phase1_steps:
                return 1
            elif self.step_count < self.phase1_steps + self.adjusted_phase2_steps:
                return 2
            else:
                return 3

        def _calculate_lr(self):
            current_phase = self._current_phase()

            # Phase 1 - linear warmup
            if current_phase == 1:
                progress = self.step_count / self.phase1_steps
                self.current_lr = self.init_lr + (self.max_lr - self.init_lr) * progress

            # Phase 2 - smooth cosine annealing
            elif current_phase == 2:
                phase_progress = (self.step_count - self.phase1_steps) / self.adjusted_phase2_steps
                self.current_lr = self.max_lr * (0.2 + 0.8 * (1 + np.cos(np.pi * phase_progress)) / 2)

            # Phase 3 - linear decay to minimum LR
            else:
                phase_progress = (self.step_count - self.phase1_steps - self.adjusted_phase2_steps) / self.phase3_steps
                self.current_lr = self.max_lr_phase3 * (1 - phase_progress) + self.final_lr * phase_progress

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        self.optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # Mixed precision forward
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / CONFIG['accumulation_steps']
            # Backward
            self.scaler.scale(loss).backward()
            # Gradient accumulation update
            if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                # Relaxed gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()  # Move out of gradient accumulation condition
            # Statistics
            total_loss += loss.item() * CONFIG['accumulation_steps'] * inputs.size(0)
            all_preds.extend(torch.argmax(outputs.detach(), dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return {
            'loss': total_loss / len(self.train_loader.dataset),
            'acc': balanced_accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro')
        }

    def validate(self, loader=None):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_probs = []
        eval_loader = loader if loader is not None else self.val_loader
        with torch.no_grad():
            for inputs, labels in eval_loader:  # Only this loop
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item() * inputs.size(0)

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        # New code segment (compute confidence intervals)
        from sklearn.utils import resample

        def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=95):
            stats = []
            for _ in range(n_bootstrap):
                indices = resample(np.arange(len(y_true)))
                acc = balanced_accuracy_score(y_true[indices], y_pred[indices])
                stats.append(acc)
            lower = (100 - ci) / 2
            upper = 100 - lower
            return np.percentile(stats, [lower, upper])

        # Compute 95% confidence interval
        ci_low, ci_high = bootstrap_ci(all_labels, all_preds)
        # Compute metrics
        metrics = {
            'loss': total_loss / len(eval_loader.dataset),  # Key fix point
            'acc': balanced_accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average=CONFIG['eval_metrics']['average_type']),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'labels': all_labels,
            'probs': all_probs,
            'preds': all_preds
        }
        # Confusion matrix
        if CONFIG['eval_metrics']['show_confusion_matrix']:
            metrics['confusion_matrix'] = confusion_matrix(
                all_labels, all_preds,
                labels=np.arange(CONFIG['num_classes'])
            )
        # AUC-ROC (multi-class)
        try:
            if CONFIG['num_classes'] == 2:
                metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Exact multi-class calculation (class-wise then macro average)
                y_true_bin = label_binarize(all_labels, classes=np.arange(CONFIG['num_classes']))
                class_auc = []
                for i in range(CONFIG['num_classes']):
                    if np.sum(y_true_bin[:, i]) == 0:  # Handle case with no positive samples
                        class_auc.append(float('nan'))
                        continue
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
                    class_auc.append(auc(fpr, tpr))
                # Exclude invalid values and compute macro average
                valid_auc = np.array(class_auc)[~np.isnan(class_auc)]
                metrics['auc'] = np.mean(valid_auc) if len(valid_auc) > 0 else float('nan')
        except Exception as e:
            print(f"AUC calculation failed: {str(e)}")
            metrics['auc'] = -1.0
        return metrics

    def train(self):
        print(f"\nStarting training, using device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}\n")
        for epoch in range(CONFIG['epochs']):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            # Pass validation loss to scheduler
            self.scheduler.step(val_loss=val_metrics['loss'])
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), 'best_model.hqd_r.pth')
            else:
                self.early_stop_counter += 1

            # Print during training loop
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['acc']:.2%} | F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['acc']:.2%} (95% CI: {val_metrics['ci_low']:.2%}~{val_metrics['ci_high']:.2%}) | "  # Modified line
                  f"F1: {val_metrics['f1']:.4f}")
            print("-" * 70)
            # Evaluate challenge set every 5 epochs (check existence)
            if (epoch + 1) % 5 == 0 and self.hard_val_loader is not None:
                hard_val_metrics = self.validate(self.hard_val_loader)
                print(f"\nChallenge validation set | Acc: {hard_val_metrics['acc']:.2%} | F1: {hard_val_metrics['f1']:.4f}")
                print("-" * 70)
            # Early stopping check
            if self.early_stop_counter >= CONFIG['early_stop_patience']:
                print(f"Early stopping triggered, best validation F1: {self.best_f1:.4f}")
                break


def plot_evaluation_curves(metrics):
    """Plot confusion matrix and ROC curve"""
    plt.figure(figsize=(15, 6))
    # Confusion matrix heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(metrics['confusion_matrix'],
                annot=True, fmt='d',
                cmap='Blues',
                xticklabels=CONFIG['eval_metrics']['class_names'],
                yticklabels=CONFIG['eval_metrics']['class_names'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    # ROC curve plotting part
    if CONFIG['eval_metrics']['plot_roc_curve'] and CONFIG['num_classes'] > 2:
        plt.subplot(1, 2, 2)
        y_true_bin = label_binarize(metrics['labels'], classes=np.arange(CONFIG['num_classes']))
        fpr, tpr, roc_auc = {}, {}, {}
        # Exact per-class calculation
        for i in range(CONFIG['num_classes']):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], metrics['probs'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot curves
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink'])
        for i, color in zip(range(CONFIG['num_classes']), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{CONFIG["eval_metrics"]["class_names"][i]} (AUC={roc_auc[i]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC curve (macro average AUC={metrics["auc"]:.4f})')
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# %% Main process
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Automatically select optimal algorithm
    # Data file path
    H5_PATH = "ecoli_hqd_1000(9-).h5"

    # # Data leakage check
    # if check_data_leak(H5_PATH):
    #     raise RuntimeError("Data leakage check failed, please verify data preprocessing pipeline!")

    # New cross-dataset leakage check
    def check_inter_dataset_leak(source_group, target_group):
        leak_count = 0
        with h5py.File(H5_PATH, 'r') as h5:
            for cls in h5[source_group]:
                source_data = h5[f'{source_group}/{cls}/data']
                source_hashes = set(d.tobytes() for d in source_data)

                for t_cls in h5[target_group]:
                    target_data = h5[f'{target_group}/{t_cls}/data']
                    for d in target_data:
                        if d.tobytes() in source_hashes:
                            leak_count += 1
        return leak_count

    print("\n=== Cross-dataset leakage check ===")
    print(f"Test set -> training set leakage count: {check_inter_dataset_leak('test', 'train')}")
    print(f"Test set -> validation set leakage count: {check_inter_dataset_leak('test', 'val')}")
    print(f"Validation set -> training set leakage count: {check_inter_dataset_leak('val', 'train')}")

    # Data loading
    # Training set (original data + training augmentation)
    train_dataset = BioDataset(H5_PATH, 'train')
    # Validation set (original data + basic augmentation)
    val_dataset = BioDataset(H5_PATH, 'val')
    # Standard test set (original data)
    standard_test_dataset = BioDataset(H5_PATH, 'test')

    # Print first sample from training and validation sets
    train_sample, train_label = train_dataset[0]
    val_sample, val_label = val_dataset[0]
    print(f"Training set first sample shape: {train_sample.shape}, label: {train_label}")
    print(f"Validation set first sample shape: {val_sample.shape}, label: {val_label}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['train_batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['val_batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    # Standard test set loader (original data)
    standard_test_loader = DataLoader(
        standard_test_dataset,
        batch_size=CONFIG['val_batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    # Model initialization
    model = InceptionTime(num_classes=CONFIG['num_classes'])
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Delete old model file
    import os

    if os.path.exists('best_model.hqd_r.pth'):
        os.remove('best_model.hqd_r.pth')

    # Reinitialize and train
    model = InceptionTime(num_classes=CONFIG['num_classes'])
    trainer = BioTrainer(model, train_loader, val_loader)
    trainer.train()

    # Final evaluation (new test set evaluation)
    # ----------------------------------------------------------------
    print("\n=== Validation set final evaluation ===")
    model.load_state_dict(torch.load('best_model.hqd_r.pth'))
    val_metrics = trainer.validate()

    # Standard test set evaluation
    print("\n=== Standard test set evaluation ===")
    standard_metrics = trainer.validate(standard_test_loader)

    # Results output
    print("\n=== Validation set results ===")
    print(f"Balanced accuracy: {val_metrics['acc']:.4f} (95% CI: {val_metrics['ci_low']:.4f}~{val_metrics['ci_high']:.4f})")
    print(f"F1 score ({CONFIG['eval_metrics']['average_type']}): {val_metrics['f1']:.4f}")
    print(f"AUC-ROC ({CONFIG['eval_metrics']['average_type']}): {val_metrics['auc']:.4f}")

    print("\n=== Standard test set results ===")
    print(f"Balanced accuracy: {standard_metrics['acc']:.4f} (95% CI: {standard_metrics['ci_low']:.4f}~{standard_metrics['ci_high']:.4f})")
    print(f"F1 score ({CONFIG['eval_metrics']['average_type']}): {standard_metrics['f1']:.4f}")
    print(f"AUC-ROC ({CONFIG['eval_metrics']['average_type']}): {standard_metrics['auc']:.4f}")

    # Visualization
    plot_evaluation_curves(standard_metrics)  # Standard test set charts