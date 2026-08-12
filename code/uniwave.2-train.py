#!/usr/bin/env python3
"""
UniWave-2 Training Script
Usage:
    python train.py --data data.h5 --output_dir results/ [--seed 726]
"""
import os
import argparse
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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.utils import resample

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- Configuration (can be overridden by command line) --------------------
DEFAULT_CONFIG = {
    'seed': 726,
    'num_workers': 6,
    'seq_length': 2000,
    'train_batch_size': 256,
    'val_batch_size': 512,
    'num_classes': None,   # will be inferred from data
    'weight_decay': 5e-4,
    'epochs': 100,
    'early_stop_patience': 14,
    'accumulation_steps': 2,
    'label_smoothing': 0.15,
    'dropout_rate': 0.4,
    'grad_clip': 5.0,
    'max_lr': 1e-3,
    'div_factor': 15.0,
    'init_lr': 1e-4,
    'eval_metrics': {
        'show_confusion_matrix': True,
        'plot_roc_curve': True,
        'average_type': 'macro',
        'class_names': []  # will be set from HDF5 metadata
    },
    'embedding': {
        'enable': True,
        'dim': 128,
        'kernel_size': 3
    }
}

# -------------------- Data Loading --------------------
class BioDataset(Dataset):
    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode = mode
        with h5py.File(h5_path, 'r') as h5:
            self.classes = sorted([c for c in h5[mode].keys() if c.startswith('class_')])
            self.samples = []
            for cls in self.classes:
                num = h5[f'{mode}/{cls}/data'].shape[0]
                self.samples.extend([(cls, i) for i in range(num)])
            print(f"Loaded {len(self.samples)} samples from {mode} split")
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, sample_idx = self.samples[idx]
        with h5py.File(self.h5_path, 'r') as h5:
            data = h5[f'{self.mode}/{cls}/data'][sample_idx]
            data = torch.FloatTensor(data).unsqueeze(0)  # (1, seq_len)
            label = int(cls.split('_')[1])
            return data, torch.tensor(label, dtype=torch.long)

# -------------------- Model Definitions (unchanged from original) --------------------
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
        batch_size, C, L = y.shape
        device = y.device
        dtype = y.dtype
        freq = self.base_freq.to(device).expand(batch_size, -1, -1)
        phase = self.base_phase.to(device).expand(batch_size, -1, -1)
        pos = torch.linspace(0.0, 1.0, steps=L, device=device, dtype=dtype).view(1, 1, L)
        pos_encoding = torch.sin(2 * torch.pi * freq * pos + phase)
        if pos_encoding.shape[1] != C:
            pos_encoding = pos_encoding.expand(batch_size, C, L)
        combined = torch.cat([y, pos_encoding], dim=1)
        attn = self.attention(combined)
        return y * attn + pos_encoding * (1 - attn)

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
                          padding=(k + (k-1)*(dilation-1) - 1)//2,
                          dilation=dilation, groups=base_channels),
                nn.BatchNorm1d(base_channels),
                nn.GELU()
            ) for k, dilation in zip([3, 21], [1, 3])
        ])
        self.channel_red = nn.Conv1d(base_channels*3, base_channels*2, 1)
        self.res_conv = nn.Conv1d(in_channels, base_channels*2, 1) if in_channels != base_channels*2 else None

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)
        x = self.bottleneck(x)
        branches = [conv(x) for conv in self.convs]
        branches.append(nn.AdaptiveAvgPool1d(x.size(-1))(x))
        out = self.channel_red(torch.cat(branches, dim=1))
        return out + residual

class InceptionTime(nn.Module):
    def __init__(self, num_classes, seq_len=2000, embedding_dim=128, config=None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.embedding_dim = embedding_dim if self.config['embedding']['enable'] else 1
        kernel_size = self.config['embedding']['kernel_size']
        padding = (kernel_size - 1) // 2
        self.embedding = nn.Conv1d(1, self.embedding_dim, kernel_size, padding=padding, bias=False)
        self.wave_encoder = WaveEncoder(seq_len=seq_len, latent_dim=self.embedding_dim)
        self.input_norm = nn.BatchNorm1d(self.embedding_dim, eps=1e-3, momentum=0.1, affine=True)

        self.inception_blocks = nn.ModuleList([
            self._make_inception(self.embedding_dim, 32, pool=True),
            self._make_inception(64, 32, pool=False),
            self._make_inception(64, 32, pool=False),
            self._make_inception(64, 32, pool=False)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        self.gru = nn.GRU(
            input_size=64, hidden_size=64, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.gru_fusion = nn.Sequential(
            nn.Linear(64*2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def _make_inception(self, in_c, base_c, pool=False):
        layers = [InceptionModule(in_c, base_c)]
        if pool:
            layers.append(nn.MaxPool1d(3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.wave_encoder(x)
        x = self.input_norm(x)
        inception_feats = []
        for block in self.inception_blocks:
            x = block(x)
            inception_feats.append(x)
        x_global = self.adaptive_pool(x).squeeze(-1)
        gru_input = inception_feats[-1].transpose(1, 2)
        gru_out, _ = self.gru(gru_input)
        gru_feat = self.gru_fusion(gru_out[:, -1, :])
        fused = torch.cat([x_global, gru_feat], dim=1)
        return self.classifier(fused)

# -------------------- Trainer --------------------
class BioTrainer:
    def __init__(self, model, train_loader, val_loader, config, output_dir, hard_val_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hard_val_loader = hard_val_loader
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Optimizer
        params = [
            {'params': [p for n, p in model.named_parameters() if 'norm' not in n],
             'weight_decay': config['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if 'norm' in n]}
        ]
        self.optimizer = optim.AdamW(params, lr=config['init_lr'])
        steps_per_epoch = len(train_loader) // config['accumulation_steps']
        self.scheduler = self.ThreePhaseLRScheduler(
            self.optimizer, config['epochs'], steps_per_epoch,
            config['max_lr'], config['div_factor']
        )
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        self.best_f1 = 0.0
        self.early_stop_counter = 0
        self.best_model_path = os.path.join(output_dir, 'best_model.pth')

    class ThreePhaseLRScheduler:
        def __init__(self, optimizer, num_epochs, steps_per_epoch, max_lr, div_factor):
            self.optimizer = optimizer
            self.steps_per_epoch = max(1, steps_per_epoch)
            self.total_steps = num_epochs * self.steps_per_epoch
            self.step_count = 0
            self.phase1_ratio = 0.15
            self.phase2_ratio = 0.70
            self.phase1_steps = int(self.phase1_ratio * self.total_steps)
            self.phase2_steps = int(self.phase2_ratio * self.total_steps)
            self.phase3_steps = self.total_steps - self.phase1_steps - self.phase2_steps
            self.init_lr = max_lr / div_factor
            self.max_lr = max_lr
            self.final_lr = max(self.max_lr / 100, 2e-5)
            self.max_lr_phase3 = self.max_lr * 0.1
            self.last_val_loss = float('inf')
            self.current_lr = self.init_lr
            self.adjusted_phase2_steps = self.phase2_steps

        def step(self, val_loss=None):
            if val_loss is not None:
                self._update_phase2_progress(val_loss)
            self._calculate_lr()
            for pg in self.optimizer.param_groups:
                pg['lr'] = min(max(self.current_lr, self.final_lr), self.max_lr)
            self.step_count += 1

        def _update_phase2_progress(self, val_loss):
            phase = self._current_phase()
            if phase != 2: return
            allowed_increase = max(0.01 * self.last_val_loss, 0.05)
            if val_loss > self.last_val_loss + allowed_increase:
                self.adjusted_phase2_steps = max(int(self.adjusted_phase2_steps * 0.95),
                                                 self.phase2_steps // 5)
            self.last_val_loss = val_loss

        def _current_phase(self):
            if self.step_count < self.phase1_steps:
                return 1
            elif self.step_count < self.phase1_steps + self.adjusted_phase2_steps:
                return 2
            else:
                return 3

        def _calculate_lr(self):
            phase = self._current_phase()
            if phase == 1:
                progress = self.step_count / self.phase1_steps
                self.current_lr = self.init_lr + (self.max_lr - self.init_lr) * progress
            elif phase == 2:
                progress = (self.step_count - self.phase1_steps) / self.adjusted_phase2_steps
                self.current_lr = self.max_lr * (0.2 + 0.8 * (1 + np.cos(np.pi * progress)) / 2)
            else:
                progress = (self.step_count - self.phase1_steps - self.adjusted_phase2_steps) / self.phase3_steps
                self.current_lr = self.max_lr_phase3 * (1 - progress) + self.final_lr * progress

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        self.optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / self.config['accumulation_steps']
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            total_loss += loss.item() * self.config['accumulation_steps'] * inputs.size(0)
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
            for inputs, labels in eval_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item() * inputs.size(0)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Bootstrap CI for balanced accuracy
        def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=95):
            stats = []
            for _ in range(n_bootstrap):
                idx = resample(np.arange(len(y_true)))
                stats.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
            lower = (100 - ci) / 2
            upper = 100 - lower
            return np.percentile(stats, [lower, upper])

        ci_low, ci_high = bootstrap_ci(all_labels, all_preds)
        metrics = {
            'loss': total_loss / len(eval_loader.dataset),
            'acc': balanced_accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average=self.config['eval_metrics']['average_type']),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'labels': all_labels,
            'probs': all_probs,
            'preds': all_preds
        }
        if self.config['eval_metrics']['show_confusion_matrix']:
            metrics['confusion_matrix'] = confusion_matrix(
                all_labels, all_preds, labels=np.arange(self.config['num_classes'])
            )
        # AUC
        try:
            if self.config['num_classes'] == 2:
                metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                y_bin = label_binarize(all_labels, classes=np.arange(self.config['num_classes']))
                class_auc = []
                for i in range(self.config['num_classes']):
                    if np.sum(y_bin[:, i]) == 0:
                        class_auc.append(np.nan)
                        continue
                    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                    class_auc.append(auc(fpr, tpr))
                valid = np.array(class_auc)[~np.isnan(class_auc)]
                metrics['auc'] = np.mean(valid) if len(valid) > 0 else np.nan
        except Exception as e:
            print(f"AUC error: {e}")
            metrics['auc'] = -1.0
        return metrics

    def train(self):
        print(f"Training on {self.device}")
        for epoch in range(self.config['epochs']):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step(val_loss=val_metrics['loss'])

            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}/{self.config['epochs']} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.2%} | F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.2%} (CI: {val_metrics['ci_low']:.2%}~{val_metrics['ci_high']:.2%}) | F1: {val_metrics['f1']:.4f}")
            print("-"*70)

            if self.early_stop_counter >= self.config['early_stop_patience']:
                print(f"Early stopping triggered. Best F1: {self.best_f1:.4f}")
                break

    def evaluate_test(self, test_loader):
        print("\n=== Evaluating on test set ===")
        metrics = self.validate(test_loader)
        print(f"Balanced accuracy: {metrics['acc']:.4f} (95% CI: {metrics['ci_low']:.4f}~{metrics['ci_high']:.4f})")
        print(f"Macro-F1: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        # Plot if requested
        if self.config['eval_metrics']['show_confusion_matrix']:
            self._plot_curves(metrics)
        return metrics

    def _plot_curves(self, metrics):
        plt.figure(figsize=(15,6))
        # Confusion matrix
        plt.subplot(1,2,1)
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.config['eval_metrics']['class_names'],
                    yticklabels=self.config['eval_metrics']['class_names'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        # ROC
        if self.config['num_classes'] > 2:
            plt.subplot(1,2,2)
            y_bin = label_binarize(metrics['labels'], classes=np.arange(self.config['num_classes']))
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(self.config['num_classes']):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], metrics['probs'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink'])
            for i, color in zip(range(self.config['num_classes']), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'{self.config["eval_metrics"]["class_names"][i]} (AUC={roc_auc[i]:.3f})')
            plt.plot([0,1], [0,1], 'k--', lw=2)
            plt.xlim([0,1]); plt.ylim([0,1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC (macro AUC={metrics["auc"]:.3f})')
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'evaluation_plots.png'), dpi=300)
        plt.show()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to HDF5 file')
    parser.add_argument('--output_dir', default='./results', help='Directory to save model and plots')
    parser.add_argument('--seed', type=int, default=726, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Train batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data and infer classes
    with h5py.File(args.data, 'r') as h5:
        num_classes = len([c for c in h5['train'].keys() if c.startswith('class_')])
        class_names = [h5.attrs.get('class_labels', [f'Class{i}' for i in range(num_classes)])]
        if isinstance(class_names, np.ndarray):
            class_names = class_names.tolist()
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['seed'] = args.seed
    config['num_classes'] = num_classes
    config['train_batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['eval_metrics']['class_names'] = class_names

    # Create datasets
    train_dataset = BioDataset(args.data, 'train')
    val_dataset = BioDataset(args.data, 'val')
    test_dataset = BioDataset(args.data, 'test')

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],
                              shuffle=True, num_workers=config['num_workers'],
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'],
                            num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['val_batch_size'],
                             num_workers=config['num_workers'], pin_memory=True)

    # Model
    model = InceptionTime(num_classes=num_classes, seq_len=config['seq_length'],
                          embedding_dim=config['embedding']['dim'], config=config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = BioTrainer(model, train_loader, val_loader, config, args.output_dir)
    trainer.train()

    # Load best model and evaluate test
    model.load_state_dict(torch.load(trainer.best_model_path))
    test_metrics = trainer.evaluate_test(test_loader)

    # Save metrics
    import json
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in test_metrics.items()
                   if k not in ['labels', 'probs', 'preds', 'confusion_matrix']}, f, indent=2)

if __name__ == '__main__':
    main()