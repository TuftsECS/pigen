import os
import logging
from datetime import datetime
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def configure_optimizers(pinn_model, mlp_model, config):
    pinn_lr = config.get('learning_rate_pinn', 1e-3)
    pinn_weight_decay = config.get('weight_decay_pinn', 1e-2)
    mlp_lr = config.get('learning_rate_mlp', 1e-3)
    mlp_weight_decay = config.get('weight_decay_mlp', 1e-2)

    pinn_optimizer = optim.AdamW(
        pinn_model.parameters(),
        lr=pinn_lr,
        weight_decay=pinn_weight_decay
    )

    mlp_optimizer = optim.AdamW(
        mlp_model.parameters(),
        lr=mlp_lr,
        weight_decay=mlp_weight_decay
    )

    return {
        "pinn_optimizer": pinn_optimizer,
        "mlp_optimizer": mlp_optimizer
    }

def calculate_accuracy(outputs, targets, time_real, scalers=None):
    device = outputs.device

    if scalers is not None:
        current_scale = torch.tensor(scalers['current'].scale_[0], device=device)
        current_mean = torch.tensor(scalers['current'].mean_[0], device=device)
        outputs = outputs * current_scale + current_mean
        targets = targets * current_scale + current_mean

    time_np = time_real.cpu().numpy() if torch.is_tensor(time_real) else time_real
    outputs_np = outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets

    seq_len = len(targets_np)
    if seq_len < 2:
        return 0.0, True, 0.0, True, 0.0, True, 0.0, -1, -1, -1, -1, -1, -1

    relative_errors_seq = np.abs(outputs_np - targets_np) / (np.abs(targets_np) + 1e-10)
    mean_error = np.mean(relative_errors_seq) * 100

    target_start_current = targets_np[0]
    target_end_current = targets_np[-1]
    pred_start_current = outputs_np[0]
    pred_end_current = outputs_np[-1]

    target_is_increasing = target_end_current > target_start_current
    pred_is_increasing = pred_end_current > pred_start_current

    def find_index(array, threshold, is_increasing):
        if is_increasing:
            indices = np.where(array >= threshold)[0]
            return indices[0] if len(indices) > 0 else -1
        else:
            indices = np.where(array <= threshold)[0]
            return indices[0] if len(indices) > 0 else -1

    is_switch_time_accurate = False
    switch_time_error = float('inf')
    is_switch_duration_accurate = False
    switch_duration_error = float('inf')
    is_plateau_current_accurate = False
    plateau_current_error = float('inf')

    target_delta = target_end_current - target_start_current
    pred_delta = pred_end_current - pred_start_current

    target_mid_current = target_start_current + 0.5 * target_delta
    pred_mid_current = pred_start_current + 0.5 * pred_delta

    true_idx_mid = find_index(targets_np, target_mid_current, target_is_increasing)
    pred_idx_mid = find_index(outputs_np, pred_mid_current, pred_is_increasing)

    seq_length = len(time_real)
    true_relative_pos = true_idx_mid / seq_length
    pred_relative_pos = pred_idx_mid / seq_length

    switch_time_error = abs(pred_relative_pos - true_relative_pos) * 100
    is_switch_time_accurate = switch_time_error <= 3.0

    target_threshold_10 = target_start_current + 0.1 * target_delta
    target_threshold_90 = target_start_current + 0.9 * target_delta
    pred_threshold_10 = pred_start_current + 0.1 * pred_delta
    pred_threshold_90 = pred_start_current + 0.9 * pred_delta

    true_idx_10 = find_index(targets_np, target_threshold_10, target_is_increasing)
    true_idx_90 = find_index(targets_np, target_threshold_90, target_is_increasing)
    pred_idx_10 = find_index(outputs_np, pred_threshold_10, pred_is_increasing)
    pred_idx_90 = find_index(outputs_np, pred_threshold_90, pred_is_increasing)

    true_duration = abs(true_idx_90 - true_idx_10)
    pred_duration = abs(pred_idx_90 - pred_idx_10)

    true_duration_relative = true_duration / seq_length
    pred_duration_relative = pred_duration / seq_length

    switch_duration_error = abs(pred_duration_relative - true_duration_relative) * 100
    is_switch_duration_accurate = switch_duration_error <= 10.0

    diff_true = np.diff(targets_np)
    non_zero_indices = np.where(diff_true != 0)[0]

    if len(non_zero_indices) > 0:
        final_current_idx = non_zero_indices[-1] + 1
    else:
        final_current_idx = 1

    if final_current_idx < seq_len:
        true_plateau_mean = np.mean(targets_np[final_current_idx:])
        pred_plateau_mean = np.mean(outputs_np[final_current_idx:])

        if np.abs(true_plateau_mean) > 1e-10:
            plateau_current_error = abs(pred_plateau_mean - true_plateau_mean) / abs(true_plateau_mean) * 100
        else:
            plateau_current_error = 0.0 if np.abs(pred_plateau_mean) < 1e-10 else float('inf')

        is_plateau_current_accurate = plateau_current_error <= 5.0
    else:
        true_plateau_mean = np.nan
        pred_plateau_mean = np.nan
        plateau_current_error = float('inf')
        is_plateau_current_accurate = False

    return (mean_error,
            is_switch_time_accurate, switch_time_error,
            is_switch_duration_accurate, switch_duration_error,
            is_plateau_current_accurate, plateau_current_error,
            true_idx_10, true_idx_mid, true_idx_90,
            pred_idx_10, pred_idx_mid, pred_idx_90)

def calculate_epoch_metrics(errors, switch_time_accuracies, switch_duration_accuracies, plateau_current_accuracies):
    num_sequences = len(errors)
    if num_sequences == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    epoch_mean_error = sum(errors) / num_sequences
    epoch_max_error = max(errors) if errors else 0.0

    epoch_switch_time_accuracy = sum(switch_time_accuracies) / num_sequences * 100
    epoch_switch_duration_accuracy = sum(switch_duration_accuracies) / num_sequences * 100
    epoch_plateau_current_accuracy = sum(plateau_current_accuracies) / num_sequences * 100

    return (epoch_mean_error, epoch_max_error, epoch_switch_time_accuracy,
            epoch_switch_duration_accuracy, epoch_plateau_current_accuracy)

def calculate_metric_range(ranges):
    all_mins, all_maxs = zip(*ranges)
    return (min(all_mins), max(all_maxs))

def save_checkpoint(
    save_dir: str,
    epoch: int,
    pinn_model: torch.nn.Module,
    mlp_model: torch.nn.Module,
    optimizer_pinn: torch.optim.Optimizer,
    optimizer_mlp: torch.optim.Optimizer,
    val_metrics: dict,
    scalers: dict,
    metric_type='mean_error'
):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'pinn_model_state_dict': pinn_model.state_dict(),
        'mlp_model_state_dict': mlp_model.state_dict(),
        'pinn_optimizer_state_dict': optimizer_pinn.state_dict(),
        'mlp_optimizer_state_dict': optimizer_mlp.state_dict(),
        'best_valid_mean_error': val_metrics['mean_error'],
        'best_valid_max_error': val_metrics['max_error'],
        'best_valid_accuracy': val_metrics['plateau_current_accuracy'],
        'scalers': scalers
    }

    filename = f'best_{metric_type}_checkpoint.pth'
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)

    existing_backups = [f for f in os.listdir(save_dir) if f.startswith('model_epoch_')]
    existing_backups.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)

    for old_backup in existing_backups[3:]:
        try:
            os.remove(os.path.join(save_dir, old_backup))
        except OSError:
            pass

def get_standardized_gap(value, target_min=-1, target_max=1, gap_min=-1, gap_max=1):
    gap_scale = (target_max - target_min) / (gap_max - gap_min)
    gap_normalized = target_min + (value - gap_min) * gap_scale
    return gap_normalized

def plot_gap_sequence(time_seq, gap_pred, gap_physical, true_current, pred_current, attention, save_dir, sequence_idx, const, scalers, use_pde=True):

    if scalers and pred_current is not None:
        current_scale = scalers['current'].scale_[0]
        current_mean = scalers['current'].mean_[0]
        pred_current_real = pred_current.detach().cpu().numpy() * current_scale + current_mean

    n_plots = 3 if use_pde else 2
    plt.figure(figsize=(12, 3*n_plots))

    plt.subplot(n_plots, 1, 1)
    plt.semilogx(time_seq.cpu().numpy(), gap_pred.detach().cpu().numpy(), label='Predicted Gap', color='red', linestyle='--')
    if use_pde and gap_physical is not None:
        gap_standardized = get_standardized_gap(gap_physical.detach().cpu().numpy(), -1, 1, const.gap_min, const.gap_max)
        plt.semilogx(time_seq.cpu().numpy(), gap_standardized, label='Physical Gap', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Gap')
    plt.legend()
    plt.grid(True)

    plt.subplot(n_plots, 1, 2)
    plt.semilogx(time_seq.cpu().numpy(), true_current.detach().cpu().numpy(), label='True Current', color='green')
    if pred_current is not None:
        plt.semilogx(time_seq.cpu().numpy(), pred_current_real, label='Predicted Current', color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)

    if use_pde and attention is not None:
        plt.subplot(n_plots, 1, 3)
        plt.semilogx(time_seq.cpu().numpy(), attention.detach().cpu().numpy(), label='Attention', color='purple')
        plt.xlabel('Time')
        plt.ylabel('Attention')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, 'gap_plots'), exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'gap_plots', f'gap_sequence_{sequence_idx}.png'))
    plt.close()
    