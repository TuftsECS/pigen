import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from .utils import calculate_accuracy, calculate_epoch_metrics, plot_gap_sequence
from .loss import simulate_rram_wrapper

class DynamicGradientClipper:
    def __init__(self, initial_max_norm=1.0, momentum=0.95):
        self.max_norm = initial_max_norm
        self.momentum = momentum
        self.running_grad_norm = initial_max_norm
        
    def update(self, parameters):
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), norm_type=2)
        self.running_grad_norm = (self.momentum * self.running_grad_norm + 
                                (1 - self.momentum) * grad_norm.item())
        self.max_norm = min(max(self.running_grad_norm * 1.2, 0.1), 5.0)
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, norm_type=2)
        
        return self.max_norm

class AdaptivePDEScheduler:
    def __init__(self, alpha=0.95, min_weight=0.1, max_weight=5.0):
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.data_grad_ema = None
        self.pde_grad_ema = None
        
    def update(self, data_grad_norm: float, pde_grad_norm: float):
        if self.data_grad_ema is None:
            self.data_grad_ema = data_grad_norm
            self.pde_grad_ema = pde_grad_norm
        else:
            self.data_grad_ema = self.alpha * self.data_grad_ema + (1-self.alpha) * data_grad_norm
            self.pde_grad_ema = self.alpha * self.pde_grad_ema + (1-self.alpha) * pde_grad_norm
            
    def get_weight(self) -> float:
        if self.pde_grad_ema == 0 or self.data_grad_ema is None:
            return 1.0
            
        ratio = self.data_grad_ema / self.pde_grad_ema
        return max(min(ratio, self.max_weight), self.min_weight)
            
def setup_logger(save_dir: str, exp_name: str) -> logging.Logger:
    log_file = os.path.join(save_dir, 'training.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('RRAM_Training')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    logger.info(f"{'='*50}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*50}\n")
    
    return logger

def get_standardized_gap(value, target_min=-1, target_max=1, gap_min=-1, gap_max=1):
    gap_scale = (target_max - target_min) / (gap_max - gap_min)
    gap_normalized = target_min + (value - gap_min) * gap_scale
    return gap_normalized
    
def compute_pde_loss(gap_pred, gap_physical, const):
    gap_standardized = get_standardized_gap(gap_physical, -1, 1, const.gap_min, const.gap_max)
    gap_loss = F.mse_loss(gap_pred, gap_standardized)
    return gap_loss

def compute_grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
    return grad_norm

def train_sequence(pinn_model, mlp_model, dataset, pinn_optimizer, mlp_optimizer, 
                   const, scalers, use_pde=True, pde_scheduler=None, 
                   pinn_scheduler=None, mlp_scheduler=None):
    pinn_model.train()
    mlp_model.train() 
    
    total_mlp_loss = 0
    total_pinn_loss = 0
    total_data_loss = 0
    total_pde_loss = 0
    device = next(pinn_model.parameters()).device
    
    pinn_clipper = DynamicGradientClipper(initial_max_norm=1.0)
    mlp_clipper = DynamicGradientClipper(initial_max_norm=1.0)
    
    for sequence in dataset:
        time_seq_full = sequence['time'].to(device)
        dt_seq_full = sequence['dt'].to(device)
        voltage_seq_full = sequence['voltage'].to(device)
        true_current_full = sequence['current'].to(device)
        material_idx = sequence['material_idx'].to(device)
        material = sequence['material']
        
        time_seq = time_seq_full[2:]
        voltage_seq = voltage_seq_full[2:]
        true_current = true_current_full[2:]
        initial_I = torch.ones_like(voltage_seq) * true_current[0]

        time_scale = torch.tensor(scalers['time'].scale_[0], device=device)
        time_mean = torch.tensor(scalers['time'].mean_[0], device=device)
        time_real_full = time_seq_full * time_scale + time_mean
        voltage_scale = torch.tensor(scalers['voltage'].scale_[0], device=device)
        voltage_mean = torch.tensor(scalers['voltage'].mean_[0], device=device)
        voltage_real_full = voltage_seq_full * voltage_scale + voltage_mean
        dt_scale = torch.tensor(scalers['dt'].scale_[0], device=device)
        dt_mean = torch.tensor(scalers['dt'].mean_[0], device=device)
        dt_real_full = dt_seq_full * dt_scale + dt_mean
        true_current_scale = torch.tensor(scalers['current'].scale_[0], device=device)
        true_current_mean = torch.tensor(scalers['current'].mean_[0], device=device)
        true_current_real_full = true_current_full * true_current_scale + true_current_mean

        time_real = time_real_full[2:]
        voltage_real = voltage_real_full[2:]
        dt_real = dt_real_full[2:]
        true_current_real = true_current_real_full[2:]

        if use_pde:
            with torch.no_grad():
                gap = pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            
            mlp_optimizer.zero_grad()
            pred_current = mlp_model(gap, voltage_seq, initial_I, material_idx)
            mlp_loss = F.mse_loss(pred_current, true_current)
            mlp_loss.backward()
            max_norm_mlp = mlp_clipper.update(mlp_model.parameters())
            mlp_optimizer.step()
            if mlp_scheduler:
                mlp_scheduler.step()

            gap_physical_full, temperature = simulate_rram_wrapper(dt_real_full, voltage_real_full, true_current_real_full, const, material)
            gap_physical = gap_physical_full[2:]
            
            pinn_optimizer.zero_grad()
            gap = pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            
            pde_loss = compute_pde_loss(gap, gap_physical, const)
            
            pred_current = mlp_model(gap, voltage_seq, initial_I, material_idx)
            data_loss = F.mse_loss(pred_current, true_current)
            
            if pde_scheduler:
                data_grad_norm = compute_grad_norm(data_loss, pinn_model.parameters())
                pde_grad_norm = compute_grad_norm(pde_loss, pinn_model.parameters())
                
                pde_scheduler.update(data_grad_norm.item(), pde_grad_norm.item())
                pde_weight = pde_scheduler.get_weight()
            else:
                pde_weight = 1.0
                
            pinn_loss = data_loss + pde_weight * pde_loss
            pinn_loss.backward()
            max_norm_pinn = pinn_clipper.update(pinn_model.parameters())
            pinn_optimizer.step()
            if pinn_scheduler:
                pinn_scheduler.step()
            
            total_data_loss += data_loss.item()
            total_pde_loss += pde_loss.item()
                
        else:
            with torch.no_grad():
                gap = pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            
            mlp_optimizer.zero_grad()
            pred_current = mlp_model(gap, voltage_seq, initial_I, material_idx)
            mlp_loss = F.mse_loss(pred_current, true_current)
            mlp_loss.backward()
            
            max_norm_mlp = mlp_clipper.update(mlp_model.parameters())
            
            mlp_optimizer.step()
            if mlp_scheduler:
                mlp_scheduler.step()

            pinn_optimizer.zero_grad()
            gap = pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            pred_current = mlp_model(gap, voltage_seq, initial_I, material_idx)
            data_loss = F.mse_loss(pred_current, true_current)
            
            pinn_loss = data_loss
            pinn_loss.backward()
            max_norm_pinn = pinn_clipper.update(pinn_model.parameters())
            pinn_optimizer.step()
            if pinn_scheduler:
                pinn_scheduler.step()
                
        total_mlp_loss += mlp_loss.item()
        total_pinn_loss += pinn_loss.item()
        
    num_sequences = len(dataset)
    return (total_mlp_loss/num_sequences, 
            total_pinn_loss/num_sequences, 
            {
                'learning_rates': {
                    'pinn_lr': pinn_optimizer.param_groups[0]['lr'],
                    'mlp_lr': mlp_optimizer.param_groups[0]['lr']
                },
                'grad_norms': {
                    'pinn_max_norm': max_norm_pinn,
                    'mlp_max_norm': max_norm_mlp
                },
                'pde_weight': pde_scheduler.get_weight() if pde_scheduler and use_pde else 0.0,
            })
    
def validate_sequence(pinn_model, mlp_model, dataset, const, scalers, use_pde=True, pde_scheduler=None, save_dir=None):
    pinn_model.eval()
    mlp_model.eval()
    
    total_mlp_loss = 0
    total_pinn_loss = 0
    total_data_loss = 0
    total_pde_loss = 0
    errors = []
    switch_time_accuracies = []
    switch_duration_accuracies = []
    plateau_current_accuracies = []

    device = next(pinn_model.parameters()).device

    for sequence_idx, sequence in enumerate(dataset):
        time_seq_full = sequence['time'].to(device)
        dt_seq_full = sequence['dt'].to(device)
        voltage_seq_full = sequence['voltage'].to(device)
        true_current_full = sequence['current'].to(device)
        material_idx = sequence['material_idx'].to(device)
        material = sequence['material']
        
        time_seq = time_seq_full[2:]
        voltage_seq = voltage_seq_full[2:]
        true_current = true_current_full[2:]
        initial_I = torch.ones_like(voltage_seq) * true_current[0]

        time_scale = torch.tensor(scalers['time'].scale_[0], device=device)
        time_mean = torch.tensor(scalers['time'].mean_[0], device=device)
        time_real_full = time_seq_full * time_scale + time_mean
        voltage_scale = torch.tensor(scalers['voltage'].scale_[0], device=device)
        voltage_mean = torch.tensor(scalers['voltage'].mean_[0], device=device)
        voltage_real_full = voltage_seq_full * voltage_scale + voltage_mean
        dt_scale = torch.tensor(scalers['dt'].scale_[0], device=device)
        dt_mean = torch.tensor(scalers['dt'].mean_[0], device=device)
        dt_real_full = dt_seq_full * dt_scale + dt_mean
        true_current_scale = torch.tensor(scalers['current'].scale_[0], device=device)
        true_current_mean = torch.tensor(scalers['current'].mean_[0], device=device)
        true_current_real_full = true_current_full * true_current_scale + true_current_mean

        time_real = time_real_full[2:]
        voltage_real = voltage_real_full[2:]
        dt_real = dt_real_full[2:]
        true_current_real = true_current_real_full[2:]
        
        with torch.no_grad():
            gap = pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            pred_current = mlp_model(gap, voltage_seq, initial_I, material_idx)
            mlp_loss = F.mse_loss(pred_current, true_current)
            data_loss = mlp_loss
            
            gap_physical_full, temperature = simulate_rram_wrapper(dt_real_full, voltage_real_full, true_current_real_full, const, material)
            gap_physical = gap_physical_full[2:]
            
            pde_loss = compute_pde_loss(gap, gap_physical, const)

            pde_weight = pde_scheduler.get_weight() if pde_scheduler else 1.0
            pinn_loss = data_loss + pde_weight * pde_loss
            
            total_data_loss += data_loss.item()
            total_pde_loss += pde_loss.item()
                
            (
                mean_error,
                is_switch_time_accurate, _,
                is_switch_duration_accurate, _,
                is_plateau_current_accurate, _,
                true_idx_10, true_idx_mid, true_idx_90,
                pred_idx_10, pred_idx_mid, pred_idx_90
            ) = calculate_accuracy(pred_current, true_current, time_real, scalers)
            
            errors.append(mean_error)
            switch_time_accuracies.append(is_switch_time_accurate)
            switch_duration_accuracies.append(is_switch_duration_accurate)
            plateau_current_accuracies.append(is_plateau_current_accurate)

            total_mlp_loss += mlp_loss.item()
            total_pinn_loss += pinn_loss.item()
            
            if save_dir and sequence_idx % 10 == 0:
                plot_gap_sequence(time_real, gap, gap_physical, true_current_real, 
                                pred_current, None, save_dir, sequence_idx, const,
                                scalers, use_pde=use_pde)
    
    epoch_mean_error, epoch_max_error, epoch_switch_time_accuracy, \
    epoch_switch_duration_accuracy, epoch_plateau_current_accuracy = \
        calculate_epoch_metrics(errors, switch_time_accuracies, switch_duration_accuracies, plateau_current_accuracies)
    
    num_sequences = len(dataset)
    
    metrics = {
        'mlp_loss': total_mlp_loss / num_sequences,
        'pinn_loss': total_pinn_loss / num_sequences,
        'data_loss': total_data_loss / num_sequences,
        'pde_loss': total_pde_loss / num_sequences if use_pde else 0.0,
        'mean_error': epoch_mean_error,
        'max_error': epoch_max_error,
        'switch_time_accuracy': epoch_switch_time_accuracy,
        'switch_duration_accuracy': epoch_switch_duration_accuracy,
        'plateau_current_accuracy': epoch_plateau_current_accuracy,
        'pde_weight': pde_scheduler.get_weight() if pde_scheduler and use_pde else 0.0
    }
    
    return metrics