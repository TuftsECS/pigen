import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from .cache import PhysicsSimulationCache
from .metrics import accuracy_percents, log_time_weights, mean_relative_error, switching_errors
from .plotting import plot_state_sequence

# Leading samples of every sequence (the voltage ramp-up) that are excluded from model inputs and losses.
RAMP_SAMPLES = 2

_physics_cache = PhysicsSimulationCache()

class DynamicGradientClipper:
    def __init__(self, initial_max_norm=1.0, momentum=0.95):
        self.max_norm = initial_max_norm
        self.momentum = momentum
        self.running_grad_norm = initial_max_norm
        
    def update(self, parameters):
        """Clips gradients in place. Returns None if the gradient is non-finite and the step must be skipped."""
        parameters = list(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), norm_type=2)
        if not torch.isfinite(grad_norm):
            return None
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
        if not (np.isfinite(data_grad_norm) and np.isfinite(pde_grad_norm)):
            return
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
    logger = logging.getLogger('PIGen')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file, mode='w')
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

def state_loss(state_pred, state_physical, physics):
    return F.mse_loss(state_pred, physics.normalize_state(state_physical))

def compute_grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
    return grad_norm

def prepare_sequence(sequence, scalers, device):
    """Standardized model inputs (ramp-up samples dropped) and full-length physical-unit tensors of one sequence."""
    def to_real(key, values):
        scale = torch.tensor(scalers[key].scale_[0], device=device)
        mean = torch.tensor(scalers[key].mean_[0], device=device)
        return values * scale + mean

    time_full = sequence['time'].to(device)
    dt_full = sequence['dt'].to(device)
    voltage_full = sequence['voltage'].to(device)
    current_full = sequence['current'].to(device)
    voltage = voltage_full[RAMP_SAMPLES:]
    current = current_full[RAMP_SAMPLES:]
    dt_real_full = to_real('dt', dt_full)
    return {
        'material': sequence['material'],
        'material_idx': sequence['material_idx'].to(device),
        'time': time_full[RAMP_SAMPLES:],
        'voltage': voltage,
        'current': current,
        'initial_current': torch.ones_like(voltage) * current[0],
        'dt_real': dt_real_full[RAMP_SAMPLES:],
        'time_real_full': to_real('time', time_full),
        'dt_real_full': dt_real_full,
        'voltage_real_full': to_real('voltage', voltage_full),
        'current_real_full': to_real('current', current_full),
    }

def physical_state(physics, s):
    full = _physics_cache.simulate(physics, s['dt_real_full'], s['voltage_real_full'], s['current_real_full'], s['material'])
    return full[RAMP_SAMPLES:]

def train_epoch(pinn_model, mlp_model, dataset, pinn_optimizer, mlp_optimizer, physics, scalers,
                use_pde=True, pde_scheduler=None):
    pinn_model.train()
    mlp_model.train() 
    
    total_mlp_loss = 0
    total_pinn_loss = 0
    skipped_steps = 0
    device = next(pinn_model.parameters()).device
    
    pinn_clipper = DynamicGradientClipper(initial_max_norm=1.0)
    mlp_clipper = DynamicGradientClipper(initial_max_norm=1.0)
    
    for idx in torch.randperm(len(dataset)).tolist():
        s = prepare_sequence(dataset[idx], scalers, device)
        inputs = (s['time'], s['dt_real'], s['voltage'], s['material_idx'])

        with torch.no_grad():
            state = pinn_model(*inputs)

        mlp_optimizer.zero_grad()
        pred_current = mlp_model(state, s['voltage'], s['initial_current'], s['material_idx'])
        mlp_loss = F.mse_loss(pred_current, s['current'])
        mlp_loss.backward()
        if mlp_clipper.update(mlp_model.parameters()) is not None:
            mlp_optimizer.step()
        else:
            skipped_steps += 1

        pinn_optimizer.zero_grad()
        state = pinn_model(*inputs)
        if use_pde:
            pde_loss = state_loss(state, physical_state(physics, s), physics)

        pred_current = mlp_model(state, s['voltage'], s['initial_current'], s['material_idx'])
        data_loss = F.mse_loss(pred_current, s['current'])

        if use_pde:
            if pde_scheduler:
                data_grad_norm = compute_grad_norm(data_loss, pinn_model.parameters())
                pde_grad_norm = compute_grad_norm(pde_loss, pinn_model.parameters())
                pde_scheduler.update(data_grad_norm.item(), pde_grad_norm.item())
                pde_weight = pde_scheduler.get_weight()
            else:
                pde_weight = 1.0
            pinn_loss = data_loss + pde_weight * pde_loss
        else:
            pinn_loss = data_loss

        pinn_loss.backward()
        if pinn_clipper.update(pinn_model.parameters()) is not None:
            pinn_optimizer.step()
        else:
            skipped_steps += 1
                
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
                'pde_weight': pde_scheduler.get_weight() if pde_scheduler and use_pde else 0.0,
                'skipped_steps': skipped_steps,
            })
    
def validate_epoch(pinn_model, mlp_model, dataset, physics, scalers, use_pde=True, pde_scheduler=None, save_dir=None):
    pinn_model.eval()
    mlp_model.eval()
    
    total_mlp_loss = 0
    total_pinn_loss = 0
    total_logt_mse = 0
    errors = []
    records = []

    device = next(pinn_model.parameters()).device
    current_scale = float(scalers['current'].scale_[0])
    current_mean = float(scalers['current'].mean_[0])

    for sequence_idx, sequence in enumerate(dataset):
        s = prepare_sequence(sequence, scalers, device)
        time_real = s['time_real_full'][RAMP_SAMPLES:]
        
        with torch.no_grad():
            state = pinn_model(s['time'], s['dt_real'], s['voltage'], s['material_idx'])
            pred_current = mlp_model(state, s['voltage'], s['initial_current'], s['material_idx'])
            mlp_loss = F.mse_loss(pred_current, s['current'])
            total_logt_mse += (log_time_weights(s['time_real_full']) * (pred_current - s['current']) ** 2).mean().item()
            
            state_phys = physical_state(physics, s)
            pde_weight = pde_scheduler.get_weight() if pde_scheduler else 1.0
            pinn_loss = mlp_loss + pde_weight * state_loss(state, state_phys, physics)

            pred_real = (pred_current * current_scale + current_mean).detach().cpu().numpy()
            true_real = (s['current'] * current_scale + current_mean).detach().cpu().numpy()
            errors.append(mean_relative_error(pred_real, true_real))
            records.append(switching_errors(time_real.detach().cpu().numpy(), true_real, pred_real))

            total_mlp_loss += mlp_loss.item()
            total_pinn_loss += pinn_loss.item()
            
            if save_dir and sequence_idx % 10 == 0:
                plot_state_sequence(time_real, state, state_phys, s['current_real_full'][RAMP_SAMPLES:],
                                    pred_current, save_dir, sequence_idx, physics, scalers, use_pde=use_pde)
    
    num_sequences = len(dataset)
    return {
        'mlp_loss': total_mlp_loss / num_sequences,
        'logt_mse': total_logt_mse / num_sequences,
        'pinn_loss': total_pinn_loss / num_sequences,
        'mean_error': float(np.mean(errors)) if errors else 0.0,
        'max_error': float(np.max(errors)) if errors else 0.0,
        **accuracy_percents(records),
    }
