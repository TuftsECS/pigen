import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import jit

_physics_cache = None

def get_physics_cache():
    global _physics_cache
    if _physics_cache is None:
        from .cache import PhysicsSimulationCache
        _physics_cache = PhysicsSimulationCache(enabled=True)
    return _physics_cache

def simulate_rram_wrapper(dt, v, current, const, material, use_cache=True):
    device = dt.device
    
    if use_cache:
        cache = get_physics_cache()
        cached_result = cache.get(dt, v, current, material, device)
        if cached_result is not None:
            return cached_result
    
    const.update_material(material)
    dtype = dt.dtype
    
    beta = torch.tensor(const.beta, device=device, dtype=dtype)
    g1 = torch.tensor(const.g1, device=device, dtype=dtype)
    q = torch.tensor(const.q, device=device, dtype=dtype)
    Eag = torch.tensor(const.Eag, device=device, dtype=dtype)
    kb = torch.tensor(const.kb, device=device, dtype=dtype)
    a0 = torch.tensor(const.a0, device=device, dtype=dtype)
    tox = torch.tensor(const.tox, device=device, dtype=dtype)
    Ear = torch.tensor(const.Ear, device=device, dtype=dtype)
    gap_min = torch.tensor(const.gap_min, device=device, dtype=dtype)
    gap_max = torch.tensor(const.gap_max, device=device, dtype=dtype)
    T0 = torch.tensor(const.T0, device=device, dtype=dtype)
    Cth = torch.tensor(const.Cth, device=device, dtype=dtype)
    Tau_th = torch.tensor(const.Tau_th, device=device, dtype=dtype)
    
    if v[-1] > 0:
        Vel0 = torch.tensor(const.Vel0_pos, device=device, dtype=dtype)
        gamma0 = torch.tensor(const.gamma0_pos, device=device, dtype=dtype)
    else:
        Vel0 = torch.tensor(const.Vel0_neg, device=device, dtype=dtype)
        gamma0 = torch.tensor(const.gamma0_neg, device=device, dtype=dtype)
        
    gap, temperature = simulate_rram(
        dt, v, current, 
        gamma0, beta, g1, q, Eag, 
        kb, a0, tox, Ear, Vel0,
        gap_min, gap_max, T0, Cth, Tau_th
    )
    
    if use_cache:
        cache.put(dt, v, current, material, gap, temperature)
    
    return gap, temperature


@jit.script
def simulate_rram(dt, v, current, gamma0, beta, g1, q, Eag, kb, a0, tox, Ear, Vel0, 
                 gap_min, gap_max, T0, Cth, Tau_th):
    seq_len = dt.shape[0]
    gap = torch.zeros_like(dt)
    temperature = torch.zeros_like(dt)
    
    is_positive_voltage = torch.mean(v) > 0
    gap[0] = gap_max if is_positive_voltage else gap_min
    temperature[0] = T0
    
    inv_kb = 1.0 / kb
    q_inv_kb = q * inv_kb
    q_a0_tox_kb = q * a0 / tox * inv_kb
    
    for i in range(1, seq_len):
        dt_i = dt[i]
        prev_gap = gap[i-1]
        prev_temp = temperature[i-1]
        
        gamma = gamma0 - beta * (prev_gap / g1)**3
        
        inv_temp = 1.0 / prev_temp
        v_i = v[i]
        
        exp_forward = torch.exp(-q_inv_kb * Eag * inv_temp + 
                               gamma * q_a0_tox_kb * v_i * inv_temp)
        exp_reverse = torch.exp(-q_inv_kb * Ear * inv_temp - 
                               gamma * q_a0_tox_kb * v_i * inv_temp)
        
        gap_ddt = -Vel0 * (exp_forward - exp_reverse)
        gap[i] = torch.clamp(prev_gap + gap_ddt * dt_i, min=gap_min, max=gap_max)
        
        power_i = torch.abs(v_i * current[i])
        temperature[i] = (prev_temp + dt_i * (power_i / Cth + T0 / Tau_th)) / (1 + dt_i / Tau_th)
    return gap, temperature