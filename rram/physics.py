import torch
from torch import jit

from core.physics import DevicePhysics


class RRAMPhysics(DevicePhysics):
    """Filament-gap dynamics with Joule heating of a metal-oxide RRAM cell (Stanford RRAM model, Jiang et al. 2016)."""

    state_name = 'gap'
    materials = ('HfO2', 'Al2O3', 'TiO2')
    split_voltages = {'HfO2': (1.27, -1.32), 'Al2O3': (1.0, -1.18), 'TiO2': (1.54, -1.45)}
    initial_time = 7e-12

    kb = 1.380649e-23       # Boltzmann constant [J/K]
    q = 1.60217663e-19      # Elementary charge [C]
    T0 = 273 + 25           # Ambient temperature [K]
    tox = 5e-9              # Oxide thickness [m]
    a0 = 0.25e-9            # Atomic distance [m]
    gap_min = 0.1e-9        # Minimum gap [m]
    gap_max = 1.7e-9        # Maximum gap [m]
    Tau_th = 2.3e-10        # Effective thermal time constant [s]
    g1 = 1e-9               # Length scale for gamma calculation [m]
    beta = 1.25             # Field enhancement coefficient
    gamma0 = {'pos': 15, 'neg': 8.5}    # Field enhancement factor, by voltage polarity
    Vel0 = {'pos': 120, 'neg': 150}     # Base velocity [m/s], by voltage polarity
    material_params = {
        'HfO2': {'Eag': 1.241, 'Ear': 1.24, 'Cth': 3.05e-18},
        'Al2O3': {'Eag': 1.001, 'Ear': 1.0, 'Cth': 2.98e-18},
        'TiO2': {'Eag': 1.501, 'Ear': 1.50, 'Cth': 3.18e-18},
    }

    def simulate(self, dt, voltage, current, material):
        params = self.material_params[material]
        polarity = 'pos' if voltage[-1] > 0 else 'neg'
        as_tensor = lambda value: torch.tensor(value, device=dt.device, dtype=dt.dtype)
        gap, _ = simulate_rram(
            dt, voltage, current,
            as_tensor(self.gamma0[polarity]), as_tensor(self.beta), as_tensor(self.g1), as_tensor(self.q),
            as_tensor(params['Eag']), as_tensor(self.kb), as_tensor(self.a0), as_tensor(self.tox),
            as_tensor(params['Ear']), as_tensor(self.Vel0[polarity]), as_tensor(self.gap_min),
            as_tensor(self.gap_max), as_tensor(self.T0), as_tensor(params['Cth']), as_tensor(self.Tau_th),
        )
        return gap

    def normalize_state(self, gap):
        return -1 + (gap - self.gap_min) * (2 / (self.gap_max - self.gap_min))


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
