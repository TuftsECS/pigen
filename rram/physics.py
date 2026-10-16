import torch
from torch import jit

from core.physics import DevicePhysics
from .endurance import EnduranceCalculator


class RRAMPhysics(DevicePhysics):
    """Filament-gap dynamics with Joule heating of a metal-oxide RRAM cell (Stanford RRAM model, Jiang et al. 2016)."""

    state_name = 'gap'
    materials = ('HfO2', 'Al2O3', 'TiO2')
    split_voltages = {'HfO2': (1.5, -1.5), 'Al2O3': (1.38, -1.38), 'TiO2': (1.65, -1.65)}
    initial_time = 7e-12
    # SET/RESET pulses from each material's switching threshold to 2 V.
    design_space = {
        'HfO2': {'pos': (1.5, 2.0), 'neg': (-2.0, -1.5)},
        'Al2O3': {'pos': (1.38, 2.0), 'neg': (-2.0, -1.38)},
        'TiO2': {'pos': (1.65, 2.0), 'neg': (-2.0, -1.65)},
    }

    kb = 1.380649e-23       # Boltzmann constant [J/K]
    q = 1.60217663e-19      # Elementary charge [C]
    T0 = 273 + 25           # Ambient temperature [K]
    tox = 5e-9              # Oxide thickness [m]
    a0 = 0.25e-9            # Atomic distance [m]
    gap_min = 0.1e-9        # Minimum gap [m]
    gap_max = 1.7e-9        # Maximum gap [m]
    Tau_th = 3.5e-10        # Effective thermal time constant [s]
    g1 = 1e-9               # Length scale for gamma calculation [m]
    beta = 0.3              # Field enhancement coefficient
    gamma0 = 4.0            # Field enhancement factor
    Vel0 = 100              # Base velocity [m/s]
    material_params = {
        'HfO2': {'Eag': 1.241, 'Ear': 1.24, 'Cth': 2.17e-17},
        'Al2O3': {'Eag': 1.001, 'Ear': 1.0, 'Cth': 2.12e-17},
        'TiO2': {'Eag': 1.701, 'Ear': 1.70, 'Cth': 2.26e-17},
    }

    def __init__(self):
        self.endurance_model = EnduranceCalculator({m: p['Eag'] for m, p in self.material_params.items()})

    def simulate(self, dt, voltage, current, material):
        params = self.material_params[material]
        as_tensor = lambda value: torch.tensor(value, device=dt.device, dtype=dt.dtype)
        gap, _ = simulate_rram(
            dt, voltage, current,
            as_tensor(self.gamma0), as_tensor(self.beta), as_tensor(self.g1), as_tensor(self.q),
            as_tensor(params['Eag']), as_tensor(self.kb), as_tensor(self.a0), as_tensor(self.tox),
            as_tensor(params['Ear']), as_tensor(self.Vel0), as_tensor(self.gap_min),
            as_tensor(self.gap_max), as_tensor(self.T0), as_tensor(params['Cth']), as_tensor(self.Tau_th),
        )
        return gap

    def normalize_state(self, gap):
        return -1 + (gap - self.gap_min) * (2 / (self.gap_max - self.gap_min))

    def endurance(self, current, voltage, dt, material):
        return self.endurance_model.calculate_endurance(current, voltage, dt, material)


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
