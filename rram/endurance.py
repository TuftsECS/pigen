import numpy as np
import torch
from typing import Tuple, Dict


class EnduranceCalculator:
    """RRAM write endurance: failure time over switching time at the pulse's peak temperature."""

    def __init__(self, switching_barriers):
        """``switching_barriers``: per material Us [eV] (RRAMPhysics' Eag)."""
        self.kb = 1.380649e-23        # Boltzmann constant [J/K]
        self.q = 1.60217663e-19       # Elementary charge [C]
        self.T0 = 273 + 25            # Ambient temperature [K]
        self.a0 = 0.25e-9             # Atomic distance [m]
        self.tox = 5e-9               # Oxide thickness [m]
        self.f0 = 1e13                # Attempt frequency [Hz]
        self.Tau_th = 2.3e-10         # Effective thermal time constant [s]
        self.material_params = {
            'HfO2': {
                'Us': switching_barriers['HfO2'],  # Switching barrier = Eag
                'Uf': 1.74,           # Failure barrier > Eag
                'Cth': 3.1825e-16,    # Effective thermal capacitance [J/K]
            },
            'Al2O3': {
                'Us': switching_barriers['Al2O3'],  # Switching barrier = Eag
                'Uf': 1.35,           # Failure barrier > Eag
                'Cth': 3.112e-16,
            },
            'TiO2': {
                'Us': switching_barriers['TiO2'],  # Switching barrier = Eag
                'Uf': 1.96,           # Failure barrier > Eag
                'Cth': 3.3234e-16,
            }
        }
        self.current_material = 'HfO2'
        
    def calculate_temperature(self, current: torch.Tensor, voltage: torch.Tensor, dt: torch.Tensor) -> float:
        """Peak Joule-heating temperature of one pulse (implicit Euler, as in rram.va)."""
        cth = self.material_params[self.current_material]['Cth']
        power = (voltage * current).abs().detach().cpu().double().numpy()
        temperature = peak = self.T0
        for dt_i, power_i in zip(dt.detach().cpu().double().numpy(), power):
            temperature = (temperature + dt_i * (power_i / cth + self.T0 / self.Tau_th)) / (1 + dt_i / self.Tau_th)
            peak = max(peak, temperature)
        return float(peak)

    def calculate_run_time(self, voltage: float, temperature: float) -> float:
        Us_joules = self.material_params[self.current_material]['Us'] * self.q  # Convert eV to Joules
        
        return (2 * self.tox) / (self.f0 * self.a0) * \
            np.exp(Us_joules / (self.kb * temperature)) * \
            np.exp(-self.q * abs(voltage) * self.a0 / (2 * self.kb * temperature * self.tox))

    def calculate_failure_time(self, voltage: float, temperature: float) -> float:
        Uf_joules = self.material_params[self.current_material]['Uf'] * self.q  # Convert eV to Joules
        
        return (2 * self.tox) / (self.f0 * self.a0) * \
            np.exp(Uf_joules / (self.kb * temperature)) * \
            np.exp(-self.q * abs(voltage) * self.a0 / (2 * self.kb * temperature * self.tox))

    def calculate_endurance(self, current: torch.Tensor, voltage: torch.Tensor, dt: torch.Tensor,
                          material: str = 'HfO2') -> Tuple[float, Dict]:
        """Endurance of one operation from its current, voltage and time-step sequences in physical units."""
        self.update_material(material)
        
        temperature = self.calculate_temperature(current, voltage, dt)
        mean_current, mean_voltage = current.mean().item(), voltage.mean().item()
        ts = self.calculate_run_time(mean_voltage, temperature)
        tf = self.calculate_failure_time(mean_voltage, temperature)
        endurance = tf / ts

        metrics = {
            'temperature': temperature,
            'run_time': ts,
            'failure_time': tf,
            'endurance_cycles': endurance,
            'current': mean_current,
            'voltage': mean_voltage,
            'run_energy': abs(mean_current * mean_voltage * ts),
            'failure_energy': abs(mean_current * mean_voltage * tf)
        }

        return endurance, metrics

    def update_material(self, material: str):
        if material not in self.material_params:
            raise ValueError(f"Unknown material: {material}")
        self.current_material = material
