import os
import numpy as np
import torch
import argparse
from typing import Tuple, Dict
from src.data import RRAMDataset, Constants
from src.models import RRAM_PINN, MLP_Current

class RRAMEvaluator:
    def __init__(self, model_path, data_path, output_dir, device=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.const = Constants()
        
        self.checkpoint, self.pinn_model, self.mlp_model = self.load_model(model_path)
        self.dataset = self.load_data(data_path, self.checkpoint['scalers'])
        self.scalers = self.checkpoint['scalers']
        
        self.materials = ['HfO2', 'Al2O3', 'TiO2']
        self.material_to_idx = {mat: idx for idx, mat in enumerate(self.materials)}
        
        self.endurance_calculator = EnduranceCalculator()

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        state = checkpoint['pinn_model_state_dict']
        embedding_size = state['material_embedding.weight'].shape[1]
        hidden_size = state['gru.weight_hh_l0'].shape[1]
        
        pinn_model = RRAM_PINN(
            hidden_size=hidden_size, 
            embedding_size=embedding_size,
            const=self.const
        ).to(self.device)
        
        mlp_model = MLP_Current(
            hidden_size=hidden_size,
            embedding_size=embedding_size
        ).to(self.device)
        
        pinn_model.load_state_dict(checkpoint['pinn_model_state_dict'])
        mlp_model.load_state_dict(checkpoint['mlp_model_state_dict'])
        
        pinn_model.eval()
        mlp_model.eval()
        
        print(f"Successfully loaded model - Epoch: {checkpoint['epoch']+1}")
        print(f"Validation accuracy: {checkpoint['best_valid_accuracy']:.2f}%")
        print(f"Validation mean error: {checkpoint['best_valid_mean_error']:.2f}%")
        
        return checkpoint, pinn_model, mlp_model
    
    def load_data(self, data_path, scalers):
        print(f"Loading dataset from {data_path}")
        dataset = RRAMDataset(
            data_path=data_path,
            fit_scaler=False,
            is_train=True,
            use_full_dataset=True,
            split_ratio=1.0,
            seed=42
        )
        dataset.set_scalers(scalers)

        print(f"Loaded {len(dataset)} sequences")
        material_counts = dataset.get_material_distribution()
        for material, count in material_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"{material}: {count} sequences ({percentage:.1f}%)")
            
        return dataset
    
    def predict_sequence(self, sequence):
        time_seq_full = sequence['time'].to(self.device)
        dt_seq_full = sequence['dt'].to(self.device)
        voltage_seq_full = sequence['voltage'].to(self.device)
        true_current_full = sequence['current'].to(self.device)
        material_idx = sequence['material_idx'].to(self.device)
        
        time_seq = time_seq_full[2:]
        dt_seq = dt_seq_full[2:]
        voltage_seq = voltage_seq_full[2:]
        true_current = true_current_full[2:]
        initial_I = torch.ones_like(voltage_seq) * true_current[0]
        
        time_scale = torch.tensor(self.scalers['time'].scale_[0], device=self.device)
        time_mean = torch.tensor(self.scalers['time'].mean_[0], device=self.device)
        time_real_full = time_seq_full * time_scale + time_mean
        
        voltage_scale = torch.tensor(self.scalers['voltage'].scale_[0], device=self.device)
        voltage_mean = torch.tensor(self.scalers['voltage'].mean_[0], device=self.device)
        voltage_real_full = voltage_seq_full * voltage_scale + voltage_mean
        
        dt_scale = torch.tensor(self.scalers['dt'].scale_[0], device=self.device)
        dt_mean = torch.tensor(self.scalers['dt'].mean_[0], device=self.device)
        dt_real_full = dt_seq_full * dt_scale + dt_mean
        
        current_scale = torch.tensor(self.scalers['current'].scale_[0], device=self.device)
        current_mean = torch.tensor(self.scalers['current'].mean_[0], device=self.device)
        true_current_real_full = true_current_full * current_scale + current_mean
        
        time_real = time_real_full[2:]
        voltage_real = voltage_real_full[2:]
        dt_real = dt_real_full[2:]
        true_current_real = true_current_real_full[2:]
        
        with torch.no_grad():
            gap = self.pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            pred_current = self.mlp_model(gap, voltage_seq, initial_I, material_idx)
            pred_current_real = pred_current * current_scale + current_mean

        return time_real, dt_real, voltage_real, true_current_real, pred_current_real, gap
    
    def evaluate(self, material, pos_voltage, neg_voltage):
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}. Supported materials: {self.materials}")
            
        material_idx = self.material_to_idx[material]
        voltage_scale = self.scalers['voltage'].scale_[0]
        voltage_mean = self.scalers['voltage'].mean_[0]
        
        available_voltages = []
        for sequence in self.dataset:
            if sequence['material_idx'].item() != material_idx:
                continue
            
            seq_voltage = sequence['voltage'][-1].item()
            seq_voltage_real = seq_voltage * voltage_scale + voltage_mean
            available_voltages.append(seq_voltage_real)
        
        available_voltages = sorted(list(set([round(v, 2) for v in available_voltages])))
        pos_voltages = [v for v in available_voltages if v > 0]
        neg_voltages = [v for v in available_voltages if v < 0]
        
        pos_matched_sequence, pos_matched_idx, pos_actual_voltage = self._find_closest_sequence(material_idx, pos_voltage, "positive")
        neg_matched_sequence, neg_matched_idx, neg_actual_voltage = self._find_closest_sequence(material_idx, neg_voltage, "negative")
        
        pos_result = self._evaluate_single_sequence(pos_matched_sequence, pos_matched_idx, material, "SET")
        neg_result = self._evaluate_single_sequence(neg_matched_sequence, neg_matched_idx, material, "RESET")
        
        merged_result = self._merge_results(pos_result, neg_result)
        
        return merged_result
    
    def _find_closest_sequence(self, material_idx, target_voltage, voltage_type):
        voltage_scale = self.scalers['voltage'].scale_[0]
        voltage_mean = self.scalers['voltage'].mean_[0]
        
        closest_sequence = None
        closest_idx = -1
        closest_voltage = None
        min_diff = float('inf')
        
        for idx, sequence in enumerate(self.dataset):
            if sequence['material_idx'].item() != material_idx:
                continue
            seq_voltage = sequence['voltage'][-1].item()
            seq_voltage_real = seq_voltage * voltage_scale + voltage_mean
            
            if (voltage_type == "positive" and seq_voltage_real <= 0) or (voltage_type == "negative" and seq_voltage_real >= 0):
                continue
            diff = abs(seq_voltage_real - target_voltage)
            if diff < min_diff:
                min_diff = diff
                closest_sequence = sequence
                closest_idx = idx
                closest_voltage = seq_voltage_real
        
        if closest_sequence is None:
            raise ValueError(f"No {voltage_type} voltage sequence found for the material. Please check if the voltage value is reasonable.")
        
        return closest_sequence, closest_idx, closest_voltage
    
    def _evaluate_single_sequence(self, sequence, sequence_idx, material, operation_type):
        material_idx = self.material_to_idx[material]
        
        time_real, dt_real, voltage_real, true_current_real, pred_current_real, gap = self.predict_sequence(sequence)
        stable_idx = get_stable_index(pred_current_real)
        
        if stable_idx < len(time_real):
            actual_switching_time = time_real[stable_idx] - time_real[0]
            actual_switching_time = actual_switching_time.item()
        else:
            actual_switching_time = (time_real[-1] - time_real[0]).item()
        
        endurance, metrics = self.endurance_calculator.calculate_endurance(
            pred_current_real.mean().item(), 
            voltage_real.mean().item(), 
            dt_real,
            material
        )
        threshold_times, total_energy = get_energy_consumption(
            time_real, voltage_real, pred_current_real, material_idx
        )
        
        result = {
            'sequence_id': sequence_idx,
            'material': material,
            'voltage': voltage_real.mean().item(),
            'operation_type': operation_type,
            'endurance': endurance,
            'actual_switching_time': actual_switching_time,
            'total_energy': total_energy,
            'threshold_times': threshold_times,
            'temperature': metrics['temperature'],
            'stable_index': stable_idx,
            'metrics': metrics,
            'time_real': time_real,
            'voltage_real': voltage_real,
            'true_current_real': true_current_real,
            'pred_current_real': pred_current_real,
            'gap': gap
        }
        
        return result
    
    def _merge_results(self, pos_result, neg_result):
        avg_endurance = min(pos_result['endurance'], neg_result['endurance'])
        total_switching_time = pos_result['actual_switching_time'] + neg_result['actual_switching_time']
        frequency = get_frequency(total_switching_time)
        total_energy = pos_result['total_energy'] + neg_result['total_energy']
        avg_temperature = (pos_result['temperature'] + neg_result['temperature']) / 2
        
        merged_result = {
            'material': pos_result['material'],
            'pos_voltage': pos_result['voltage'],
            'neg_voltage': neg_result['voltage'],
            'avg_endurance': avg_endurance,
            'total_switching_time': total_switching_time,
            'pos_switching_time': pos_result['actual_switching_time'],
            'neg_switching_time': neg_result['actual_switching_time'],
            'frequency': frequency,
            'total_energy': total_energy,
            'avg_temperature': avg_temperature,
            'pos_result': pos_result,
            'neg_result': neg_result
        }
        
        return merged_result

def get_energy_consumption(time_sequence, voltage_sequence, current_sequence, material_idx):
    materials = ['HfO2', 'Al2O3', 'TiO2']
    material = materials[material_idx]
    
    energy_thresholds = {
        'HfO2': [0.1e-12, 1e-12, 5e-12],
        'Al2O3': [0.1e-12, 1e-12, 5e-12],
        'TiO2': [0.1e-12, 1e-12, 5e-12]
    }
    
    if material not in energy_thresholds:
        print(f"Unknown material: {material}, using HfO2 thresholds")
        material = 'HfO2'
        
    threshold_times = {th: None for th in energy_thresholds[material]}

    cumulative_energy = 0
    for i in range(1, len(time_sequence)):
        delta_t = time_sequence[i] - time_sequence[i-1]
        avg_power = voltage_sequence[i] * current_sequence[i]
        energy_step = avg_power * delta_t
        cumulative_energy += energy_step
        for threshold in energy_thresholds[material]:
            if threshold_times[threshold] is None and cumulative_energy >= threshold:
                threshold_times[threshold] = time_sequence[i]
    
    return threshold_times, cumulative_energy

class EnduranceCalculator:
    def __init__(self):
        self.kb = 1.380649e-23        # Boltzmann constant [J/K]
        self.q = 1.60217663e-19       # Elementary charge [C]
        self.T0 = 273 + 25            # Ambient temperature [K]
        self.a0 = 0.25e-9             # Atomic distance [m]
        self.tox = 5e-9               # Oxide thickness [m]
        self.f0 = 1e13                # Attempt frequency [Hz]
        
        self.material_params = {
            'HfO2': {
                'Us': 1.2,            # Same as Eag in Constants
                'Uf': 1.6,            # Failure barrier > Eag
                'Cth': 2.17e-17,
                'Tau_th': 3.5e-10      # Thermal time constant [s]
            },
            'Al2O3': {
                'Us': 1.0,             # Same as Eag in Constants
                'Uf': 1.6,             # Failure barrier > Eag  
                'Cth': 2.12e-17,
                'Tau_th': 3.5e-10
            },
            'TiO2': {
                'Us': 1.7,            # Same as Eag in Constants
                'Uf': 2.3,            # Failure barrier > Eag
                'Cth': 2.26e-17,
                'Tau_th': 3.5e-10
            }
        }
        self.current_material = 'HfO2'
        
    def calculate_temperature(self, current: float, voltage: float, dt: float) -> float:
        params = self.material_params[self.current_material]
        Cth = params['Cth']
        tau_th = params['Tau_th']
        
        if isinstance(dt, torch.Tensor):
            if dt.device.type != 'cpu':
                dt = dt.cpu()
            dt = dt.mean().item()
        else:
            dt = np.mean(dt)
            
        power = abs(voltage * current)
        delta_T = dt * (power/Cth + self.T0/tau_th) / (1 + dt/tau_th) /100
        return self.T0 + delta_T

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

    def calculate_endurance(self, current: float, voltage: float, time_step: float,
                          material: str = 'HfO2') -> Tuple[float, Dict]:
        self.update_material(material)
        
        temperature = self.calculate_temperature(current, voltage, time_step)
        ts = self.calculate_run_time(voltage, temperature)
        tf = self.calculate_failure_time(voltage, temperature)
        endurance = tf / ts

        metrics = {
            'temperature': temperature,
            'run_time': ts,
            'failure_time': tf,
            'endurance_cycles': endurance,
            'current': current,
            'voltage': voltage,
            'run_energy': abs(current * voltage * ts),
            'failure_energy': abs(current * voltage * tf)
        }

        return endurance, metrics

    def update_material(self, material: str):
        if material not in self.material_params:
            raise ValueError(f"Unknown material: {material}")
        self.current_material = material
        
def find_index(array, threshold, is_increasing):
    if is_increasing:
        indices = np.where(array >= threshold)[0]
        return indices[0] if len(indices) > 0 else -1
    else:
        indices = np.where(array <= threshold)[0]
        return indices[0] if len(indices) > 0 else -1
        
def get_stable_index(current_sequence):
    if isinstance(current_sequence, torch.Tensor):
        current_sequence_np = current_sequence.detach().cpu().numpy()
    else:
        current_sequence_np = np.asarray(current_sequence)

    end_current = current_sequence_np[-1]
    start_current = current_sequence_np[0]
    is_increasing = end_current > start_current
    delta_current = end_current - start_current
    threshold_90 = start_current + 0.9 * delta_current
    idx_90 = find_index(current_sequence_np, threshold_90, is_increasing)
    return idx_90

def get_frequency(switching_time):
    frequency = 1.0 / switching_time
    
    max_frequency = 1e9
    if frequency > max_frequency:
        frequency = max_frequency
    
    return frequency

def main():
    parser = argparse.ArgumentParser(description='RRAM model evaluation tool')
    parser.add_argument('--model_path', type=str, default='checkpoints/pinn_sparse.pth', help='model checkpoint path')
    parser.add_argument('--data_path', type=str, default='data/rram_data.mat', help='dataset path')
    parser.add_argument('--output_dir', type=str, default='test_results', help='output directory')
    parser.add_argument('--device', type=str, default='cuda', help='calculation device (cuda/cpu)')
    parser.add_argument('--material', type=str, required=True, choices=['HfO2', 'Al2O3', 'TiO2'], help='material type')
    parser.add_argument('--pos_voltage', type=float, required=True, help='positive voltage value (V)')
    parser.add_argument('--neg_voltage', type=float, required=True, help='negative voltage value (V)')
    
    args = parser.parse_args()
    
    evaluator = RRAMEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )
    try:
        result = evaluator.evaluate(args.material, args.pos_voltage, args.neg_voltage)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"\nError occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
