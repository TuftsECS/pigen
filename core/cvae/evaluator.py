import os
import numpy as np
import torch
from .. import CurrentReadout, SequenceDataset, StatePINN, load_checkpoint, prepare_sequence
from ..training import RAMP_SAMPLES

# Cumulative energies [J] whose crossing times are reported for each pulse.
ENERGY_THRESHOLDS = (0.1e-12, 1e-12, 5e-12)

class DeviceEvaluator:
    """Performance of a device written by one positive and one negative constant-voltage pulse, predicted by a trained PINN.

    A requested pulse voltage is matched to the dataset sequence with the closest final voltage of the same polarity.
    """

    def __init__(self, physics, model_path, data_path, output_dir, device=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.physics = physics
        self.materials = list(self.physics.materials)
        self.material_to_idx = {mat: idx for idx, mat in enumerate(self.materials)}
        
        self.checkpoint, self.pinn_model, self.mlp_model = self.load_model(model_path)
        self.dataset = self.load_data(data_path, self.checkpoint['scalers'])
        self.scalers = self.checkpoint['scalers']

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        checkpoint = load_checkpoint(model_path, self.device)
        
        state = checkpoint['pinn_model_state_dict']
        embedding_size = state['material_embedding.weight'].shape[1]
        hidden_size = state['gru.weight_hh_l0'].shape[1]
        
        pinn_model = StatePINN(len(self.materials), hidden_size=hidden_size, embedding_size=embedding_size).to(self.device)
        mlp_model = CurrentReadout(len(self.materials), hidden_size=hidden_size, embedding_size=embedding_size).to(self.device)
        
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
        dataset = SequenceDataset(
            data_path=data_path,
            physics=self.physics,
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
        s = prepare_sequence(sequence, self.scalers, self.device)
        current_scale = torch.tensor(self.scalers['current'].scale_[0], device=self.device)
        current_mean = torch.tensor(self.scalers['current'].mean_[0], device=self.device)
        
        with torch.no_grad():
            state = self.pinn_model(s['time'], s['dt_real'], s['voltage'], s['material_idx'])
            pred_current = self.mlp_model(state, s['voltage'], s['initial_current'], s['material_idx'])
            pred_current_real = pred_current * current_scale + current_mean

        return (s['time_real_full'][RAMP_SAMPLES:], s['dt_real'], s['voltage_real_full'][RAMP_SAMPLES:],
                s['current_real_full'][RAMP_SAMPLES:], pred_current_real, state)
    
    def evaluate(self, material, pos_voltage, neg_voltage):
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}. Supported materials: {self.materials}")
            
        material_idx = self.material_to_idx[material]
        
        pos_matched_sequence, pos_matched_idx, pos_actual_voltage = self._find_closest_sequence(material_idx, pos_voltage, "positive")
        neg_matched_sequence, neg_matched_idx, neg_actual_voltage = self._find_closest_sequence(material_idx, neg_voltage, "negative")
        
        pos_result = self._evaluate_single_sequence(pos_matched_sequence, pos_matched_idx, material, "positive")
        neg_result = self._evaluate_single_sequence(neg_matched_sequence, neg_matched_idx, material, "negative")
        
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
        time_real, dt_real, voltage_real, true_current_real, pred_current_real, state = self.predict_sequence(sequence)
        stable_idx = get_stable_index(pred_current_real)
        
        if stable_idx < len(time_real):
            actual_switching_time = time_real[stable_idx] - time_real[0]
            actual_switching_time = actual_switching_time.item()
        else:
            actual_switching_time = (time_real[-1] - time_real[0]).item()
        
        endurance, metrics = self.physics.endurance(pred_current_real, voltage_real, dt_real, material)
        threshold_times, total_energy = get_energy_consumption(time_real, voltage_real, pred_current_real)
        
        result = {
            'sequence_id': sequence_idx,
            'material': material,
            'voltage': voltage_real.mean().item(),
            'operation_type': operation_type,
            'endurance': endurance,
            'actual_switching_time': actual_switching_time,
            'total_energy': total_energy,
            'threshold_times': threshold_times,
            'stable_index': stable_idx,
            'metrics': metrics,
            'time_real': time_real,
            'voltage_real': voltage_real,
            'true_current_real': true_current_real,
            'pred_current_real': pred_current_real,
            'state': state
        }
        
        return result
    
    def _merge_results(self, pos_result, neg_result):
        avg_endurance = min(pos_result['endurance'], neg_result['endurance'])
        total_switching_time = pos_result['actual_switching_time'] + neg_result['actual_switching_time']
        frequency = get_frequency(total_switching_time)
        total_energy = pos_result['total_energy'] + neg_result['total_energy']
        
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
            'pos_result': pos_result,
            'neg_result': neg_result
        }
        
        return merged_result

def get_energy_consumption(time_sequence, voltage_sequence, current_sequence):
    threshold_times = {th: None for th in ENERGY_THRESHOLDS}

    cumulative_energy = 0
    for i in range(1, len(time_sequence)):
        delta_t = time_sequence[i] - time_sequence[i-1]
        avg_power = voltage_sequence[i] * current_sequence[i]
        energy_step = avg_power * delta_t
        cumulative_energy += energy_step
        for threshold in ENERGY_THRESHOLDS:
            if threshold_times[threshold] is None and cumulative_energy >= threshold:
                threshold_times[threshold] = time_sequence[i]
    
    return threshold_times, cumulative_energy

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
