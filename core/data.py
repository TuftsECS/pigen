import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
from collections import defaultdict
import logging

def format_voltage(voltage: Union[float, np.ndarray, torch.Tensor], precision: int = 2) -> float:
    if isinstance(voltage, (np.ndarray, torch.Tensor)):
        voltage = float(voltage)
    return round(voltage, precision)

class SequenceDataset(Dataset):
    """Voltage/current switching sequences of one device, read from a .mat file and split by `physics` conventions."""

    def __init__(
        self, 
        data_path: str,
        physics,
        fit_scaler: bool = False,
        is_train: bool = True,
        seed: Optional[int] = None,
        logger = None,
        split_ratio: float = 0.8,
        use_full_dataset: bool = True,
        voltage_stride: float = 0.2
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.scalers = None
        
        data = loadmat(data_path)
        num_sequences = len([k for k in data.keys() if k.startswith('time_')])
        self.num_total = num_sequences
        # Training voltages per material of a stride split (identical for the training and validation sets).
        self.train_voltages = {}
        
        self.sequences = []
        sequence_voltage_map = {}
        
        self.material_to_idx = {material: idx for idx, material in enumerate(physics.materials)}
        
        for i in range(num_sequences):
            time_orig = data[f'time_{i}'].flatten()
            initial_time = np.array([physics.initial_time])
            time_with_initial = np.concatenate([initial_time, time_orig])
            dt = time_with_initial[1:] - time_with_initial[:-1]
            
            voltage_data = data[f'v_{i}'].flatten()
            formatted_voltage = np.array([format_voltage(v) for v in voltage_data], dtype=np.float32)
            
            sequence = {
                'time': torch.FloatTensor(time_orig),
                'dt': torch.FloatTensor(dt),
                'voltage': torch.tensor(formatted_voltage, dtype=torch.float32),
                'current': torch.FloatTensor(data[f'current_{i}'].flatten()),
                'material': data[f'material_{i}'][0],
                'material_idx': torch.tensor(self.material_to_idx[data[f'material_{i}'][0]])
            }
            voltage_val = format_voltage(sequence['voltage'][-1])
            sequence_voltage_map[i] = voltage_val
            self.sequences.append(sequence)
        
        material_to_voltages_map = defaultdict(set)
        material_to_indices_map = defaultdict(list)
        for i, voltage in sequence_voltage_map.items():
            if i < len(self.sequences):
                material = self.sequences[i]['material']
                material_to_voltages_map[material].add(voltage)
                material_to_indices_map[material].append(i)
            else:
                self.logger.warning(f"Index {i} out of bounds when building material maps.")
        for material in list(material_to_voltages_map.keys()): 
            sorted_voltages = sorted(list(material_to_voltages_map[material]))
            if sorted_voltages:
                material_to_voltages_map[material] = sorted_voltages
            else:
                 del material_to_voltages_map[material] 
                 if material in material_to_indices_map:
                     del material_to_indices_map[material]
                 self.logger.info(f"Removed material {material} from maps: No associated voltages found.")

        self.logger.debug(f"Pre-calculated available voltages per material: {material_to_voltages_map}")
        self.logger.debug(f"Pre-calculated indices per material: {material_to_indices_map}")

        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        if use_full_dataset:
            all_indices = np.arange(len(self.sequences))
            rng.shuffle(all_indices)
            train_size = int(len(self.sequences) * split_ratio)
            if is_train:
                final_indices = all_indices[:train_size]
            else:
                final_indices = all_indices[train_size:]
        else:
            target_step = voltage_stride
            
            material_boundaries = physics.split_voltages
            final_selected_indices = set()
            max_gap_factor = 1.25

            for material, material_voltages in material_to_voltages_map.items():
                material_indices = material_to_indices_map.get(material, [])
                if not material_voltages or not material_indices:
                    self.logger.debug(f"Skipping {material}: No voltages or indices found.")
                    continue 

                self.logger.debug(f"Processing {material} (Indices: {len(material_indices)}, Voltages: {len(material_voltages)})" )

                boundaries = material_boundaries.get(material, [])
                formatted_boundaries_for_material = {format_voltage(b) for b in boundaries}
                min_v, max_v = material_voltages[0], material_voltages[-1]
                endpoints = {min_v, max_v}
                effective_step = abs(target_step) if target_step != 0 else 0.2 
                ideal_steps = np.arange(min_v, max_v + 0.1 * effective_step, effective_step)
                
                stepped_voltages_for_material = set()
                if len(ideal_steps) > 0:
                    for ideal_v in ideal_steps:
                        closest_v = min(material_voltages, key=lambda x: abs(x - ideal_v))
                        stepped_voltages_for_material.add(closest_v)
                
                initial_targets = stepped_voltages_for_material.union(formatted_boundaries_for_material).union(endpoints)
                filtered_initial_targets = {v for v in initial_targets if v in material_voltages}
                self.logger.debug(f"  a. Initial targets for {material}: {sorted(list(filtered_initial_targets))}")

                pruned_targets = set(filtered_initial_targets)
                if len(filtered_initial_targets) >= 2:
                    mandatory_points = {v for v in formatted_boundaries_for_material.union(endpoints) if v in material_voltages}
                    points_to_remove = set()
                    current_selection_sorted = sorted(list(filtered_initial_targets))
                    min_gap_factor = 0.75
                    min_allowed_gap = target_step * min_gap_factor
                    self.logger.debug(f"  b. Pruning {material} (min gap: {min_allowed_gap:.3f}). Mandatory: {mandatory_points}")

                    i = 0
                    temp_sorted_list = list(current_selection_sorted)
                    while i < len(temp_sorted_list) - 1:
                        v1 = temp_sorted_list[i]
                        v2 = temp_sorted_list[i+1]
                        gap = abs(v2 - v1)

                        if gap < min_allowed_gap:
                            v1_is_mandatory = v1 in mandatory_points
                            v2_is_mandatory = v2 in mandatory_points
                            point_removed = None

                            if v1_is_mandatory and not v2_is_mandatory:
                                points_to_remove.add(v2)
                                point_removed = v2
                            elif not v1_is_mandatory and v2_is_mandatory:
                                points_to_remove.add(v1)
                                point_removed = v1
                            elif not v1_is_mandatory and not v2_is_mandatory:
                                midpoint = v1 + gap / 2
                                if abs(v1 - midpoint) < abs(v2 - midpoint):
                                     points_to_remove.add(v1)
                                     point_removed = v1
                                else:
                                     points_to_remove.add(v2)
                                     point_removed = v2

                            if point_removed is not None:
                                self.logger.debug(f"    Pruning ({material}): Marked {point_removed:.2f} for removal (gap {gap:.3f} < {min_allowed_gap:.3f} between {v1:.2f} and {v2:.2f}).")
                                if point_removed == v1:
                                    temp_sorted_list.pop(i)
                                    continue 
                                else:
                                     temp_sorted_list.pop(i+1)
                                     continue 
                        i += 1

                    pruned_targets = filtered_initial_targets - points_to_remove
                    self.logger.debug(f"  b. Pruned targets for {material}: {sorted(list(pruned_targets))}")
                else:
                     self.logger.debug(f"  b. Skipping pruning for {material}: Not enough points.")
                     pruned_targets = filtered_initial_targets

                final_material_target_voltages = set(pruned_targets)
                if len(pruned_targets) >= 2:
                    current_selection_sorted = sorted(list(pruned_targets))
                    points_added_count = 0
                    max_gap_threshold = target_step * max_gap_factor
                    self.logger.debug(f"  c. Filling gaps for {material} > {max_gap_threshold:.3f} V (on pruned set).")
                    
                    processed_indices = set()
                    needs_recheck = True
                    while needs_recheck:
                        needs_recheck = False
                        current_selection_sorted = sorted(list(final_material_target_voltages))
                        if len(current_selection_sorted) < 2: break
                        
                        for i in range(len(current_selection_sorted) - 1):
                            if i in processed_indices: continue
                            
                            v1, v2 = current_selection_sorted[i], current_selection_sorted[i+1]
                            gap = v2 - v1
                            if gap > max_gap_threshold:
                                midpoint = v1 + gap / 2
                                closest_available_to_midpoint = min(material_voltages, key=lambda x: abs(x - midpoint))
                                
                                if (closest_available_to_midpoint != v1 and 
                                    closest_available_to_midpoint != v2 and
                                    closest_available_to_midpoint not in final_material_target_voltages):
                                    
                                    final_material_target_voltages.add(closest_available_to_midpoint)
                                    points_added_count += 1
                                    self.logger.debug(f"    Gap Fill ({material}): Added {closest_available_to_midpoint:.2f} between {v1:.2f} and {v2:.2f}")
                                    needs_recheck = True
                                    processed_indices.clear()
                                    break
                            processed_indices.add(i)
                            
                    if points_added_count > 0:
                         self.logger.debug(f"  c. Added {points_added_count} points via gap filling for {material}.")
                else:
                    self.logger.debug(f"  c. Skipping gap filling for {material}: Not enough points after pruning.")

                final_voltages_sorted = sorted(list(final_material_target_voltages))
                self.train_voltages[material] = final_voltages_sorted
                self.logger.debug(f"  Final target voltages for {material}: {final_voltages_sorted}")

                for idx in material_indices:
                    v = sequence_voltage_map.get(idx)
                    if v is not None and v in final_material_target_voltages:
                        final_selected_indices.add(idx)

            final_selected_indices_array = np.array(list(final_selected_indices), dtype=int)
            all_indices = np.arange(len(self.sequences))

            if is_train:
                final_indices = final_selected_indices_array
            else:
                final_indices = np.setdiff1d(all_indices, final_selected_indices_array, assume_unique=True)
                
            rng.shuffle(final_indices)
            
        self.sequences = [self.sequences[i] for i in final_indices]
        
        if fit_scaler:
            self.scalers = self._fit_scalers()
            self._apply_scaling()
            
    def _fit_scalers(self):
        scalers = {}
        
        all_time = torch.cat([seq['time'] for seq in self.sequences])
        all_dt = torch.cat([seq['dt'] for seq in self.sequences])
        all_voltage = torch.cat([seq['voltage'] for seq in self.sequences])
        all_current = torch.cat([seq['current'] for seq in self.sequences])
        
        scalers['time'] = StandardScaler().fit(all_time.reshape(-1, 1))
        scalers['dt'] = StandardScaler().fit(all_dt.reshape(-1, 1))
        scalers['voltage'] = StandardScaler().fit(all_voltage.reshape(-1, 1))
        scalers['current'] = StandardScaler().fit(all_current.reshape(-1, 1))
        
        return scalers
        
    def _apply_scaling(self):
        if self.scalers is None:
            raise ValueError("Scalers not fitted yet!")
            
        for seq in self.sequences:
            seq['time'] = torch.FloatTensor(
                self.scalers['time'].transform(seq['time'].reshape(-1, 1))
            ).flatten()
            seq['dt'] = torch.FloatTensor(
                self.scalers['dt'].transform(seq['dt'].reshape(-1, 1))
            ).flatten()
            seq['voltage'] = torch.FloatTensor(
                self.scalers['voltage'].transform(seq['voltage'].reshape(-1, 1))
            ).flatten()
            seq['current'] = torch.FloatTensor(
                self.scalers['current'].transform(seq['current'].reshape(-1, 1))
            ).flatten()
    
    def get_scalers(self):
        if self.scalers is None:
            raise ValueError("Dataset was not initialized with fit_scaler=True")
        return self.scalers
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

    def set_scalers(self, scalers):
        self.scalers = scalers
        self._apply_scaling()
        
    def get_material_distribution(self):
        material_counts = {}
        for sequence in self.sequences:
            material = sequence['material']
            material_counts[material] = material_counts.get(material, 0) + 1
        return material_counts
