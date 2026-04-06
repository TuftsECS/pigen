import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import tqdm
import math

from .metric_eval import RRAMEvaluator, EnduranceCalculator

class RRAM_CVAE(nn.Module):
    def __init__(self, condition_dim=3, latent_dim=8, hidden_dim=128):
        super(RRAM_CVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(6 + condition_dim, hidden_dim),  # 6: material(3) + pos_voltage + neg_voltage + pulse_width
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.material_out = nn.Linear(hidden_dim, 3)
        self.pos_voltage_out = nn.Linear(hidden_dim, 1)
        self.neg_voltage_out = nn.Linear(hidden_dim, 1)
        self.pulse_width_out = nn.Linear(hidden_dim, 1)
    
    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h = self.decoder(z_c)
        material_logits = self.material_out(h)
        pos_voltage = self.pos_voltage_out(h)
        neg_voltage = self.neg_voltage_out(h)
        pulse_width = self.pulse_width_out(h)
        return material_logits, pos_voltage, neg_voltage, pulse_width
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        material_logits, pos_voltage, neg_voltage, pulse_width = self.decode(z, c)
        return material_logits, pos_voltage, neg_voltage, pulse_width, mu, logvar

def cvae_loss_function(material_pred, voltage_pos_pred, voltage_neg_pred, pulse_width_pred,
                      material_true, voltage_pos_true, voltage_neg_true, pulse_width_true,
                      mu, logvar, kl_weight=0.01):
    material_loss = F.cross_entropy(material_pred, material_true)
    voltage_pos_loss = F.mse_loss(voltage_pos_pred, voltage_pos_true)
    voltage_neg_loss = F.mse_loss(voltage_neg_pred, voltage_neg_true)
    pulse_width_loss = F.mse_loss(pulse_width_pred, pulse_width_true)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = material_loss + voltage_pos_loss + voltage_neg_loss + pulse_width_loss + kl_weight * kl_loss
    
    return total_loss, material_loss, voltage_pos_loss, voltage_neg_loss, pulse_width_loss, kl_loss

class RRAMParameterDataset(Dataset):
    def __init__(self, evaluator, num_samples=500, create_new=False, save_path='rram_cvae_dataset.pt',
                min_pulse_width=10e-9):
        self.evaluator = evaluator
        self.materials = evaluator.materials
        self.material_to_idx = evaluator.material_to_idx
        self.device = evaluator.device
        self.min_pulse_width = min_pulse_width
        
        self.voltage_ranges = {
            'HfO2': {'pos': (1.5, 2.0), 'neg': (-2.0, -1.5)},
            'TiO2': {'pos': (1.65, 2.0), 'neg': (-2.0, -1.65)},
            'Al2O3': {'pos': (1.38, 2.0), 'neg': (-2.0, -1.38)}
        }
        
        if os.path.exists(save_path) and not create_new:
            print(f"Loading existing dataset from {save_path}")
            saved_data = torch.load(save_path)
            self.samples = saved_data['samples']
            self.performance_scaler = saved_data['performance_scaler']
            self.voltage_scaler = saved_data['voltage_scaler']
            self.pulse_width_scaler = saved_data['pulse_width_scaler']
            print(f"Loaded {len(self.samples)} samples")
        else:
            print(f"Generating new dataset with {num_samples} samples...")
            self.samples = self.generate_samples(num_samples)
            self._convert_tensors_to_cpu()
            
            self.performance_scaler = StandardScaler()
            self.voltage_scaler = StandardScaler()
            self.pulse_width_scaler = StandardScaler()
            all_performances = np.array([s['performance'] for s in self.samples])
            self.performance_scaler.fit(all_performances)
            
            all_voltages = np.array([[s['pos_voltage'], s['neg_voltage']] for s in self.samples])
            self.voltage_scaler.fit(all_voltages)
            all_pulse_widths = np.array([[s['pulse_width']] for s in self.samples])
            self.pulse_width_scaler.fit(all_pulse_widths)
            
            save_data = {
                'samples': self.samples,
                'performance_scaler': self.performance_scaler,
                'voltage_scaler': self.voltage_scaler,
                'pulse_width_scaler': self.pulse_width_scaler
            }
            torch.save(save_data, save_path)
            print(f"Dataset saved to {save_path}")
    
    def _convert_tensors_to_cpu(self):
        for i, sample in enumerate(self.samples):
            if isinstance(sample['performance'], torch.Tensor):
                sample['performance'] = sample['performance'].cpu().numpy().tolist()
            
            if isinstance(sample['pos_voltage'], torch.Tensor):
                sample['pos_voltage'] = float(sample['pos_voltage'].cpu().item())
            if isinstance(sample['neg_voltage'], torch.Tensor):
                sample['neg_voltage'] = float(sample['neg_voltage'].cpu().item())
            if 'pulse_width' in sample and isinstance(sample['pulse_width'], torch.Tensor):
                sample['pulse_width'] = float(sample['pulse_width'].cpu().item())
                
            self.samples[i] = sample
    
    def generate_samples(self, num_samples):
        samples = []
        
        for material_idx, material in enumerate(self.materials):
            material_samples = num_samples // len(self.materials)
            
            print(f"Generating {material_samples} samples for {material}...")
            for i in tqdm.tqdm(range(material_samples)):
                voltage_range = self.voltage_ranges[material]
                pos_min, pos_max = voltage_range['pos']
                neg_min, neg_max = voltage_range['neg']
                
                pos_voltage = np.random.uniform(pos_min, pos_max)
                neg_voltage = np.random.uniform(neg_min, neg_max)
                
                try:
                    result = self.evaluator.evaluate(material, pos_voltage, neg_voltage)
                    avg_endurance = result['avg_endurance']
                    frequency = result['frequency']
                    total_energy = result['total_energy']
                    total_switching_time = result['total_switching_time']
                    pos_switching_time = result['pos_switching_time']
                    neg_switching_time = result['neg_switching_time']
                    if isinstance(avg_endurance, torch.Tensor):
                        avg_endurance = float(avg_endurance.cpu().item())
                    if isinstance(frequency, torch.Tensor):
                        frequency = float(frequency.cpu().item())
                    if isinstance(total_energy, torch.Tensor):
                        total_energy = float(total_energy.cpu().item())
                    if isinstance(total_switching_time, torch.Tensor):
                        total_switching_time = float(total_switching_time.cpu().item())
                    if isinstance(pos_switching_time, torch.Tensor):
                        pos_switching_time = float(pos_switching_time.cpu().item())
                    if isinstance(neg_switching_time, torch.Tensor):
                        neg_switching_time = float(neg_switching_time.cpu().item())
                    
                    pulse_width = max(self.min_pulse_width, total_switching_time * 1.1)
                    samples.append({
                        'material': material,
                        'material_idx': material_idx,
                        'pos_voltage': pos_voltage,
                        'neg_voltage': neg_voltage,
                        'pulse_width': pulse_width,
                        'performance': [
                            avg_endurance,
                            total_switching_time,
                            total_energy
                        ]
                    })
                except Exception as e:
                    print(f"Error evaluating {material} at {pos_voltage}V/{neg_voltage}V: {e}")
                    continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        material_idx = torch.tensor(sample['material_idx'], dtype=torch.long)
        material_onehot = F.one_hot(material_idx, num_classes=3).float()
        
        pos_voltage = torch.tensor([sample['pos_voltage']], dtype=torch.float32)
        neg_voltage = torch.tensor([sample['neg_voltage']], dtype=torch.float32)
        pulse_width = torch.tensor([sample['pulse_width']], dtype=torch.float32)
        performance = np.array([sample['performance']])
        performance_normalized = torch.tensor(
            self.performance_scaler.transform(performance),
            dtype=torch.float32
        ).squeeze(0)
        
        voltages = np.array([[sample['pos_voltage'], sample['neg_voltage']]])
        voltages_normalized = self.voltage_scaler.transform(voltages)[0]
        pos_voltage_normalized = torch.tensor([voltages_normalized[0]], dtype=torch.float32)
        neg_voltage_normalized = torch.tensor([voltages_normalized[1]], dtype=torch.float32)
        pulse_width_array = np.array([[sample['pulse_width']]])
        pulse_width_normalized = self.pulse_width_scaler.transform(pulse_width_array)[0]
        pulse_width_normalized = torch.tensor([pulse_width_normalized[0]], dtype=torch.float32)
        
        return {
            'material_idx': material_idx,
            'material_onehot': material_onehot,
            'pos_voltage': pos_voltage,
            'neg_voltage': neg_voltage,
            'pulse_width': pulse_width,
            'pos_voltage_normalized': pos_voltage_normalized,
            'neg_voltage_normalized': neg_voltage_normalized,
            'pulse_width_normalized': pulse_width_normalized,
            'performance': performance_normalized
        }

def train_cvae(evaluator, num_epochs=50, batch_size=32, latent_dim=8, hidden_dim=128,
               dataset_path='rram_cvae_dataset.pt', model_save_path='cvae_model.pth',
               create_new_dataset=False, min_pulse_width=10e-9,
               early_stopping_patience=20, early_stopping_delta=1e-4):
    dataset = RRAMParameterDataset(
        evaluator, 
        num_samples=500, 
        create_new=create_new_dataset, 
        save_path=dataset_path,
        min_pulse_width=min_pulse_width
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = evaluator.device
    condition_dim = 3  # endurance, switching time, energy
    model = RRAM_CVAE(condition_dim, latent_dim, hidden_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0

    print(f"Starting CVAE training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        material_loss_total = 0
        voltage_pos_loss_total = 0
        voltage_neg_loss_total = 0
        pulse_width_loss_total = 0
        kl_loss_total = 0
        
        for batch in dataloader:
            material_idx = batch['material_idx'].to(device)
            material_onehot = batch['material_onehot'].to(device)
            pos_voltage = batch['pos_voltage_normalized'].to(device)
            neg_voltage = batch['neg_voltage_normalized'].to(device)
            pulse_width = batch['pulse_width_normalized'].to(device)
            
            x = torch.cat([material_onehot, pos_voltage, neg_voltage, pulse_width], dim=1)
            performance = batch['performance'].to(device)
            
            material_pred, pos_voltage_pred, neg_voltage_pred, pulse_width_pred, mu, logvar = model(x, performance)
            
            loss, mat_loss, pos_loss, neg_loss, pulse_loss, kl = cvae_loss_function(
                material_pred, pos_voltage_pred, neg_voltage_pred, pulse_width_pred,
                material_idx, pos_voltage, neg_voltage, pulse_width,
                mu, logvar
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            material_loss_total += mat_loss.item()
            voltage_pos_loss_total += pos_loss.item()
            voltage_neg_loss_total += neg_loss.item()
            pulse_width_loss_total += pulse_loss.item()
            kl_loss_total += kl.item()
        
        avg_loss = train_loss / len(dataloader)
        avg_material_loss = material_loss_total / len(dataloader)
        avg_voltage_pos_loss = voltage_pos_loss_total / len(dataloader)
        avg_voltage_neg_loss = voltage_neg_loss_total / len(dataloader)
        avg_pulse_width_loss = pulse_width_loss_total / len(dataloader)
        avg_kl_loss = kl_loss_total / len(dataloader)
        
        train_losses.append(avg_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, '
              f'Material: {avg_material_loss:.4f}, Pos V: {avg_voltage_pos_loss:.4f}, '
              f'Neg V: {avg_voltage_neg_loss:.4f}, Pulse: {avg_pulse_width_loss:.4f}, '
              f'KL: {avg_kl_loss:.4f}')

        if avg_loss < best_loss - early_stopping_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1} as loss did not improve.")
            break

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'condition_dim': condition_dim
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    loss_plot_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.'
    plt.savefig(os.path.join(loss_plot_dir, 'cvae_training_loss.png'))
    plt.close()
    
    return model, dataset

class RRAMParameterRecommender:
    def __init__(self, model_path, evaluator, dataset_path='rram_cvae_dataset.pt', min_pulse_width=10e-9, similarity_threshold=0.05):
        self.evaluator = evaluator
        self.device = evaluator.device
        self.materials = evaluator.materials
        self.min_pulse_width = min_pulse_width
        self.similarity_threshold = similarity_threshold
        
        self.voltage_ranges = {
            'HfO2': {'pos': (1.5, 2.0), 'neg': (-2.0, -1.5)},
            'TiO2': {'pos': (1.65, 2.0), 'neg': (-2.0, -1.65)},
            'Al2O3': {'pos': (1.38, 2.0), 'neg': (-2.0, -1.38)}
        }
        
        print(f"Loading CVAE model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.latent_dim = checkpoint.get('latent_dim', 8)
            self.hidden_dim = checkpoint.get('hidden_dim', 128)
            self.condition_dim = checkpoint.get('condition_dim', 3)
            
            self.model = RRAM_CVAE(
                self.condition_dim, self.latent_dim, self.hidden_dim
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.latent_dim = 8
            self.hidden_dim = 128
            self.condition_dim = 3
            
            self.model = RRAM_CVAE(
                self.condition_dim, self.latent_dim, self.hidden_dim
            ).to(self.device)
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            saved_data = torch.load(dataset_path)
            self.performance_scaler = saved_data['performance_scaler']
            self.voltage_scaler = saved_data['voltage_scaler']
            self.pulse_width_scaler = saved_data['pulse_width_scaler']
        else:
            print(f"Dataset not found at {dataset_path}. Creating a small dataset to get scalers...")
            dataset = RRAMParameterDataset(
                evaluator, 
                num_samples=30, 
                create_new=True, 
                save_path=dataset_path,
                min_pulse_width=min_pulse_width
            )
            self.performance_scaler = dataset.performance_scaler
            self.voltage_scaler = dataset.voltage_scaler
            self.pulse_width_scaler = dataset.pulse_width_scaler
    
    def _are_recommendations_similar(self, rec1, rec2, voltage_threshold=None, performance_threshold=None, pulse_width_threshold=None):
        """Determine if two recommendations are similar."""
        if voltage_threshold is None:
            voltage_threshold = self.similarity_threshold
        if performance_threshold is None:
            performance_threshold = self.similarity_threshold * 2
        if pulse_width_threshold is None:
            pulse_width_threshold = self.similarity_threshold * 2
            
        if rec1['material'] != rec2['material']:
            return False
        pos_voltage_diff = abs(rec1['pos_voltage'] - rec2['pos_voltage']) / max(abs(rec1['pos_voltage']), 1e-10)
        neg_voltage_diff = abs(rec1['neg_voltage'] - rec2['neg_voltage']) / max(abs(rec1['neg_voltage']), 1e-10)
        
        avg_pw1 = (rec1.get('set_pulse_width', 0) + rec1.get('reset_pulse_width', 0)) / 2.0
        avg_pw2 = (rec2.get('set_pulse_width', 0) + rec2.get('reset_pulse_width', 0)) / 2.0
        denominator = max(avg_pw1, 1e-10)
        pulse_width_diff = abs(avg_pw1 - avg_pw2) / denominator if denominator > 0 else 0.0
        
        time_diff = abs(rec1['predicted_performance']['total_switching_time'] - rec2['predicted_performance']['total_switching_time']) / max(rec1['predicted_performance']['total_switching_time'], 1e-10)
        energy_diff = abs(rec1['predicted_performance']['energy'] - rec2['predicted_performance']['energy']) / max(rec1['predicted_performance']['energy'], 1e-10)
        voltage_similar = (pos_voltage_diff < voltage_threshold and neg_voltage_diff < voltage_threshold)
        pulse_width_similar = (pulse_width_diff < pulse_width_threshold)
        performance_similar = (time_diff < performance_threshold and energy_diff < performance_threshold)
        
        return voltage_similar and (pulse_width_similar or performance_similar)
    
    def _generate_diverse_candidates(self, material, target_performance_normalized, num_voltages=4):
        """Generate diverse voltage parameter candidates for given material."""
        candidates = []
        voltage_range = self.voltage_ranges[material]
        pos_min, pos_max = voltage_range['pos']
        neg_min, neg_max = voltage_range['neg']
        
        pos_voltages = np.linspace(pos_min, pos_max, num_voltages)
        neg_voltages = np.linspace(neg_min, neg_max, num_voltages)
        pos_voltages = np.append(pos_voltages, 
                               np.random.uniform(pos_min + 0.6*(pos_max-pos_min), 
                                               pos_max, 
                                               num_voltages//2))
        neg_voltages = np.append(neg_voltages, 
                               np.random.uniform(neg_min, 
                                               neg_min + 0.4*(neg_max-neg_min), 
                                               num_voltages//2))
        
        pos_voltages = np.unique(pos_voltages)
        neg_voltages = np.unique(neg_voltages)
        
        for pos_v in pos_voltages:
            for neg_v in neg_voltages:
                z = torch.randn(1, self.latent_dim).to(self.device)
                with torch.no_grad():
                    material_logits, pos_voltage_norm, neg_voltage_norm, pulse_width_norm = self.model.decode(z, target_performance_normalized)
                
                pulse_width_array = np.array([[pulse_width_norm.cpu().item()]])
                pulse_width_denorm = self.pulse_width_scaler.inverse_transform(pulse_width_array)[0][0]
                candidates.append({
                    'pos_voltage': float(pos_v),
                    'neg_voltage': float(neg_v),
                    'pulse_width': float(pulse_width_denorm)
                })
        
        return candidates
    
    def recommend_parameters(self, target_endurance, target_switching_time, target_energy, num_recommendations=5, num_samples=20, energy_penalty_factor=3.0, max_energy_error_ratio=2.0):
        """Recommend parameters based on target performance metrics."""
        print("\nGenerating parameter recommendations...")
        
        num_recommendations = 4
        num_samples = max(num_samples, num_recommendations * 15)  # increase more sample points to improve diversity
        
        print(f"Target switching time: {target_switching_time*1e9:.2f} ns")
        
        target_performance = np.array([[target_endurance, target_switching_time, target_energy]])
        target_performance_normalized = torch.tensor(
            self.performance_scaler.transform(target_performance),
            dtype=torch.float32
        ).to(self.device)
        
        # Phase 1: CVAE-based candidate generation
        candidates = []
        for _ in range(num_samples // 2):
            for material_idx, material in enumerate(self.materials):
                z = torch.randn(1, self.latent_dim).to(self.device)
                with torch.no_grad():
                    material_logits, pos_voltage_norm, neg_voltage_norm, pulse_width_norm = self.model.decode(z, target_performance_normalized)
                
                voltages = np.array([[pos_voltage_norm.cpu().item(), neg_voltage_norm.cpu().item()]])
                voltages_denorm = self.voltage_scaler.inverse_transform(voltages)[0]
                pos_voltage_denorm = voltages_denorm[0]
                neg_voltage_denorm = voltages_denorm[1]
                
                pulse_width_array = np.array([[pulse_width_norm.cpu().item()]])
                pulse_width_denorm = self.pulse_width_scaler.inverse_transform(pulse_width_array)[0][0]
                
                voltage_range = self.voltage_ranges[material]
                pos_min, pos_max = voltage_range['pos']
                neg_min, neg_max = voltage_range['neg']
                
                pos_voltage_denorm = np.clip(pos_voltage_denorm, pos_min, pos_max)
                neg_voltage_denorm = np.clip(neg_voltage_denorm, neg_min, neg_max)
                
                try:
                    result = self.evaluator.evaluate(material, pos_voltage_denorm, neg_voltage_denorm)
                    
                    avg_endurance = result['avg_endurance']
                    total_energy = result['total_energy']
                    actual_total_switching_time = result['total_switching_time']
                    pos_switching_time = result.get('pos_switching_time')
                    neg_switching_time = result.get('neg_switching_time')
                    
                    if isinstance(avg_endurance, torch.Tensor):
                        avg_endurance = float(avg_endurance.cpu().item())
                    if isinstance(total_energy, torch.Tensor):
                        total_energy = float(total_energy.cpu().item())
                    if isinstance(actual_total_switching_time, torch.Tensor):
                        actual_total_switching_time = float(actual_total_switching_time.cpu().item())
                    if isinstance(pos_switching_time, torch.Tensor):
                        pos_switching_time = float(pos_switching_time.cpu().item())
                    if isinstance(neg_switching_time, torch.Tensor):
                        neg_switching_time = float(neg_switching_time.cpu().item())
                    
                    set_pulse_width = max(self.min_pulse_width, pos_switching_time * 1.1)
                    reset_pulse_width = max(self.min_pulse_width, neg_switching_time * 1.1)
                    
                    candidates.append({
                        'material': material,
                        'pos_voltage': pos_voltage_denorm,
                        'neg_voltage': neg_voltage_denorm,
                        'set_pulse_width': set_pulse_width,                         'reset_pulse_width': reset_pulse_width,                         'predicted_performance': {
                            'endurance': avg_endurance,
                            'total_switching_time': actual_total_switching_time,
                            'pos_switching_time': pos_switching_time,                             'neg_switching_time': neg_switching_time,                             'energy': total_energy
                        },
                        'target_performance': {
                            'endurance': target_endurance,
                            'total_switching_time': target_switching_time,
                            'energy': target_energy
                        }
                    })
                    
                except Exception as e:
                    print(f"Error evaluating recommendation for {material} at {pos_voltage_denorm:.2f}V/{neg_voltage_denorm:.2f}V: {e}")
                    continue
        
        # Phase 2: Grid-based diverse candidate generation
        for material_idx, material in enumerate(self.materials):
            diverse_candidates = self._generate_diverse_candidates(material, target_performance_normalized, num_voltages=8)
            
            for candidate in diverse_candidates:
                pos_voltage = candidate['pos_voltage']
                neg_voltage = candidate['neg_voltage']
                pulse_width = candidate['pulse_width']
                
                try:
                    result = self.evaluator.evaluate(material, pos_voltage, neg_voltage)
                    
                    avg_endurance = result['avg_endurance']
                    total_energy = result['total_energy']
                    actual_total_switching_time = result['total_switching_time']
                    pos_switching_time = result['pos_switching_time']
                    neg_switching_time = result['neg_switching_time']
                    if isinstance(avg_endurance, torch.Tensor):
                        avg_endurance = float(avg_endurance.cpu().item())
                    if isinstance(total_energy, torch.Tensor):
                        total_energy = float(total_energy.cpu().item())
                    if isinstance(actual_total_switching_time, torch.Tensor):
                        actual_total_switching_time = float(actual_total_switching_time.cpu().item())
                    if isinstance(pos_switching_time, torch.Tensor):
                        pos_switching_time = float(pos_switching_time.cpu().item())
                    if isinstance(neg_switching_time, torch.Tensor):
                        neg_switching_time = float(neg_switching_time.cpu().item())
                    
                    set_pulse_width = max(self.min_pulse_width, pos_switching_time * 1.1)
                    reset_pulse_width = max(self.min_pulse_width, neg_switching_time * 1.1)
                    
                    candidates.append({
                        'material': material,
                        'pos_voltage': pos_voltage,
                        'neg_voltage': neg_voltage,
                        'set_pulse_width': set_pulse_width,                         'reset_pulse_width': reset_pulse_width,                         'predicted_performance': {
                            'endurance': avg_endurance,
                            'total_switching_time': actual_total_switching_time,
                            'pos_switching_time': pos_switching_time,                             'neg_switching_time': neg_switching_time,                             'energy': total_energy
                        },
                        'target_performance': {
                            'endurance': target_endurance,
                            'total_switching_time': target_switching_time,
                            'energy': target_energy
                        }
                    })
                    
                except Exception as e:
                    print(f"Error evaluating diverse recommendation for {material} at {pos_voltage:.2f}V/{neg_voltage:.2f}V: {e}")
                    continue
        
        if not candidates:
            print("Failed to generate any valid recommendations. Please try different target parameters.")
            return []
        
        # Phase 3: Filter by endurance and score candidates
        print("\nFiltering and scoring candidates...")
        print(f"Initial number of candidates generated: {len(candidates)}")

        material_counts = {}
        for c in candidates:
            material = c['material']
            material_counts[material] = material_counts.get(material, 0) + 1
        
        print("Candidates by material before filtering:")
        for material, count in material_counts.items():
            print(f"  {material}: {count} candidates")

        initial_count = len(candidates)
        candidates = [
            c for c in candidates
            if c['predicted_performance']['endurance'] >= target_endurance
        ]
        filtered_count = initial_count - len(candidates)
        print(f"Filtered out {filtered_count} candidates due to not meeting target endurance ({target_endurance:.2e}).")

        if not candidates:
            print("No candidates meet the target endurance requirement. Cannot generate recommendations.")
            return []
            
        material_counts = {}
        for c in candidates:
            material = c['material']
            material_counts[material] = material_counts.get(material, 0) + 1
        
        print("Candidates by material after endurance filtering:")
        for material, count in material_counts.items():
            print(f"  {material}: {count} candidates")

        for candidate in candidates:
            predicted_time = candidate['predicted_performance']['total_switching_time']
            target_time = candidate['target_performance']['total_switching_time']
            time_score = 0.0
            if predicted_time <= target_time:
                time_ratio = min(target_time / max(predicted_time, 1e-15), 2.0)
                time_score = 0.5 + 0.5 * (time_ratio - 1.0) / 1.0
            else:
                time_score = 0.5 * (target_time / predicted_time)
            time_score = max(0.0, min(1.0, time_score))

            energy = candidate['predicted_performance']['energy']
            target_energy_val = candidate['target_performance']['energy']
            energy_score = 0.0
            if energy <= target_energy_val:
                energy_ratio = min(target_energy_val / max(energy, 1e-15), 2.0)
                energy_score = 0.5 + 0.5 * (energy_ratio - 1.0) / 1.0
            else:
                energy_score = 0.5 * (target_energy_val / energy)
            energy_score = max(0.0, min(1.0, energy_score))

            predicted_endurance = candidate['predicted_performance']['endurance']
            log_improvement = 0.0
            if predicted_endurance > 0 and target_endurance > 0:
                try:
                    if predicted_endurance / target_endurance > 0:
                         log_improvement = math.log10(predicted_endurance / target_endurance)
                    else:
                        log_improvement = -float('inf')
                except ValueError:
                    log_improvement = -float('inf')
                    
            log_improvement = max(0.0, log_improvement) 

            k = 0.35
            endurance_score = 0.5 + 0.5 * math.tanh(k * log_improvement)
            
            candidate['score'] = {
                'switching_time': time_score,
                'energy': energy_score,
                'endurance': endurance_score,
            }
            
            candidate['score']['combined'] = (time_score + energy_score + endurance_score) / 3.0
            
            candidate['endurance_ratio'] = candidate['predicted_performance']['endurance'] / target_endurance if target_endurance > 0 else 0

        # Sort candidates by different optimization objectives
        candidates_by_material = {}
        for material in self.materials:
            material_candidates = [c for c in candidates if c['material'] == material]
            if material_candidates:
                candidates_by_material[material] = {
                    'combined': sorted(material_candidates, key=lambda c: c['score']['combined'], reverse=True),
                    'endurance': sorted(material_candidates, key=lambda c: c['score']['endurance'], reverse=True),
                    'energy': sorted(material_candidates, key=lambda c: (c['score']['energy'], -c['predicted_performance']['energy']), reverse=True),
                    'switching_time': sorted(material_candidates, key=lambda c: c['predicted_performance']['total_switching_time'])
                }
        
        candidates_by_combined = sorted(candidates, key=lambda c: c['score']['combined'], reverse=True)
        candidates_by_endurance = sorted(candidates, key=lambda c: c['score']['endurance'], reverse=True)
        candidates_by_energy = sorted(candidates, 
                                      key=lambda c: (c['score']['energy'], -c['predicted_performance']['energy']), 
                                      reverse=True)
        candidates_by_switching_time = sorted(candidates,
                                              key=lambda c: c['predicted_performance']['total_switching_time'])
        
        print("\nTop candidates for each optimization target:")
        
        print("Best combined score (all materials):")
        top_combined = candidates_by_combined[0]
        print(f"  {top_combined['material']}, {top_combined['pos_voltage']:.2f}V/{top_combined['neg_voltage']:.2f}V, "
              f"Combined: {top_combined['score']['combined']:.2f}")
        
        print("Best combined score by material:")
        for material, material_data in candidates_by_material.items():
            if material_data['combined']:
                best = material_data['combined'][0]
                print(f"  {material}: Combined={best['score']['combined']:.2f}, "
                      f"Time={best['predicted_performance']['total_switching_time']*1e9:.2f}ns, "
                      f"Energy={best['predicted_performance']['energy']*1e12:.2f}pJ, "
                      f"Endurance={best['predicted_performance']['endurance']:.2e}")
        
        print("\nBest endurance (all materials):")
        top_endurance = candidates_by_endurance[0]
        print(f"  {top_endurance['material']}, {top_endurance['pos_voltage']:.2f}V/{top_endurance['neg_voltage']:.2f}V, "
              f"Endurance: {top_endurance['predicted_performance']['endurance']:.2e}, Score: {top_endurance['score']['endurance']:.2f}")
        
        print("\nBest energy efficiency (all materials):")
        top_energy = candidates_by_energy[0]
        print(f"  {top_energy['material']}, {top_energy['pos_voltage']:.2f}V/{top_energy['neg_voltage']:.2f}V, "
              f"Energy: {top_energy['predicted_performance']['energy']*1e12:.2f} pJ, Score: {top_energy['score']['energy']:.2f}")
        
        print("\nBest switching time (all materials):")
        top_switching_time = candidates_by_switching_time[0]
        print(f"  {top_switching_time['material']}, {top_switching_time['pos_voltage']:.2f}V/{top_switching_time['neg_voltage']:.2f}V, "
              f"Switching Time: {top_switching_time['predicted_performance']['total_switching_time']*1e9:.2f} ns, Score: {top_switching_time['score']['switching_time']:.2f}")
        
        # Phase 4: Select diverse recommendations across optimization targets
        final_recommendations = []
        selected_materials = set()
        selected_targets = set()

        optimization_targets = [
            ('Overall Performance', candidates_by_combined),
            ('Endurance Optimization', candidates_by_endurance),
            ('Energy Optimization', candidates_by_energy),
            ('Switching Time Optimization', candidates_by_switching_time)
        ]
        
        similarity_threshold = self.similarity_threshold
        voltage_difference_threshold = 0.15

        for target_name, sorted_candidates in optimization_targets:
            if len(final_recommendations) >= 4:
                break

            if target_name in selected_targets:
                continue
                
            best_candidate = None

            for candidate in sorted_candidates:
                if candidate['material'] not in selected_materials:
                    is_different_enough = True
                    for existing in final_recommendations:
                        if self._are_recommendations_similar(candidate, existing, voltage_threshold=voltage_difference_threshold):
                            is_different_enough = False
                            break
                    
                    if is_different_enough:
                        best_candidate = candidate
                        break
            
            if best_candidate is None:
                for candidate in sorted_candidates:
                    is_different_enough = True
                    for existing in final_recommendations:
                        if self._are_recommendations_similar(candidate, existing, voltage_threshold=voltage_difference_threshold):
                            is_different_enough = False
                            break
                    
                    if is_different_enough:
                        best_candidate = candidate
                        break
            
            if best_candidate is None and sorted_candidates:
                best_candidate = sorted_candidates[0]
            
            if best_candidate:
                final_recommendations.append({
                    **best_candidate,
                    'optimization_target': target_name
                })
                selected_materials.add(best_candidate['material'])
                selected_targets.add(target_name)
        
        # Ensure material diversity in final selection
        if len(set(c['material'] for c in candidates)) > 1 and len(selected_materials) < 2:
            missing_materials = set(self.materials) - selected_materials

            if not missing_materials:
                missing_materials = set(self.materials)
            
            for material in missing_materials:
                if material not in candidates_by_material:
                    continue
                    
                material_candidates = candidates_by_material[material]['combined']
                if not material_candidates:
                    continue
                
                if len(final_recommendations) >= 4:
                    break
                
                best_candidate = material_candidates[0]
                
                is_different_enough = True
                for existing in final_recommendations:
                    if self._are_recommendations_similar(best_candidate, existing, voltage_threshold=voltage_difference_threshold):
                        is_different_enough = False
                        break
                
                if is_different_enough:
                    available_targets = set(t[0] for t in optimization_targets) - selected_targets
                    target_name = next(iter(available_targets)) if available_targets else "Material Diversity"
                    
                    final_recommendations.append({
                        **best_candidate,
                        'optimization_target': target_name
                    })
                    selected_materials.add(material)
                    if target_name != "Material Diversity":
                        selected_targets.add(target_name)
        
        # Fill remaining slots from top candidates
        if len(final_recommendations) < 4 and len(candidates) >= 4:
            used_candidates = set((c['material'], c['pos_voltage'], c['neg_voltage']) for c in final_recommendations)
            remaining_candidates = [
                c for c in candidates 
                if (c['material'], c['pos_voltage'], c['neg_voltage']) not in used_candidates
            ]
            
            remaining_candidates.sort(key=lambda c: c['score']['combined'], reverse=True)
            
            unused_targets = set(t[0] for t in optimization_targets) - selected_targets

            for candidate in remaining_candidates:
                if len(final_recommendations) >= 4:
                    break
                
                is_different_enough = True
                for existing in final_recommendations:
                    if self._are_recommendations_similar(candidate, existing, voltage_threshold=voltage_difference_threshold):
                        is_different_enough = False
                        break
                
                if is_different_enough:
                    target_name = next(iter(unused_targets)) if unused_targets else "Material Diversity"
                    
                    final_recommendations.append({
                        **candidate,
                        'optimization_target': target_name
                    })
                    if target_name in unused_targets:
                        unused_targets.remove(target_name)
        
        print(f"\nGenerated {len(candidates)} candidates, selected {len(final_recommendations)} recommendations with different optimization targets")
        print("Final recommendations:")
        for i, rec in enumerate(final_recommendations):
            print(f" {i+1}. {rec['optimization_target']}: {rec['material']}, "
                  f"{rec['pos_voltage']:.2f}V/{rec['neg_voltage']:.2f}V, "
                  f"Endurance={rec['predicted_performance']['endurance']:.2e}, "
                  f"Time={rec['predicted_performance']['total_switching_time']*1e9:.2f}ns, "
                  f"Energy={rec['predicted_performance']['energy']*1e12:.2f}pJ")
        
        return final_recommendations[:4]
