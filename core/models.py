import torch
import torch.nn as nn

def init_weights(module):
    if isinstance(module, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(module.weight, gain=gain/2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class StatePINN(nn.Module):
    """GRU predicting the device's normalized internal state in [-1, 1] from time, time step, voltage and material."""

    def __init__(self, num_materials, hidden_size=32, embedding_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.material_embedding = nn.Embedding(num_materials, self.embedding_size)

        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.LeakyReLU(0.2)
        )
        
        self.gru = nn.GRU(
            input_size=self.embedding_size + 2 + 2,  # [time(1), voltage(1), material_embedding, timestep_encoding(2)]
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
            nn.Tanh()
        )
        
        self.apply(init_weights)
    
    def forward(self, t, dt, v, material_idx):
        material_emb = self.material_embedding(material_idx)  # [1, embedding_size]
        material_emb = material_emb.expand(len(t), -1)  # [seq_len, embedding_size]
        
        dt = dt.unsqueeze(-1)  # [seq_len, 1]
        
        dt_encoded = torch.sign(dt) * torch.log1p(torch.abs(dt) * 1e12)
        dt_features = self.timestep_encoder(dt_encoded)  # [seq_len, 2]
        
        t = t.unsqueeze(-1)  # [seq_len, 1]
        v = v.unsqueeze(-1)  # [seq_len, 1]
        
        x = torch.cat([t, v, material_emb, dt_features], dim=-1).unsqueeze(0)  # [1, seq_len, input_size]
        
        gru_out, _ = self.gru(x)  # [1, seq_len, hidden_size]
        gru_features = gru_out.squeeze(0)  # [seq_len, hidden_size]
        
        enhanced_features = self.feature_enhancer(
            torch.cat([gru_features, dt_features], dim=-1)
        )
        
        return self.state_head(enhanced_features).squeeze(-1)  # [seq_len]

    
class CurrentReadout(nn.Module):
    def __init__(self, num_materials, hidden_size=32, embedding_size=8, current_mean=0.0, current_scale=1.0):
        """Predicts current as I_init * exp(net(...)) in real units; current_mean/scale undo the current standardization."""
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.register_buffer('current_mean', torch.tensor(float(current_mean)))
        self.register_buffer('current_scale', torch.tensor(float(current_scale)))
        self.material_embedding = nn.Embedding(num_materials, self.embedding_size)

        input_size = self.embedding_size + 3  # material_emb + state + voltage + initial_I
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.apply(init_weights)
   
    def forward(self, state, v, initial_I, material_idx):
        material_emb = self.material_embedding(material_idx)
        material_emb = material_emb.expand(len(state), -1)
        
        x = torch.cat([state.unsqueeze(-1), v.unsqueeze(-1), initial_I.unsqueeze(-1), material_emb], dim=-1)
        log_ratio = self.net(x).squeeze(-1)
        initial_I_real = initial_I * self.current_scale + self.current_mean
        current_real = initial_I_real * torch.exp(log_ratio.clamp(-10, 10))
        return (current_real - self.current_mean) / self.current_scale
