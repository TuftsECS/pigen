import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    if isinstance(module, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(module.weight, gain=gain/2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class RRAM_PINN(nn.Module):
    def __init__(self, hidden_size=32, embedding_size=5, const=None):
        super(RRAM_PINN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.const = const
        
        self.material_embedding = nn.Embedding(3, self.embedding_size)

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
        
        self.gap_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
            nn.Tanh()
        )
        
        self.apply(init_weights)
    
    def forward(self, t, dt, v, material_idx, hidden=None):
        material_emb = self.material_embedding(material_idx)  # [1, embedding_size]
        material_emb = material_emb.expand(len(t), -1)  # [seq_len, embedding_size]
        
        dt = dt.unsqueeze(-1)  # [seq_len, 1]
        
        dt_encoded = torch.sign(dt) * torch.log1p(torch.abs(dt) * 1e12)
        dt_features = self.timestep_encoder(dt_encoded)  # [seq_len, 8]
        
        t = t.unsqueeze(-1)  # [seq_len, 1]
        v = v.unsqueeze(-1)  # [seq_len, 1]
        
        x = torch.cat([t, v, material_emb, dt_features], dim=-1).unsqueeze(0)  # [1, seq_len, input_size]
        
        gru_out, _ = self.gru(x)  # [1, seq_len, hidden_size]
        gru_features = gru_out.squeeze(0)  # [seq_len, hidden_size]
        
        enhanced_features = self.feature_enhancer(
            torch.cat([gru_features, dt_features], dim=-1)
        )
        
        gap_pred = self.gap_predictor(enhanced_features).squeeze(-1)  # [seq_len]
        
        return gap_pred

    
class MLP_Current(nn.Module):
    def __init__(self, hidden_size=32, embedding_size=8):
        super(MLP_Current, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.material_embedding = nn.Embedding(3, self.embedding_size)

        input_size = self.embedding_size + 3  # material_emb + gap + voltage + initial_I
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.apply(init_weights)
   
    def forward(self, gap, v, initial_I, material_idx):
        material_emb = self.material_embedding(material_idx)
        material_emb = material_emb.expand(len(gap), -1)
        
        x = torch.cat([gap.unsqueeze(-1), v.unsqueeze(-1), initial_I.unsqueeze(-1), material_emb], dim=-1)
        current = self.net(x)
        return current.squeeze(-1)
    