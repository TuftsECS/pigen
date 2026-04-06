import torch
from typing import Dict, Tuple
import hashlib

class PhysicsSimulationCache:    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _compute_key(self, dt, v, current, material):
        data = torch.cat([dt, v, current]).cpu().numpy()
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        return f"{material}_{data_hash}"
    
    def get(self, dt, v, current, material, device):
        if not self.enabled:
            return None
        
        key = self._compute_key(dt, v, current, material)
        
        if key in self.cache:
            self.hit_count += 1
            gap_cached, temp_cached = self.cache[key]
            return gap_cached.to(device), temp_cached.to(device)
        
        self.miss_count += 1
        return None
    
    def put(self, dt, v, current, material, gap, temperature):
        if not self.enabled:
            return
        
        key = self._compute_key(dt, v, current, material)
        self.cache[key] = (gap.cpu().clone(), temperature.cpu().clone())
    
    def clear(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total * 100 if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

