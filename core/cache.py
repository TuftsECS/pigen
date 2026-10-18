import hashlib
import torch


class PhysicsSimulationCache:
    """Memoizes device simulations; the physics target of a sequence never changes during training."""

    def __init__(self):
        self.cache = {}

    def simulate(self, physics, dt, voltage, current, material):
        data = torch.cat([dt, voltage, current]).cpu().numpy()
        key = f"{type(physics).__name__}_{material}_{hashlib.md5(data.tobytes()).hexdigest()}"
        if key not in self.cache:
            self.cache[key] = physics.simulate(dt, voltage, current, material).cpu().clone()
        return self.cache[key].to(dt.device)
