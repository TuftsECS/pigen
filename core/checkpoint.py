import os
import torch
import torch.optim as optim

def configure_optimizers(pinn_model, mlp_model, config):
    pinn_optimizer = optim.AdamW(
        pinn_model.parameters(),
        lr=config['learning_rate_pinn'],
        weight_decay=config['weight_decay_pinn']
    )
    mlp_optimizer = optim.AdamW(
        mlp_model.parameters(),
        lr=config['learning_rate_mlp'],
        weight_decay=config['weight_decay_mlp']
    )
    return pinn_optimizer, mlp_optimizer

def save_checkpoint(save_dir, epoch, pinn_model, mlp_model, pinn_optimizer, mlp_optimizer, val_metrics, scalers, metric):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'pinn_model_state_dict': pinn_model.state_dict(),
        'mlp_model_state_dict': mlp_model.state_dict(),
        'pinn_optimizer_state_dict': pinn_optimizer.state_dict(),
        'mlp_optimizer_state_dict': mlp_optimizer.state_dict(),
        'best_valid_mean_error': val_metrics['mean_error'],
        'best_valid_max_error': val_metrics['max_error'],
        'best_valid_accuracy': val_metrics['final_current_accuracy_5'],
        'scalers': scalers
    }
    torch.save(checkpoint, os.path.join(save_dir, f'best_{metric}_checkpoint.pth'))

def load_checkpoint(path, device):
    """Checkpoints hold fitted sklearn scalers, so they cannot be loaded with torch's weights-only mode."""
    return torch.load(path, map_location=device, weights_only=False)
