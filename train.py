import argparse
import importlib

import torch
import torch.optim as optim

from core import (
    AdaptivePDEScheduler, CurrentReadout, SequenceDataset, StatePINN,
    configure_optimizers, save_checkpoint, setup_logger, train_epoch, validate_epoch,
)

# Validation metrics tracked for checkpointing, with whether higher is better.
CHECKPOINT_METRICS = [
    ('mean_error', False),
    ('logt_mse', False),
    ('final_current_accuracy_5', True),
    ('t90_accuracy', True),
]
# Start learning rates validated on rram_ICCAD, used when --learning_rate_* is not given.
DEFAULT_LR = {'pde': 1e-3, 'no_pde': 3e-4}

def get_args():
    parser = argparse.ArgumentParser(description='Train the state PINN and current readout of one memory device.')
    parser.add_argument('--physics', type=str, default='rram', help='Device package providing the physics core')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset (.mat)')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--use_pde', action='store_true', help='Supervise the PINN state with the device physics')
    parser.add_argument('--use_full_dataset', action='store_true', help='Random 80/20 split instead of a voltage-stride split')
    parser.add_argument('--voltage_stride', type=float, default=0.2, help='Voltage stride for dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000, help='Maximum epochs (training usually stops early)')
    parser.add_argument('--learning_rate_pinn', type=float, default=None, help='Default: 1e-3 with --use_pde, else 3e-4')
    parser.add_argument('--learning_rate_mlp', type=float, default=None, help='Default: 1e-3 with --use_pde, else 3e-4')
    parser.add_argument('--weight_decay_pinn', type=float, default=1e-5)
    parser.add_argument('--weight_decay_mlp', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear LR warmup epochs')
    parser.add_argument('--lr_patience', type=int, default=20, help='Epochs without train loss improvement before halving LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='LR floor')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='Epochs without improvement at the LR floor before stopping')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=5)
    args = parser.parse_args()
    default_lr = DEFAULT_LR['pde' if args.use_pde else 'no_pde']
    if args.learning_rate_pinn is None:
        args.learning_rate_pinn = default_lr
    if args.learning_rate_mlp is None:
        args.learning_rate_mlp = default_lr
    return args

def log_data_summary(logger, config, physics, train_dataset, val_dataset):
    split = ('random 80/20 split' if config['use_full_dataset']
             else f"voltage-stride split (step {config['voltage_stride']} V)")
    logger.info(f"Data: {config['data_path']} ({train_dataset.num_total} sequences), {split}")
    if not config['use_full_dataset']:
        logger.info("  Training voltages [V]:")
        width = max(len(material) for material in physics.materials) + 1
        for material in physics.materials:
            voltages = ', '.join(f'{v:g}' for v in train_dataset.train_voltages.get(material, []))
            logger.info(f"    {material + ':':<{width}} {voltages}")
    for title, dataset in (('Training', train_dataset), ('Validation', val_dataset)):
        counts = dataset.get_material_distribution()
        per_material = ', '.join(f'{material} {counts.get(material, 0)}' for material in physics.materials)
        logger.info(f"  {title + ':':<11} {len(dataset):>4} sequences ({len(dataset) / dataset.num_total * 100:4.1f}%): {per_material}")
    logger.info("")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    config = vars(args)
    torch.manual_seed(config['seed'])
    physics = importlib.import_module(config['physics']).Physics()
    
    logger = setup_logger(config['save_dir'], config['exp_name'])
    logger.info(f"Physics: {type(physics).__name__} | Device: {device}")
    logger.info(f"Configuration: {config}")
    
    pde_scheduler = None
    if config['use_pde']:
        pde_scheduler = AdaptivePDEScheduler(alpha=0.95, min_weight=0.1, max_weight=1.0)
        logger.info(f"Mode: PDE (adaptive weight, EMA alpha {pde_scheduler.alpha}, "
                    f"range [{pde_scheduler.min_weight}, {pde_scheduler.max_weight}])")
    else:
        logger.info("Mode: no PDE")
    
    split = dict(physics=physics, seed=config['seed'], logger=logger,
                 use_full_dataset=config['use_full_dataset'], voltage_stride=config['voltage_stride'])
    train_dataset = SequenceDataset(config['data_path'], fit_scaler=True, is_train=True, **split)
    val_dataset = SequenceDataset(config['data_path'], fit_scaler=False, is_train=False, **split)
    scalers = train_dataset.get_scalers()
    val_dataset.set_scalers(scalers)
    log_data_summary(logger, config, physics, train_dataset, val_dataset)

    num_materials = len(physics.materials)
    pinn_model = StatePINN(num_materials, hidden_size=config['hidden_size'], embedding_size=config['embedding_size']).to(device)
    mlp_model = CurrentReadout(num_materials, hidden_size=config['hidden_size'], embedding_size=config['embedding_size'],
                               current_mean=scalers['current'].mean_[0], current_scale=scalers['current'].scale_[0]).to(device)
    pinn_optimizer, mlp_optimizer = configure_optimizers(pinn_model, mlp_model, config)

    plateau_kwargs = dict(mode='min', factor=0.5, patience=config['lr_patience'],
                          threshold=1e-2, min_lr=config['min_lr'])
    pinn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pinn_optimizer, **plateau_kwargs)
    mlp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mlp_optimizer, **plateau_kwargs)
    base_lrs = [(pinn_optimizer, config['learning_rate_pinn']), (mlp_optimizer, config['learning_rate_mlp'])]
    best_train_loss = float('inf')
    epochs_without_improvement = 0
    best = {metric: (-float('inf') if higher else float('inf')) for metric, higher in CHECKPOINT_METRICS}

    for epoch in range(config['epochs']):
        if epoch < config['warmup_epochs']:
            for optimizer, base_lr in base_lrs:
                for group in optimizer.param_groups:
                    group['lr'] = base_lr * (epoch + 1) / config['warmup_epochs']

        train_mlp_loss, train_pinn_loss, train_metrics = train_epoch(
            pinn_model, mlp_model, train_dataset, pinn_optimizer, mlp_optimizer, physics, scalers,
            use_pde=config['use_pde'], pde_scheduler=pde_scheduler)
        val_metrics = validate_epoch(
            pinn_model, mlp_model, val_dataset, physics, scalers,
            use_pde=config['use_pde'], pde_scheduler=pde_scheduler)

        logger.info(
            f"Epoch [{epoch+1}/{config['epochs']}]:\n"
            f"  Train - MLP Loss: {train_mlp_loss:.6f}, PINN Loss: {train_pinn_loss:.6f}\n"
            f"  Valid - MLP Loss: {val_metrics['mlp_loss']:.6f}, Log-t MSE: {val_metrics['logt_mse']:.6f}, PINN Loss: {val_metrics['pinn_loss']:.6f}\n"
            f"  Valid Metrics - Mean Error: {val_metrics['mean_error']:.2f}% (Max: {val_metrics['max_error']:.2f}%)\n"
            f"                - Final Current Acc: {val_metrics['final_current_accuracy_5']:.2f}% (5%), {val_metrics['final_current_accuracy_10']:.2f}% (10%)\n"
            f"                - t90 Acc: {val_metrics['t90_accuracy']:.2f}% (0.25 dex)\n"
            f"  LRs - PINN: {train_metrics['learning_rates']['pinn_lr']:.2e}, MLP: {train_metrics['learning_rates']['mlp_lr']:.2e}"
            + (f"\n  PDE Weight: {train_metrics['pde_weight']:.3f}" if pde_scheduler and config['use_pde'] else "")
            + (f"\n  Skipped non-finite steps: {train_metrics['skipped_steps']}" if train_metrics['skipped_steps'] else "")
        )

        for metric, higher in CHECKPOINT_METRICS:
            value = val_metrics[metric]
            if (value > best[metric]) if higher else (value < best[metric]):
                best[metric] = value
                save_checkpoint(config['save_dir'], epoch, pinn_model, mlp_model, pinn_optimizer, mlp_optimizer,
                                val_metrics, scalers, metric)
                logger.info(f"New best {metric}: {value:.6g} (checkpoint saved)")
                if metric == 'logt_mse':
                    validate_epoch(pinn_model, mlp_model, val_dataset, physics, scalers, use_pde=config['use_pde'],
                                   pde_scheduler=pde_scheduler, save_dir=config['save_dir'])

        if epoch + 1 >= config['warmup_epochs']:
            pinn_scheduler.step(train_mlp_loss)
            mlp_scheduler.step(train_mlp_loss)

        if pinn_scheduler.is_better(train_mlp_loss, best_train_loss):
            best_train_loss = train_mlp_loss
            epochs_without_improvement = 0
        elif all(g['lr'] <= config['min_lr'] for o in (pinn_optimizer, mlp_optimizer) for g in o.param_groups):
            epochs_without_improvement += 1
        if epochs_without_improvement >= config['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}: LR at floor {config['min_lr']:.1e}, "
                        f"no train loss improvement for {epochs_without_improvement} epochs")
            break
            
    logger.info("Training completed!")
    logger.info(f"Best mean error achieved: {best['mean_error']:.2f}%")
    logger.info(f"Best log-t MSE achieved: {best['logt_mse']:.6f}")
    if pde_scheduler and config['use_pde']:
        logger.info(f"Final PDE weight: {pde_scheduler.get_weight():.3f}")
    
if __name__ == '__main__':
    main()
