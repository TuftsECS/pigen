import argparse
import importlib
import os

import torch
import torch.optim as optim

from core import (
    AdaptivePDEScheduler, CurrentReadout, SequenceDataset, StatePINN,
    configure_optimizers, load_checkpoint, save_checkpoint, setup_logger, train_epoch, validate_epoch,
)

# Start learning rates validated on rram.mat, used when --learning_rate_* is not given.
DEFAULT_LR = {'pde': 1e-3, 'no_pde': 3e-4}
# Shared dataset on the Tufts HPC cluster.
DEFAULT_DATA_PATH = '/cluster/tufts/donatolab/shared/datasets/pigen/rram.mat'

def get_args():
    parser = argparse.ArgumentParser(description='Train the state PINN and current readout of one memory device.')
    parser.add_argument('--physics', type=str, default='rram', help='Device package providing the physics core')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Dataset (.mat)')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--use_pde', action='store_true', help='Supervise the PINN state with the device physics')
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--use_full_dataset', action='store_true', help='Random 80/20 split instead of a voltage-stride split')
    split_group.add_argument('--voltage_stride', type=float, default=0.2, help='Voltage stride for dataset')
    split_group.add_argument('--train_fraction', type=float, default=None,
                        help='Group split: train on this fraction of the pool left after 15%% validation and 15%% test')
    parser.add_argument('--split_seed', type=int, default=0, help='Seed of the group split')
    parser.add_argument('--eval_data', type=str, default=None, help='Also score the selected checkpoint on every sequence of this .mat file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000, help='Maximum epochs (training usually stops early)')
    parser.add_argument('--learning_rate_pinn', type=float, default=None, help='Default: 1e-3 with --use_pde, else 3e-4')
    parser.add_argument('--learning_rate_mlp', type=float, default=None, help='Default: 1e-3 with --use_pde, else 3e-4')
    parser.add_argument('--weight_decay_pinn', type=float, default=1e-5)
    parser.add_argument('--weight_decay_mlp', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear LR warmup epochs')
    parser.add_argument('--lr_patience', type=int, default=20, help='Epochs without train loss improvement before halving LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='LR floor')
    parser.add_argument('--early_stop_patience', type=int, default=200, help='Epochs without a new best checkpoint before stopping')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=5)
    args = parser.parse_args()
    if args.train_fraction is not None and not 0 < args.train_fraction <= 1:
        parser.error('--train_fraction must be in (0, 1]')
    if args.eval_data and not os.path.isfile(args.eval_data):
        parser.error(f'--eval_data file not found: {args.eval_data}')
    default_lr = DEFAULT_LR['pde' if args.use_pde else 'no_pde']
    if args.learning_rate_pinn is None:
        args.learning_rate_pinn = default_lr
    if args.learning_rate_mlp is None:
        args.learning_rate_mlp = default_lr
    return args

def format_metrics(m):
    rate = lambda value: 'n/a' if value is None else f'{value:.2f}%'
    return (f"Final Current Acc: {m['final_current_accuracy_5']:.2f}% (5%), {m['final_current_accuracy_10']:.2f}% (10%), "
            f"t90 Acc: {m['t90_accuracy']:.2f}% (0.25 dex), Joint Pass: {rate(m['joint_pass'])}, Missed: {rate(m['missed_rate'])}, "
            f"False: {rate(m['false_rate'])}, TB log-MAE: {m['time_balanced_logmae']:.6f}, Balanced: {m['balanced_score']:.6f}")

def log_data_summary(logger, config, physics, train_dataset, val_dataset, test_dataset):
    split = (f"group split (training fraction {config['train_fraction']:g}, split seed {config['split_seed']})"
             if config['train_fraction'] is not None
             else 'random 80/20 split' if config['use_full_dataset']
             else f"voltage-stride split (step {config['voltage_stride']} V)")
    logger.info(f"Data: {config['data_path']} ({train_dataset.num_total} sequences), {split}")
    if config['train_fraction'] is None and not config['use_full_dataset']:
        logger.info("  Training voltages [V]:")
        width = max(len(material) for material in physics.materials) + 1
        for material in physics.materials:
            voltages = ', '.join(f'{v:g}' for v in train_dataset.train_voltages.get(material, []))
            logger.info(f"    {material + ':':<{width}} {voltages}")
    for title, dataset in (('Training', train_dataset), ('Validation', val_dataset), ('Test', test_dataset)):
        if dataset is None:
            continue
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
    
    split = dict(physics=physics, seed=config['seed'], use_full_dataset=config['use_full_dataset'],
                 voltage_stride=config['voltage_stride'], train_fraction=config['train_fraction'], split_seed=config['split_seed'])
    train_dataset = SequenceDataset(config['data_path'], fit_scaler=True, is_train=True, **split)
    val_dataset = SequenceDataset(config['data_path'], fit_scaler=False, is_train=False, **split)
    scalers = train_dataset.get_scalers()
    val_dataset.set_scalers(scalers)
    test_dataset = None
    if config['train_fraction'] is not None:
        test_dataset = SequenceDataset(config['data_path'], fit_scaler=False, is_train=False, test=True, **split)
        test_dataset.set_scalers(scalers)
    log_data_summary(logger, config, physics, train_dataset, val_dataset, test_dataset)

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
    best_score, best_epoch, best_metrics = (float('inf'), float('inf')), None, None

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
            f"  Valid - MLP Loss: {val_metrics['mlp_loss']:.6f}, PINN Loss: {val_metrics['pinn_loss']:.6f}\n"
            f"  Valid Metrics - {format_metrics(val_metrics)}\n"
            f"  LRs - PINN: {train_metrics['learning_rates']['pinn_lr']:.2e}, MLP: {train_metrics['learning_rates']['mlp_lr']:.2e}"
            + (f"\n  PDE Weight: {train_metrics['pde_weight']:.3f}" if pde_scheduler and config['use_pde'] else "")
            + (f"\n  Skipped non-finite steps: {train_metrics['skipped_steps']}" if train_metrics['skipped_steps'] else "")
        )

        score = (val_metrics['balanced_score'], val_metrics['time_balanced_logmae'])
        if score < best_score:
            best_score, best_epoch, best_metrics = score, epoch, val_metrics
            save_checkpoint(config['save_dir'], epoch, pinn_model, mlp_model, pinn_optimizer, mlp_optimizer, val_metrics, scalers)
            logger.info(f"New best checkpoint: balanced score {score[0]:.6g}, TB log-MAE {score[1]:.6g} (saved)")
            validate_epoch(pinn_model, mlp_model, val_dataset, physics, scalers, use_pde=config['use_pde'],
                           pde_scheduler=pde_scheduler, save_dir=config['save_dir'])

        if epoch + 1 >= config['warmup_epochs']:
            pinn_scheduler.step(train_mlp_loss)
            mlp_scheduler.step(train_mlp_loss)

        if epoch - best_epoch >= config['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}: no new best checkpoint for {config['early_stop_patience']} epochs")
            break
            
    logger.info("Training completed!")
    logger.info(f"Selected epoch {best_epoch + 1} - {format_metrics(best_metrics)}")
    held_out = [('Test', test_dataset)] if test_dataset is not None else []
    if config['eval_data']:
        eval_dataset = SequenceDataset(config['eval_data'], physics=physics, seed=config['seed'], split_ratio=1.0)
        eval_dataset.set_scalers(scalers)
        held_out.append((f"Eval {config['eval_data']}", eval_dataset))
    if held_out:
        best = load_checkpoint(os.path.join(config['save_dir'], 'best_checkpoint.pth'), device)
        pinn_model.load_state_dict(best['pinn_model_state_dict'])
        mlp_model.load_state_dict(best['mlp_model_state_dict'])
        for name, dataset in held_out:
            metrics = validate_epoch(pinn_model, mlp_model, dataset, physics, scalers, use_pde=config['use_pde'],
                                     pde_scheduler=pde_scheduler, pooled=True)
            logger.info(f"{name} at selected epoch {best_epoch + 1} ({len(dataset)} sequences) - {format_metrics(metrics)}")
    if pde_scheduler and config['use_pde']:
        logger.info(f"Final PDE weight: {pde_scheduler.get_weight():.3f}")
    
if __name__ == '__main__':
    main()
