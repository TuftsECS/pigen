import os
import torch
import argparse
import numpy as np
import torch.optim as optim

from src import (
    # Models
    RRAM_PINN, MLP_Current,
    # Data
    Constants, RRAMDataset, collate_sequences,
    # Training
    setup_logger, train_sequence, validate_sequence,
    # Utils
    configure_optimizers, save_checkpoint, calculate_metric_range,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate_pinn', type=float, default=1e-4)
    parser.add_argument('--learning_rate_mlp', type=float, default=1e-4)
    parser.add_argument('--weight_decay_pinn', type=float, default=1e-5)
    parser.add_argument('--weight_decay_mlp', type=float, default=1e-5)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--embedding_size', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--data_path', type=str, default='data/rram_data.mat')
    
    parser.add_argument('--use_pde', action='store_true', help='Whether to use PDE loss')
    parser.add_argument('--use_full_dataset', action='store_true', help='Whether to use full dataset')
    parser.add_argument('--voltage_stride', type=float, default=0.2, help='Voltage stride for dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def main():
    const = Constants()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    config = vars(args)
    
    logger = setup_logger(config['save_dir'], config['exp_name'])
    logger.info("="*50)
    logger.info("Starting RRAM model training...")
    logger.info("="*50)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")
    logger.info("\n" + "="*50)
    
    pde_scheduler = None
    if config['use_pde']:
        from src.training import AdaptivePDEScheduler
        pde_scheduler = AdaptivePDEScheduler(alpha=0.95, min_weight=0.1, max_weight=10.0)
        logger.info("Initialized AdaptivePDEScheduler with parameters:")
        logger.info(f"- alpha: 0.95 (EMA smoothing factor)")
        logger.info(f"- min_weight: 0.1")
        logger.info(f"- max_weight: 10.0")
    
    train_dataset = RRAMDataset(
        data_path=config['data_path'],
        fit_scaler=True,
        is_train=True,
        seed=config['seed'],
        logger=logger,
        use_full_dataset=config['use_full_dataset'],
        voltage_stride=config['voltage_stride']

    )
    
    val_dataset = RRAMDataset(
        data_path=config['data_path'],
        fit_scaler=False,
        is_train=False,
        seed=config['seed'],
        logger=logger,
        use_full_dataset=config['use_full_dataset'],
        voltage_stride=config['voltage_stride']
    )
    
    scalers = train_dataset.get_scalers()
    val_dataset.set_scalers(scalers)
    
    logger.info("\nDataset Material Distribution for Training:")
    material_counts = train_dataset.get_material_distribution()
    for material, count in material_counts.items():
        percentage = (count / len(train_dataset)) * 100
        logger.info(f"{material}: {count} sequences ({percentage:.1f}%)")
        
    logger.info("\nDataset Material Distribution for Validation:")
    material_counts = val_dataset.get_material_distribution()
    for material, count in material_counts.items():
        percentage = (count / len(val_dataset)) * 100
        logger.info(f"{material}: {count} sequences ({percentage:.1f}%)")

    pinn_model = RRAM_PINN(hidden_size=config['hidden_size'], embedding_size=config['embedding_size'], const=const).to(device)
    mlp_model = MLP_Current(hidden_size=config['hidden_size'], embedding_size=config['embedding_size']).to(device)
    
    optimizers_dict = configure_optimizers(pinn_model, mlp_model, config)
    pinn_optimizer = optimizers_dict['pinn_optimizer']
    mlp_optimizer = optimizers_dict['mlp_optimizer']

    steps_per_epoch = len(train_dataset)
    total_steps = args.epochs * steps_per_epoch
    
    pinn_max_lr = pinn_optimizer.param_groups[0]['lr'] * 100 
    mlp_max_lr = mlp_optimizer.param_groups[0]['lr'] * 100

    pinn_scheduler = optim.lr_scheduler.OneCycleLR(
        pinn_optimizer, 
        max_lr=pinn_max_lr, 
        total_steps=total_steps
    )
    mlp_scheduler = optim.lr_scheduler.OneCycleLR(
        mlp_optimizer, 
        max_lr=mlp_max_lr, 
        total_steps=total_steps
    )
    
    best_mean_error = float('inf')
    best_switch_duration_accuracy = 0.0
    best_switch_time_accuracy = 0.0
    best_plateau_current_accuracy = 0.0

    history = {
        'train_mlp_losses': [],
        'train_pinn_losses': [],
        'valid_mlp_losses': [],
        'valid_pinn_losses': [],
        'valid_mean_errors': [],
        'valid_max_errors': [], 
        'valid_switch_time_accuracies': [],
        'valid_switch_duration_accuracies': [],
        'valid_plateau_current_accuracies': [],
        'learning_rates': [],
        'pde_weights': []
    }
    
    for epoch in range(config['epochs']):
        pinn_model.train()
        mlp_model.train()
        
        train_mlp_loss, train_pinn_loss, train_metrics = train_sequence(
            pinn_model=pinn_model,
            mlp_model=mlp_model,
            dataset=train_dataset,
            pinn_optimizer=pinn_optimizer,
            mlp_optimizer=mlp_optimizer,
            pinn_scheduler=pinn_scheduler,
            mlp_scheduler=mlp_scheduler,
            const=const,
            scalers=scalers,
            use_pde=config['use_pde'],
            pde_scheduler=pde_scheduler,
        )
        
        val_metrics = validate_sequence(
            pinn_model=pinn_model,
            mlp_model=mlp_model,
            dataset=val_dataset,
            const=const,
            scalers=scalers,
            use_pde=config['use_pde'],
            pde_scheduler=pde_scheduler,
            save_dir=config['save_dir']
        )
        
        history['train_mlp_losses'].append(train_mlp_loss)
        history['train_pinn_losses'].append(train_pinn_loss)
        history['valid_mlp_losses'].append(val_metrics['mlp_loss'])
        history['valid_pinn_losses'].append(val_metrics['pinn_loss'])
        history['valid_mean_errors'].append(val_metrics['mean_error'])
        history['valid_max_errors'].append(val_metrics['max_error'])
        history['valid_switch_time_accuracies'].append(val_metrics['switch_time_accuracy'])
        history['valid_switch_duration_accuracies'].append(val_metrics['switch_duration_accuracy'])
        history['valid_plateau_current_accuracies'].append(val_metrics['plateau_current_accuracy'])
        history['learning_rates'].append(train_metrics['learning_rates'])
        if pde_scheduler and config['use_pde']:
            history['pde_weights'].append(train_metrics['pde_weight'])
    
        logger.info(
            f"Epoch [{epoch+1}/{config['epochs']}]:\n"
            f"  Train - MLP Loss: {train_mlp_loss:.6f}, PINN Loss: {train_pinn_loss:.6f}\n"
            f"  Valid - MLP Loss: {val_metrics['mlp_loss']:.6f}, PINN Loss: {val_metrics['pinn_loss']:.6f}\n"
            f"  Valid Metrics - Mean Error: {val_metrics['mean_error']:.2f}% (Max: {val_metrics['max_error']:.2f}%)\n"
            f"                - Switch Time Acc: {val_metrics['switch_time_accuracy']:.2f}%\n"
            f"                - Switch Duration Acc: {val_metrics['switch_duration_accuracy']:.2f}%\n"
            f"                - Plateau Current Acc: {val_metrics['plateau_current_accuracy']:.2f}%\n"
            f"  LRs - PINN: {train_metrics['learning_rates']['pinn_lr']:.2e}, MLP: {train_metrics['learning_rates']['mlp_lr']:.2e}"
            + (f"\n  PDE Weight: {train_metrics['pde_weight']:.3f}" if pde_scheduler and config['use_pde'] else "")
        )
        
        if val_metrics['mean_error'] < best_mean_error:
            best_mean_error = val_metrics['mean_error']
            save_checkpoint(
                save_dir=config['save_dir'],
                epoch=epoch,
                pinn_model=pinn_model,
                mlp_model=mlp_model,
                optimizer_pinn=pinn_optimizer,
                optimizer_mlp=mlp_optimizer,
                val_metrics=val_metrics,
                scalers=scalers,
                metric_type='mean_error'
            )
            logger.info(f"New best model (mean_error) saved: {best_mean_error:.2f}%\n")

        if val_metrics['switch_time_accuracy'] > best_switch_time_accuracy:
            best_switch_time_accuracy = val_metrics['switch_time_accuracy']
            save_checkpoint(
                save_dir=config['save_dir'],
                epoch=epoch,
                pinn_model=pinn_model,
                mlp_model=mlp_model,
                optimizer_pinn=pinn_optimizer,
                optimizer_mlp=mlp_optimizer,
                val_metrics=val_metrics,
                scalers=scalers,
                metric_type='accuracy'
            )
            logger.info(f"New best model saved with switch time accuracy: {best_switch_time_accuracy:.2f}%")
            
        if val_metrics['switch_duration_accuracy'] > best_switch_duration_accuracy:
            best_switch_duration_accuracy = val_metrics['switch_duration_accuracy']
            save_checkpoint(
                save_dir=config['save_dir'],
                epoch=epoch,
                pinn_model=pinn_model,
                mlp_model=mlp_model,
                optimizer_pinn=pinn_optimizer,
                optimizer_mlp=mlp_optimizer,
                val_metrics=val_metrics,
                scalers=scalers,
                metric_type='switch_accuracy'
            )
            logger.info(f"New best model saved with switch duration accuracy: {best_switch_duration_accuracy:.2f}%")
            
        if val_metrics['plateau_current_accuracy'] > best_plateau_current_accuracy:
            best_plateau_current_accuracy = val_metrics['plateau_current_accuracy']
            save_checkpoint(
                save_dir=config['save_dir'],
                epoch=epoch,
                pinn_model=pinn_model,
                mlp_model=mlp_model,
                optimizer_pinn=pinn_optimizer,
                optimizer_mlp=mlp_optimizer,
                val_metrics=val_metrics,
                scalers=scalers,
                metric_type='plateau_current_accuracy'
            )
            logger.info(f"New best model saved with plateau current accuracy: {best_plateau_current_accuracy:.2f}%")
            
    logger.info("Training completed!")
    logger.info(f"Best mean error achieved: {best_mean_error:.2f}%")
    if pde_scheduler and config['use_pde']:
        logger.info(f"Final PDE weight: {pde_scheduler.get_weight():.3f}")
    
if __name__ == '__main__':
    main()