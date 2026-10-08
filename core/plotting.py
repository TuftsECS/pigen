import os
import matplotlib.pyplot as plt

def plot_state_sequence(time_seq, state_pred, state_physical, true_current, pred_current, save_dir, sequence_idx,
                        physics, scalers, use_pde=True):
    name = physics.state_name
    current_scale = scalers['current'].scale_[0]
    current_mean = scalers['current'].mean_[0]
    pred_current_real = pred_current.detach().cpu().numpy() * current_scale + current_mean
    time = time_seq.cpu().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.semilogx(time, state_pred.detach().cpu().numpy(), label=f'Predicted {name}', color='red', linestyle='--')
    if use_pde:
        plt.semilogx(time, physics.normalize_state(state_physical).detach().cpu().numpy(),
                     label=f'Physical {name}', color='blue')
    plt.xlabel('Time')
    plt.ylabel(name.capitalize())
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogx(time, true_current.detach().cpu().numpy(), label='True Current', color='green')
    plt.semilogx(time, pred_current_real, label='Predicted Current', color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_dir = os.path.join(save_dir, f'{name}_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{name}_sequence_{sequence_idx}.png'))
    plt.close()
