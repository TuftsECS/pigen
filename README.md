# PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)

This repository contains the official implementation of our ICCAD 2025 paper:

> **[PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling](https://ieeexplore.ieee.org/document/11240964)**
> Z. Zhang and M. Donato
> *IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2025*

PIGen is a framework that combines **Physics-Informed Neural Networks (PINNs)** with a **Conditional Variational Autoencoder (CVAE)** to accelerate the design optimization of Resistive Random-Access Memory (ReRAM) devices. The PINN component models the physical switching behavior governed by ion migration and vacancy generation in metal-oxide layers, while the CVAE leverages the trained PINN to suggest optimal device parameters (material, operating voltages, pulse width) for desired performance targets such as endurance, switching speed, and energy efficiency.

## Overview

PIGen integrates two components to bridge physical simulation and design optimization:

**1. Physics-Informed Simulation (PINN)**: A recurrent network predicts the device's internal state (for ReRAM, the conductive-filament gap) from the applied voltage over time, and a readout network maps that state to current. With `--use_pde`, the predicted state is supervised by the device physics, which matters most when only a few operating voltages are measured. For ReRAM, the physics is the Stanford ReRAM model (Jiang et al., 2016): ion kinetics with voltage-driven hopping, thermal activation and Joule heating.

**2. Generative Parameter Recommendation (CVAE)**: A Conditional Variational Autoencoder trained on PINN simulation data that generates candidate parameter sets (material, voltages, pulse width) optimized for user-specified performance targets. Candidates are validated through the PINN model, and multi-objective Pareto analysis identifies optimal trade-offs between energy, latency, and endurance.

The training framework and the CVAE in `core/` are shared by every device; each memory device contributes its physics core in its own package (currently `rram/`, see [Adding a new memory device](#adding-a-new-memory-device)).

**Supported ReRAM materials**: HfO2, Al2O3, TiO2

## Updates

- **Oct 2026** — Updated the evaluation metrics. See [Validation metrics](#1-training-the-pinn-model).

## Project Structure

```
pigen/
├── core/                    # Shared training framework
│   ├── physics.py           # DevicePhysics: the interface a device's physics core implements
│   ├── data.py              # Sequence dataset: .mat loading, full/voltage-stride splits, scaling
│   ├── models.py            # StatePINN (GRU state predictor) and CurrentReadout
│   ├── training.py          # Training/validation epochs, gradient clipping, adaptive PDE weight
│   ├── metrics.py           # Final-current and t90 accuracies, log-time-weighted MSE
│   ├── plotting.py          # State and current waveform plots
│   ├── checkpoint.py        # Optimizers and checkpoint I/O
│   ├── cache.py             # Physics simulation cache
│   └── cvae/                # CVAE parameter recommendation, device evaluation with the PINN, Pareto front
├── rram/                    # ReRAM device
│   ├── physics.py           # RRAMPhysics: material parameters, gap dynamics, Joule heating, CVAE design space
│   └── endurance.py         # Endurance model (switching vs. failure barrier at the pulse's peak temperature)
├── train.py                 # Train one model
├── generate.py              # Train the CVAE, recommend parameters and plot the Pareto front
├── compare.py               # Launch PDE vs no-PDE runs (locally or on SLURM) and summarize them
├── environment.yaml         # Conda environment
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- A CUDA GPU is recommended (CPU fallback supported)

### Setup

```bash
git clone https://github.com/TuftsECS/pigen.git
cd pigen
conda env create -f environment.yaml
conda activate pigen
```

### Data Preparation

PIGen expects a MATLAB `.mat` file of switching sequences, for ReRAM generated from the [Stanford RRAM Model](https://nano.stanford.edu/downloads/stanford-rram-model). Place datasets under `data/` and pass them with `--data_path` (`--data` for `compare.py`); without it, the scripts read the shared copy on the Tufts HPC cluster, `/cluster/tufts/donatolab/shared/datasets/pigen/rram.mat`.

Each sequence `i` should have the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `time_{i}` | `(N,)` | Time points (seconds) |
| `v_{i}` | `(N,)` | Applied voltage waveform (V) |
| `current_{i}` | `(N,)` | Measured current response (A) |
| `material_{i}` | string | Material name, one of the device's materials (ReRAM: `HfO2`, `Al2O3`, `TiO2`) |

where `N` is the number of time steps per sequence and `i` is a zero-indexed sequence identifier. The total number of sequences is inferred from the number of `time_*` keys in the file. The first two samples of each sequence are the voltage ramp-up and are not used as model inputs.

## Usage

Run all commands from the repository root.

### 1. Training the PINN Model

```bash
python train.py --data_path data/rram.mat --exp_name my_experiment --use_pde --voltage_stride 0.2
```

**Key arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Tufts HPC shared copy | Dataset (`.mat`), see [Data Preparation](#data-preparation) |
| `--exp_name` | (required) | Experiment name for logging |
| `--physics` | `rram` | Device package providing the physics core |
| `--use_pde` | False | Supervise the PINN state with the device physics |
| `--use_full_dataset` | False | Random 80/20 split instead of a voltage-stride split |
| `--voltage_stride` | 0.2 | Voltage step between training voltages; all other voltages are validation |
| `--save_dir` | `checkpoints` | Directory for the log, checkpoints and plots |
| `--seed` | 42 | Random seed (data split, initialization, data order) |
| `--epochs` | 2000 | Maximum epochs; training usually stops early |
| `--learning_rate_pinn` / `--learning_rate_mlp` | 1e-3 with `--use_pde`, else 3e-4 | Start learning rates |
| `--warmup_epochs` | 5 | Linear learning-rate warmup epochs |
| `--lr_patience` | 20 | Epochs without training-loss improvement before halving the learning rate |
| `--min_lr` | 1e-6 | Learning-rate floor |
| `--early_stop_patience` | 200 | Epochs without a new best checkpoint (validation balanced score) before stopping |
| `--current_loss` | `relative` | Current data loss: `mse` of the standardized current, `relative` (mean squared pointwise relative error) or `log` (MSE of ln\|current\|) |
| `--hidden_size` | 128 | Hidden layer size |

**Validation metrics** (logged every epoch):

| Metric | Definition |
|--------|------------|
| MLP Loss | Current data loss selected by `--current_loss` (training loss) |
| Final Current Acc | Share of sequences whose mean \|current\| over the last 10 % of the time axis is within 5 % or 10 % of the measured mean |
| t90 Acc | Share of switching sequences whose 90 % crossing is within 0.25 decade (about 1.8×) of the measured crossing. This is when the transition finishes |
| Joint Pass | Share of sequences with the final current within 10 % and t90 within 0.25 decade; a sequence that does not switch passes only if the prediction does not switch |
| Missed / False | Share of switching sequences predicted not to switch / of non-switching sequences predicted to switch |
| TB log-MAE | Mean \|log10 I\| error, each point weighted half by the time and half by the log time it covers |
| Balanced | Selection score: final-current and t90 errors in units of their tolerances (10 %, 0.25 decade), capped at 4, averaged per sequence and then over switching and non-switching sequences |

Both curves are measured on \|current\|, each between its own first sample and its own trailing mean, so an error in amplitude does not move the crossing time. A measured curve that changes by at most 10 % of its larger endpoint is not a switch and is scored only on final current. If the measurement switches and the prediction does not, that sequence counts as inaccurate. The accuracies and TB log-MAE are averaged equally over (material, SET/RESET) groups; Joint Pass, Missed, False and Balanced pool all sequences. These are the evaluation metrics of PGNN.

The checkpoint with the lowest Balanced score (ties broken by TB log-MAE) is saved to `<save_dir>/best_checkpoint.pth`, and its metrics are logged at the end of training as `Selected epoch ...`. Waveform plots under `<save_dir>/<state>_plots/` (for ReRAM, `gap_plots/`) are refreshed whenever this checkpoint improves.

### 2. Comparing PDE and no-PDE Training

`compare.py run` trains every combination of datasets, modes (`pde`, `no_pde`), splits (`full`, `stride_<step>`) and seeds, storing each run in `<out>/<dataset>/<mode>/<split>/seed_<seed>`. By default it runs 50 trainings (5 splits × 5 seeds × 2 modes) one after another; `--slurm` submits them as a single SLURM job array instead.

```bash
# Preview the runs and the generated job script
python compare.py run --data data/rram.mat --slurm --dry_run

# Submit to SLURM (cluster-specific settings are options; "--setup" runs before each training)
python compare.py run --data data/rram.mat --slurm --partition gpu --gres gpu:a100:1 \
                      --setup "module load anaconda; source activate pigen" --out results/compare

# A smaller study; arguments after "--" are passed to every train.py call
python compare.py run --data data/rram.mat --splits full stride_0.4 --seeds 42 43 -- --epochs 500

# Paired PDE vs no-PDE summary of finished runs (best validation epoch of each run)
python compare.py summarize --out results/compare
```

`summarize` prints one row per split. Each cell is `PDE / no-PDE (wins/seeds)`: the two numbers are means over seeds of each run's selected checkpoint, and the fraction is how many seeds PDE wins. Errors and the missed-switch rate count a win when PDE is lower; accuracies count a win when PDE is higher. Equal values are not wins.

With `--setup`, jobs run `python` from the environment the setup commands activate; without it, they use the interpreter that runs `compare.py` (override either with `--python`). SLURM job scripts, task lists and logs are written to `<out>/.submit/<timestamp>/`. Re-using an `--out` directory overwrites earlier runs with the same name. Run `python compare.py run --help` for all options.

### 3. Training the CVAE Model (Optional)

If retraining the CVAE with a new PINN checkpoint (`--physics` selects the device, `rram` by default):

```bash
python generate.py --train_cvae \
                   --model_path checkpoints/best_checkpoint.pth \
                   --data_path data/rram.mat \
                   --cvae_model_path recommendation_results/cvae_model.pth \
                   --dataset_path recommendation_results/cvae_dataset.pt
```

### 4. Generating Parameter Recommendations

```bash
python generate.py --model_path checkpoints/best_checkpoint.pth \
                   --data_path data/rram.mat \
                   --cvae_model_path recommendation_results/cvae_model.pth \
                   --dataset_path recommendation_results/cvae_dataset.pt \
                   --target_endurance 1e5 \
                   --target_switching_time 5e-9 \
                   --target_energy 1e-12
```

The system generates 4 diverse recommendations optimized for different objectives (overall performance, endurance, energy efficiency, switching speed). Results are saved to `recommendation_results/recommendations.json`.

### 5. Generating Pareto Front Plots

```bash
python generate.py --generate_pareto \
                   --model_path checkpoints/best_checkpoint.pth \
                   --data_path data/rram.mat \
                   --output_dir recommendation_results \
                   --target_switching_time 10e-9 \
                   --target_energy 5e-10 \
                   --target_endurance 1e5
```

Generates `<physics>_pareto_front.png` and `.pdf` (for ReRAM, `rram_pareto_front.png`) visualizing energy vs. latency trade-offs across materials, with Pareto-optimal points highlighted.

## Adding a New Memory Device

The models, training loop, metrics, CVAE, `compare.py` and `generate.py` only talk to a device through `core.physics.DevicePhysics`. To support another device (for example phase-change memory):

1. Create a package next to `rram/`, e.g. `pcm/`.
2. In `pcm/physics.py`, subclass `DevicePhysics`:
   - `state_name`: name of the internal state (used for plots), e.g. `'crystalline_fraction'`
   - `materials`: material names in dataset order
   - `split_voltages`: per material, voltages a stride split always keeps for training (e.g. switching thresholds)
   - `initial_time`: time before the first sample, used to compute the first time step
   - `simulate(dt, voltage, current, material)`: physical state at every time step (1-D tensors in physical units)
   - `normalize_state(state)`: map the physical state onto [-1, 1]
   - For parameter recommendation only (`generate.py`):
     - `design_space`: per material, the voltage ranges of the positive and negative write pulses, e.g. `{'HfO2': {'pos': (1.5, 2.0), 'neg': (-2.0, -1.5)}}`
     - `endurance(current, voltage, dt, material)`: endurance [cycles] of one write pulse and a dict of device-specific metrics
3. Export it in `pcm/__init__.py` as `from .physics import PCMPhysics as Physics`.
4. Train with `python train.py --physics pcm ...` or `python compare.py run --physics pcm ...`, then recommend parameters with `python generate.py --physics pcm ...`.

`rram/physics.py` (with `rram/endurance.py`) is a complete reference implementation.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{zhang2025pigen,
  author={Zhang, Zihan and Donato, Marco},
  booktitle={2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD)},
  title={PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling},
  year={2025},
  pages={1-9},
  doi={10.1109/ICCAD66269.2025.11240964},
  address={Munich, Germany},
  keywords={Performance evaluation; Computational modeling; Neural networks; Autoencoders; Switches; Predictive models; Aerodynamics; Space exploration; Physics; Optimization; Resistive Random-Access Memory; Physics-Informed Neural Network; Generative Model}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

[1] Z. Jiang et al., "A Compact Model for Metal-Oxide Resistive Random Access Memory With Experiment Verification," *IEEE Transactions on Electron Devices*, vol. 63, no. 5, pp. 1884-1892, 2016.
