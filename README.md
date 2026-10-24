# PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)

Official implementation of the ICCAD 2025 paper:

> **[PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling](https://ieeexplore.ieee.org/document/11240964)**
> Z. Zhang and M. Donato
> *IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2025*

PIGen couples a **physics-informed neural network (PINN)** that simulates ReRAM switching with a **conditional variational autoencoder (CVAE)** that recommends device parameters (material, write voltages, pulse width) for target endurance, switching time and energy.

## Overview

- **PINN**: a GRU predicts the device's normalized internal state (for ReRAM, the filament gap) over time, and a readout gives the current as `I = I_init · exp(k · (s_init − s))`, where `s_init` is the physical initial state and `k > 0` is learned from the voltage, initial current and material. With `--use_pde`, the predicted state is supervised by the device physics (for ReRAM, the Stanford model [1]).
- **CVAE**: trained on PINN evaluations, it proposes candidate parameters, validates them with the PINN, and plots the energy-latency Pareto front.

The training framework and the CVAE in `core/` are device-independent; each device provides its physics in its own package (currently `rram/`, materials HfO2, Al2O3 and TiO2).

## Project Structure

```
pigen/
├── core/
│   ├── physics.py      # DevicePhysics interface
│   ├── data.py         # .mat loading, random and voltage-stride splits, scaling
│   ├── models.py       # StatePINN and CurrentReadout
│   ├── training.py     # Training and validation epochs, adaptive PDE weight
│   ├── metrics.py      # Final-current and t90 metrics, selection score
│   ├── plotting.py     # State and current waveform plots
│   ├── checkpoint.py   # Optimizers and checkpoint I/O
│   ├── cache.py        # Physics simulation cache
│   └── cvae/           # CVAE, PINN-based device evaluation, Pareto front
├── rram/               # ReRAM physics and endurance model
├── train.py            # Train one PINN
├── compare.py          # PDE vs no-PDE runs (local or SLURM) and their summary
├── generate.py         # Train the CVAE, recommend parameters, plot the Pareto front
└── environment.yaml
```

## Installation

```bash
git clone https://github.com/TuftsECS/pigen.git
cd pigen
conda env create -f environment.yaml
conda activate pigen
```

A CUDA GPU is recommended; the code also runs on CPU.

## Data

PIGen reads a MATLAB `.mat` file of switching sequences (for ReRAM, generated with the [Stanford RRAM Model](https://nano.stanford.edu/downloads/stanford-rram-model)). Pass it with `--data_path` (`--data` for `compare.py`); without it, the scripts use the shared copy on the Tufts HPC cluster, `/cluster/tufts/donatolab/shared/datasets/pigen/rram.mat`.

| Key | Shape | Description |
|-----|-------|-------------|
| `time_{i}` | `(N,)` | Time points [s] |
| `v_{i}` | `(N,)` | Applied voltage [V] |
| `current_{i}` | `(N,)` | Current [A] |
| `material_{i}` | string | Material name (ReRAM: `HfO2`, `Al2O3`, `TiO2`) |

`i` runs from 0; the first two samples of each sequence are the voltage ramp and are not model inputs.

## Usage

Run all commands from the repository root.

### Train a PINN

```bash
python train.py --data_path data/rram.mat --exp_name my_experiment --use_pde --voltage_stride 0.2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Tufts HPC copy | Dataset (`.mat`) |
| `--exp_name` | (required) | Experiment name |
| `--physics` | `rram` | Device package |
| `--use_pde` | off | Supervise the PINN state with the device physics |
| `--use_full_dataset` | off | Random 80/20 split instead of a voltage-stride split |
| `--voltage_stride` | 0.2 | Voltage step between training voltages; the other voltages are validation |
| `--train_fraction` | off | Group split instead (below; not with `--use_full_dataset` or `--voltage_stride`): train on this fraction (e.g. 0.05, 0.1, 1.0) of the training pool |
| `--split_seed` | 0 | Seed of the group split |
| `--eval_data` | off | Also score the selected checkpoint on every sequence of this `.mat` file (e.g. records at new voltages) |
| `--save_dir` | `checkpoints` | Output directory |
| `--seed` | 42 | Random seed |
| `--epochs` | 2000 | Maximum epochs |
| `--learning_rate_pinn` / `--learning_rate_mlp` | 1e-3 with `--use_pde`, else 3e-4 | Start learning rates |
| `--warmup_epochs` | 5 | Linear learning-rate warmup |
| `--lr_patience` | 20 | Epochs without training-loss improvement before halving the learning rate |
| `--min_lr` | 1e-6 | Learning-rate floor |
| `--early_stop_patience` | 200 | Epochs without a new best checkpoint before stopping |
| `--hidden_size` / `--embedding_size` | 128 / 5 | Model sizes |

The training loss is the mean squared relative error of the current. Validation metrics, logged every epoch:

| Metric | Definition |
|--------|------------|
| Final Current Acc | Share of sequences whose mean \|current\| over the last 10 % of the time axis is within 5 % / 10 % of the measurement |
| t90 Acc | Share of switching sequences whose 90 % crossing time is within 0.25 decade of the measurement |
| Joint Pass | Final current within 10 % and t90 within 0.25 decade |
| Missed / False | Switching sequences predicted not to switch / non-switching sequences predicted to switch |
| TB log-MAE | Mean \|log10 I\| error, weighted half by time and half by log time |
| Balanced | Selection score: final-current and t90 errors in units of their tolerances, capped at 4 |

Accuracies and TB log-MAE are averaged over (material, SET/RESET) groups. The checkpoint with the lowest Balanced score is saved to `<save_dir>/best_checkpoint.pth` and reported at the end of the log as `Selected epoch ...`; waveform plots are written to `<save_dir>/<state>_plots/` (ReRAM: `gap_plots/`). The metrics read the measured current on the time axis of the file, counted from each sequence's first sample.

**Group split.** With `--train_fraction`, the sequences of each (material, SET/RESET) are grouped by voltage and shuffled with `--split_seed`. The first 15 % of the voltages form the validation set and the next 15 % the test set; training uses the given fraction of the remaining voltages. The checkpoint is selected on the validation set only. Once training ends, the selected checkpoint is scored once on the test set and on `--eval_data`, written as `Test at selected epoch ...` and `Eval <file> at selected epoch ...`. Every rate on these two lines pools all sequences.

### Compare PDE and no-PDE training

`compare.py run` trains every combination of `--modes` (default `pde no_pde`), `--splits` (default `full stride_0.1 stride_0.2 stride_0.4 stride_0.8`) and `--seeds` (default 42 to 46) into `<out>/<dataset>/<mode>/<split>/seed_<seed>`; arguments after `--` go to every `train.py` call. Each split name sets the split of `train.py`:

| `--splits` value | `train.py` options |
|---|---|
| `full` | `--use_full_dataset` |
| `stride_<step>`, e.g. `stride_0.4` | `--voltage_stride <step>` |
| `frac_<fraction>_split_<k>`, e.g. `frac_0.1_split_0` | `--train_fraction <fraction> --split_seed <k>` (group split) |

```bash
python compare.py run --data data/rram.mat --slurm --dry_run
python compare.py run --data data/rram.mat --slurm --partition gpu --gres gpu:a100:1 \
    --setup "module load anaconda; source activate pigen" \
    --seeds 42 43 44 58 59 60 122 456 789 1120 --out results/compare -- --epochs 4000
python compare.py summarize --out results/compare
```

`summarize` prints, per split, `PDE / no-PDE (runs where PDE is better)` at each run's selected checkpoint; `--part test` or `--part eval` reads the test or `--eval_data` line instead of the validation one, and the two split seeds of a group-split fraction are averaged together. See `python compare.py run --help` for the SLURM options.

Group splits with two split seeds and three seeds each (`--eval_data` adds records at new voltages):

```bash
python compare.py run --data data/rram.mat --slurm --seeds 0 1 2 --out results/compare_group \
    --splits frac_0.05_split_0 frac_0.05_split_1 frac_0.1_split_0 frac_0.1_split_1 frac_1.0_split_0 frac_1.0_split_1 -- --epochs 4000
python compare.py summarize --out results/compare_group --part test
```

### Recommend parameters with the CVAE

Train the CVAE on a PINN checkpoint (`--create_new_dataset` rebuilds the CVAE dataset, needed whenever the PINN changes):

```bash
python generate.py --train_cvae --create_new_dataset \
    --model_path checkpoints/best_checkpoint.pth --data_path data/rram.mat \
    --cvae_model_path recommendation_results/cvae_model.pth \
    --dataset_path recommendation_results/cvae_dataset.pt
```

Recommend `--num_recommendations` (default 4) parameter sets for given targets, written to `recommendation_results/recommendations.json`:

```bash
python generate.py --model_path checkpoints/best_checkpoint.pth --data_path data/rram.mat \
    --cvae_model_path recommendation_results/cvae_model.pth \
    --dataset_path recommendation_results/cvae_dataset.pt \
    --target_endurance 1e5 --target_switching_time 5e-9 --target_energy 1e-12
```

Plot the energy-latency Pareto front (`rram_pareto_front.png` / `.pdf`):

```bash
python generate.py --generate_pareto --model_path checkpoints/best_checkpoint.pth --data_path data/rram.mat \
    --output_dir recommendation_results \
    --target_switching_time 10e-9 --target_energy 5e-10 --target_endurance 1e5
```

## Adding a New Memory Device

`core` talks to a device only through `core.physics.DevicePhysics`. For a new device (e.g. phase-change memory):

1. Create a package next to `rram/`, e.g. `pcm/`, and subclass `DevicePhysics` in `pcm/physics.py`:
   - `state_name`, `materials`, `split_voltages` (voltages a stride split always trains on) and `initial_time`
   - `simulate(dt, voltage, current, material)`: physical state at every time step
   - `normalize_state(state)`: map the physical state onto [-1, 1]; a larger normalized state must mean a lower current
   - `initial_state(voltage)`: normalized state at the start of a pulse
   - for `generate.py`: `design_space` (write-voltage ranges per material) and `endurance(current, voltage, dt, material)`
2. Export it in `pcm/__init__.py` as `from .physics import PCMPhysics as Physics`.
3. Run the scripts with `--physics pcm`.

`rram/physics.py` and `rram/endurance.py` are a complete example.

## Citation

```bibtex
@INPROCEEDINGS{zhang2025pigen,
  author={Zhang, Zihan and Donato, Marco},
  booktitle={2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD)},
  title={PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling},
  year={2025},
  pages={1-9},
  doi={10.1109/ICCAD66269.2025.11240964},
  address={Munich, Germany}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## References

[1] Z. Jiang et al., "A Compact Model for Metal-Oxide Resistive Random Access Memory With Experiment Verification," *IEEE Transactions on Electron Devices*, vol. 63, no. 5, pp. 1884-1892, 2016.
