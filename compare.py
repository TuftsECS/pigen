"""Launch PIGen training runs (PDE vs no-PDE over datasets, splits and seeds) and summarize them.

Run from the repository root:
  python compare.py run --data data/rram.mat               # every run, one after another on this machine
  python compare.py run --data data/rram.mat --slurm       # one SLURM job array, one task per run
  python compare.py run ... --dry_run                            # only print what would run
  python compare.py run ... -- --epochs 500 --hidden_size 64     # arguments after -- go to every train.py call
  python compare.py summarize --out results/compare              # PDE vs no-PDE, paired by seed
  python compare.py summarize --out results/compare --part test  # the same on the test records of group splits
"""
import argparse
import itertools
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
# Shared dataset on the Tufts HPC cluster.
DEFAULT_DATA = '/cluster/tufts/donatolab/shared/datasets/pigen/rram.mat'
SPLIT_PATTERN = re.compile(r'full|stride_\d*\.?\d+|frac_\d*\.?\d+_split_\d+')
METRICS_PATTERN = (r"Final Current Acc: (\S+)% \(5%\), (\S+)% \(10%\), t90 Acc: (\S+)% \(0\.25 dex\), "
                   r"Joint Pass: ([^,]+), Missed: ([^,]+), False: ([^,]+), TB log-MAE: (\S+), Balanced:")
# The selected checkpoint's metrics line on the validation set, on the test set of a group split, and on an --eval_data file.
PART_PATTERNS = {part: re.compile(prefix + METRICS_PATTERN) for part, prefix in (
    ('validation', r"Selected epoch \d+ - "), ('test', r"Test at selected epoch \d+ \(\d+ sequences\) - "),
    ('eval', r"Eval .+ at selected epoch \d+ \(\d+ sequences\) - "))}
# Summary columns at each run's selected checkpoint: (label, lower is better), in the order of METRICS_PATTERN groups.
SUMMARY_METRICS = [('final current 5%', False), ('final current 10%', False), ('t90 0.25 dex %', False),
                   ('joint pass %', False), ('missed switch %', True), ('false switch %', True), ('TB log-MAE', True)]
# Columns printed together; the summary is split into blocks to stay readable in a terminal.
SUMMARY_BLOCKS = [[0, 1, 2], [3, 4, 5, 6]]


def build_tasks(args, extra):
    tasks = []
    for data, mode, split, seed in itertools.product(args.data, args.modes, args.splits, args.seeds):
        name = Path(data).stem
        save_dir = Path(args.out).resolve() / name / mode / split / f'seed_{seed}'
        cmd = [args.python, str(REPO / 'train.py'), '--physics', args.physics, '--data_path', str(Path(data).resolve()),
               '--exp_name', f'{name}_{mode}_{split}_seed_{seed}', '--save_dir', str(save_dir), '--seed', str(seed)]
        if mode == 'pde':
            cmd.append('--use_pde')
        if split == 'full':
            cmd.append('--use_full_dataset')
        elif split.startswith('frac_'):
            fraction, split_seed = split.removeprefix('frac_').split('_split_')
            cmd += ['--train_fraction', fraction, '--split_seed', split_seed]
        else:
            cmd += ['--voltage_stride', split.removeprefix('stride_')]
        tasks.append(cmd + extra)
    return tasks


def slurm_script(args, submit_dir, n_tasks):
    lines = ['#!/bin/bash', '#SBATCH -J pigen', f'#SBATCH --array=0-{n_tasks - 1}',
             f'#SBATCH --time={args.time}', f'#SBATCH --mem={args.mem}', f'#SBATCH --cpus-per-task={args.cpus}',
             f'#SBATCH --output={submit_dir}/logs/%A_%a.out', f'#SBATCH --error={submit_dir}/logs/%A_%a.err']
    for option, value in (('partition', args.partition), ('gres', args.gres), ('exclude', args.exclude)):
        if value:
            lines.append(f'#SBATCH --{option}={value}')
    lines += ['', args.setup or '', f'cd {shlex.quote(str(REPO))}']
    if args.gres:
        lines.append(f'{shlex.quote(args.python)} -c "import torch; torch.zeros(1, device=\'cuda\')" '
                     '|| { echo "ERROR: CUDA not usable on $(hostname)" >&2; exit 1; }')
    lines.append(f'eval "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {shlex.quote(str(submit_dir / "tasks.txt"))})"')
    return '\n'.join(lines) + '\n'


def run(args, extra):
    bad = [s for s in args.splits if not SPLIT_PATTERN.fullmatch(s)]
    if bad:
        sys.exit(f'Unknown split(s) {bad}: use "full" or "stride_<voltage step>", e.g. stride_0.2')
    tasks = build_tasks(args, extra)
    commands = [shlex.join(task) for task in tasks]

    if args.slurm:
        submit_dir = Path(args.out).resolve() / '.submit' / time.strftime('%Y%m%d-%H%M%S')
        script = slurm_script(args, submit_dir, len(tasks))
        if args.dry_run:
            print(script, *commands, sep='\n')
            return
        (submit_dir / 'logs').mkdir(parents=True)
        (submit_dir / 'tasks.txt').write_text('\n'.join(commands) + '\n')
        (submit_dir / 'job.sh').write_text(script)
        result = subprocess.run(['sbatch', str(submit_dir / 'job.sh')], capture_output=True, text=True)
        if result.returncode:
            sys.exit(f'sbatch failed: {result.stderr.strip()}')
        print(f'{result.stdout.strip()} ({len(tasks)} tasks). Job files and logs: {submit_dir}')
        return

    if args.dry_run:
        print(*commands, sep='\n')
        return
    failed = []
    for i, (task, command) in enumerate(zip(tasks, commands), 1):
        print(f'\n[{i}/{len(tasks)}] {command}', flush=True)
        if subprocess.run(task, cwd=REPO).returncode:
            failed.append(command)
    if failed:
        sys.exit('Failed runs:\n' + '\n'.join(failed))


def selected_metrics(log, part):
    match = PART_PATTERNS[part].search(log.read_text()) if log.exists() else None
    if match is None:
        return None
    return [float('nan') if value == 'n/a' else float(value.rstrip('%')) for value in match.groups()]


def split_order(split):
    if split == 'full':
        return (0, 0.0)
    kind, value = split.split('_')[:2]
    return (1 if kind == 'stride' else 2, float(value))


def summarize(args):
    root = Path(args.out)
    datasets = sorted(p for p in root.iterdir() if (p / 'pde').is_dir() and (p / 'no_pde').is_dir()) if root.is_dir() else []
    if not datasets:
        sys.exit(f'No pde/no_pde runs found under {root}')
    for data_dir in datasets:
        splits = {p.name for p in (data_dir / 'pde').iterdir()} & {p.name for p in (data_dir / 'no_pde').iterdir()}
        pairs_by_split = {}
        for split in sorted(splits, key=split_order):
            pairs = []
            for seed_dir in sorted((data_dir / 'pde' / split).glob('seed_*')):
                pde = selected_metrics(seed_dir / 'training.log', args.part)
                no_pde = selected_metrics(data_dir / 'no_pde' / split / seed_dir.name / 'training.log', args.part)
                if pde and no_pde:
                    pairs.append((pde, no_pde))
            pairs_by_split.setdefault(re.sub(r'_split_\d+$', '', split), []).extend(pairs)

        print(f'\n== {data_dir.name}: selected checkpoint on the {args.part} records, mean over runs, PDE / no-PDE (runs where PDE is better)')
        for block in SUMMARY_BLOCKS:
            print(f'{"split":<12}{"runs":>6}  ' + ''.join(f'{SUMMARY_METRICS[j][0]:>28}' for j in block))
            for split, pairs in pairs_by_split.items():
                if not pairs:
                    print(f'{split:<12}{0:>6}  (no finished PDE/no-PDE pairs)')
                    continue
                pde, no_pde = np.array([p for p, _ in pairs]), np.array([q for _, q in pairs])
                cells = []
                for j in block:
                    lower = SUMMARY_METRICS[j][1]
                    wins = int(((pde[:, j] < no_pde[:, j]) if lower else (pde[:, j] > no_pde[:, j])).sum())
                    cells.append(f'{pde[:, j].mean():.4g} / {no_pde[:, j].mean():.4g} ({wins}/{len(pairs)})')
                print(f'{split:<12}{len(pairs):>6}  ' + ''.join(f'{cell:>28}' for cell in cells))
            print()


def main():
    argv, extra = (sys.argv[1:], [])
    if '--' in argv:
        cut = argv.index('--')
        argv, extra = argv[:cut], argv[cut + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    run_parser = commands.add_parser('run', help='Train every dataset x mode x split x seed combination')
    run_parser.add_argument('--data', nargs='+', default=[DEFAULT_DATA], help='Dataset .mat file(s)')
    run_parser.add_argument('--physics', default='rram', help='Device package providing the physics core')
    run_parser.add_argument('--modes', nargs='+', default=['pde', 'no_pde'], choices=['pde', 'no_pde'])
    run_parser.add_argument('--splits', nargs='+', default=['full', 'stride_0.1', 'stride_0.2', 'stride_0.4', 'stride_0.8'],
                            help='"full" (random 80/20), "stride_<voltage step>" or "frac_<training fraction>_split_<split seed>" '
                                 '(group split)')
    run_parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    run_parser.add_argument('--out', default='results/compare', help='Results root; one sub-folder per run')
    run_parser.add_argument('--python', default=None,
                            help='Python interpreter for train.py (default: "python" from the --setup environment if '
                                 '--setup is given, else the interpreter running compare.py)')
    run_parser.add_argument('--dry_run', action='store_true', help='Print the runs (and SLURM job script) only')
    slurm = run_parser.add_argument_group('SLURM (with --slurm)')
    slurm.add_argument('--slurm', action='store_true', help='Submit all runs as one SLURM job array')
    slurm.add_argument('--partition', default=None)
    slurm.add_argument('--gres', default='gpu:1', help='Generic resources per task; empty string for CPU-only')
    slurm.add_argument('--time', default='2-00:00:00')
    slurm.add_argument('--mem', default='10G')
    slurm.add_argument('--cpus', type=int, default=6)
    slurm.add_argument('--exclude', default=None, help='Nodes to avoid')
    slurm.add_argument('--setup', default=None,
                       help='Shell commands run before training, e.g. "module load anaconda; source activate pigen"')

    summary_parser = commands.add_parser('summarize', help='Compare finished PDE and no-PDE runs, paired by seed')
    summary_parser.add_argument('--out', default='results/compare', help='Results root used by "run"')
    summary_parser.add_argument('--part', default='validation', choices=list(PART_PATTERNS),
                                help='Records the selected checkpoint is scored on: the validation set, the test set of a group split '
                                     'or the --eval_data file')

    args = parser.parse_args(argv)
    if args.command == 'run':
        if args.python is None:
            args.python = 'python' if args.setup else sys.executable
        run(args, extra)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
