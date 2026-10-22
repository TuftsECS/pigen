import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import torch
import os
import matplotlib.ticker as ticker
from functools import partial
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.transforms import Bbox

from .generative_model import ParameterRecommender
from .evaluator import DeviceEvaluator
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 'FreeSans', 'sans-serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15
})

# Marker colors, assigned to materials in order.
MATERIAL_PALETTE = [
    {'fill': '#FBD178', 'edge': '#EE8227'},
    {'fill': '#F9C3BF', 'edge': '#EB716B'},
    {'fill': '#BFE4EE', 'edge': '#79ADD6'},
    {'fill': '#C7E9C0', 'edge': '#41AB5D'},
    {'fill': '#DADAEB', 'edge': '#807DBA'},
]
COLORS = {
    'pareto': '#556B2F',
    'target': '#1E803D'
}
AXIS_MARGIN = 1.3  # Log-axis padding factor around the data points and targets

def validate_data(df, default_material):
    """Drop points that cannot be drawn on log axes (non-finite or non-positive values)."""
    df_clean = df.copy()
    if 'total_switching_time' not in df_clean.columns:
        print("Error: 'total_switching_time' column not found in DataFrame for validation.")
        df_clean['total_switching_time'] = 10e-9
    if 'is_recommendation' in df_clean.columns:
        df_clean['is_recommendation'] = df_clean['is_recommendation'].fillna(False)
    if 'is_other_recommendation' in df_clean.columns:
        df_clean['is_other_recommendation'] = df_clean['is_other_recommendation'].fillna(False)
    
    df_clean = df_clean[
        np.isfinite(df_clean['energy']) & (df_clean['energy'] > 0) &
        np.isfinite(df_clean['total_switching_time']) & (df_clean['total_switching_time'] > 0) &
        np.isfinite(df_clean['endurance']) & (df_clean['endurance'] > 0)
    ]
    
    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} data points with invalid values")
        
    if len(df_clean) == 0:
        print("Error: No valid data points for plotting")
        return pd.DataFrame({
            'material': [default_material],
            'energy': [1000],
            'total_switching_time': [10e-9],
            'endurance': [1e6],
            'is_recommendation': [False],
            'is_other_recommendation': [False]
        })
    
    return df_clean

def get_safe_axis_limits(df, target_energy_pJ, target_switching_time_ns):
    """Log-axis limits that contain every point and both targets, padded by AXIS_MARGIN."""
    energy = np.append(df['energy'].to_numpy(), target_energy_pJ)
    time_ns = np.append(df['switching_time_ns'].to_numpy(), target_switching_time_ns)
    return (energy.min() / AXIS_MARGIN, energy.max() * AXIS_MARGIN,
            time_ns.min() / AXIS_MARGIN, time_ns.max() * AXIS_MARGIN)

def calculate_size_for_endurance(endurance):
    """Calculate marker size based on endurance."""
    log_endurance = np.log10(endurance)
    size_factor = 8
    return size_factor * (log_endurance ** 2.2)

class PlainLogFormatter(ticker.LogFormatter):
    """LogFormatter that writes the labels it keeps as plain numbers."""

    def __call__(self, x, pos=None):
        return f'{x:.15g}' if super().__call__(x, pos) else ''

def place_clear(ax, make, candidates, points, radii, avoid):
    """Draw make(**candidate) for the candidate inside the axes with the fewest `avoid` overlaps, then the fewest covered points."""
    renderer = RendererAgg(ax.figure.bbox.width, ax.figure.bbox.height, ax.figure.dpi)
    centers = ax.transData.transform(points)
    blocked = [item if isinstance(item, Bbox) else item.get_window_extent(renderer) for item in avoid]

    def cost(candidate):
        probe = make(**candidate)
        box = probe.get_window_extent(renderer)
        probe.remove()
        outside = not (ax.bbox.x0 <= box.x0 and box.x1 <= ax.bbox.x1 and ax.bbox.y0 <= box.y0 and box.y1 <= ax.bbox.y1)
        covered = np.sum((centers[:, 0] + radii > box.x0) & (centers[:, 0] - radii < box.x1)
                         & (centers[:, 1] + radii > box.y0) & (centers[:, 1] - radii < box.y1))
        return int(outside) + box.count_overlaps(blocked), int(covered)

    return make(**min(candidates, key=cost))

def generate_pareto_plot(physics,
                        data_path,
                        model_path='checkpoints/best_checkpoint.pth',
                        output_dir='recommendation_results',
                        cvae_model_path='recommendation_results/cvae_model.pth',
                        dataset_path='recommendation_results/cvae_dataset.pt',
                        target_switching_time=10e-9,  # s
                        target_energy=5e-10,  # J
                        target_endurance=1e5,  # cycles
                        num_sample_points=50,
                        energy_penalty_factor=3.0,
                        max_energy_error_ratio=1.0,
                        min_pulse_width=1e-9,
                        diverse_candidates=40,
                        num_recommendations=4):
    """Energy-latency Pareto front of one device, evaluated with its trained PINN; arguments as in generate.py."""
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing device evaluator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        evaluator = DeviceEvaluator(
            physics,
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device
        )
    except Exception as e:
        print(f"Error initializing device evaluator: {e}")
        return None, None, None
    
    print("Initializing parameter recommender...")
    try:
        recommender = ParameterRecommender(
            cvae_model_path, 
            evaluator,
            dataset_path=dataset_path,
            min_pulse_width=min_pulse_width
        )
    except Exception as e:
        print(f"Error initializing parameter recommender: {e}")
        return None, None, None
    
    print(f"Generating parameter recommendations for target performance: endurance={target_endurance:.2e}, switching_time={target_switching_time*1e9:.2f}ns, energy={target_energy:.2e}")
    try:
        recommendations = recommender.recommend_parameters(
            target_endurance,
            target_switching_time,
            target_energy,
            num_recommendations=num_recommendations,
            num_samples=diverse_candidates,
            energy_penalty_factor=energy_penalty_factor,
            max_energy_error_ratio=float('inf') if max_energy_error_ratio < 0 else max_energy_error_ratio
        )
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        recommendations = []

    if recommendations:
        print("\n--- Recommendations Selected for Plot ---")
        for i, rec in enumerate(recommendations):
            print(f" Recommendation #{i+1}:")
            print(f"  Material: {rec['material']}")
            print(f"  SET Voltage: {rec['pos_voltage']:.2f} V")
            print(f"  RESET Voltage: {rec['neg_voltage']:.2f} V")
            print(f"  SET Pulse Width: {rec.get('set_pulse_width', 0)*1e9:.2f} ns")
            print(f"  RESET Pulse Width: {rec.get('reset_pulse_width', 0)*1e9:.2f} ns")
            print(f"  Predicted Performance:")
            print(f"    Endurance: {rec['predicted_performance']['endurance']:.2e} cycles")
            print(f"    Switching Time: {rec['predicted_performance']['total_switching_time']*1e9:.2f} ns")
            print(f"    Energy: {rec['predicted_performance']['energy']*1e12:.2f} pJ")
        print("---")
    else:
        print("\nNo recommendations generated or selected for plotting.")
    
    # Sample the design space across all materials
    materials = evaluator.materials
    data = []

    print(f"Generating {num_sample_points} sampling points for each material...")
    num_sample_points = min(num_sample_points, 100)
    points_per_dim = max(2, int(np.sqrt(num_sample_points)))
    
    for material in materials:
        voltage_range = recommender.voltage_ranges[material]
        pos_min, pos_max = voltage_range['pos']
        neg_min, neg_max = voltage_range['neg']
        
        pos_voltages = np.linspace(pos_min, pos_max, points_per_dim)
        neg_voltages = np.linspace(neg_min, neg_max, points_per_dim)
        
        for pos_v in pos_voltages:
            for neg_v in neg_voltages:
                try:
                    result = evaluator.evaluate(material, pos_v, neg_v)
                    
                    avg_endurance = result['avg_endurance']
                    total_energy = result['total_energy']
                    total_switching_time = result['total_switching_time']
                    if isinstance(avg_endurance, torch.Tensor):
                        avg_endurance = float(avg_endurance.cpu().item())
                    if isinstance(total_energy, torch.Tensor):
                        total_energy = float(total_energy.cpu().item())
                    if isinstance(total_switching_time, torch.Tensor):
                        total_switching_time = float(total_switching_time.cpu().item())
                    
                    data.append({
                        'material': material,
                        'energy': total_energy * 1e12,
                        'total_switching_time': total_switching_time,
                        'endurance': avg_endurance,
                        'is_recommendation': False,
                        'is_other_recommendation': False,
                        'pos_voltage': pos_v,
                        'neg_voltage': neg_v
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {material} at {pos_v:.2f}V/{neg_v:.2f}V: {e}")
                    continue
    
    # Add recommendation points to the dataset
    for i, rec in enumerate(recommendations):
        try:
            is_best_overall = rec.get('optimization_target', '') == 'Overall Performance'
            
            data.append({
                'material': rec['material'],
                'energy': rec['predicted_performance']['energy'] * 1e12,
                'total_switching_time': rec['predicted_performance']['total_switching_time'],
                'endurance': rec['predicted_performance']['endurance'],
                'is_recommendation': is_best_overall,
                'is_other_recommendation': not is_best_overall and rec.get('optimization_target', '') != '',
                'rec_num': i + 1,
                'pos_voltage': rec['pos_voltage'],
                'neg_voltage': rec['neg_voltage'],
                'optimization_target': rec.get('optimization_target', '')
            })
        except Exception as e:
            print(f"Error adding recommendation {i+1}: {e}")
    
    df = pd.DataFrame(data)
    df = validate_data(df, materials[0])
    if 'total_switching_time' in df.columns:
        df['switching_time_ns'] = df['total_switching_time'] * 1e9
    else:
        print("Error: 'total_switching_time' not found after validation. Plotting may fail.")
        df['switching_time_ns'] = 10.0 

    print(f"Generated {len(df)} data points, including {len(recommendations)} recommendation points")
    
    target_energy_pJ = target_energy * 1e12
    target_switching_time_ns = target_switching_time * 1e9
    
    try:
        colors = {m: MATERIAL_PALETTE[i % len(MATERIAL_PALETTE)] for i, m in enumerate(materials)}
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_axes([0.12, 0.12, 0.85, 0.7])
        for material in materials:
            material_data = df[df['material'] == material]
            
            normal_points = material_data[~(material_data['is_recommendation'] | material_data['is_other_recommendation'])]
            
            if not normal_points.empty:
                sizes = normal_points['endurance'].apply(calculate_size_for_endurance)
                
                ax.scatter(
                    normal_points['energy'], 
                    normal_points['switching_time_ns'],
                    s=sizes,
                    color=colors[material]['fill'],
                    edgecolor=colors[material]['edge'],
                    alpha=0.7,
                    linewidth=1.5,
                    label=material
                )
            
            rec_points = material_data[material_data['is_recommendation']]
            if not rec_points.empty:
                sizes = rec_points['endurance'].apply(lambda x: calculate_size_for_endurance(x) * 1.5)
                
                ax.scatter(
                    rec_points['energy'], 
                    rec_points['switching_time_ns'],
                    s=sizes,
                    color=colors[material]['fill'],
                    edgecolor=colors[material]['edge'],
                    alpha=0.9,
                    linewidth=2,
                    marker='*',
                    zorder=4
                )
            
            other_points = material_data[material_data['is_other_recommendation']]
            if not other_points.empty:
                sizes = other_points['endurance'].apply(calculate_size_for_endurance) * 0.5
                
                ax.scatter(
                    other_points['energy'],
                    other_points['switching_time_ns'],
                    s=sizes,
                    color=colors[material]['fill'],
                    edgecolor=colors[material]['edge'],
                    alpha=0.9,
                    linewidth=2,
                    marker='D',
                    zorder=3
                )
        
        # Compute Pareto front via dominance check
        df_non_dominated = df.copy()
        df_non_dominated['dominated'] = False

        for i, row_i in df.iterrows():
            for j, row_j in df.iterrows():
                if i != j:
                    if (row_j['energy'] <= row_i['energy'] and row_j['switching_time_ns'] <= row_i['switching_time_ns']
                        and (row_j['energy'] < row_i['energy'] or row_j['switching_time_ns'] < row_i['switching_time_ns'])):
                        df_non_dominated.loc[i, 'dominated'] = True
                        break
        
        pareto_points = df_non_dominated[df_non_dominated.get('dominated', False) != True]
        pareto_points = pareto_points.sort_values(by='energy')
        if not pareto_points.empty:
            ax.plot(
                pareto_points['energy'], 
                pareto_points['switching_time_ns'],
                color=COLORS['pareto'],
                linestyle='--',
                linewidth=1.5,
                label='Pareto Front'
            )
        
        ax.axhline(
            y=target_switching_time_ns,
            color=COLORS['target'],
            linestyle=':',
            linewidth=2,
            alpha=0.8
        )
        ax.axvline(
            x=target_energy_pJ,
            color=COLORS['target'],
            linestyle=':',
            linewidth=2,
            alpha=0.8
        )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (pJ)')
        ax.set_ylabel('Latency (ns)')
        
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.xaxis.set_major_formatter(PlainLogFormatter())
        ax.xaxis.set_minor_formatter(PlainLogFormatter())
        
        x_min, x_max, y_min, y_max = get_safe_axis_limits(df, target_energy_pJ, target_switching_time_ns)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        try:
            points = np.column_stack([df['energy'], df['switching_time_ns']])
            radii_pt = 0.5 * np.sqrt(df['endurance'].apply(calculate_size_for_endurance).to_numpy()
                                     * np.where(df['is_recommendation'], 1.5, 1.0)) + 1
            radii = radii_pt * fig.dpi / 72
            x_line, y_line = ax.transData.transform((target_energy_pJ, target_switching_time_ns))
            pad = fig.dpi / 72
            labels = [Bbox.from_extents(x_line - pad, ax.bbox.y0, x_line + pad, ax.bbox.y1),
                      Bbox.from_extents(ax.bbox.x0, y_line - pad, ax.bbox.x1, y_line + pad)]
            style = dict(textcoords='offset points', annotation_clip=False, fontsize=16, fontweight='bold')
            recommended = (df['is_recommendation'] | df['is_other_recommendation']).to_numpy(dtype=bool)
            for (_, row), gap in zip(df[recommended].iterrows(), radii_pt[recommended] + 3):
                labels.append(place_clear(
                    ax,
                    partial(ax.annotate, f"#{int(row['rec_num'])}", (row['energy'], row['switching_time_ns']), **style),
                    [dict(xytext=(gap, 0), ha='left', va='center'),
                     dict(xytext=(-gap, 0), ha='right', va='center'),
                     dict(xytext=(0, gap), ha='center', va='bottom'),
                     dict(xytext=(0, -gap), ha='center', va='top')],
                    points, radii, labels
                ))
            energy_anchors = [(y_min, 6, 'bottom'), (y_max, -6, 'top'),
                              *((y_min * (y_max / y_min) ** f, 0, 'center') for f in (0.25, 0.5, 0.75))]
            labels.append(place_clear(
                ax,
                partial(ax.annotate, 'Target Energy', color=COLORS['target'], **style),
                [dict(xy=(target_energy_pJ, y), xytext=(dx, dy), ha=ha, va=va, rotation=90)
                 for dx, ha in ((6, 'left'), (-6, 'right')) for y, dy, va in energy_anchors]
                + [dict(xy=(target_energy_pJ, y), xytext=(dx, dy), ha=ha, va=va, rotation=0)
                   for y, dy, va in ((y_max, -6, 'top'), (y_min, 6, 'bottom')) for dx, ha in ((6, 'left'), (-6, 'right'))],
                points, radii, labels
            ))
            latency_anchors = [(x_min, 6, 'left'), (target_energy_pJ, -6, 'right'), (target_energy_pJ, 6, 'left'), (x_max, -6, 'right'),
                               *((x_min * (x_max / x_min) ** f, 0, 'center') for f in (0.25, 0.5, 0.75))]
            place_clear(
                ax,
                partial(ax.annotate, 'Target Latency', color=COLORS['target'], **style),
                [dict(xy=(x, target_switching_time_ns), xytext=(dx, dy), ha=ha, va=va)
                 for dy, va in ((4, 'bottom'), (-4, 'top')) for x, dx, ha in latency_anchors],
                points, radii, labels
            )
        except Exception as e:
            print(f"Warning: Could not add labels: {e}")

        ax.grid(True, linestyle='--', alpha=0.3)
        
        legend_elements = [
            *[Line2D([0], [0], marker='o', color='w', label=material,
                     markerfacecolor=colors[material]['fill'], markeredgecolor=colors[material]['edge'], markersize=12)
              for material in materials],
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='Best Overall', linestyle='None'),
        ]
        if df['is_other_recommendation'].any():
            legend_elements.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10,
                                          label='Other Recommendations', linestyle='None'))
        
        ax.legend(
            handles=legend_elements, 
            loc='lower right',
            bbox_to_anchor=(0.65, 1.02),
            ncol=2,
            framealpha=0.9,
            columnspacing=1.0
        )
        
        ax_endurance = fig.add_axes([0.7, 0.85, 0.25, 0.1], frame_on=False)
        ax_endurance.set_title('Endurance (cycles)', fontsize=16)
        ax_endurance.set_xlim(0, 1)
        ax_endurance.set_ylim(0, 1)
        ax_endurance.set_axis_off()
        # Up to four evenly spaced whole decades spanning the plotted endurance range.
        log_endurance = np.log10(df['endurance'])
        low, high = int(np.floor(log_endurance.min())), int(np.ceil(log_endurance.max()))
        step = max(1, int(np.ceil((high - low) / 3)))
        decades = np.arange(low, high + step, step)
        endurance_values = [10.0 ** d for d in decades]
        endurance_labels = [f'$10^{{{d}}}$' for d in decades]
        x_positions = np.linspace(0.1, 0.9, len(endurance_values))
        
        for x, value, label in zip(x_positions, endurance_values, endurance_labels):
            size = calculate_size_for_endurance(value)
            ax_endurance.scatter(x, 0.5, s=size, color='gray', edgecolor='darkgray')
            ax_endurance.text(x, 0.001, label, ha='center', va='center', fontsize=18)
        
        try:
            device_name = type(physics).__module__.split('.')[0]
            png_path = os.path.join(output_dir, f'{device_name}_pareto_front.png')
            pdf_path = os.path.join(output_dir, f'{device_name}_pareto_front.pdf')
            
            plt.savefig(png_path, dpi=200, bbox_inches='tight')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Pareto front plot saved to: {png_path} and {pdf_path}")
        except ValueError as e:
            print(f"Error saving at high resolution: {e}")
            try:
                plt.figure(figsize=(6, 4.5), dpi=100)
                plt.savefig(png_path, dpi=100, bbox_inches='tight')
                print(f"Saved lower resolution plot to: {png_path}")
            except Exception as e2:
                print(f"Failed to save plot: {e2}")
        
        return fig, ax, df
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None, None, df 