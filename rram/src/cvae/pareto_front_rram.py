import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import torch
import os
import matplotlib.ticker as ticker

from .generative_model import RRAMParameterRecommender
from .metric_eval import RRAMEvaluator
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

COLORS = {
    'HfO2': {'fill': '#FBD178', 'edge': '#EE8227'},
    'TiO2': {'fill': '#BFE4EE', 'edge': '#79ADD6'},
    'Al2O3': {'fill': '#F9C3BF', 'edge': '#EB716B'},
    'pareto': '#556B2F',
    'target': '#1E803D'
}
ENERGY_MIN_VALID = 10       # pJ
ENERGY_MAX_VALID = 5000     # pJ
TIME_MIN_VALID = 0.1e-9     # s (0.1 ns)
TIME_MAX_VALID = 100e-9     # s (100 ns)
ENDURANCE_MIN_VALID = 1e3   # cycles
ENDURANCE_MAX_VALID = 1e12  # cycles

def validate_data(df):
    """Clean and validate data to prevent plotting issues, using switching time."""
    df_clean = df.copy()
    if 'total_switching_time' not in df_clean.columns:
        print("Error: 'total_switching_time' column not found in DataFrame for validation.")
        df_clean['total_switching_time'] = 10e-9
    if 'is_recommendation' in df_clean.columns:
        df_clean['is_recommendation'] = df_clean['is_recommendation'].fillna(False)
    if 'is_other_recommendation' in df_clean.columns:
        df_clean['is_other_recommendation'] = df_clean['is_other_recommendation'].fillna(False)
    
    df_clean = df_clean[
        (df_clean['energy'] >= ENERGY_MIN_VALID) & 
        (df_clean['energy'] <= ENERGY_MAX_VALID) &
        (df_clean['total_switching_time'] >= TIME_MIN_VALID) & 
        (df_clean['total_switching_time'] <= TIME_MAX_VALID) & 
        (df_clean['endurance'] >= ENDURANCE_MIN_VALID) & 
        (df_clean['endurance'] <= ENDURANCE_MAX_VALID) &
        np.isfinite(df_clean['energy']) &
        np.isfinite(df_clean['total_switching_time']) &
        np.isfinite(df_clean['endurance'])
    ]
    
    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} data points with invalid values")
        
    if len(df_clean) == 0:
        print("Error: No valid data points for plotting")
        return pd.DataFrame({
            'material': ['HfO2'],
            'energy': [1000],
            'total_switching_time': [10e-9],
            'endurance': [1e6],
            'is_recommendation': [False],
            'is_other_recommendation': [False]
        })
    
    return df_clean

def get_safe_axis_limits(df, target_energy_pJ, target_switching_time_ns):
    """Calculate safe axis limits for energy and switching time."""
    DEFAULT_ENERGY_MIN = 150
    DEFAULT_ENERGY_MAX = 2000
    DEFAULT_TIME_MIN_NS = 1
    DEFAULT_TIME_MAX_NS = 50 
    
    try:
        if 'switching_time_ns' not in df.columns:
             print("Warning: 'switching_time_ns' not found in DataFrame for axis limits. Using defaults.")
             return DEFAULT_ENERGY_MIN, DEFAULT_ENERGY_MAX, DEFAULT_TIME_MIN_NS, DEFAULT_TIME_MAX_NS
        if len(df) > 0:
            x_min = max(np.percentile(df['energy'], 1), DEFAULT_ENERGY_MIN / 2)
            x_max = min(np.percentile(df['energy'], 99) * 1.2, DEFAULT_ENERGY_MAX * 2)
            y_min = max(np.percentile(df['switching_time_ns'], 1) / 1.2, DEFAULT_TIME_MIN_NS / 2)
            y_max = min(np.percentile(df['switching_time_ns'], 99) * 1.2, DEFAULT_TIME_MAX_NS * 2)
            
            if target_energy_pJ is not None:
                x_min = min(x_min, target_energy_pJ * 0.8)
                x_max = max(x_max, target_energy_pJ * 1.2)
            
            if target_switching_time_ns is not None:
                y_min = min(y_min, target_switching_time_ns * 0.8)
                y_max = max(y_max, target_switching_time_ns * 1.2)
            
            x_min = max(x_min, DEFAULT_ENERGY_MIN)
            x_max = min(x_max, DEFAULT_ENERGY_MAX)
            y_min = max(y_min, DEFAULT_TIME_MIN_NS)
            y_max = min(y_max, DEFAULT_TIME_MAX_NS)
            
            if x_max / x_min < 2:
                x_min = max(DEFAULT_ENERGY_MIN / 2, x_min / 2)
                x_max = min(DEFAULT_ENERGY_MAX * 2, x_max * 2)
            
            if y_max <= y_min:
                y_max = y_min * 2

            if y_max / y_min < 2:
                 y_min = max(DEFAULT_TIME_MIN_NS / 2, y_min / 2)
                 y_max = min(DEFAULT_TIME_MAX_NS * 2, y_max * 2)
                 if y_max <= y_min:
                     y_max = y_min * 1.1


            return x_min, x_max, y_min, y_max
            
    except Exception as e:
        print(f"Error calculating axis limits: {e}")
    
    return DEFAULT_ENERGY_MIN, DEFAULT_ENERGY_MAX, DEFAULT_TIME_MIN_NS, DEFAULT_TIME_MAX_NS

def calculate_size_for_endurance(endurance):
    """Calculate marker size based on endurance."""
    log_endurance = np.log10(endurance)
    size_factor = 8
    return size_factor * (log_endurance ** 2.2)

def generate_pareto_plot(model_path='checkpoints/pinn_sparse.pth',
                        data_path='data/rram_data.mat',
                        output_dir='recommendation_results',
                        cvae_model_path='recommendation_results/cvae_model.pth',
                        dataset_path='recommendation_results/rram_cvae_dataset.pt',
                        target_switching_time=10e-9, # s, default 10 ns
                        target_energy=5e-10,  # J, equivalent to 500 pJ
                        target_endurance=1e6,  # cycles
                        num_sample_points=50,
                        energy_penalty_factor=3.0,
                        max_energy_error_ratio=1.0,
                        min_pulse_width=1e-9,
                        diverse_candidates=40):
    """
    Generate Pareto front plot using the real RRAM model
    
    Parameters:
        model_path: Path to trained RRAM model
        data_path: Path to RRAM dataset
        output_dir: Output directory
        cvae_model_path: Path to CVAE model
        dataset_path: Path to CVAE dataset
        target_switching_time: Target switching time (s)
        target_energy: Target energy consumption (J)
        target_endurance: Target endurance (cycles)
        num_sample_points: Number of sample points to generate for each material
        energy_penalty_factor: Penalty factor for exceeding target energy in recommendations
        max_energy_error_ratio: Maximum allowed ratio of predicted energy over target energy for recommendations
        min_pulse_width: Minimum pulse width for recommendations (default: 1ns)
        diverse_candidates: Number of diverse candidates for recommendation generation (default: 40)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing RRAM evaluator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        evaluator = RRAMEvaluator(
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device
        )
    except Exception as e:
        print(f"Error initializing RRAM evaluator: {e}")
        return None, None, None
    
    print("Initializing parameter recommender...")
    try:
        recommender = RRAMParameterRecommender(
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
            num_recommendations=3,
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
    df = validate_data(df)
    if 'total_switching_time' in df.columns:
        df['switching_time_ns'] = df['total_switching_time'] * 1e9
    else:
        print("Error: 'total_switching_time' not found after validation. Plotting may fail.")
        df['switching_time_ns'] = 10.0 

    print(f"Generated {len(df)} data points, including {len(recommendations)} recommendation points")
    
    target_energy_pJ = target_energy * 1e12
    target_switching_time_ns = target_switching_time * 1e9
    
    try:
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
                    color=COLORS[material]['fill'],
                    edgecolor=COLORS[material]['edge'],
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
                    color=COLORS[material]['fill'],
                    edgecolor=COLORS[material]['edge'],
                    alpha=0.9,
                    linewidth=2,
                    marker='*'
                )
            
            recommendation_mask = material_data['is_recommendation'].fillna(False)
            for idx, row in material_data[recommendation_mask].iterrows():
                if row['is_recommendation'] and row.get('rec_num') == 1:
                    ax.annotate(
                        f"#1",
                        (row['energy'], row['switching_time_ns']),
                        xytext=(10, 0),
                        textcoords='offset points',
                        fontsize=16,
                        fontweight='bold'
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
        
        x_min, x_max, y_min, y_max = get_safe_axis_limits(df, target_energy_pJ, target_switching_time_ns)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        try:
            ax.text(
                target_energy_pJ * 1.05,
                y_min * 1.1,
                'Target Energy',
                color=COLORS['target'],
                fontsize=16,
                fontweight='bold',
                rotation=90,
                va='bottom'
            )
            ax.text(
                x_min * 1.05,
                target_switching_time_ns * 1.05,
                'Target Latency',
                color=COLORS['target'],
                fontsize=16,
                fontweight='bold',
                ha='left'
            )
        except Exception as e:
            print(f"Warning: Could not add target labels: {e}")
        
        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            axis.set_major_formatter(formatter)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='HfO2', 
                   markerfacecolor=COLORS['HfO2']['fill'], markeredgecolor=COLORS['HfO2']['edge'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='TiO2', 
                   markerfacecolor=COLORS['TiO2']['fill'], markeredgecolor=COLORS['TiO2']['edge'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Al2O3', 
                   markerfacecolor=COLORS['Al2O3']['fill'], markeredgecolor=COLORS['Al2O3']['edge'], markersize=12),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='Best Overall', linestyle='None'),
        ]
        
        main_legend = ax.legend(
            handles=legend_elements, 
            loc='upper center',
            bbox_to_anchor=(0.35, 1.2),
            ncol=2,
            framealpha=0.9,
            columnspacing=1.0
        )
        
        ax_endurance = fig.add_axes([0.7, 0.85, 0.25, 0.1], frame_on=False)
        ax_endurance.set_title('Endurance (cycles)', fontsize=16)
        ax_endurance.set_xlim(0, 1)
        ax_endurance.set_ylim(0, 1)
        ax_endurance.set_axis_off()
        endurance_values = [1e5, 1e6, 1e7, 1e8]
        endurance_labels = ['10⁵', '10⁶', '10⁷', '10⁸']
        x_positions = np.linspace(0.1, 0.9, len(endurance_values))
        
        for x, value, label in zip(x_positions, endurance_values, endurance_labels):
            size = calculate_size_for_endurance(value)
            ax_endurance.scatter(x, 0.5, s=size, color='gray', edgecolor='darkgray')
            ax_endurance.text(x, 0.001, label, ha='center', va='center', fontsize=18)
        
        try:
            png_path = os.path.join(output_dir, 'rram_pareto_front.png')
            pdf_path = os.path.join(output_dir, 'rram_pareto_front.pdf')
            
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