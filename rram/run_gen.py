import argparse
import os
import torch

from src.cvae.generative_model import RRAMParameterRecommender, train_cvae
from src.cvae.metric_eval import RRAMEvaluator
from src.cvae.pareto_front_rram import generate_pareto_plot

def main():
    parser = argparse.ArgumentParser(description='RRAM CVAE Model and Pareto Front Generation Tool')

    parser.add_argument('--model_path', type=str, default='checkpoints/pinn_sparse.pth', help='Path to the trained PINN model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/rram_data.mat', help='Path to the RRAM dataset')
    parser.add_argument('--output_dir', type=str, default='recommendation_results', help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation (cuda/cpu)')
    parser.add_argument('--generate_pareto', action='store_true', help='Generate and save the Pareto front plot.')

    cvae_group = parser.add_argument_group('CVAE Training and Recommendation')
    cvae_group.add_argument('--train_cvae', action='store_true', help='Train a new CVAE model')
    cvae_group.add_argument('--create_new_dataset', action='store_true', help='Create a new dataset for CVAE training')
    cvae_group.add_argument('--cvae_model_path', type=str, default='recommendation_results/cvae_model.pth', help='Path to CVAE model')
    cvae_group.add_argument('--dataset_path', type=str, default='recommendation_results/rram_cvae_dataset.pt', help='Path to CVAE dataset')
    cvae_group.add_argument('--cvae_epochs', type=int, default=3000, help='Number of epochs for CVAE training')
    cvae_group.add_argument('--early_stopping_patience', type=int, default=100, help='Patience for early stopping (epochs)')
    cvae_group.add_argument('--early_stopping_delta', type=float, default=1e-5, help='Minimum change in loss to qualify as improvement for early stopping')
    
    rec_group = parser.add_argument_group('Recommendation and Pareto Plot Parameters')
    rec_group.add_argument('--target_endurance', type=float, default=1e6, help='Target endurance (cycles)')
    rec_group.add_argument('--target_switching_time', type=float, default=10e-9, help='Target switching time (s), default 10ns')
    rec_group.add_argument('--target_energy', type=float, default=5e-10, help='Target energy consumption (J)')
    rec_group.add_argument('--min_pulse_width', type=float, default=1e-9, help='Minimum pulse width for recommendations (default: 1ns)')
    rec_group.add_argument('--diverse_candidates', type=int, default=50, help='Number of diverse candidates for recommendation generation')
    rec_group.add_argument('--energy_penalty_factor', type=float, default=3.0, help='Penalty factor for exceeding target energy in recommendations.')
    rec_group.add_argument('--max_energy_error_ratio', type=float, default=1.0, help='Maximum allowed ratio of predicted energy over target energy. Set to -1 to disable.')
    rec_group.add_argument('--num_recommendations', type=int, default=4, help='Number of recommendations to generate')
    rec_group.add_argument('--num_sample_points', type=int, default=50, help='Number of sample points per material for Pareto plot (max 100)')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.generate_pareto:
        print("--- Generating Pareto Front Plot ---")
        generate_pareto_plot(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            cvae_model_path=args.cvae_model_path,
            dataset_path=args.dataset_path,
            target_switching_time=args.target_switching_time,
            target_energy=args.target_energy,
            target_endurance=args.target_endurance,
            num_sample_points=args.num_sample_points,
            energy_penalty_factor=args.energy_penalty_factor,
            max_energy_error_ratio=args.max_energy_error_ratio,
            min_pulse_width=args.min_pulse_width,
            diverse_candidates=args.diverse_candidates
        )
    else:
        print("--- Running CVAE Parameter Recommendation ---")
        device = torch.device(args.device)
        evaluator = RRAMEvaluator(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=device
        )

        if args.train_cvae:
            print("Training CVAE model...")
            train_cvae(
                evaluator,
                num_epochs=args.cvae_epochs,
                dataset_path=args.dataset_path,
                model_save_path=args.cvae_model_path,
                create_new_dataset=args.create_new_dataset,
                min_pulse_width=args.min_pulse_width,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_delta=args.early_stopping_delta
            )
            print(f"CVAE model training complete. Model saved to {args.cvae_model_path}")

        print("\nRecommending parameters for target performance...")
        recommender = RRAMParameterRecommender(
            args.cvae_model_path,
            evaluator,
            dataset_path=args.dataset_path,
            min_pulse_width=args.min_pulse_width
        )
        recommendations = recommender.recommend_parameters(
            args.target_endurance,
            args.target_switching_time,
            args.target_energy,
            num_recommendations=args.num_recommendations,
            num_samples=args.diverse_candidates,
            energy_penalty_factor=args.energy_penalty_factor,
            max_energy_error_ratio=float('inf') if args.max_energy_error_ratio < 0 else args.max_energy_error_ratio
        )

        if recommendations:
            import json
            print("\n--- Top Recommendations ---")
            for i, rec in enumerate(recommendations):
                print(f"  #{i+1}: {rec['material']} - Vset={rec['pos_voltage']:.2f}V, Vreset={rec['neg_voltage']:.2f}V")
                print(f"     Endurance: {rec['predicted_performance']['endurance']:.2e} cycles")
                print(f"     Latency: {rec['predicted_performance']['total_switching_time']*1e9:.2f} ns")
                print(f"     Energy: {rec['predicted_performance']['energy']*1e12:.2f} pJ")

            filepath = os.path.join(args.output_dir, 'recommendations.json')
            with open(filepath, 'w') as f:
                json.dump(recommendations, f, indent=4)
            print(f"\nSaved recommendations to {filepath}")
        else:
            print("No suitable recommendations found for the given targets.")


if __name__ == "__main__":
    main()
