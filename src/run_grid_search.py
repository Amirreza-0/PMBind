#!/usr/bin/env python
"""
Simplified Parameter Grid Search Script for PMBind
Searches for optimal label_smoothing and asymmetry_strength parameters by calling run_training_down.py
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from pathlib import Path
import glob
import time

def run_single_training(fold, label_smoothing, asymmetry_strength, epochs=20, subset=1.0):
    """
    Run a single training with specific parameters by calling run_training_down.py

    Returns:
        dict: Results containing validation metrics
    """

    print(f"\n{'='*80}")
    print(f"Training: fold={fold}, label_smoothing={label_smoothing:.3f}, asymmetry_strength={asymmetry_strength:.3f}")
    print(f"{'='*80}")

    try:
        # Build command to run training
        cmd = [
            "python", "src/run_training_down.py",
            "--fold", str(fold),
            "--ls_param", str(label_smoothing),
            "--as_param", str(asymmetry_strength),
            "--subset", str(subset)
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run the training with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        # Find the results directory created by the training script
        results_pattern = f"../results/PMBind_runs_optimized4/run_*_fold{fold}"
        result_dirs = glob.glob(results_pattern)

        if not result_dirs:
            print(f"No results directory found for fold {fold}")
            return None

        # Get the most recent results directory
        latest_dir = max(result_dirs, key=os.path.getctime)
        print(f"Found results directory: {latest_dir}")

        # Read training history
        history_file = os.path.join(latest_dir, "training_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

            # Extract best validation metrics
            best_val_mcc = max(history['val_mcc']) if history['val_mcc'] else 0.0
            best_val_auc = max(history['val_auc']) if history['val_auc'] else 0.0
            final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
            final_val_f1 = history['val_f1'][-1] if history['val_f1'] else 0.0

            result_data = {
                'label_smoothing': label_smoothing,
                'asymmetry_strength': asymmetry_strength,
                'fold': fold,
                'best_val_mcc': best_val_mcc,
                'best_val_auc': best_val_auc,
                'final_val_acc': final_val_acc,
                'final_val_f1': final_val_f1,
                'epochs_trained': len(history['val_mcc']),
                'output_dir': latest_dir
            }

            print(f"✓ Training completed successfully:")
            print(f"  Best Val MCC: {best_val_mcc:.4f}")
            print(f"  Best Val AUC: {best_val_auc:.4f}")
            print(f"  Epochs trained: {len(history['val_mcc'])}")

            return result_data
        else:
            print(f"No training history found at {history_file}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Training timed out for parameters: ls={label_smoothing}, as={asymmetry_strength}")
        return None
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def plot_grid_results(results_df, output_dir):
    """
    Create comprehensive heatmap plots of the grid search results.
    """

    # Aggregate results by parameter combination (mean across fold)
    agg_results = results_df.groupby(['label_smoothing', 'asymmetry_strength']).agg({
        'best_val_mcc': ['mean', 'std', 'count'],
        'best_val_auc': ['mean', 'std'],
        'final_val_acc': ['mean', 'std'],
        'final_val_f1': ['mean', 'std']
    }).round(4)

    # Create pivot tables for heatmaps
    pivot_mcc = results_df.groupby(['label_smoothing', 'asymmetry_strength'])['best_val_mcc'].mean().reset_index()
    pivot_table_mcc = pivot_mcc.pivot(index='asymmetry_strength', columns='label_smoothing', values='best_val_mcc')

    pivot_auc = results_df.groupby(['label_smoothing', 'asymmetry_strength'])['best_val_auc'].mean().reset_index()
    pivot_table_auc = pivot_auc.pivot(index='asymmetry_strength', columns='label_smoothing', values='best_val_auc')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MCC Heatmap
    sns.heatmap(pivot_table_mcc, annot=True, fmt='.4f', cmap='viridis',
                ax=axes[0,0], cbar_kws={'label': 'Mean Validation MCC'})
    axes[0,0].set_title('Mean Validation MCC Across Fold', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Label Smoothing', fontsize=12)
    axes[0,0].set_ylabel('Asymmetry Strength', fontsize=12)

    # Plot 2: AUC Heatmap
    sns.heatmap(pivot_table_auc, annot=True, fmt='.4f', cmap='plasma',
                ax=axes[0,1], cbar_kws={'label': 'Mean Validation AUC'})
    axes[0,1].set_title('Mean Validation AUC Across Fold', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Label Smoothing', fontsize=12)
    axes[0,1].set_ylabel('Asymmetry Strength', fontsize=12)

    # Plot 3: Label Smoothing Effect
    ls_effect = results_df.groupby('label_smoothing')['best_val_mcc'].agg(['mean', 'std'])
    axes[1,0].errorbar(ls_effect.index, ls_effect['mean'], yerr=ls_effect['std'],
                      marker='o', linewidth=2, capsize=5, capthick=2)
    axes[1,0].set_xlabel('Label Smoothing', fontsize=12)
    axes[1,0].set_ylabel('Mean Validation MCC', fontsize=12)
    axes[1,0].set_title('Effect of Label Smoothing on Performance', fontsize=14, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Asymmetry Strength Effect
    as_effect = results_df.groupby('asymmetry_strength')['best_val_mcc'].agg(['mean', 'std'])
    axes[1,1].errorbar(as_effect.index, as_effect['mean'], yerr=as_effect['std'],
                      marker='s', color='orange', linewidth=2, capsize=5, capthick=2)
    axes[1,1].set_xlabel('Asymmetry Strength', fontsize=12)
    axes[1,1].set_ylabel('Mean Validation MCC', fontsize=12)
    axes[1,1].set_title('Effect of Asymmetry Strength on Performance', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grid_search_results.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'grid_search_results.pdf'), bbox_inches='tight')
    plt.show()

    # Create detailed results table
    detailed_table = agg_results[('best_val_mcc', 'mean')].reset_index()
    detailed_table.columns = ['Label_Smoothing', 'Asymmetry_Strength', 'Mean_Val_MCC']
    detailed_table['Std_Val_MCC'] = agg_results[('best_val_mcc', 'std')].values
    detailed_table['Count'] = agg_results[('best_val_mcc', 'count')].values
    detailed_table = detailed_table.sort_values('Mean_Val_MCC', ascending=False)

    # Find and print best parameters
    best_row = detailed_table.iloc[0]

    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"BEST PARAMETERS:")
    print(f"  Label Smoothing: {best_row['Label_Smoothing']}")
    print(f"  Asymmetry Strength: {best_row['Asymmetry_Strength']}")
    print(f"  Mean Validation MCC: {best_row['Mean_Val_MCC']:.4f} ± {best_row['Std_Val_MCC']:.4f}")
    print(f"  Number of fold: {int(best_row['Count'])}")
    print(f"\nTOP 5 PARAMETER COMBINATIONS:")
    print(detailed_table.head().to_string(index=False))
    print(f"{'='*80}")

    # Save detailed results
    detailed_table.to_csv(os.path.join(output_dir, 'ranked_results.csv'), index=False)

    return best_row['Label_Smoothing'], best_row['Asymmetry_Strength']

def main():
    parser = argparse.ArgumentParser(description="Parameter grid search for PMBind")
    parser.add_argument("--output_dir", type=str, default="./grid_search_results",
                       help="Directory to save results")
    parser.add_argument("--fold", type=int, default=1,
                          help="Single fold to run")
    parser.add_argument("--subset", type=float, default=1.0,
                       help="Subset percentage of training data to use")
    parser.add_argument("--ls_params", type=float, nargs="+",
                       default=[0.05, 0.1, 0.15, 0.2],
                       help="Label smoothing values to test")
    parser.add_argument("--as_params", type=float, nargs="+",
                       default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                       help="Asymmetry strength values to test")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate parameter combinations
    param_combinations = list(product(args.ls_params, args.as_params))
    total_runs = len(param_combinations)

    print(f"Starting parameter grid search:")
    print(f"  Label smoothing values: {args.ls_params}")
    print(f"  Asymmetry strength values: {args.as_params}")
    print(f"  Fold: {args.fold}")
    print(f"  Parameter combinations: {len(param_combinations)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Using subset: {args.subset}")

    # Save configuration
    config = {
        'ls_params': args.ls_params,
        'as_params': args.as_params,
        'fold': args.fold,
        'subset': args.subset,
        'total_combinations': len(param_combinations),
        'total_runs': total_runs,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Run grid search
    results = []
    run_count = 0

    print(f"\n{'='*100}")
    print(f"PROCESSING FOLD {fold}")
    print(f"{'='*100}")

    for label_smoothing, asymmetry_strength in param_combinations:
        run_count += 1
        print(f"\n[Run {run_count}/{total_runs}]")

        result = run_single_training(
            fold=args.fold,
            label_smoothing=label_smoothing,
            asymmetry_strength=asymmetry_strength,
            subset=args.subset
        )

        if result:
            results.append(result)
            # Save intermediate results
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_csv(os.path.join(args.output_dir, 'intermediate_results.csv'), index=False)
        else:
            print(f"  ✗ Training failed")

    # Process and analyze results
    if results:
        print(f"\nGrid search completed with {len(results)} successful runs out of {total_runs} total runs")

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)

        # Create summary statistics
        summary = results_df.groupby(['label_smoothing', 'asymmetry_strength']).agg({
            'best_val_mcc': ['mean', 'std', 'count'],
            'best_val_auc': ['mean', 'std'],
            'final_val_acc': ['mean', 'std'],
            'final_val_f1': ['mean', 'std'],
            'epochs_trained': 'mean'
        }).round(4)

        summary.to_csv(os.path.join(args.output_dir, 'summary_results.csv'))

        # Create plots and find best parameters
        best_ls, best_as = plot_grid_results(results_df, args.output_dir)

        # Save final results summary
        final_summary = {
            'total_runs_attempted': total_runs,
            'successful_runs': len(results),
            'success_rate': len(results) / total_runs,
            'best_label_smoothing': best_ls,
            'best_asymmetry_strength': best_as,
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=4)

        print(f"\nAll results saved to: {args.output_dir}")
        print(f"Success rate: {len(results)}/{total_runs} ({100*len(results)/total_runs:.1f}%)")

    else:
        print("No successful training runs! Check your setup and try again.")

if __name__ == "__main__":
    main()