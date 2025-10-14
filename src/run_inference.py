#!/usr/bin/env python
"""
run_inference.py: Orchestrates inference runs for trained models.
- Loads paths from training_paths.json
- Supports running inference on training, validation, and benchmark datasets
- Automatically finds model directories and configurations
"""
import os
import sys
import json
import pandas as pd
import subprocess
import re

# ==============================================================================
# --- CONFIGURATION: LOAD FROM training_paths.json ---
# ==============================================================================
# Load paths from training_paths.json
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "infer_paths.json")

with open(config_path, "r") as f:
    PATHS_CONFIG = json.load(f)

# Configuration for inference
CONFIG = {
    # Directory containing all your trained model folders
    "BASE_MODEL_DIR": "/media/amirreza/Crucial-500/BASE_MODEL_DIR",
    # Base directory where all inference output will be saved
    "BASE_OUTPUT_FOLDER": "/media/amirreza/Crucial-500/BASE_MODEL_DIR/Inference_Results",
    # Paths loaded from training_paths.json
    "ALLELE_SEQ_PATH": PATHS_CONFIG["allele_seq_path"],
    "EMBEDDING_KEY_PATH": PATHS_CONFIG["embedding_key_path"],
    "EMBEDDING_NPZ_PATH": PATHS_CONFIG["embedding_table_path"],
    "TRAIN_PARQUET_PATH": PATHS_CONFIG["train_parquet_path"],
    "VAL_PARQUET_PATH": PATHS_CONFIG["val_parquet_path"],
    "BENCH1_PARQUET_PATH": PATHS_CONFIG["bench1_parquet_path"],
    "BENCH2_PARQUET_PATH": PATHS_CONFIG["bench2_parquet_path"],
    "BENCH3_PARQUET_PATH": PATHS_CONFIG["bench3_parquet_path"],
    # MHC Class (1 or 2)
    "MHC_CLASS": 1,
    # Batch size for inference
    "BATCH_SIZE": 10
}


# ==============================================================================

class Tee:
    """Helper class to redirect stdout to both console and file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def find_model_dir_for_fold(base_dir, fold_num):
    """Finds the model directory corresponding to a specific fold."""
    for dirname in os.listdir(base_dir):
        path = os.path.join(base_dir, dirname)
        if os.path.isdir(path) and f"_fold{fold_num}" in dirname:
            return path
    return None


def run_inference_for_model(model_dir, base_output_folder):
    """Sets up paths and orchestrates calls to infer.py for a trained model."""

    config_path = os.path.join(model_dir, "run_config.json")
    # Check for both keras and weights.h5 formats
    model_weights_path = os.path.join(model_dir, "best_model.weights.h5")
    if not os.path.exists(model_weights_path):
        model_weights_path = os.path.join(model_dir, "best_model.keras")

    if not os.path.exists(model_weights_path):
        print(f"ERROR: Model weights not found at {model_weights_path}. Skipping.")
        return

    # Use paths from training_paths.json
    paths = {
        "train": CONFIG["TRAIN_PARQUET_PATH"],
        "val": CONFIG["VAL_PARQUET_PATH"],
        "bench1": CONFIG["BENCH1_PARQUET_PATH"],
        "bench2": CONFIG["BENCH2_PARQUET_PATH"],
        "bench3": CONFIG["BENCH3_PARQUET_PATH"],
        "embed_npz": CONFIG["EMBEDDING_NPZ_PATH"],
        "embed_key": CONFIG["EMBEDDING_KEY_PATH"],
        "allele_seq": CONFIG["ALLELE_SEQ_PATH"],
    }

    # --- Run Inference on Standard Datasets ---
    for dset_name in ["bench1", "bench2", "bench3", "val", "train"]:
        print(f"\n--- Starting Inference on {dset_name.upper()} Set ---")
        data_path = paths[dset_name]

        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found at {data_path}. Skipping {dset_name}.")
            continue

        # Create subset for large files
        if dset_name in ["train", "val"]:
            df_full = pd.read_parquet(data_path)
            if len(df_full) > 100000:
                print(f"Creating a stratified subset of 100k samples for {dset_name}...")
                df_subset = df_full.groupby('assigned_label', group_keys=False).apply(
                    lambda x: x.sample(n=50000, random_state=42) if len(x) > 50000 else x
                ).reset_index(drop=True)
                subset_path = os.path.join(base_output_folder, f"{dset_name}_subset.parquet")
                df_subset.to_parquet(subset_path)
                data_path = subset_path

        infer_out_dir = os.path.join(base_output_folder, f"inference_{dset_name}")

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "infer.py"),
            "--model_weights_path", model_weights_path, "--config_path", config_path,
            "--df_path", data_path, "--out_dir", infer_out_dir, "--name", dset_name,
            "--allele_seq_path", paths["allele_seq"],
            "--embedding_key_path", paths["embed_key"], "--embedding_npz_path", paths["embed_npz"],
            "--batch_size", str(CONFIG["BATCH_SIZE"])
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on trained PMBind models")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory containing run_config.json and best_model weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for inference results (default: BASE_OUTPUT_FOLDER/model_name)")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Get model name from directory
    model_name = os.path.basename(model_dir)

    # Setup output directory
    if args.output_dir:
        output_folder = os.path.abspath(args.output_dir)
    else:
        output_folder = os.path.join(CONFIG["BASE_OUTPUT_FOLDER"], model_name)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"STARTING INFERENCE FOR MODEL: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_folder}")
    print(f"{'=' * 80}")

    log_file_path = os.path.join(output_folder, "inference.log")
    original_stdout = sys.stdout

    try:
        with open(log_file_path, 'w') as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            run_inference_for_model(
                model_dir=model_dir,
                base_output_folder=output_folder
            )
    finally:
        sys.stdout = original_stdout

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print(f"Results saved to: {output_folder}")
    print("=" * 80)