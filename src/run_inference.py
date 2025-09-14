#!/usr/bin/env python
"""
run_all_folds_inference.py: Orchestrates inference runs for ALL folds.
- NO command-line arguments are needed.
- Edit the CONFIG dictionary below to set your paths.
- The script automatically finds the model directory for each fold.
"""
import os
import sys
import json
import pandas as pd
import subprocess
import re

# ==============================================================================
# --- CONFIGURATION: EDIT THESE PATHS ---
# ==============================================================================
CONFIG = {
    # Directory containing all your trained model folders (e.g., run_..._fold0, run_..._fold1)
    "BASE_MODEL_DIR": "../results/PMBind_best/",
    # Base directory where all inference output will be saved.
    "BASE_OUTPUT_FOLDER": "../results/PMBind_results",
    # Root directory of the cross-validation dataset.
    "DATA_ROOT": "../data/cross_validation_dataset",
    # Directory containing the ESM embedding files (.npz and .csv).
    "EMBEDDING_DIR": "../results/ESM/esm3-open/PMGen_whole_seq_/",
    # Full path to the aligned allele sequence CSV file.
    "ALLELE_SEQ_PATH": "../data/alleles/aligned_PMGen_class_1.csv",
    # List of folds to run inference on. range(5) means folds 1, 2, 3, 4, 5
    "FOLDS_TO_RUN": range(1, 2),
    # MHC Class (1 or 2)
    "MHC_CLASS": 1
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


def run_inference_for_fold(model_dir, base_output_folder, fold):
    """Sets up paths and orchestrates calls to simple_infer.py for a single fold."""

    mhc_class_str = f"mhc{CONFIG['MHC_CLASS']}"
    config_path = os.path.join(model_dir, "config.json")
    model_weights_path = os.path.join(model_dir, "best_model.weights.h5")

    if not os.path.exists(model_weights_path):
        print(f"ERROR: Model weights not found at {model_weights_path}. Skipping.")
        return

    fold_dir_base = os.path.join(CONFIG['DATA_ROOT'], mhc_class_str, "cv_folds")

    paths = {
        "train": os.path.join(fold_dir_base, f"fold_{fold:02d}_train.parquet"),
        "val": os.path.join(fold_dir_base, f"fold_{fold:02d}_val.parquet"),
        "test": os.path.join(os.path.dirname(fold_dir_base), "test_set_rarest_alleles.parquet"),
        "tval": os.path.join(os.path.dirname(fold_dir_base), "tval_set_rarest_alleles.parquet"),
        "embed_npz": os.path.join(CONFIG['EMBEDDING_DIR'], f"{mhc_class_str}_encodings.npz"),
        "embed_key": os.path.join(CONFIG['EMBEDDING_DIR'], f"{mhc_class_str}_encodings.csv"),
    }

    # --- Run Inference on Standard Datasets ---
    for dset_name in ["test", "tval", "val", "train"]:
        print(f"\n--- Starting Inference on {dset_name.upper()} Set ---")
        data_path = paths[dset_name]

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
            "python", "infer.py",
            "--model_weights_path", model_weights_path, "--config_path", config_path,
            "--df_path", data_path, "--out_dir", infer_out_dir, "--name", dset_name,
            "--allele_seq_path", CONFIG['ALLELE_SEQ_PATH'],
            "--embedding_key_path", paths["embed_key"], "--embedding_npz_path", paths["embed_npz"],
        ]
        subprocess.run(cmd, check=True)

    # --- Run Joint Inference on Train + Test ---
    print(f"\n--- Starting Joint Inference on TRAIN+TEST ---")
    df_train = pd.read_parquet(paths["train"])
    df_train['source'] = 'train'
    df_test = pd.read_parquet(paths["test"])
    df_test['source'] = 'test'
    df_joint = pd.concat([df_train, df_test], ignore_index=True)
    joint_data_path = os.path.join(base_output_folder, "joint_train_test_data.parquet")
    df_joint.to_parquet(joint_data_path)

    joint_infer_out_dir = os.path.join(base_output_folder, "inference_train_test_joint")
    cmd = [
        "python", "infer.py",
        "--model_weights_path", model_weights_path, "--config_path", config_path,
        "--df_path", joint_data_path, "--out_dir", joint_infer_out_dir, "--name", "train_test_joint",
        "--allele_seq_path", CONFIG['ALLELE_SEQ_PATH'],
        "--embedding_key_path", paths["embed_key"], "--embedding_npz_path", paths["embed_npz"],
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    os.makedirs(CONFIG["BASE_OUTPUT_FOLDER"], exist_ok=True)

    for fold_number in CONFIG["FOLDS_TO_RUN"]:
        print(f"\n{'=' * 80}")
        print(f"STARTING INFERENCE FOR FOLD {fold_number}")
        print(f"{'=' * 80}")

        # Find the model directory for the current fold
        model_directory = find_model_dir_for_fold(CONFIG["BASE_MODEL_DIR"], fold_number)

        if not model_directory:
            print(
                f"WARNING: Could not find a trained model directory for fold {fold_number} in '{CONFIG['BASE_MODEL_DIR']}'. Skipping.")
            continue

        print(f"Found model for fold {fold_number} at: {model_directory}")

        # Setup a specific output directory and log file for this fold
        fold_output_folder = os.path.join(CONFIG["BASE_OUTPUT_FOLDER"], f"inference_run_fold_{fold_number}")
        os.makedirs(fold_output_folder, exist_ok=True)

        log_file_path = os.path.join(fold_output_folder, "orchestrator.log")
        original_stdout = sys.stdout

        try:
            with open(log_file_path, 'w') as log_file:
                sys.stdout = Tee(original_stdout, log_file)
                run_inference_for_fold(
                    model_dir=model_directory,
                    base_output_folder=fold_output_folder,
                    fold=fold_number
                )
        finally:
            sys.stdout = original_stdout
            print(f"Completed inference for fold {fold_number}. Log saved to {log_file_path}")

    print(f"\n{'=' * 80}")
    print("ALL FOLDS PROCESSED.")
    print(f"{'=' * 80}")