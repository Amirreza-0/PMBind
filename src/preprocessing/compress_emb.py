import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_embedding_db(npz_path: str):
    """Load embedding database with memory mapping."""
    try:
        # Try loading without explicit allow_pickle (defaults to True)
        return np.load(npz_path, mmap_mode="r")
    except ValueError as e:
        if "allow_pickle" in str(e):
            # If pickle error, try with explicit allow_pickle=True
            print(f"Warning: NPZ file contains pickled data, loading with allow_pickle=True")
            return np.load(npz_path, mmap_mode="r", allow_pickle=True)
        else:
            raise e

def compress_npz(npz_path):
    """Compress embedding database by pooling statistics across embedding dimension.
    Args:
        npz_path (str): Path to the .npz file containing the embedding database.
    """
    EMB_DB = load_embedding_db(npz_path)
    # Convert to array in the same order as .files
    arr_emb = np.stack([EMB_DB[k] for k in tqdm(EMB_DB.files, desc="Loading embeddings")], axis=0)  # (B, seq, emb)

    # Scale across all batches & timesteps
    print("Scaling embeddings...")
    arr_emb_r = arr_emb.reshape(-1, arr_emb.shape[-1])  # (B*seq, emb)
    scaler = StandardScaler()
    arr_emb_r = scaler.fit_transform(arr_emb_r)
    arr_emb_s = arr_emb_r.reshape(arr_emb.shape)        # (B, seq, emb)

    # Pool statistics per token
    print("Pooling statistics...")
    arr_min  = np.min(arr_emb_s, axis=-1, keepdims=True)
    arr_max  = np.max(arr_emb_s, axis=-1, keepdims=True)
    arr_q25   = np.percentile(arr_emb_s, 25, axis=-1, keepdims=True)
    arr_q50   = np.percentile(arr_emb_s, 50, axis=-1, keepdims=True) # Median
    arr_q75   = np.percentile(arr_emb_s, 75, axis=-1, keepdims=True)
    arr_skew  = np.mean((arr_emb_s - arr_q50)**3, axis=-1, keepdims=True) / (np.std(arr_emb_s, axis=-1, keepdims=True)**3 + 1e-10)
    arr_outlier = np.mean((arr_emb_s < arr_q25 - 1.5 * (arr_q75 - arr_q25)) | (arr_emb_s > arr_q75 + 1.5 * (arr_q75 - arr_q25)), axis=-1, keepdims=True)
    # Concatenate pooled statistics
    arr_concat = np.concatenate([arr_min, arr_q25, arr_q50, arr_q75, arr_max, arr_skew, arr_outlier], axis=-1)  # (B, seq, 7)
    # Rebuild dict, aligned with keys
    EMB_DB_pooled = {k: arr_concat[i] for i, k in enumerate(EMB_DB.files)}

    # Save compressed database
    print("Saving compressed database...")
    np.savez_compressed(npz_path.replace('.npz', '_pooled.npz'), **EMB_DB_pooled)
    print(f"Saved compressed embedding database to {npz_path.replace('.npz', '_pooled.npz')}")
    print(f"Original shape: {arr_emb.shape}, Compressed shape: {arr_concat.shape}")
    print(f"Original size: {len(EMB_DB.files)}, Compressed size: {len(EMB_DB_pooled.keys())}")
    print(f"Compression factor: {arr_emb.shape[-1] / arr_concat.shape[-1]:.2f}x")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compress embedding database by pooling statistics.")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file containing the embedding database.")
    args = parser.parse_args()

    compress_npz(args.npz_path)