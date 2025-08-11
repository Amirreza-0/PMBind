#!/usr/bin/env python
"""
requires python3.10+
============
Generate protein embeddings with ESM-C or any other EvolutionaryScale/Meta
protein language model.

Examples
--------
# 1. Local ESM-C-300M on GPU 0, output NPZ
python esm_embed.py --input proteins.fa --model esmc_300m \
                    --device cuda:0 --outfile embeddings.npz

# 2. Remote ESM-3-large (98 B) via Forge API
export ESM_API_TOKEN="hf_xxxxxxxxxxxxxxxxxx"
python esm_embed.py --input proteins.fa --model esm3-98b-2024-08 \
                    --remote --outfile embeds.parquet
"""
from __future__ import annotations
import argparse, os, sys, json, time, itertools, pathlib, warnings
from typing import List, Tuple, Dict, Iterable, Any

import torch
import numpy as np
import pandas as pd
import tqdm
import csv


# ---------------------------  I/O utilities  ---------------------------------
def read_dat(path: str) -> List[Tuple[str, str]]:
    """Read tab-separated file: first col is id, second is sequence."""
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                line = line.split("\t", 1)
            elif " " in line:
                line = line.split(" ")
            else:
                line = line.split(maxsplit=1)
            if not line or len(line) < 2:
                print(line)
                continue

            if len(line) == 2:
                seqs.append((line[0], line[1]))

    return seqs


def read_csv(path: str, mhc_class: int):
    selected_cols = ["allele", "mhc_sequence", "mhc_class"]
    file = pd.read_csv(path, sep=",", usecols=selected_cols)
    file = file[file["mhc_class"] == mhc_class]
    print(file.columns)
    # convert simple allele, remove * and : to mtach with netmhcpan
    file[selected_cols[0]] = file[selected_cols[0]].str.replace("*", "")
    file[selected_cols[0]] = file[selected_cols[0]].str.replace(":", "")

    return file


# -------------  Local model loader (ESM-C, ESM-2, ESM3-open)  ----------------
def load_local_model(model_name: str, device: str):
    """
    Return (model, to_tensor_fn) for **new** ESM-C / ESM-3
    or (model, batch_converter_fn) for **legacy** ESM-2.
    """
    try:                                              # new-style ESM-C
        if model_name.startswith("esmc"):
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            model = ESMC.from_pretrained(model_name).to(device)
            def embed_one(seq: str):
                protein = ESMProtein(sequence=seq)
                t = model.encode(protein)
                out = model.logits(t, LogitsConfig(sequence=True,
                                                   return_embeddings=True))
                return out.embeddings.mean(0).cpu().numpy()
            return embed_one
        # new-style ESM-3 open weight
        if model_name.startswith("esm3"):
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig
            model = ESM3.from_pretrained(model_name).to(device)
            def embed_one(seq: str):
                protein = ESMProtein(sequence=seq)
                t = model.encode(protein)
                out = model.logits(t, LogitsConfig(sequence=True,
                                                   return_embeddings=True))
                return out.embeddings.mean(0).cpu().numpy()
            return embed_one
    except ImportError:
        pass  # will try legacy route below

    # ---------- legacy facebookresearch/esm (ESM-1/2) ----------
    try:
        from esm import pretrained
        create = getattr(pretrained, model_name,
                         pretrained.load_model_and_alphabet)
        model, alphabet = create(model_name) if callable(create) else create()
        model.eval().to(device)
        converter = alphabet.get_batch_converter()
        def embed_one(seq: str):
            _, _, toks = converter([("x", seq)])
            with torch.inference_mode():
                rep = model(toks.to(device),
                            repr_layers=[model.num_layers])["representations"]
            return rep[model.num_layers][0, 1:len(seq)+1].mean(0).cpu().numpy()
        return embed_one
    except Exception as e:
        raise RuntimeError(f"Don’t know how to load {model_name}: {e}")


# Remote (Forge / AWS) wrapper
def load_remote_client(model_name: str, token: str):
    import esm
    client = esm.sdk.client(model_name, token=token)
    from esm.sdk.api import ESMProtein, LogitsConfig
    def embed_one(seq: str):
        protein = ESMProtein(sequence=seq)
        t = client.encode(protein)
        out = client.logits(t, LogitsConfig(sequence=True,
                                            return_embeddings=True))
        return out.embeddings.mean(0)
    return embed_one


def embed_remote(
    client,
    sequences: List[Tuple[str, str]],
    batch_size: int = 16,
    pooling: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Use cloud client; supports .embed() (ESM-C) and .generate_embeddings() (ESM-3).
    """
    embeds = {}
    for i in range(0, len(sequences), batch_size):
        sub = sequences[i : i + batch_size]
        ids, seqs = zip(*sub)
        if hasattr(client, "embed"):
            out = client.embed(seqs, pooling)
        else:
            out = client.generate_embeddings(seqs, pooling)
        embeds.update({idx: emb for idx, emb in zip(ids, out)})
    return embeds


def get_chain_(file: pd.DataFrame) -> tuple[list[Any], list[Any], list[list[bool]]]:
    """
    Get chains from the file based on MHC class.
    For MHC class I, return chain A and indices.
    For MHC class II, return chain A, chain B, and their indices.
    """
    # MHC Class I: single chain (chain A)
    # split the mhc_sequence column into a single column for chain A
    chain = file['allele'].tolist()
    # clean up allele names by removing * and : and _
    chain = [allele.replace('*', '').replace(':', '').replace('_', '') for allele in chain]
    # MHC Class I: single sequence (chain A)
    seqs = file['mhc_sequence'].tolist()
    # create a mask that shows which positions are not gaps like a binary mask
    mask = [list(map(lambda x: x != '-', seq)) for seq in seqs]
    # remove gaps from sequences
    seqs = [seq.replace('-', '') for seq in seqs]
    return chain, seqs, mask


# --------------------------  Main CLI handler  ------------------------------
def main(**local_args):
    parser = argparse.ArgumentParser(description="ESM embedding generator")

    if local_args:
        args = parser.parse_args([])
        for k, v in local_args.items():
            setattr(args, k, v)
    else:
        parser.add_argument("--input", required=True, help="dat file")
        parser.add_argument("--model", default="esmc_300m", help="Model name/id")
        parser.add_argument("--outfile", required=True, help="Output .npz or .parquet")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
        parser.add_argument("--remote", action="store_true", help="Use cloud API")
        parser.add_argument("--api_token", default=os.getenv("ESM_API_TOKEN"), help="Forge/NIM token")
        parser.add_argument("--mhc_class", type=int, default=1, help="MHC class (1 or 2) for CSV input")
        args = parser.parse_args()

    sequences = {}
    print("MHC Class I selected (single chain)")
    file = read_csv(args.input, mhc_class=args.mhc_class)
    chain_a, seqs_a, mask_a = get_chain_(file)
    print("number of unique chains in chain_a:", len(set(chain_a)))
    # ‑-----------------------------  BEGIN CHANGE  ‑-----------------------------
    # Build a mapping: sequence  →  list of (allele_id, mask)
    seq_to_alleles: dict[str, list[tuple[str, list[bool]]]] = {}
    for allele, seq, m in zip(chain_a, seqs_a, mask_a):
        seq_to_alleles.setdefault(seq, []).append((allele, m))

    print("unique sequences to embed:", len(seq_to_alleles))

    print("embeddings with", args.model)
    # Load (or connect to) the model only once
    embed_one = load_local_model(args.model, args.device)

    # Embed every UNIQUE sequence
    seq_embeddings: dict[str, np.ndarray] = {}
    for seq in tqdm.tqdm(seq_to_alleles, desc="Embedding unique sequences"):
        seq_embeddings[seq] = embed_one(seq)  # (L, d) or (d,) – whatever embed_one returns

    # Now propagate the embedding to every allele that shares the sequence
    embeddings_a: dict[str, np.ndarray] = {}
    masks_a: dict[str, list[bool]] = {}
    for seq, alleles in seq_to_alleles.items():
        for allele, m in alleles:
            embeddings_a[allele] = seq_embeddings[seq]
            masks_a[allele] = m
            sequences[allele] = seq

    print("number of embeddings (after expansion):", len(embeddings_a))

    # # Get embeddings with local model
    # # run local model
    # embed_one = load_local_model(args.model, args.device)
    # print("embeddings with", args.model)
    # # print len of sequences
    # print(f"Number of sequences: {len(seqs_a)}")
    # # print first 5 sequences
    # print("First 5 sequences:", seqs_a[:5])
    # embeddings_a = {}
    # masks_a = {}
    # # Process all sequences and ensure no seq_id is skipped
    # skipped_ids = []
    # for i, (seq_id, seq, mask) in enumerate(
    #         tqdm.tqdm(zip(chain_a, seqs_a, mask_a), total=len(seqs_a), desc="Embedding chain A sequences")):
    #     # Handle potential problematic seq_ids
    #     if seq_id is None or seq_id == "":
    #         seq_id = f"unnamed_sequence_{i}"
    #         skipped_ids.append(i)
    #
    #     emb = embed_one(seq)
    #     embeddings_a[seq_id] = emb
    #     sequences[seq_id] = seq
    #     masks_a[seq_id] = mask
    # if skipped_ids:
    #     print(f"Warning: {len(skipped_ids)} sequences had missing IDs and were given automatic names")

    print("number of embeddings: ",len(embeddings_a.keys()))
    max_len_a = max(len(seq) for seq in mask_a)
    # For each sequence, create a zero array of shape (max_len_a, embedding_dim) and fill positions using mask_a
    embeddings_a_padded = {}
    for key in embeddings_a.keys():
        if key in chain_a:
            arr = np.zeros((max_len_a, embeddings_a[key].shape[-1]), dtype=embeddings_a[key].dtype)
            emb_idx = 0
            for i, m in enumerate(masks_a[key]):
                if m and emb_idx < len(embeddings_a[key]):
                    arr[i] = embeddings_a[key][emb_idx]
                    emb_idx += 1
            embeddings_a_padded[key] = arr
        else:
            print(f"Warning: seq_id {key} not found in chain_a list.")

    # use embeddings_a_padded as the final embeddings
    embeddings = embeddings_a_padded

    # ------------------  save ------------------
    out_path = pathlib.Path(args.outfile)
    # make directory if it does not exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".npz":
        for k, v in embeddings.items():
            if isinstance(v, torch.Tensor):
                embeddings[k] = v.cpu().numpy()
            # save as .npz file
        np.savez_compressed(out_path, **embeddings)
        with open(out_path.with_suffix(".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "mhc_sequence"])
            for seq_id, seq in sequences.items():
                writer.writerow([seq_id, seq])
    elif out_path.suffix == ".parquet":
        df = pd.DataFrame(
            [(k, v.astype(np.float32)) for k, v in embeddings.items()],
            columns=["key", "embedding"],
        )
        df.to_parquet(out_path, index=False)
    else:
        sys.exit("outfile must end with .npz or .parquet")

    print(f"[✓] Saved embeddings for {len(embeddings)} sequences to {out_path}")


if __name__ == "__main__":
    model = "esmc_600m"                 # or whichever model you want
    remote = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    pooling = "mean"

    for mhc_class in (1, 2):            # ← run class-I then class-II
        dat_path = f"/scratch-scc/users/u15472/PMBind/data/alleles/aligned_PMGen_class_{mhc_class}.csv"
        out_path = (
            f"/scratch-scc/users/u15472/PMBind/data/ESM/{model}/PMGen_whole_seq/mhc{mhc_class}_encodings.npz"
        )

        main(
            input=dat_path,
            model=model,
            outfile=out_path,
            remote=remote,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            mhc_class=mhc_class,
        )
