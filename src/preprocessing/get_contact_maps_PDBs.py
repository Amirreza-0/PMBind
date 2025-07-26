#!/usr/bin/env python3
"""
Process PMGen PDB files:

    • peptide sequences are taken from sequences.tsv (columns: id, peptide)
    • two distance-derived matrices are computed per PDB
        – Cα channel (carbon='CA')
        – Cβ channel (carbon='CB')
    • matrices are stacked along the last axis → shape (*, *, 2)
    • the stack is averaged across all PDBs of the same id
    • result written to <output_dir>/<id>.npy
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
# from Bio.PDB import PDBParser
# from scipy.spatial.distance import pdist, squareform
from src.utils import process_pdb_distance_matrix

# ───────────────────────────────────────────────────────────────────────
# NEW distance-matrix function supplied by the user
# ───────────────────────────────────────────────────────────────────────
# def process_pdb_distance_matrix(
#         pdb_path: str | Path,
#         threshold: float,
#         peptide: str,
#         chainid: str = "A",
#         carbon: str = "CB"
#     ) -> np.ndarray:
#     """
#     Return a normalised inverse-distance matrix for the chosen
#     carbon type (CA or CB).
#
#     Rows correspond to the peptide residues (last len(peptide) of chain),
#     columns correspond to the rest of the protein that precedes the peptide.
#     """
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure("protein", pdb_path)
#     chain = structure[0][chainid]
#
#     coords = []
#     carbon_not = "CA" if carbon == "CB" else "CB"
#     for res in chain:
#         if res.get_resname() == "GLY":        # Glycine => Cα only
#             atom = res["CA"]
#         else:
#             atom = res[carbon] if carbon in res else res[carbon_not]
#         coords.append(atom.get_coord())
#     coords = np.asarray(coords, dtype=np.float32)
#
#     # pairwise distances
#     dist_matrix = squareform(pdist(coords, metric="euclidean"))
#
#     # binary mask: 1 if distance >= threshold   (will be zeroed later)
#     mask = np.where(dist_matrix < threshold, 0.0, 1.0)
#     mask += np.eye(mask.shape[0], dtype=np.float32)     # diagonal = 1
#     mask = np.where(mask == 1, 0.0, 1.0)                # invert
#
#     # inverse distances
#     result = 1.0 / (dist_matrix + 1e-9)
#     result *= mask                                      # apply mask
#
#     # keep only contacts between peptide (rows) and rest (cols)
#     pep_len = len(peptide)
#     result = result[-pep_len:, : -pep_len]
#
#     # 0-1 normalisation
#     result = (result - np.min(result)) / (np.ptp(result) + 1e-9)
#     return result
# ───────────────────────────────────────────────────────────────────────


def read_id_to_peptide(tsv_path: str | Path) -> Dict[str, str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if {"id", "peptide"}.difference(df.columns):
        sys.exit(f"{tsv_path} must contain at least the columns 'id' and 'peptide'")
    return dict(zip(df["id"].astype(str), df["peptide"].astype(str)))


def split_folder_name(folder: Path) -> str:
    """
    Return the id part in a folder name of the form  \<id\>_\<anything\>.
    If no “_” is present the full folder name is returned unchanged.
    """
    return folder.name.split("_", 1)[0]


def collect_id_pdbs(id_name: str, root_dir: Path, id_to_pep: dict) -> tuple[List[Path], List[str]]:
    """
    Sample folders are named like <id>_number (e.g. ida_1, ida_2, …).
    from the sample root directory, collect all PDB files in side sample folders
    get pdb files and peptide sequences
    """
    #split folder name by _, if part1 == id_name, add the pdb
    pdb_files = []
    peptides = []
    for sample_dir in root_dir.iterdir():
        id_name_sample = split_folder_name(sample_dir)
        if sample_dir.is_dir() and id_name == id_name_sample:
            pdb_files.extend(sample_dir.glob("*.pdb"))
            peptide = id_to_pep.get(sample_dir.name, "")
            if peptide:
                peptides.append(peptide)

    if not pdb_files:
        print(f"No PDB files found for id '{id_name}' in {root_dir}")
    return sorted(pdb_files), peptides



def main(
        root_dir: str | Path,
        sequences_tsv: str | Path,
        chainid: str = "A",
        threshold: float = 9.0,
        output_dir: str | Path | None = None
) -> None:

    root_dir = Path(root_dir).expanduser().resolve()
    out_dir  = Path(output_dir or root_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_pep = read_id_to_peptide(sequences_tsv)
    carbon_types = ("CA", "CB")

    unique_ids = []
    # define unique id names from the root directory
    for sample_name in sorted(p.name for p in root_dir.iterdir() if p.is_dir()):
        id_name = split_folder_name(Path(sample_name))
        if sample_name not in id_to_pep:
            print(f"[{sample_name}] id '{id_name}' not in TSV – skipping")
            continue
        if id_name not in unique_ids:
            unique_ids.append(id_name)

    for id_name in unique_ids:
        pdb_files, peptides = collect_id_pdbs(id_name, root_dir=root_dir, id_to_pep=id_to_pep)
        if not pdb_files:
            print(f"[{id_name}] no PDB files – skipping")
            continue

        per_pdb_arrays: List[np.ndarray] = []
        for pdb, pep in zip(pdb_files, peptides):
            print(f"[{id_name}] processing {pdb.name} with peptide '{pep}'")
            try:
                # build both CA and CB channels
                channel_mats = []
                for carbon in carbon_types:
                    channel_mats.append(
                        np.expand_dims(
                            process_pdb_distance_matrix(
                                pdb, threshold, pep, chainid, carbon
                            ),
                            axis=-1
                        )
                    )
                combined = np.concatenate(channel_mats, axis=-1)  # (*, *, 2)
                per_pdb_arrays.append(combined)
            except Exception as exc:
                print(f"   -> failed on {pdb.name}: {exc}")

        if not per_pdb_arrays:
            print(f"[{id_name}] all PDBs failed – skipping")
            continue

        # verify consistent shapes
        shapes = {arr.shape for arr in per_pdb_arrays}
        if len(shapes) != 1:
            print(f"[{id_name}] shape mismatch {shapes} – skipping")
            continue

        avg_mat = np.mean(np.stack(per_pdb_arrays, axis=0), axis=0).astype(np.float32)
        np.save(out_dir / f"{id_name}.npy", avg_mat)
        print(f"[{id_name}] saved → {out_dir / (id_name + '.npy')}")


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hard-coded demo parameters  – replace by argparse for production use
    ROOT_DIR   = "/media/amirreza/lasse/PMGen/results-h2/alphafold"        # folder with sub-dirs id_1, id_2, …
    TSV_FILE   = "/media/amirreza/lasse/PMGen/data/example/h2_updated.tsv"  # mapping id \<-\> peptide
    CHAIN_ID   = "A"                # The chain id in the PDB files (AlphaFold gets only one chain)
    THRESHOLD  = 9.0                # Distance threshold for contacts (in Angstrom)
    OUTPUT_DIR = "../../data/contact_maps"    # Output directory for the resulting .npy files

    main(ROOT_DIR, TSV_FILE, CHAIN_ID, THRESHOLD, OUTPUT_DIR)