#!/usr/bin/env python3
import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess, os, sys, shutil

# ----------------------------------------------------------------------
# Parameters
infile  = "../../data/alleles/PMGen_pseudoseq_extended_26_07_2025.csv"
classes_to_run = (1, 2)          # run both classes
out_tmpl = "../../data/alleles/aligned_PMGen_class_{cls}.csv"
# ----------------------------------------------------------------------

# Read once
df_full = pd.read_csv(infile)

# Work directory for all temporary files # out_tmpl/tmp
tmp_root = os.path.join(os.path.dirname(os.path.abspath(out_tmpl)), "tmp")
if not os.path.exists(tmp_root):
    os.makedirs(tmp_root)


try:
    for mhc_class in classes_to_run:
        print(f"\n=== Processing class {mhc_class} ===")
        df = df_full[df_full["mhc_types"] == mhc_class].copy()

        if df.empty:
            print(f"  No entries for class {mhc_class}, skipping.")
            continue

        # Remove dashes and keep only unique allele–sequence pairs
        df["mhc_sequence"] = df["sequence"].str.replace("-", "", regex=False)
        df = df.drop_duplicates(subset=["simple_allele"])    # or ["simple_allele","mhc_sequence"]

        print(f"  Unique alleles: {len(df)}")

        # ------------------------------------------------------------------
        # Write FASTA
        print("  Writing sequences to FASTA for alignment ...")
        fasta_path = os.path.join(tmp_root, f"class_{mhc_class}.fa")
        aln_path   = os.path.join(tmp_root, f"class_{mhc_class}.aln")

        records = [SeqRecord(Seq(seq), id=allele)
                   for allele, seq in zip(df["simple_allele"], df["mhc_sequence"])]
        # Sort longest first (helps some aligners)
        records.sort(key=lambda r: len(r.seq), reverse=True)
        SeqIO.write(records, fasta_path, "fasta")

        # ------------------------------------------------------------------
        # Align
        print("  Aligning sequences ...")
        alignment = None
        try:                                    # first try MAFFT
            cmd = f"mafft --auto {fasta_path}"
            with open(aln_path, "w") as fout:
                subprocess.run(cmd, shell=True, check=True, stdout=fout, stderr=subprocess.DEVNULL)
            alignment = AlignIO.read(aln_path, "fasta")

        except (subprocess.SubprocessError, FileNotFoundError):
            print("  MAFFT not available")

        # ------------------------------------------------------------------
        # Save to CSV
        aligned_rows = [{"allele": rec.id,
                         "mhc_sequence": str(rec.seq),
                         "mhc_class": mhc_class} for rec in alignment]

        out_path = os.path.join(os.getcwd(), out_tmpl.format(cls=mhc_class))
        pd.DataFrame(aligned_rows).to_csv(out_path, index=False)
        print(f"  Alignment written to {out_path}")

finally:
    # Clean up temp directory
    shutil.rmtree(tmp_root, ignore_errors=True)