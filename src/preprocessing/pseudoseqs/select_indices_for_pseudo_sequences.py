# 1. load input2.tsv [peptide,mhc_seq,mhc_type,anchors,id,hla]
# 2. load foldmasson_aa.fsa and foldmason_ss.fsa
# 3. replace the sample id with allele name from input2.tsv
# 4. use the aligned sequences from foldmasson_aa.fsa and foldmason_ss.fsa
# 5. select these indices from aligned sequences:
#   selected_indices = [26, 28, 30, 45, 46, 47, 54, 55, 66, 80, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 101, 102, 105, 116, 118, 120, 135, 137, 138, 139, 144, 145, 154, 163, 164, 167, 168, 173, 176, 177, 178, 180, 181, 184, 185, 188, 189, 192]
# 6. print if any of these indices are gaps in aligned sequences
# 7. output pseudo_seq.tsv [allele,pseudo_aa,pseudo_3di,mhc_class]


import csv
import sys

# Configuration
INPUT_TSV = 'mhc_seqs_input_foldmason.tsv'
AA_FSA = 'foldmason_aa.fa'
SS_FSA = 'foldmason_ss.fa'
OUTPUT_TSV = 'pseudo_seq.csv'

SELECTED_INDICES = [
    26, 28, 30, 45, 46, 47, 54, 55, 66, 80, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 97, 98, 99, 101, 102, 105, 116, 118, 120, 135, 137,
    138, 139, 144, 145, 154, 163, 164, 167, 168, 173, 176, 177, 178, 180,
    181, 184, 185, 188, 189, 192
]


def clean_header(header_str):
    """
    Cleans FASTA header to match TSV IDs.
    1. Takes first word (split by space)
    2. Removes file extensions (.pdb, .cif, .ent)
    """
    # Take the part before the first space
    s = header_str.split()[0]
    # Remove common extensions often left by FoldMason
    for ext in ['pdb', '.cif', '.ent', '.af2', '.json']:
        if s.endswith(ext):
            s = s.replace(ext, '')
    return s


def read_fasta_to_dict(filepath):
    sequences = {}
    current_header = None
    current_seq = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header:
                        sequences[current_header] = "".join(current_seq)

                    # CLEAN THE ID HERE
                    raw_header = line[1:]
                    current_header = clean_header(raw_header)

                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header:
                sequences[current_header] = "".join(current_seq)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

    return sequences


def extract_pseudo_sequence(full_seq, indices, seq_name, type_label):
    pseudo = []
    has_gap = False
    seq_len = len(full_seq)

    for idx in indices:
        if idx >= seq_len:
            # Fail silently or verbose? Let's just return None to skip this row safely
            print(f"  -> Error: Index {idx} out of bounds for {seq_name} (Len: {seq_len})")
            return None

        char = full_seq[idx]
        pseudo.append(char)
        if char == '-':
            has_gap = True

    if has_gap:
        print(f"  -> Warning: Gap in {type_label} for {seq_name}")

    return "".join(pseudo)


def main():
    # 1. Load Metadata
    metadata = {}
    print(f"Loading metadata from {INPUT_TSV}...")

    try:
        with open(INPUT_TSV, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                # Clean TSV ID just in case
                sample_id = row['id'].strip()
                metadata[sample_id] = {
                    'hla': row['hla'],
                    'mhc_class': row['mhc_type']
                }
                # DEBUG: Print first 3 IDs from TSV
                if i < 3:
                    print(f"  [DEBUG TSV ID #{i}]: '{sample_id}'")

    except Exception as e:
        print(f"Error reading TSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(metadata)} IDs from TSV.")

    # 2. Load FASTAs
    print(f"Loading Amino Acid sequences from {AA_FSA}...")
    aa_sequences = read_fasta_to_dict(AA_FSA)
    # DEBUG: Print first 3 keys from FASTA
    print(f"  [DEBUG FASTA KEYS]: {list(aa_sequences.keys())[:3]} ...")

    print(f"Loading Structure sequences from {SS_FSA}...")
    ss_sequences = read_fasta_to_dict(SS_FSA)

    # 3. Process
    output_rows = []
    print("\nProcessing sequences...")

    found_count = 0
    missing_count = 0

    for sample_id, info in metadata.items():
        # Perform checks
        if sample_id not in aa_sequences:
            print(f"Skipping '{sample_id}': Not found in AA FASTA (Check ID matching)")
            missing_count += 1
            continue
        if sample_id not in ss_sequences:
            print(f"Skipping '{sample_id}': Not found in SS FASTA")
            missing_count += 1
            continue

        aa_aligned = aa_sequences[sample_id]
        ss_aligned = ss_sequences[sample_id]
        allele_name = info['hla']

        pseudo_aa = extract_pseudo_sequence(aa_aligned, SELECTED_INDICES, sample_id, "AA")
        pseudo_ss = extract_pseudo_sequence(ss_aligned, SELECTED_INDICES, sample_id, "3Di")

        if pseudo_aa and pseudo_ss:
            output_rows.append({
                'allele': allele_name,
                'pseudo_aa': pseudo_aa,
                'pseudo_3di': pseudo_ss,
                'mhc_class': info['mhc_class']
            })
            found_count += 1

    # 4. Write Output
    print(f"\nWriting {len(output_rows)} results to {OUTPUT_TSV}...")
    print(f"Summary: {found_count} matched, {missing_count} missing/skipped.")

    with open(OUTPUT_TSV, 'w', newline='') as f:
        fieldnames = ['allele', 'pseudo_aa', 'pseudo_3di', 'mhc_class']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print("Done.")


if __name__ == "__main__":
    main()