# 14D comprehensive physicochemical encoding
"""
Feature Sources (by column index):

Columns 0-4: Atchley Factors (Atchley1-5)
    Source: Atchley, W.R., Zhao, J., Fernandes, A.D., and Drüke, T. (2005).
            "Solving the protein sequence metric problem."
            Proc. Natl. Acad. Sci. USA 102(18): 6395-6400.
            DOI: 10.1073/pnas.0408677102
    Description: Five multidimensional patterns from factor analysis of
                ~500 amino acid physicochem properties:
                - Factor 1 (PAH): Polarity, Accessibility, Hydrophobicity
                - Factor 2 (PSS): Propensity for Secondary Structure
                - Factor 3 (MS): Molecular Size
                - Factor 4 (CC): Codon Composition
                - Factor 5 (EC): Electrostatic Charge
    Values: Directly from Table 2 of the original publication

Column 5: Molecular Weight (normalized)
    Source: Standard biochemistry values
    Reference: IUPAC amino acid data
    Normalization: Divided by maximum MW (Trp = 204.23 Da) to get [0,1] range
    Raw values (Da): A=89, C=121, D=133, E=147, F=165, G=75, H=155, I=131,
                     K=146, L=131, M=149, N=132, P=115, Q=146, R=174, S=105,
                     T=119, V=117, W=204, Y=181

Column 6: Van der Waals Volume (normalized)
    Source: Zamyatnin, A.A. (1972).
            "Protein volume in solution."
            Prog. Biophys. Mol. Biol. 24: 107-123.
            DOI: 10.1016/0079-6107(72)90005-3
    Also: Zamyatnin, A.A. (1984).
          "Amino Acid, Peptide, and Protein Volume in Solution."
          Annu. Rev. Biophys. Bioeng. 13: 145-165.
          DOI: 10.1146/annurev.bb.13.060184.001045
    Description: Van der Waals volumes from crystallographic data
    Normalization: Divided by maximum volume (Trp = 163 Ų) to get [0,1] range
    Raw values (ų): A=67, C=86, D=91, E=109, F=135, G=48, H=118, I=124,
                    K=135, L=124, M=124, N=96, P=90, Q=114, R=148, S=73,
                    T=93, V=105, W=163, Y=141

Column 7: Aromaticity
    Source: Lobry, J.R. (1994).
            "Properties influencing the success of commercial genetic algorithms."
            Referenced in BioPython ProtParam module
    Description: Binary indicator for aromatic residues
    Values: 1 for F, Y, W; 0 for all others

Columns 8-9: Hydrogen Bond Donors and Acceptors
    Source: Chemical structure analysis (standard organic chemistry)
    Description: Number of H-bond donors/acceptors in amino acid side chain
    Donors: Count of -NH, -NH2, -OH groups that can donate H-bonds
    Acceptors: Count of =O, -OH, =N groups that can accept H-bonds

Column 10: Flexibility
    Source: Vihinen, M., Torkkila, E., and Riikonen, P. (1994).
            "Accuracy of protein flexibility predictions."
            Proteins 19(2): 141-149.
            DOI: 10.1002/prot.340190207
    Description: Normalized B-factors from 92 refined protein structures
                 Window size = 9 residues (optimized parameter)
    Implementation: BioPython ProtParam uses Vihinen's optimized parameters
    Raw scale: ~0.3 to ~0.6 (normalized B-factors)

Column 11: Polarity (normalized)
    Source: Grantham, R. (1974).
            "Amino acid difference formula to help explain protein evolution."
            Science 185(4154): 862-864.
            DOI: 10.1126/science.185.4154.862
    Also: ExPASy ProtScale: https://web.expasy.org/protscale/pscale/PolarityGrantham.html
    Description: One of three properties (polarity, volume, composition)
                 correlating with protein substitution frequencies
    Normalization: Divided by maximum value (D = 13.0) to get [0,1] range
    Raw values: A=8.1, C=5.5, D=13.0, E=12.3, F=5.2, G=9.0, H=10.4, I=5.2,
                K=11.3, L=4.9, M=5.7, N=11.6, P=8.0, Q=10.5, R=10.5, S=9.2,
                T=8.6, W=5.4, Y=6.2, V=5.9

Column 12: Isoelectric Point (normalized)
    Source: Standard biochemistry tables
    Description: pH at which amino acid has no net charge
    Normalization: Mapped to [0,1] range using min-max scaling
                  (pI - min_pI) / (max_pI - min_pI)
                  where min_pI ≈ 2.77 (D), max_pI ≈ 10.76 (R, K)
    Raw pI values: A=6.00, C=5.07, D=2.77, E=3.22, F=5.48, G=5.97, H=7.59,
                   I=6.02, K=9.74, L=5.98, M=5.74, N=5.41, P=6.30, Q=5.65,
                   R=10.76, S=5.68, T=5.60, V=5.96, W=5.89, Y=5.66

Column 13: Miyazawa-Jernigan Hydrophobicity (contact energy)
    Source: Miyazawa, S. and Jernigan, R.L. (1996).
            "Residue-residue potentials with a favorable contact pair term
             and an unfavorable high packing density term, for simulation and threading."
            J. Mol. Biol. 256(3): 623-644.
            DOI: 10.1006/jmbi.1996.0114
    Description: Statistical potential representing contact energies between
                 amino acid pairs in folded proteins
                 Miyazawa-Jernigan hydrophobicity contact energy scale (kcal/mol)
    Values: More negative = more hydrophobic Range: ~ -2.2 (F) to ~ +1.1 (R)

Special amino acids B and Z:
    B = Asx (Aspartic acid OR Asparagine) - ambiguous position in sequence
    Z = Glx (Glutamic acid OR Glutamine) - ambiguous position in sequence
    Values: Average of the two possible residues
"""

# ============================================================================
# RAW VALUES WITH SOURCES
# ============================================================================

# Raw amino acid properties with full source attribution
peptide_binding_encoding_raw = {
    # Standard 20 amino acids
    # Format: [Atch1, Atch2, Atch3, Atch4, Atch5, MW, Vol, Arom, HDon, HAcc, Flex, Pol, pI, MJ_Hydro]
    "A": [-0.591, -1.302, -0.733,  1.570, -0.146,  89, 67, 0, 1, 1, 0.357,  8.1, 6.00, -0.62],
    "C": [-1.343,  0.465, -0.862, -1.020, -0.255, 121, 86, 0, 1, 1, 0.346,  5.5, 5.07, -1.48],
    "D": [ 1.050,  0.302, -3.656, -0.259, -3.242, 133, 91, 0, 1, 4, 0.511, 13.0, 2.77, 0.45],
    "E": [ 1.357, -1.453,  1.477,  0.113, -0.837, 147,109, 0, 1, 3, 0.497, 12.3, 3.22, 0.53],
    "F": [-1.006, -0.590,  1.891, -0.397,  0.412, 165,135, 1, 1, 1, 0.314,  5.2, 5.48, -2.22],
    "G": [-0.384,  1.652,  1.330,  1.045,  2.064,  75, 48, 0, 1, 1, 0.544,  9.0, 5.97, 0.0],
    "H": [ 0.336, -0.417, -1.673, -1.474, -0.078, 155,118, 0, 2, 2, 0.323, 10.4, 7.59, -0.40],
    "I": [-1.239, -0.547,  2.131,  0.393,  0.816, 131,124, 0, 1, 1, 0.462,  5.2, 6.02, -1.90],
    "K": [ 1.831, -0.561,  0.533, -0.277,  1.648, 146,135, 0, 3, 1, 0.466, 11.3, 9.74, 0.77],
    "L": [-1.019, -0.987, -1.505,  1.266, -0.912, 131,124, 0, 1, 1, 0.365,  4.9, 5.98, -1.80],
    "M": [-0.663, -1.524,  2.219, -1.005,  1.212, 149,124, 0, 1, 1, 0.295,  5.7, 5.74, -1.40],
    "N": [ 0.945,  0.828,  1.299, -0.169,  0.933, 132, 96, 0, 2, 2, 0.463, 11.6, 5.41, 0.85],
    "P": [ 0.189,  2.081, -1.628,  0.421, -1.392, 115, 90, 0, 1, 1, 0.509,  8.0, 6.30, -0.21],
    "Q": [ 0.931, -0.179, -3.005, -0.503, -1.853, 146,114, 0, 2, 2, 0.493, 10.5, 5.65, 0.77],
    "R": [ 1.538, -0.055,  1.502,  0.440,  2.897, 174,148, 0, 4, 1, 0.529, 10.5,10.76, 1.10],
    "S": [-0.228,  1.399, -4.760,  0.670, -2.647, 105, 73, 0, 2, 2, 0.507,  9.2, 5.68, -0.12],
    "T": [-0.032,  0.326,  2.213,  0.908,  1.313, 119, 93, 0, 2, 2, 0.444,  8.6, 5.60, -0.32],
    "V": [-1.337, -0.279, -0.544,  1.242, -1.262, 117,105, 0, 1, 1, 0.386,  5.9, 5.96, -1.47],
    "W": [-0.595,  0.009,  0.672, -2.128, -0.184, 204,163, 1, 2, 1, 0.305,  5.4, 5.89, -2.09],
    "Y": [ 0.260,  0.830,  3.097, -0.838,  1.512, 181,141, 1, 2, 2, 0.420,  6.2, 5.66, -1.39],
    # Ambiguous amino acids (average of possible residues)
    "B": [ 0.998,  0.565, -1.179, -0.214, -1.155, 132.5, 93.5, 0, 1.5, 3, 0.487, 12.3, 4.09, 0.65],
    "Z": [ 1.144, -0.816, -0.764, -0.195, -1.345, 146.5,111.5, 0, 1.5, 2.5, 0.495, 11.4, 4.44, 0.65],
    # Special token
    "X": [ 0.000,  0.000,  0.000,  0.000,  0.000, 135, 105, 0, 1.5, 1.5, 0.400,  8.0, 6.00, -0.50],
}

import numpy as np
import pandas as pd

feature_names = [
    "Atchley1_PAH", "Atchley2_PSS", "Atchley3_MS", "Atchley4_CC", "Atchley5_EC",
    "Molecular_Weight", "VdW_Volume", "Aromaticity",
    "H_Bond_Donors", "H_Bond_Acceptors", "Flexibility", "Polarity",
    "Isoelectric_Point", "Miyazawa_Jernigan_Hydrophobicity"
]


def z_score_standardization(data_dict):
    """
    Apply z-score standardization: (x - mean) / std

    This is the preferred normalization for deep learning because:
    - Centers data at 0 with unit variance
    - Better gradient flow during backpropagation
    - Less sensitive to outliers than min-max
    - No artificial boundaries
    """
    # Convert to numpy array
    aa_codes = list(data_dict.keys())
    data_matrix = np.array([data_dict[aa] for aa in aa_codes])

    # Calculate mean and std for each feature (column)
    mean_values = np.mean(data_matrix, axis=0)
    std_values = np.std(data_matrix, axis=0, ddof=0)  # Population std

    # Apply z-score standardization
    standardized_matrix = np.zeros_like(data_matrix, dtype=float)

    for col in range(data_matrix.shape[1]):
        if std_values[col] == 0:
            # If std is 0 (all values same), set to 0
            standardized_matrix[:, col] = 0.0
        else:
            standardized_matrix[:, col] = (data_matrix[:, col] - mean_values[col]) / std_values[col]

    # Convert back to dictionary
    standardized_dict = {aa: standardized_matrix[i].tolist()
                         for i, aa in enumerate(aa_codes)}

    return standardized_dict, mean_values, std_values


# Perform z-score standardization
normalized_encoding, means, stds = z_score_standardization(peptide_binding_encoding_raw)

# Create DataFrame for visualization
df_normalized = pd.DataFrame.from_dict(normalized_encoding, orient='index',
                                       columns=feature_names)
df_normalized.index.name = 'AA'

print("=" * 90)
print("Z-SCORE STANDARDIZED AMINO ACID ENCODINGS (Mean=0, Std=1)")
print("OPTIMIZED FOR DEEP LEARNING")
print("=" * 90)
print()
print(df_normalized.round(4))
print()

# Display normalization parameters
print("=" * 90)
print("STANDARDIZATION PARAMETERS (for inverse transform if needed)")
print("=" * 90)
print()
df_params = pd.DataFrame({
    'Feature': feature_names,
    'Mean': means.round(4),
    'Std': stds.round(4)
})
print(df_params.to_string(index=False))
print()

# Verification
print("=" * 90)
print("VERIFICATION (should be ~0 mean, ~1 std)")
print("=" * 90)
all_values = np.array(list(normalized_encoding.values()))
print(f"Overall mean: {all_values.mean():.6f} (should be ≈ 0)")
print(f"Overall std:  {all_values.std():.6f} (should be ≈ 1)")
print(f"Min value: {all_values.min():.4f}")
print(f"Max value: {all_values.max():.4f}")
print()

# Per-feature verification
print("Per-feature statistics:")
for i, name in enumerate(feature_names):
    col_values = all_values[:, i]
    print(f"  {name:40s} | mean: {col_values.mean():7.4f}, std: {col_values.std():6.4f}")
print()

# Export ready-to-use dictionary
print("=" * 90)
print("READY-TO-USE DICTIONARY (Copy this into your code)")
print("=" * 90)
print()
print("# Z-score standardized encoding for deep learning")
print("peptide_binding_encoding_normalized = {")
for aa, values in normalized_encoding.items():
    values_str = ", ".join([f"{v:7.4f}" for v in values])
    print(f'    "{aa}": [{values_str}],')
print("}")
print()

# Example usage
print("=" * 90)
print("EXAMPLE: Encoding a peptide sequence")
print("=" * 90)
print()
peptide = "SIINFEKL"
print(f"Peptide: {peptide}")
print(f"Encoded shape: ({len(peptide)}, 14)")
print()
print("Encoded matrix (first 3 residues):")
for i, aa in enumerate(peptide[:3]):
    print(f"{aa}: {normalized_encoding[aa][:5]}...")  # Show first 5 features

# ============================================================================
# CORRELATION ANALYSIS AND VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = df_normalized.corr()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Full correlation heatmap with values
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax1)
ax1.set_title('Feature Correlation Matrix (All 14 Features)', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Features', fontsize=11)
ax1.set_ylabel('Features', fontsize=11)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)

# Plot 2: Simplified correlation heatmap (showing only strong correlations)
# Mask for values close to zero (weak correlations)
mask = np.abs(correlation_matrix) < 0.85
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            mask=mask,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax2)
ax2.set_title('Strong Correlations Only (|r| ≥ 0.3)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Features', fontsize=11)
ax2.set_ylabel('Features', fontsize=11)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('amino_acid_feature_correlation.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation plot saved as 'amino_acid_feature_correlation.png'")
plt.show()

# Print highly correlated feature pairs
print("\n" + "=" * 80)
print("HIGHLY CORRELATED FEATURE PAIRS (|r| ≥ 0.5)")
print("=" * 80)
print()

# Get upper triangle indices to avoid duplicates
upper_triangle = np.triu_indices_from(correlation_matrix, k=1)
correlations = []

for i, j in zip(*upper_triangle):
    corr_value = correlation_matrix.iloc[i, j]
    if abs(corr_value) >= 0.85:
        correlations.append({
            'Feature 1': feature_names[i],
            'Feature 2': feature_names[j],
            'Correlation': corr_value
        })

# Sort by absolute correlation value
correlations_df = pd.DataFrame(correlations)
if not correlations_df.empty:
    correlations_df['Abs_Correlation'] = correlations_df['Correlation'].abs()
    correlations_df = correlations_df.sort_values('Abs_Correlation', ascending=False)
    correlations_df = correlations_df.drop('Abs_Correlation', axis=1)
    print(correlations_df.to_string(index=False))
else:
    print("No feature pairs with |r| ≥ 0.5 found.")

print()

# Additional statistics
print("=" * 80)
print("CORRELATION STATISTICS")
print("=" * 80)
print(f"Mean absolute correlation: {np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]).mean():.3f}")
print(f"Max correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].max():.3f}")
print(f"Min correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].min():.3f}")
print(f"Number of strong correlations (|r| ≥ 0.5): {(np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]) >= 0.5).sum()}")
print(f"Number of moderate correlations (0.3 ≤ |r| < 0.5): {((np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]) >= 0.3) & (np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]) < 0.5)).sum()}")
print()


# Z-score standardized encoding for deep learning
Physicochemical_Properties = {
    "A": [-0.7115, -1.3796, -0.3237,  1.7644, -0.0249, -1.7046, -1.5572, -0.3873, -0.7860, -0.7889, -0.9536, -0.1956,  0.0770, -0.1703],
    "C": [-1.4936,  0.5087, -0.3881, -1.1136, -0.0977, -0.5661, -0.8431, -0.3873, -0.7860, -0.7889, -1.0949, -1.1785, -0.4755, -0.9901],
    "D": [ 0.9952,  0.3345, -1.7827, -0.2679, -2.0926, -0.1392, -0.6552, -0.3873, -0.7860,  2.8401,  1.0251,  1.6568, -1.8416,  0.8496],
    "E": [ 1.3145, -1.5410,  0.7795,  0.1454, -0.4864,  0.3589,  0.0212, -0.3873, -0.7860,  1.6305,  0.8452,  1.3922, -1.5743,  0.9259],
    "F": [-1.1431, -0.6188,  0.9861, -0.4213,  0.3478,  0.9993,  0.9983,  2.5820, -0.7860, -0.7889, -1.5060, -1.2919, -0.2319, -1.6955],
    "G": [-0.4962,  1.7772,  0.7061,  1.1811,  1.4511, -2.2027, -2.2712, -0.3873, -0.7860, -0.7889,  1.4490,  0.1446,  0.0591,  0.4207],
    "H": [ 0.2526, -0.4339, -0.7929, -1.6181,  0.0205,  0.6435,  0.3595, -0.3873,  0.5531,  0.4208, -1.3904,  0.6739,  1.0214,  0.0394],
    "I": [-1.3855, -0.5728,  1.1059,  0.4566,  0.6176, -0.2104,  0.5850, -0.3873, -0.7860, -0.7889,  0.3955, -1.2919,  0.0888, -1.3905],
    "K": [ 1.8074, -0.5878,  0.3083, -0.2879,  1.1733,  0.3233,  0.9983, -0.3873,  1.8922, -0.7889,  0.4469,  1.0141,  2.2985,  1.1546],
    "L": [-1.1567, -1.0430, -0.7090,  1.4266, -0.5365, -0.2104,  0.5850, -0.3873, -0.7860, -0.7889, -0.8508, -1.4053,  0.0651, -1.2951],
    "M": [-0.7864, -1.6169,  1.1498, -1.0969,  0.8821,  0.4300,  0.5850, -0.3873, -0.7860, -0.7889, -1.7501, -1.1029, -0.0775, -0.9139],
    "N": [ 0.8860,  0.8966,  0.6906, -0.1679,  0.6957, -0.1748, -0.4673, -0.3873,  0.5531,  0.4208,  0.4083,  1.1276, -0.2735,  1.2309],
    "P": [ 0.0997,  2.2356, -0.7704,  0.4877, -0.8570, -0.7796, -0.6928, -0.3873, -0.7860, -0.7889,  0.9994, -0.2334,  0.2552,  0.2205],
    "Q": [ 0.8714, -0.1795, -1.4577, -0.5391, -1.1649,  0.3233,  0.2091, -0.3873,  0.5531,  0.4208,  0.7938,  0.7117, -0.1309,  1.1546],
    "R": [ 1.5027, -0.0470,  0.7919,  0.5088,  2.0074,  1.3194,  1.4869, -0.3873,  3.2313, -0.7889,  1.2563,  0.7117,  2.9044,  1.4692],
    "S": [-0.3340,  1.5068, -2.3337,  0.7644, -1.6952, -1.1354, -1.3317, -0.3873,  0.5531,  0.4208,  0.9737,  0.2203, -0.1131,  0.3063],
    "T": [-0.1301,  0.3601,  1.1468,  1.0288,  0.9495, -0.6373, -0.5801, -0.3873,  0.5531,  0.4208,  0.1642, -0.0066, -0.1606,  0.1156],
    "V": [-1.4874, -0.2864, -0.2293,  1.4000, -0.7702, -0.7084, -0.1291, -0.3873, -0.7860, -0.7889, -0.5810, -1.0273,  0.0532, -0.9806],
    "W": [-0.7157,  0.0214,  0.3776, -2.3448, -0.0503,  2.3868,  2.0506,  2.5820,  0.5531, -0.7889, -1.6217, -1.2163,  0.0116, -1.5716],
    "Y": [ 0.1736,  0.8987,  1.5881, -0.9113,  1.0824,  1.5685,  1.2238,  2.5820,  0.5531,  0.4208, -0.1441, -0.9139, -0.1250, -0.9043],
    "B": [ 0.9411,  0.6155, -0.5463, -0.2179, -0.6988, -0.1570, -0.5613, -0.3873, -0.1164,  1.6305,  0.7167,  1.3922, -1.0576,  1.0403],
    "Z": [ 1.0929, -0.8603, -0.3391, -0.1968, -0.8257,  0.3411,  0.1152, -0.3873, -0.1164,  1.0256,  0.8195,  1.0519, -0.8497,  1.0403],
    "X": [-0.0969,  0.0118,  0.0422,  0.0199,  0.0726, -0.0681, -0.1291, -0.3873, -0.1164, -0.1841, -0.4011, -0.2334,  0.0770, -0.0560],
}