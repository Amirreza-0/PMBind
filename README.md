# PMBind: Peptide-MHC Binding Prediction with Multi-Task Deep Learning

PMBind is a deep learning framework for predicting peptide-MHC (pMHC) binding interactions using a novel multi-task transformer architecture. The model jointly learns binary binding classification and per-residue reconstruction for both peptides and MHC molecules, leveraging 2D-masked cross-attention mechanisms for enhanced interpretability and clustering capabilities.

## 🔬 Overview

PMBind addresses the critical challenge of predicting peptide binding to Major Histocompatibility Complex (MHC) molecules, which is essential for:
- **Immunotherapy Design**: Identifying neoantigens for cancer vaccines
- **Vaccine Development**: Predicting immunogenic peptides
- **Autoimmune Disease Research**: Understanding self-peptide presentation
- **Personalized Medicine**: Tailoring treatments based on MHC allele profiles

### Key Features

- **Multi-Task Learning**: Simultaneous binding prediction and sequence reconstruction
- **2D-Masked Cross-Attention**: Novel attention mechanism preventing self-attention within peptide/MHC while enabling cross-attention between them
- **Interpretability**: Attention weight visualization and per-residue contribution analysis
- **ESM-2 Embeddings**: Leverages state-of-the-art protein language model embeddings for MHC representation
- **Physicochemical Encoding**: Comprehensive 14D physicochemical properties for peptide encoding (Atchley factors, molecular weight, VdW volume, etc.)
- **Cross-Validation Framework**: Robust evaluation with allele-group-based splitting
- **Binding Affinity Analysis**: Correlation analysis with experimental measurements

## 📋 Table of Contents

- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Inference](#inference)
  - [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended for large datasets)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Amirreza-0/PMBind.git
cd PMBind/
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

Core dependencies include:
- `tensorflow` - Deep learning framework
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `esm` - ESM-2 protein language model
- `matplotlib`, `seaborn` - Visualization
- `pyarrow` - Efficient data storage
- `keras_hub` - Additional Keras utilities
- `umap-learn` - Dimensionality reduction for visualization

## 📊 Data Requirements

### Dataset Download

Download the preprocessed datasets from the data repository:

**Main Dataset**: [Download Link](https://owncloud.gwdg.de/index.php/s/tNYEyyxhUWIy4Jh)

Extract the dataset into the `data/` directory:
```bash
cd data/
# Extract downloaded files here
```

The data directory should contain:
- `PMDb_alleles_sequences_aligned.csv` - MHC allele sequences
- Binding affinity data (parquet files)
- Cross-validation splits
- Benchmark datasets

### Data Format

**Input Requirements:**
- **Peptides**: Amino acid sequences (typically 8-15 residues)
- **MHC Alleles**: Full-length or truncated MHC sequences with corresponding identifiers
- **Labels**: Binary binding labels (0/1) or binding affinity measurements (IC50, Kd, etc.)

**Preprocessing Pipeline:**
```
Raw Data → ESM-2 Embedding → Alignment (MAFFT) → TFRecord Generation → Training
```

## 🎯 Quick Start

### 1. Prepare Your Data

```bash
cd src/

# Step 1: Create cross-validation dataset splits
python create_dataset.py

# Step 2: Generate ESM-2 embeddings for MHC sequences
python preprocessing/run_ESM.py --input ../data/PMDb_alleles_sequences_aligned.csv \
                                --model esmc_300m \
                                --device cuda:0 \
                                --outfile ../data/mhc_embeddings.npz

# Step 3: Run MAFFT alignment (if needed)
python preprocessing/run_MAFFT_alignment.py

# Step 4: Create TFRecord files for efficient training
python create_tfrecords.py
```

### 2. Train the Model

```bash
# Configure training paths in training_paths.json
# Then run training
python run_training.py --epochs 100 \
                      --batch_size 512 \
                      --learning_rate 0.001 \
                      --emb_dim 64 \
                      --heads 4 \
                      --transformer_layers 3
```

### 3. Run Inference

```bash
# Configure inference paths in infer_paths.json
python run_inference.py --model_weights h5/best_model.weights.h5 \
                       --test_data ../data/test.parquet \
                       --output_dir results/
```

### 4. Visualize Results

```bash
# Analyze attention patterns
python accumulative_attention.py --model_weights h5/best_model.weights.h5 \
                                --allele HLA-A*02:01

# Binding affinity correlation analysis
python infer_binding_affinity.py --model_weights h5/best_model.weights.h5 \
                                --df_path ../data/affinity_data.parquet \
                                --measurement_col IC50
```

## 📁 Project Structure

```
PMBind/
├── data/                          # Data directory
│   ├── PMDb_alleles_sequences_aligned.csv
│   ├── README_DATA.MD
│   └── dataset_link.txt
├── src/                           # Source code
│   ├── models.py                  # PMBind model architecture
│   ├── utils.py                   # Utility functions and custom layers
│   ├── run_training.py           # Training script
│   ├── run_inference.py          # Inference script
│   ├── infer.py                  # Advanced inference with metrics
│   ├── infer_binding_affinity.py # Binding affinity analysis
│   ├── create_dataset.py         # Dataset creation and CV splits
│   ├── create_tfrecords.py       # TFRecord generation
│   ├── visualizations.py         # Visualization utilities
│   ├── accumulative_attention.py # Attention analysis
│   ├── run_grid_search.py        # Hyperparameter tuning
│   ├── run_interpolation.py      # Latent space interpolation
│   ├── run_permutation_ablation.py # Feature importance analysis
│   ├── training_paths.json       # Training configuration
│   ├── infer_paths.json          # Inference configuration
│   └── preprocessing/            # Preprocessing scripts
│       ├── run_ESM.py           # ESM-2 embedding generation
│       ├── run_MAFFT_alignment.py # Sequence alignment
│       ├── compress_emb.py      # Embedding compression
│       ├── cross_val_split.py   # Cross-validation utilities
│       ├── data_concatenations.py # Data merging
│       ├── tfrecord_split.py    # TFRecord splitting
│       └── tfrecord_analyse.py  # TFRecord analysis
├── tests/                        # Test files
│   └── amino_acid_encoding_physicochemical_properties.py
├── results/                      # Output directory (generated)
├── h5/                          # Model weights (generated)
├── requirements.txt             # Python dependencies
├── LICENSE                      # Apache 2.0 License
└── README.md                    # This file
```

## 💻 Usage

### Data Preprocessing

#### Generate ESM-2 Embeddings

```bash
cd src/preprocessing/

# For local model (recommended for large datasets)
python run_ESM.py --input ../../data/allele_sequences.fasta \
                  --model esmc_300m \
                  --device cuda:0 \
                  --outfile ../../data/embeddings.npz

# For remote API (ESM-3)
export ESM_API_TOKEN="your_huggingface_token"
python run_ESM.py --input ../../data/allele_sequences.fasta \
                  --model esm3-98b-2024-08 \
                  --remote \
                  --outfile ../../data/embeddings.parquet
```

#### Create Cross-Validation Splits

```bash
cd src/
python create_dataset.py
```

This script:
- Loads binding affinity data
- Creates K-fold cross-validation splits with allele-group stratification
- Generates train/validation/test sets
- Handles benchmark dataset separation

#### Generate TFRecords

```bash
python create_tfrecords.py
```

TFRecords provide efficient data loading during training with:
- Compact storage of peptide/MHC indices
- Pre-computed embedding lookups
- Parallel I/O operations

### Training

#### Basic Training

```bash
cd src/
python run_training.py
```

#### Advanced Training with Custom Parameters

```bash
python run_training.py \
    --epochs 150 \
    --batch_size 1024 \
    --learning_rate 0.0005 \
    --emb_dim 128 \
    --heads 8 \
    --transformer_layers 4 \
    --latent_dim 256 \
    --dropout 0.3 \
    --l2_reg 0.001 \
    --mixed_precision
```

**Training Features:**
- Mixed precision training (FP16) for faster computation
- Class-weighted loss for imbalanced datasets
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping with patience
- Model checkpointing (best validation performance)
- TensorBoard logging

#### Hyperparameter Tuning

```bash
python run_grid_search.py
```

Performs grid search over:
- Embedding dimensions
- Number of attention heads
- Transformer layers
- Learning rates
- Regularization parameters

### Inference

#### Standard Inference

```bash
python run_inference.py \
    --model_weights ../h5/best_model.weights.h5 \
    --test_data ../data/test.parquet \
    --output_dir ../results/test_predictions/
```

#### Advanced Inference with Metrics

```bash
python infer.py \
    --model_weights ../h5/best_model.weights.h5 \
    --config ../h5/model_config.json \
    --test_data ../data/benchmark.parquet \
    --output_dir ../results/benchmark/ \
    --compute_attention \
    --save_embeddings
```

**Outputs:**
- Binding predictions (CSV/Parquet)
- Performance metrics (AUROC, AUPRC, MCC)
- Confusion matrices
- Per-allele performance breakdown
- Latent representations
- Attention weights (optional)

#### Binding Affinity Analysis

```bash
python infer_binding_affinity.py \
    --model_weights ../h5/best_model.weights.h5 \
    --df_path ../data/affinity_measurements.parquet \
    --measurement_col IC50 \
    --invert_measurements \
    --output_dir ../results/affinity_analysis/
```

**Correlation Metrics:**
- Pearson correlation coefficient
- Spearman rank correlation
- Visualization of prediction vs. measurement
- Classification of binders/non-binders/noisy samples

### Visualization

#### Attention Weight Analysis

```bash
python accumulative_attention.py \
    --model_weights ../h5/best_model.weights.h5 \
    --allele HLA-A*02:01 \
    --peptides_file ../data/binder_peptides.csv \
    --output_dir ../results/attention_analysis/
```

**Generates:**
- Heatmaps of average attention patterns
- Per-position importance scores
- Anchor residue identification
- Binding pocket visualization

#### Latent Space Interpolation

```bash
python run_interpolation.py \
    --model_weights ../h5/best_model.weights.h5 \
    --peptide1 SIINFEKL \
    --peptide2 GILGFVFTL \
    --allele HLA-A*02:01 \
    --steps 20
```

Interpolates between two peptides in latent space to visualize decision boundaries.

#### Feature Importance via Permutation

```bash
python run_permutation_ablation.py \
    --model_weights ../h5/best_model.weights.h5 \
    --test_data ../data/test.parquet \
    --features peptide,mhc,both
```

## 🏗️ Model Architecture

### Overview

PMBind uses a multi-task transformer architecture with the following components:

```
Input Layer
    ├─ Peptide: BLOSUM62 + Physicochemical (14D) → Embedding
    └─ MHC: ESM-2 Pre-trained Embeddings (1536D) → Embedding
         ↓
    Positional Encoding + Gaussian Noise + Dropout
         ↓
    Concatenation [Peptide | MHC]
         ↓
    2D-Masked Cross-Attention Transformer
         ├─ Prevents peptide self-attention
         ├─ Prevents MHC self-attention
         └─ Enables peptide-MHC cross-attention
         ↓
    Latent Sequence (B, P+M, D)
         ↓
    ┌────────────┬─────────────┬────────────┐
    ↓            ↓             ↓            ↓
Binding Head  Pep Recon.  MHC Recon.   Pooled Latent
(Binary)      (Seq2Seq)   (Seq2Seq)    (Clustering)
```

### Key Components

1. **MaskedEmbedding**: Applies token masking for denoising reconstruction
2. **PositionalEncoding**: RoPE (Rotary Position Embedding) for sequence position awareness
3. **SelfAttentionWith2DMask**: Custom attention mechanism with 2D masking
4. **Multi-Task Heads**:
   - **Binding Prediction**: Binary classification (binder/non-binder)
   - **Peptide Reconstruction**: Per-residue amino acid prediction
   - **MHC Reconstruction**: Per-residue amino acid prediction

### Loss Functions

- **Binding Loss**: Asymmetric Binary Cross-Entropy (higher penalty for false negatives)
- **Reconstruction Loss**: Masked Categorical Cross-Entropy
- **Total Loss**: Weighted combination with configurable task weights

### Metrics

- **Binary MCC** (Matthews Correlation Coefficient)
- **AUROC** (Area Under ROC Curve)
- **AUPRC** (Area Under Precision-Recall Curve)
- **Reconstruction Accuracy**

## 📈 Performance

PMBind achieves competitive performance on standard benchmarks:

| Dataset | AUROC | AUPRC | MCC |
|---------|-------|-------|-----|
| MHC-I (Cross-validation) | 0.92+ | 0.89+ | 0.75+ |
| MHC-II (Cross-validation) | 0.88+ | 0.85+ | 0.68+ |
| Novel Alleles (Leave-one-out) | 0.87+ | 0.83+ | 0.65+ |

*Note: Performance varies by allele and dataset. See publication for detailed benchmarks.*

### Advantages

- **Interpretability**: Attention weights reveal binding motifs and anchor positions
- **Generalization**: Strong performance on unseen MHC alleles
- **Multi-task Learning**: Joint reconstruction improves binding prediction
- **Embedding Quality**: Latent representations cluster by allele and binding strength

## 📚 Citation

If you use PMBind in your research, please cite:

```bibtex
@software{pmBind2025,
  author = {Aleyasin, Amirreza},
  title = {PMBind: Multi-Task Deep Learning for Peptide-MHC Binding Prediction},
  year = {2025},
  url = {https://github.com/Amirreza-0/PMBind},
  orcid = {0000-0003-2742-7138}
}
```

**Author ORCID**: [0000-0003-2742-7138](https://orcid.org/0000-0003-2742-7138)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** with clear commit messages
4. **Add tests** if applicable
5. **Update documentation** as needed
6. **Submit a pull request**

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write descriptive variable names

### Reporting Issues

When reporting bugs or requesting features, please include:
- Python version and OS
- TensorFlow version
- GPU specifications (if applicable)
- Minimal reproducible example
- Error messages and stack traces

## 🔧 Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors**
```bash
# Reduce batch size
python run_training.py --batch_size 256

# Use gradient accumulation
python run_training.py --gradient_accumulation_steps 4
```

**CUDA Out of Memory**
```bash
# Use mixed precision
python run_training.py --mixed_precision

# Reduce model size
python run_training.py --emb_dim 32 --heads 2 --transformer_layers 2
```

**Slow Training**
- Ensure TFRecords are generated (`create_tfrecords.py`)
- Use SSD storage for data files
- Enable mixed precision training
- Increase batch size if GPU memory allows

**Missing Dependencies**
```bash
# Reinstall with specific versions
pip install -r requirements.txt --force-reinstall
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Amirreza Aleyasin**
- Email: amirreza.alise@gmail.com
- ORCID: [0000-0003-2742-7138](https://orcid.org/0000-0003-2742-7138)
- GitHub: [@Amirreza-0](https://github.com/Amirreza-0)

For dataset access issues or questions, please contact the author directly.

## 🙏 Acknowledgments

PMBind builds upon several excellent tools and datasets:

- **ESM-2**: Protein language models from Meta AI / EvolutionaryScale
- **IEDB**: Immune Epitope Database for binding affinity data
- **TensorFlow/Keras**: Deep learning framework
- **BioPython**: Bioinformatics tools
- **BLOSUM62**: Amino acid substitution matrix

Special thanks to the immunoinformatics community for open data sharing and collaborative research.

---

**Last Updated**: November 2025 | **Version**: 1.0.0
