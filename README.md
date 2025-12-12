# PMBind

PMBind is a deep learning framework for predicting peptide-MHC binding interactions using a multi-task transformer architecture.

> ⚠️ **Note**: This project is under active development. Features and documentation may change.

## Overview

PMBind predicts peptide binding to Major Histocompatibility Complex (MHC) molecules using:
- Multi-task learning (binding prediction + sequence reconstruction)
- 2D-masked cross-attention transformer
- ESM-C600m protein language model embeddings
- Physicochemical peptide encoding

## Installation

```bash
git clone https://github.com/Amirreza-0/PMBind.git
cd PMBind/
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, TensorFlow, CUDA-compatible GPU (recommended)

## Data

Download the dataset from the link in the `data/` folder:
- [Main Dataset](https://owncloud.gwdg.de/index.php/s/ly4uJe3CuGxWmUV)

Extract to the `data/` directory.

## Quick Start

### 1. Prepare Data
```bash
cd src/
python create_dataset.py
python preprocessing/run_ESM.py --input ../data/PMDb_alleles_sequences_aligned.csv \
                                --model esmc_600m --device cuda:0
python create_tfrecords.py
```

### 2. Train Model
```bash
python run_training.py --fold 1
```

### 3. Run Inference
```bash
python run_inference.py --model_weights best_model.weights.keras \
                       --test_data ../data/test.parquet
```

## Project Structure

```
PMBind/
├── src/              # Source code
│   ├── models.py     # Model architecture
│   ├── run_training.py
│   ├── run_inference.py
│   └── preprocessing/
├── data/             # Dataset directory
├── tests/            # Tests
└── requirements.txt
```

## Key Scripts

- `run_training.py` - Train the model
- `run_inference.py` - Run predictions

## Model Architecture

PMBind uses a transformer with:
- Peptide encoding: BLOSUM62 + physicochemical properties (14D)
- MHC encoding: ESM-C embeddings (1152D)
- 2D-masked cross-attention (prevents self-attention, enables cross-attention)
- Multi-task heads: binding prediction, peptide reconstruction, MHC reconstruction

## Citation

```bibtex
@software{pmBind2025,
  author = {Aleyasin, Amirreza},
  title = {PMBind: Multi-Task Deep Learning for Peptide-MHC Binding Prediction},
  year = {2025},
  url = {https://github.com/Amirreza-0/PMBind},
  orcid = {0000-0003-2742-7138}
}
```

**Author**: Amirreza Aleyasin ([ORCID: 0000-0003-2742-7138](https://orcid.org/0000-0003-2742-7138))

## License

Apache License 2.0 - see [LICENSE](LICENSE) file

## Contact

- Email: amirreza.alise@gmail.com
- GitHub: [@Amirreza-0](https://github.com/Amirreza-0)
