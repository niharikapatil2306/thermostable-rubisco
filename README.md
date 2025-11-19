# Thermostable RuBisCO Protein Engineering

This repository contains the implementation of a diffusion-based machine learning approach for engineering thermostable variants of RuBisCO (Ribulose-1,5-bisphosphate carboxylase/oxygenase) proteins. The project leverages advanced deep learning techniques, including transformer architectures and BLOSUM-guided diffusion models, to generate protein sequences with enhanced thermal stability.

## Overview

RuBisCO is a critical enzyme in photosynthesis, catalyzing the first major step of carbon fixation. This project aims to computationally design thermostable RuBisCO variants that maintain functionality at elevated temperatures, which has significant implications for improving crop yields under climate change conditions.

The approach combines:
- Atomic-level structural information from protein data bank (PDB) files
- BLOSUM62 substitution matrices for biologically-informed sequence generation
- Transformer-based diffusion models for protein sequence design
- Thermostability prediction using random forest regression

## Repository Structure
```
thermostable-rubisco/
├── src/
│   ├── pdb_analysis.py          # PDB structure extraction and analysis
│   ├── train_bf.py              # Diffusion model training pipeline
│   └── thermo_final.py          # Thermostable variant generation
├── container/
│   └── pdb_analysis.def         # Singularity container definition
├── jobscripts/
│   ├── thermo_final_slurm.sh    # SLURM job script for generation
│   └── train_bf_slurm.sh        # SLURM job script for training
├── dataset.csv                   # Thermophile protein dataset
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Key Features

### 1. PDB Structure Analysis (`pdb_analysis.py`)
- Extracts comprehensive structural information from PDB files
- Computes secondary structure using DSSP
- Calculates phi/psi angles and neighbor counts
- Processes atomic coordinates and confidence scores
- Outputs structured CSV data for model training

### 2. BLOSUM-Guided Diffusion Model (`train_bf.py`)
- Implements atomic-enhanced transformer architecture
- Uses BLOSUM62 matrix for biologically-informed noise addition
- Incorporates multi-task learning:
  - Sequence prediction
  - Atomic coordinate prediction
  - Secondary structure prediction
  - Dihedral angle prediction
- Supports distributed training across multiple GPUs
- Progressive diffusion schedule for improved generation

### 3. Thermostability Prediction and Generation (`thermo_final.py`)
- Random forest-based thermostability predictor
- Analyzes thermophile protein patterns
- Generates RuBisCO variants with targeted thermal stability
- Bias toward amino acid compositions found in thermophilic organisms
- Comprehensive sequence analysis and validation

## Requirements

### Python Dependencies
- PyTorch (with CUDA support)
- Biopython
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### System Requirements
- CUDA-capable GPU (recommended for training)
- SLURM cluster (for HPC execution)
- Singularity (for containerized execution)
- DSSP installation (for secondary structure calculation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/niharikapatil2306/thermostable-rubisco.git
cd thermostable-rubisco
```

2. Install Python dependencies:
```bash
pip install torch biopython pandas numpy scikit-learn matplotlib seaborn
```

3. Install DSSP (required for secondary structure analysis):
```bash
# Ubuntu/Debian
sudo apt-get install dssp

# macOS
brew install dssp
```

4. Build Singularity container (optional):
```bash
singularity build pdb_analysis.sif container/pdb_analysis.def
```

## Usage

### Step 1: Extract Structural Features from PDB Files

```bash
python src/pdb_analysis.py
```

This script processes PDB files from the `pdb_data` directory and generates a CSV file containing:
- Protein sequences
- Atomic coordinates
- Secondary structure assignments
- Dihedral angles
- Confidence scores
- Neighbor counts

### Step 2: Train the Diffusion Model

```bash
python src/train_bf.py
```

Training configuration can be modified in the `main()` function. Key parameters:
- `d_model`: Transformer dimension (default: 512)
- `num_layers`: Number of transformer layers (default: 12)
- `num_heads`: Number of attention heads (default: 16)
- `batch_size`: Training batch size (default: 4)
- `num_epochs`: Maximum training epochs (default: 500)
- `T`: Number of diffusion timesteps (default: 1000)

For distributed training on SLURM:
```bash
sbatch jobscripts/train_bf_slurm.sh
```

### Step 3: Generate Thermostable Variants

```bash
python src/thermo_final.py
```

Generation parameters:
- `target_temperature`: Desired thermostability (default: 90°C)
- `thermostable_guidance`: Guidance strength for thermostability (default: 0.95)
- `blosum_guidance`: BLOSUM matrix guidance strength (default: 0.4)
- `num_samples`: Number of sequences to generate (default: 100)
- `num_steps`: Reverse diffusion steps (default: 250)

For SLURM execution:
```bash
sbatch jobscripts/thermo_final_slurm.sh
```

## Output Files

The pipeline generates several output files:

1. **PDB Analysis Output**
   - `final_data.csv`: Processed structural features

2. **Model Training Output**
   - `best_rubisco_diffusion_model.pth`: Best model checkpoint
   - `rubisco_sequences_guidance_*.csv`: Generated sequences with different guidance strengths

3. **Thermostable Generation Output**
   - `thermostable_rubisco_final.json`: Detailed results with predictions
   - `thermostable_rubisco_final.csv`: Tabular results
   - `thermostable_rubisco_sequences.fasta`: Best sequences in FASTA format

## Model Architecture

The atomic-enhanced RuBisCO model incorporates:

- **Sequence Embedding**: Standard token embeddings with positional encoding
- **Time Embedding**: Sinusoidal timestep embeddings for diffusion
- **Atomic Feature Processing**: Multi-head attention over atomic coordinates
- **Feature Fusion**: Integration of structural, sequential, and confidence features
- **Transformer Encoder**: 12-layer transformer with 16 attention heads
- **Multi-task Heads**: Separate prediction heads for sequences, structures, and properties

## Thermostability Prediction

The thermostability predictor uses:

- Amino acid composition features
- Hydrophobicity ratios
- Charge distribution
- Secondary structure propensities
- Sequence-based features correlated with thermal stability

Random forest regression trained on thermophilic protein dataset achieves reliable temperature predictions for guiding sequence generation.

## Performance Considerations

- **Training Time**: Approximately 8-12 hours on 4x A100 GPUs for 500 epochs
- **Memory Requirements**: ~40GB GPU memory for batch size 4
- **Generation Time**: ~2-3 minutes per sequence with 250 diffusion steps
- **Recommended Batch Size**: 4-8 depending on GPU memory

## Citation

If you use this code in your research, please cite:

```bibtex
@software{thermostable_rubisco,
  author = {Patil, Niharika},
  title = {Thermostable RuBisCO Protein Engineering},
  year = {2025},
  url = {https://github.com/niharikapatil2306/thermostable-rubisco}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BLOSUM62 substitution matrix from Henikoff & Henikoff (1992)
- Diffusion model architecture inspired by recent advances in protein generation
- PDB structure processing using Biopython and DSSP

## Contact

Niharika Patil - [@niharikapatil2306](https://github.com/niharikapatil2306)

For questions or issues, please open an issue on GitHub.
