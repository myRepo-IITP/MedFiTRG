# MedFiTRG

A modular framework for clinical outcome prediction using multimodal deep learning with EHR, CXR, and clinical text data. Features TripleFiLM Network with 3-level fusion for superior performance across four clinical tasks(IHM, LOS, PHE, REA).

## 📋 Overview

This repository implements a comprehensive multimodal deep learning framework for clinical outcome prediction, combining:
- **Electronic Health Records (EHR)** - Structured patient data
- **Chest X-Ray (CXR) Images** - Visual medical imaging data  
- **Clinical Text** - Unstructured medical notes and reports

The core architecture features a **TripleFiLM Network** with 3-level feature fusion for optimal multimodal integration.

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/myRepo-IITP/MedFiTRG
cd MedFiTRG

# Create and activate conda environment
conda create -n clinical_env python=3.8
conda activate clinical_env

# Install dependencies
pip install torch torchvision
pip install transformers torchxrayvision
pip install pandas numpy scikit-learn matplotlib seaborn
pip install pillow pydicom


## Configure Paths

Before running any experiments, configure your data and output paths. Each task directory contains its own `config.py` file that you need to edit:

```python
# config.py (example for ihm directory)
# ====================================================
# DATA PATHS - UPDATE THESE TO YOUR LOCAL PATHS
# ====================================================

# Base directories for different data modalities
METADATA_DIR = "/path/to/your/mimic/metadata"      # CSV files with patient IDs and labels
EHR_BASE_DIR = "/path/to/your/mimic-iv/data"       # EHR data (CSV files from MIMIC-IV)
CXR_BASE_DIR = "/path/to/your/mimic-cxr/images"    # CXR images (PNG/DICOM format)
TEXT_BASE_DIR = "/path/to/your/clinical/notes"     # Clinical text files
OUTPUT_DIR = "/path/to/save/results"               # Where to save models and outputs

## Run Any Task

### Quick Start Commands

Get started with any clinical prediction task using these simple commands:

```bash
# Navigate to your desired task directory
cd IHM      # In-Hospital Mortality
# or
cd LOS      # Length of Stay
# or
cd PHE      # Phenotype Prediction
# or
cd REA      # Readmission

# Run training and testing in one command
python run_ihm.py --mode both --gpu 0 --batch_size 32 --epochs 50

# Or use the provided shell script in the same task directory.

