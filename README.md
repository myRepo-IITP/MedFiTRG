# **MedFiTRG**

**MedFiTRG** is a modular multimodal deep learning framework for **clinical outcome prediction** that integrates **Electronic Health Records (EHR)**, **Chest X-Ray (CXR) images**, and **clinical text**.  
It employs a **Modulated Graph Neural Network (MGNN)** with **three-level multimodal fusion** to achieve strong performance across four clinical tasks: **In-Hospital Mortality (IHM)**, **Length of Stay (LOS)**, **Phenotyping (PHE)**, and **Readmission (REA)**.

---

## 📋 Overview

This repository presents a unified and extensible framework for multimodal clinical reasoning by jointly modeling heterogeneous healthcare data:

- **Electronic Health Records (EHR):**  
  Structured temporal patient information, including vitals and laboratory measurements.

- **Chest X-Ray (CXR) Images:**  
  High-dimensional visual representations extracted from medical imaging.

- **Clinical Text:**  
  Unstructured information derived from clinical notes and diagnostic reports.

MedFiTRG adopts a **temporal–social graph-based modeling paradigm**, where **patient visits form temporal nodes** connected through both **longitudinal edges** and **social similarity edges**. These interactions are dynamically refined using a **Modulated Graph Neural Network (MGNN)**, enabling effective cross-modal fusion and improved modeling of longitudinal and population-level clinical dependencies.



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
```

### 2. Configure Paths

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
```

### 3. Run Any Task

#### Quick Start Commands

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

# Or use the provided shell script in the same task directory
```
