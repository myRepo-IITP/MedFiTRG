import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for REA (Readmission) multimodal model"""
    # Data paths
    METADATA_DIR = ""
    EHR_BASE_DIR = ""  # preprocessed with Normalized EHR directory
    CXR_BASE_DIR = ""
    
    # Readmission metadata files
    TRAIN_META = os.path.join(METADATA_DIR, "")
    VAL_META = os.path.join(METADATA_DIR, "")
    TEST_META = os.path.join(METADATA_DIR, "")
    
    # Model Hyperparameters
    EHR_EMBEDDING_DIM = 512
    CXR_EMBEDDING_DIM = 512
    TEXT_EMBEDDING_DIM = 512
    HIDDEN_DIM = 512
    DROPOUT = 0.2
    
    # EHR Transformer
    EHR_TRANSFORMER_NHEAD = 4
    EHR_TRANSFORMER_LAYERS = 2
    
    # FiLM
    FILM_HIDDEN_DIM = 256
    SOCIAL_K_NEIGHBORS = 5
    
    # Training
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True
    LEARNING_RATE = 5e-5
    FINETUNE_LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 20
    GPU_ID = 0
    
    # Text
    TEXT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
    MAX_TEXT_LENGTH = 512
    
    # Loss weights
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    CXR_AUX_WEIGHT = 0.4
    TEXT_AUX_WEIGHT = 0.4
    
    # REA Task (binary classification)
    NUM_CLASSES = 1
    
    # Output
    MODEL_SAVE_PATH = "best_model_rea.pth"
    LOG_DIR = "logs_rea"
    RESULTS_PATH = "rea_results.csv"