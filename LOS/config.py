import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for LOS multimodal model"""
    # Data paths
    METADATA_DIR = ""
    EHR_BASE_DIR = "" # preprocessed with Normalized EHR directory
    CXR_BASE_DIR = ""
    
    # Metadata files for LOS
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
    CXR_AUX_WEIGHT = 0.1
    TEXT_AUX_WEIGHT = 0.5
    
    # Focal Loss
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    
    # LOS Task (9-class classification)
    NUM_CLASSES = 9  
    CLASS_NAMES = [
        'LOS < 1 day',
        'LOS = 1 day',
        'LOS = 2 days',
        'LOS = 3 days',
        'LOS = 4 days',
        'LOS = 5 days',
        'LOS = 6 days',
        'LOS = 7 days',
        '7 < LOS ≤ 14 days'
    ]
    
    # Output
    MODEL_SAVE_PATH = "best_model_los_triplefilm.pth"
    LOG_DIR = "logs"
    HISTORY_PATH = "training_history_los.csv"
    TEST_RESULTS_PATH = "test_results_los.csv"