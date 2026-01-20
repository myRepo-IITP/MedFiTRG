import torch
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import numpy as np
import os

from config import Config
from models import TransformerEHREmbeddingExtractor, CXRFeatureExtractor, TextEmbeddingExtractor, TripleFiLMNetwork
from dataset import IHMGraphDataset, IHMGraphCollator
from losses import RobustFocalLoss
from utils import calculate_class_weights, train_epoch_ihm, evaluate_ihm

def setup_dataloaders(config, tokenizer):
    """Setup train, validation, and test dataloaders"""
    print("[DATA] Loading IHM datasets...")
    
    # Datasets for IHM
    train_ds = IHMGraphDataset(config.TRAIN_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "train", tokenizer)
    val_ds = IHMGraphDataset(config.VAL_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "val", tokenizer)
    test_ds = IHMGraphDataset(config.TEST_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "test", tokenizer)
    
    print(f"  Training samples: {len(train_ds)} patient stays")
    print(f"  Validation samples: {len(val_ds)} patient stays")
    print(f"  Test samples: {len(test_ds)} patient stays")
    
    collator = IHMGraphCollator(max_len=config.MAX_TEXT_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, test_loader, train_ds

def setup_models(config, train_ds, device):
    """Initialize all models"""
    # Feature Extractors
    ehr_input_dim = len(train_ds.feature_columns) + len(train_ds.mask_columns)
    ehr_model = TransformerEHREmbeddingExtractor(ehr_input_dim, config.EHR_EMBEDDING_DIM).to(device)
    cxr_model = CXRFeatureExtractor(config.CXR_EMBEDDING_DIM).to(device)
    text_model = TextEmbeddingExtractor(config.TEXT_MODEL_NAME, config.TEXT_EMBEDDING_DIM).to(device)
    
    cxr_model.unfreeze_layers(1)
    text_model.unfreeze_layers(1)
    
    # Fusion Model
    fusion_model = TripleFiLMNetwork(
        ehr_dim=config.EHR_EMBEDDING_DIM, cxr_dim=config.CXR_EMBEDDING_DIM, text_dim=config.TEXT_EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM, num_classes=config.NUM_CLASSES, film_hidden_dim=config.FILM_HIDDEN_DIM, dropout=config.DROPOUT
    ).to(device)
    
    return ehr_model, cxr_model, text_model, fusion_model

def setup_optimizer(config, fusion_model, ehr_model, cxr_model, text_model):
    """Setup optimizer with different learning rates"""
    optimizer = optim.AdamW([
        {'params': fusion_model.parameters(), 'lr': config.LEARNING_RATE},
        {'params': ehr_model.parameters(), 'lr': config.LEARNING_RATE},
        {'params': cxr_model.projection.parameters(), 'lr': config.LEARNING_RATE},
        {'params': text_model.projection.parameters(), 'lr': config.LEARNING_RATE},
        {'params': filter(lambda p: p.requires_grad, cxr_model.model.parameters()), 'lr': config.FINETUNE_LEARNING_RATE},
        {'params': filter(lambda p: p.requires_grad, text_model.bert_model.parameters()), 'lr': config.FINETUNE_LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    return optimizer

def train_model(config, train_loader, val_loader, device, 
                fusion_model, ehr_model, cxr_model, text_model, 
                criterion, optimizer, scheduler):
    """Main training loop"""
    best_auroc = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    
    print(f"\n{'='*80}")
    print(f"{'EPOCH':<6} | {'TRN LOSS':<9} | {'VAL LOSS':<9} | {'AUROC':<8} | {'F1':<8} | {'AUPRC':<8} | {'THRESH':<7} | {'LR':<10}")
    print(f"{'='*80}")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch_ihm(fusion_model, train_loader, optimizer, criterion, 
                                     ehr_model, cxr_model, text_model, device, config)
        
        # Validate
        val_metrics, val_threshold = evaluate_ihm(fusion_model, val_loader, criterion, 
                                                  ehr_model, cxr_model, text_model, device)
        
        scheduler.step(val_metrics['auroc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get the current learnable threshold from the social fusion layer
        learnable_threshold = fusion_model.social_fusion.threshold.item()
        
        # Print epoch summary in table format
        print(f"{epoch:<6} | {train_loss:<9.4f} | {val_metrics['loss']:<9.4f} | {val_metrics['auroc']:<8.4f} | "
              f"{val_metrics['f1_score']:<8.4f} | {val_metrics['auprc']:<8.4f} | {val_threshold:<7.3f} | {current_lr:<10.2e}")
        
        # Check for improvement in validation AUROC
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_threshold = val_threshold
            epochs_without_improvement = 0  # Reset counter
            
            checkpoint = {
                'epoch': epoch,
                'fusion_model_state_dict': fusion_model.state_dict(),
                'ehr_model_state_dict': ehr_model.state_dict(),
                'cxr_model_state_dict': cxr_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'threshold': best_threshold,
                'learnable_threshold': learnable_threshold,
                'config': config.__dict__
            }
            
            torch.save(checkpoint, config.MODEL_SAVE_PATH)
            print(f"  [✓] Model saved with AUROC: {best_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE // 2:
                print(f"  [ ] No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping check
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\n  [!] Early stopping triggered after {epoch} epochs")
            break
    
    return best_auroc, best_threshold