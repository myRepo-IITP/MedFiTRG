import torch
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import os

from config import Config
from models import TransformerEHREmbeddingExtractor, CXRFeatureExtractor, TextEmbeddingExtractor, TripleFiLMNetwork
from dataset import PatientGraphDataset, PatientGraphCollator
from losses import RobustFocalLoss
from utils import calculate_class_weights, train_epoch_phe, evaluate_phe

def setup_dataloaders(config, tokenizer, phenotype_columns):
    """Setup train, validation, and test dataloaders for phenotype prediction"""
    print("[DATA] Loading phenotype datasets...")
    
    # Datasets for phenotype prediction
    train_ds = PatientGraphDataset(config.TRAIN_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, 
                                   "train", phenotype_columns, tokenizer)
    val_ds = PatientGraphDataset(config.VAL_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, 
                                 "val", phenotype_columns, tokenizer)
    test_ds = PatientGraphDataset(config.TEST_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, 
                                  "test", phenotype_columns, tokenizer)
    
    print(f"  Training patients: {len(train_ds)}")
    print(f"  Validation patients: {len(val_ds)}")
    print(f"  Test patients: {len(test_ds)}")
    print(f"  Number of phenotypes: {len(phenotype_columns)}")
    
    collator = PatientGraphCollator(max_len=config.MAX_TEXT_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                              collate_fn=collator, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                            collate_fn=collator, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                             collate_fn=collator, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, test_loader, train_ds

def setup_models(config, train_ds, num_classes, device):
    """Initialize all models for phenotype prediction"""
    # Feature Extractors
    ehr_input_dim = len(train_ds.feature_columns) + len(train_ds.mask_columns)
    ehr_model = TransformerEHREmbeddingExtractor(ehr_input_dim, config.EHR_EMBEDDING_DIM).to(device)
    cxr_model = CXRFeatureExtractor(config.CXR_EMBEDDING_DIM).to(device)
    text_model = TextEmbeddingExtractor(config.TEXT_MODEL_NAME, config.TEXT_EMBEDDING_DIM).to(device)
    
    cxr_model.unfreeze_layers(1)
    text_model.unfreeze_layers(1)
    
    # Fusion Model for phenotypes (multi-label classification)
    fusion_model = TripleFiLMNetwork(
        ehr_dim=config.EHR_EMBEDDING_DIM, cxr_dim=config.CXR_EMBEDDING_DIM, text_dim=config.TEXT_EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM, num_classes=num_classes, film_hidden_dim=config.FILM_HIDDEN_DIM, 
        dropout=config.DROPOUT
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
    """Main training loop for phenotype prediction"""
    best_macro_auroc = 0.0
    best_threshold = 0.5
    epochs_no_imp = 0
    
    print(f"\n{'='*80}")
    print(f"{'EPOCH':<6} | {'TRN LOSS':<9} | {'VAL LOSS':<9} | {'METRIC':<10} | {'MACRO':<8} | {'MICRO':<8}")
    print(f"{'='*80}")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch_phe(fusion_model, train_loader, optimizer, criterion, 
                                     ehr_model, cxr_model, text_model, device, config)
        val_metrics, val_threshold = evaluate_phe(fusion_model, val_loader, criterion, 
                                                  ehr_model, cxr_model, text_model, device)
        
        scheduler.step(val_metrics['macro_auroc'])
        
        # Print epoch summary
        print(f"{epoch:<6} | {train_loss:<9.4f} | {val_metrics['loss']:<9.4f} | {'AUROC':<10} | "
              f"{val_metrics['macro_auroc']:<8.4f} | {val_metrics['micro_auroc']:<8.4f}")
        print(f"{'':<6} | {'':<9} | {'':<9} | {'AUPRC':<10} | "
              f"{val_metrics['macro_auprc']:<8.4f} | {val_metrics['micro_auprc']:<8.4f}")
        print(f"{'':<6} | {'':<9} | {'':<9} | {'F1':<10} | "
              f"{val_metrics['macro_f1']:<8.4f} | {val_metrics['micro_f1']:<8.4f}")
        print(f"{'-'*80}")
        
        # Check for improvement
        if val_metrics['macro_auroc'] > best_macro_auroc:
            best_macro_auroc = val_metrics['macro_auroc']
            best_threshold = val_threshold
            epochs_no_imp = 0
            
            checkpoint = {
                'epoch': epoch,
                'fusion_model': fusion_model.state_dict(),
                'ehr_model': ehr_model.state_dict(),
                'cxr_model': cxr_model.state_dict(),
                'text_model': text_model.state_dict(),
                'optimal_threshold': best_threshold,
                'val_metrics': val_metrics,
                'config': config.__dict__
            }
            
            torch.save(checkpoint, config.MODEL_SAVE_PATH)
            print(f" [✓] Model Saved (Threshold: {best_threshold:.3f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= config.PATIENCE // 2:
                print(f"  [ ] No improvement for {epochs_no_imp} epochs")
        
        # Early stopping
        if epochs_no_imp >= config.PATIENCE:
            print(" [!] Early Stopping Triggered")
            break
            
    return best_macro_auroc, best_threshold