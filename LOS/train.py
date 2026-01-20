import torch
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import os

from config import Config
from models import TransformerEHREmbeddingExtractor, CXRFeatureExtractor, TextEmbeddingExtractor, TripleFiLMNetwork
from dataset import LOSGraphDataset, LOSGraphCollator
from losses import RobustFocalLoss
from utils import calculate_class_weights, train_epoch_los, evaluate_los

def setup_dataloaders(config, tokenizer):
    """Setup train, validation, and test dataloaders for LOS"""
    print("[DATA] Loading LOS datasets...")
    
    # Datasets for LOS
    train_ds = LOSGraphDataset(config.TRAIN_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "train", tokenizer)
    val_ds = LOSGraphDataset(config.VAL_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "val", tokenizer)
    test_ds = LOSGraphDataset(config.TEST_META, config.EHR_BASE_DIR, config.CXR_BASE_DIR, "test", tokenizer)
    
    print(f"  Training patients: {len(train_ds)}")
    print(f"  Validation patients: {len(val_ds)}")
    print(f"  Test patients: {len(test_ds)}")
    
    collator = LOSGraphCollator(max_len=config.MAX_TEXT_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, test_loader, train_ds

def setup_models(config, train_ds, device):
    """Initialize all models for LOS"""
    # Feature Extractors
    ehr_input_dim = len(train_ds.feature_columns) + len(train_ds.mask_columns)
    ehr_model = TransformerEHREmbeddingExtractor(ehr_input_dim, config.EHR_EMBEDDING_DIM).to(device)
    cxr_model = CXRFeatureExtractor(config.CXR_EMBEDDING_DIM).to(device)
    text_model = TextEmbeddingExtractor(config.TEXT_MODEL_NAME, config.TEXT_EMBEDDING_DIM).to(device)
    
    cxr_model.unfreeze_layers(1)
    text_model.unfreeze_layers(1)
    
    # Fusion Model for LOS (9 classes)
    fusion_model = TripleFiLMNetwork(
        ehr_dim=config.EHR_EMBEDDING_DIM, cxr_dim=config.CXR_EMBEDDING_DIM, text_dim=config.TEXT_EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM, num_classes=config.NUM_CLASSES, film_hidden_dim=config.FILM_HIDDEN_DIM, dropout=config.DROPOUT
    ).to(device)
    
    return ehr_model, cxr_model, text_model, fusion_model

def setup_optimizer(config, fusion_model, ehr_model, cxr_model, text_model):
    """Setup optimizer with different learning rates for LOS"""
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
    """Main training loop for LOS"""
    best_macro_auroc = 0.0
    epochs_without_improvement = 0
    history = [] 
    
    print(f"\n{'='*80}")
    print(f"{'EPOCH':<6} | {'TRN LOSS':<9} | {'VAL LOSS':<9} | {'ACCURACY':<8} | {'MACRO AUROC':<12} | {'MACRO F1':<8} | {'LR':<10}")
    print(f"{'='*80}")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch_los(fusion_model, train_loader, optimizer, criterion, 
                                     ehr_model, cxr_model, text_model, device, config)
        val_metrics = evaluate_los(fusion_model, val_loader, criterion, 
                                   ehr_model, cxr_model, text_model, device, config)
        
        scheduler.step(val_metrics['macro_auroc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get the current learnable threshold from the social fusion layer
        learnable_threshold = fusion_model.social_fusion.threshold.item()
        
        # Print epoch summary in table format
        print(f"{epoch:<6} | {train_loss:<9.4f} | {val_metrics['loss']:<9.4f} | {val_metrics['accuracy']:<8.4f} | "
              f"{val_metrics['macro_auroc']:<12.4f} | {val_metrics['macro_f1']:<8.4f} | {current_lr:<10.2e}")
        
        # Save logs
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_auroc': val_metrics['macro_auroc'],
            'val_micro_auroc': val_metrics['micro_auroc'],
            'val_macro_auprc': val_metrics['macro_auprc'],
            'val_micro_auprc': val_metrics['micro_auprc'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_micro_f1': val_metrics['micro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'learnable_threshold': learnable_threshold
        }
        history.append(epoch_log)
        
        # Early Stopping Logic
        if val_metrics['macro_auroc'] > best_macro_auroc:
            best_macro_auroc = val_metrics['macro_auroc']
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'fusion_model_state_dict': fusion_model.state_dict(),
                'ehr_model_state_dict': ehr_model.state_dict(),
                'cxr_model_state_dict': cxr_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'class_weights': criterion.class_weights.cpu() if criterion.class_weights is not None else None,
                'learnable_threshold': learnable_threshold,
                'config': config.__dict__
            }
            
            torch.save(checkpoint, config.MODEL_SAVE_PATH)
            print(f"  [✓] Model saved with Macro AUROC: {best_macro_auroc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE // 2:
                print(f"  [ ] No improvement for {epochs_without_improvement} epochs")
        
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\n  [!] Early stopping triggered after {epoch} epochs")
            break
            
    return best_macro_auroc, history