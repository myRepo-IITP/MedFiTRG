import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

def calculate_class_weights(train_loader, num_classes):
    """Calculate class weights for multi-label phenotype classification"""
    print("[DATA] Calculating class weights for phenotypes...")
    
    all_labels = []
    for batch in tqdm(train_loader, desc="Collecting labels"):
        all_labels.append(batch['labels'].numpy())
    
    all_labels = np.concatenate(all_labels)
    positive_ratios = np.nanmean(all_labels, axis=0)
    
    # Calculate positive weights (inverse of positive ratio)
    pos_weights = torch.tensor(1.0 / (positive_ratios + 1e-8))
    
    print(f"  Number of phenotypes: {num_classes}")
    print(f"  Average positive ratio per phenotype: {np.mean(positive_ratios):.4f}")
    print(f"  Min positive ratio: {np.min(positive_ratios):.4f}, Max: {np.max(positive_ratios):.4f}")
    
    return pos_weights

def evaluate_phe(model, loader, criterion, ehr_model, cxr_model, text_model, device, optimal_threshold=None):
    """Evaluation function for multi-label phenotype classification"""
    model.eval()
    ehr_model.eval()
    cxr_model.eval()
    text_model.eval()
    
    total_loss = 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            ehr_data = batch['ehr_data'].to(device)
            ehr_lens = batch['ehr_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            cxr_emb = cxr_model(batch['cxr_data'].to(device)) if len(batch['cxr_indices']) > 0 else None
            text_emb = text_model(batch['text_input_ids'].to(device), batch['text_attention_masks'].to(device)) if len(batch['text_indices']) > 0 else None
            ehr_emb, ehr_raw = ehr_model(ehr_data, ehr_lens)
            
            # Pass visit_orders to forward
            results = model(
                ehr_emb, cxr_emb, batch['cxr_indices'].to(device),
                text_emb, batch['text_indices'].to(device),
                batch['patient_indices'].to(device),
                batch['visit_orders'].to(device),  # Added for temporal fusion
                ehr_raw
            )
            
            loss = criterion(results['logits'], labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(results['logits']).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Find optimal threshold if not provided
    if optimal_threshold is None:
        best_f1, best_thr = 0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > thr).astype(int)
            f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        optimal_threshold = best_thr
    
    final_preds = (all_probs > optimal_threshold).astype(int)
    
    metrics = {
        'loss': total_loss / len(loader),
        'optimal_threshold': optimal_threshold
    }
    
    # AUROC
    try:
        metrics['macro_auroc'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['micro_auroc'] = roc_auc_score(all_labels, all_probs, average='micro')
    except:
        metrics['macro_auroc'] = 0.5
        metrics['micro_auroc'] = 0.5

    # AUPRC
    try:
        metrics['macro_auprc'] = average_precision_score(all_labels, all_probs, average='macro')
        metrics['micro_auprc'] = average_precision_score(all_labels, all_probs, average='micro')
    except:
        metrics['macro_auprc'] = 0.0
        metrics['micro_auprc'] = 0.0

    # F1 Scores
    try:
        metrics['macro_f1'] = f1_score(all_labels, final_preds, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(all_labels, final_preds, average='micro', zero_division=0)
    except:
        metrics['macro_f1'] = 0.0
        metrics['micro_f1'] = 0.0
        
    return metrics, optimal_threshold

def train_epoch_phe(model, loader, optimizer, criterion, ehr_model, cxr_model, text_model, device, config):
    """Training function for multi-label phenotype classification"""
    model.train()
    ehr_model.train()
    cxr_model.train()
    text_model.train()
    
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        ehr_data = batch['ehr_data'].to(device)
        ehr_lens = batch['ehr_lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # Modality extractions
        cxr_emb = cxr_model(batch['cxr_data'].to(device)) if len(batch['cxr_indices']) > 0 else None
        text_emb = text_model(batch['text_input_ids'].to(device), batch['text_attention_masks'].to(device)) if len(batch['text_indices']) > 0 else None
        ehr_emb, ehr_raw = ehr_model(ehr_data, ehr_lens)
        
        optimizer.zero_grad()
        
        # Pass visit_orders to forward
        results = model(
            ehr_emb, cxr_emb, batch['cxr_indices'].to(device),
            text_emb, batch['text_indices'].to(device),
            batch['patient_indices'].to(device),
            batch['visit_orders'].to(device),  # Added for temporal fusion
            ehr_raw
        )
        
        # Multi-objective loss
        main_loss = criterion(results['logits'], labels)
        
        aux_loss = 0
        if results['cxr_aux'] is not None:
            aux_loss += config.CXR_AUX_WEIGHT * criterion(
                results['cxr_aux'], 
                labels[batch['cxr_indices'].to(device)]
            )
        if results['text_aux'] is not None:
            aux_loss += config.TEXT_AUX_WEIGHT * criterion(
                results['text_aux'], 
                labels[batch['text_indices'].to(device)]
            )
            
        final_loss = main_loss + aux_loss
        final_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += final_loss.item()
        
    return total_loss / len(loader)