import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

def calculate_class_weights(train_loader):
    """Calculate class weights for readmission from training data"""
    print("[DATA] Calculating class weights for readmission...")
    
    all_labels = []
    for batch in tqdm(train_loader, desc="Collecting labels"):
        all_labels.append(batch['labels'].numpy())
    
    all_labels = np.concatenate(all_labels).flatten()
    positive_ratio = np.nanmean(all_labels)
    
    if positive_ratio > 0:
        pos_weight = torch.tensor((1 - positive_ratio) / positive_ratio)
    else:
        pos_weight = torch.tensor(1.0)
    
    print(f"  Readmission Rate: {positive_ratio:.4f}")
    print(f"  Class distribution - Positive: {np.sum(all_labels)}, Negative: {len(all_labels) - np.sum(all_labels)}")
    print(f"  Positive class weight: {pos_weight.item():.2f}")
    
    return pos_weight, positive_ratio

def evaluate_rea(model, loader, criterion, ehr_model, cxr_model, text_model, device, optimal_threshold=None):
    """Evaluation function for binary readmission classification"""
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
    
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Handle NaNs
    all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)
    all_probs = np.clip(all_probs, 0.0, 1.0)
    
    # Find Optimal Threshold if not provided
    if optimal_threshold is None:
        best_f1, best_thr = 0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > thr).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        optimal_threshold = best_thr
    
    final_preds = (all_probs > optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, final_preds).ravel()
    
    metrics = {
        'loss': total_loss / len(loader),
        'optimal_threshold': optimal_threshold,
        'auroc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'auprc': average_precision_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, final_preds),
        'precision': precision_score(all_labels, final_preds, zero_division=0),
        'recall': recall_score(all_labels, final_preds, zero_division=0),
        'f1_score': f1_score(all_labels, final_preds, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    return metrics, optimal_threshold

def train_epoch_rea(model, loader, optimizer, criterion, ehr_model, cxr_model, text_model, device, config):
    """Training function for binary readmission classification"""
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
            cxr_targets = labels[batch['cxr_indices'].to(device)]
            aux_loss += config.CXR_AUX_WEIGHT * criterion(results['cxr_aux'], cxr_targets)
            
        if results['text_aux'] is not None:
            text_targets = labels[batch['text_indices'].to(device)]
            aux_loss += config.TEXT_AUX_WEIGHT * criterion(results['text_aux'], text_targets)
            
        final_loss = main_loss + aux_loss
        final_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += final_loss.item()
        
    return total_loss / len(loader)