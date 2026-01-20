import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
from tqdm import tqdm

def calculate_class_weights(dataset, config, device):
    """Calculate class weights for LOS from training dataset"""
    print("\n[DATA] Calculating Class Weights...")
    
    # Get all LOS values from the dataframe directly
    y_true_values = dataset.metadata_df['y_true'].values
    
    # Convert LOS values to class indices
    all_labels = [dataset._los_to_class(y) for y in y_true_values]
    all_labels = np.array(all_labels)
    
    unique, counts = np.unique(all_labels, return_counts=True)
    weights = torch.zeros(config.NUM_CLASSES).to(device)
    total = sum(counts)
    
    print(f"Class Distribution:")
    for i, count in zip(unique, counts):
        if i < config.NUM_CLASSES:
            class_name = config.CLASS_NAMES[i]
            percentage = (count / total) * 100
            weights[i] = total / (config.NUM_CLASSES * count)
            print(f"  {class_name}: {count} samples ({percentage:.1f}%) -> weight: {weights[i]:.2f}")
    
    return weights

def evaluate_los(model, loader, criterion, ehr_model, cxr_model, text_model, device, config):
    """Evaluation function for LOS multi-class classification"""
    model.eval()
    ehr_model.eval()
    cxr_model.eval()
    text_model.eval()
    
    total_loss = 0
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            ehr_data = batch['ehr_data'].to(device)
            ehr_lens = batch['ehr_lengths'].to(device)
            labels = batch['class_labels'].to(device)
            
            cxr_emb = cxr_model(batch['cxr_data'].to(device)) if len(batch['cxr_indices']) > 0 else None
            text_emb = text_model(batch['text_input_ids'].to(device), batch['text_attention_masks'].to(device)) if len(batch['text_indices']) > 0 else None
            ehr_emb, ehr_raw = ehr_model(ehr_data, ehr_lens)
            
            # Pass visit_orders to forward for temporal fusion
            results = model(
                ehr_emb, cxr_emb, batch['cxr_indices'].to(device),
                text_emb, batch['text_indices'].to(device),
                batch['patient_indices'].to(device),
                batch['visit_orders'].to(device),  # Added for temporal fusion
                ehr_raw
            )
            
            loss = criterion(results['logits'], labels)
            total_loss += loss.item()
            all_logits.append(results['logits'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    # Probabilities and Predictions
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    
    # One-Hot encoding for multiclass metrics
    y_onehot = np.zeros((all_labels.size, config.NUM_CLASSES))
    y_onehot[np.arange(all_labels.size), all_labels] = 1
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': (preds == all_labels).mean(),
    }
    
    # F1 Scores
    metrics['macro_f1'] = f1_score(all_labels, preds, average='macro', zero_division=0)
    metrics['micro_f1'] = f1_score(all_labels, preds, average='micro', zero_division=0)
    metrics['weighted_f1'] = f1_score(all_labels, preds, average='weighted', zero_division=0)
    
    # AUROC
    try:
        metrics['macro_auroc'] = roc_auc_score(all_labels, probs, multi_class='ovr', average='macro')
        metrics['micro_auroc'] = roc_auc_score(y_onehot, probs, average='micro')
    except Exception as e: 
        metrics['macro_auroc'] = 0.5
        metrics['micro_auroc'] = 0.5
        
    # AUPRC
    try:
        metrics['macro_auprc'] = average_precision_score(y_onehot, probs, average='macro')
        metrics['micro_auprc'] = average_precision_score(y_onehot, probs, average='micro')
    except Exception as e:
        metrics['macro_auprc'] = 0.0
        metrics['micro_auprc'] = 0.0
        
    # Class-wise metrics
    class_report = classification_report(all_labels, preds, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)
    metrics['class_report'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)
    metrics['confusion_matrix'] = cm
    
    return metrics

def train_epoch_los(model, loader, optimizer, criterion, ehr_model, cxr_model, text_model, device, config):
    """Training function for LOS multi-class classification"""
    model.train()
    ehr_model.train()
    cxr_model.train()
    text_model.train()
    
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        ehr_data = batch['ehr_data'].to(device)
        ehr_lens = batch['ehr_lengths'].to(device)
        labels = batch['class_labels'].to(device)
        
        cxr_emb = cxr_model(batch['cxr_data'].to(device)) if len(batch['cxr_indices']) > 0 else None
        text_emb = text_model(batch['text_input_ids'].to(device), batch['text_attention_masks'].to(device)) if len(batch['text_indices']) > 0 else None
        ehr_emb, ehr_raw = ehr_model(ehr_data, ehr_lens)
        
        optimizer.zero_grad()
        
        # Pass visit_orders to forward for temporal fusion
        results = model(
            ehr_emb, cxr_emb, batch['cxr_indices'].to(device),
            text_emb, batch['text_indices'].to(device),
            batch['patient_indices'].to(device),
            batch['visit_orders'].to(device),  # Added for temporal fusion
            ehr_raw
        )
        
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