import torch
import numpy as np
import pandas as pd
from utils import evaluate_los

def test_model(config, test_loader, device, fusion_model, ehr_model, cxr_model, text_model, criterion):
    """Test the trained LOS model"""
    print(f"\n{'='*80}")
    print("FINAL TESTING WITH BEST MODEL")
    print(f"{'='*80}")
    
    # Load best model
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
    ehr_model.load_state_dict(checkpoint['ehr_model_state_dict'])
    cxr_model.load_state_dict(checkpoint['cxr_model_state_dict'])
    text_model.load_state_dict(checkpoint['text_model_state_dict'])
    
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation Macro AUROC: {checkpoint['val_metrics']['macro_auroc']:.4f}")
    print(f"  Learnable social threshold: {checkpoint['learnable_threshold']:.4f}")
    
    # Test evaluation
    test_metrics = evaluate_los(fusion_model, test_loader, criterion, 
                                ehr_model, cxr_model, text_model, device, config)
    
    print_results(test_metrics, config)
    
    return test_metrics

def print_results(test_metrics, config):
    """Print test results in a formatted way for LOS"""
    print(f"\n{'='*46}")
    print(f"{'TEST RESULTS':^46}")
    print(f"{'='*46}")
    print(f"{'METRIC':<20} | {'VALUE':<10}")
    print(f"{'-'*46}")
    print(f"{'Test Loss':<20} | {test_metrics['loss']:<10.4f}")
    print(f"{'Accuracy':<20} | {test_metrics['accuracy']:<10.4f}")
    print(f"{'Macro AUROC':<20} | {test_metrics['macro_auroc']:<10.4f}")
    print(f"{'Micro AUROC':<20} | {test_metrics['micro_auroc']:<10.4f}")
    print(f"{'Macro AUPRC':<20} | {test_metrics['macro_auprc']:<10.4f}")
    print(f"{'Micro AUPRC':<20} | {test_metrics['micro_auprc']:<10.4f}")
    print(f"{'Macro F1':<20} | {test_metrics['macro_f1']:<10.4f}")
    print(f"{'Micro F1':<20} | {test_metrics['micro_f1']:<10.4f}")
    print(f"{'Weighted F1':<20} | {test_metrics['weighted_f1']:<10.4f}")
    print(f"{'-'*46}")
    
    # Class-wise performance
    print(f"\n{'='*80}")
    print(f"{'CLASS-WISE PERFORMANCE':^80}")
    print(f"{'='*80}")
    print(f"{'CLASS':<30} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print(f"{'-'*80}")
    
    class_report = test_metrics['class_report']
    for i, class_name in enumerate(config.CLASS_NAMES):
        if str(i) in class_report:
            metrics = class_report[str(i)]
            print(f"{class_name:<30} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f}")
    
    # Confusion Matrix Summary
    cm = test_metrics['confusion_matrix']
    print(f"\n{'='*80}")
    print(f"{'CONFUSION MATRIX SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"Total Predictions: {np.sum(cm)}")
    print(f"Correct Predictions: {np.trace(cm)}")
    print(f"Overall Accuracy: {np.trace(cm)/np.sum(cm):.4f}")
    
    # Save final results
    test_results = {
        'metric': list(test_metrics.keys()), 
        'value': list(test_metrics.values())
    }
    pd.DataFrame(test_results).to_csv(config.TEST_RESULTS_PATH, index=False)

def save_history(history, config):
    """Save training history"""
    if history:
        pd.DataFrame(history).to_csv(config.HISTORY_PATH, index=False)
        print(f"  Training history saved to {config.HISTORY_PATH}")