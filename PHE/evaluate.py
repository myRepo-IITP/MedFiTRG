import torch
import numpy as np
import pandas as pd
from utils import evaluate_phe

def test_model(config, test_loader, device, fusion_model, ehr_model, cxr_model, text_model, criterion, best_threshold):
    """Test the trained phenotype model"""
    print(f"\n{'='*80}")
    print("FINAL TESTING WITH BEST MODEL")
    print(f"{'='*80}")
    
    # Load best model
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    fusion_model.load_state_dict(checkpoint['fusion_model'])
    ehr_model.load_state_dict(checkpoint['ehr_model'])
    cxr_model.load_state_dict(checkpoint['cxr_model'])
    text_model.load_state_dict(checkpoint['text_model'])
    
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation Macro AUROC: {checkpoint['val_metrics']['macro_auroc']:.4f}")
    print(f"  Optimal threshold: {checkpoint['optimal_threshold']:.3f}")
    
    # Test evaluation
    test_metrics, _ = evaluate_phe(
        fusion_model, test_loader, criterion, 
        ehr_model, cxr_model, text_model, device, 
        optimal_threshold=best_threshold
    )
    
    print_results(test_metrics)
    save_results(test_metrics, config)
    
    return test_metrics

def print_results(test_metrics):
    """Print test results in a formatted way"""
    print(f"\n{'='*46}")
    print(f"{'TEST RESULTS':^46}")
    print(f"{'='*46}")
    print(f"{'METRIC':<20} | {'MACRO':<10} | {'MICRO':<10}")
    print(f"{'-'*46}")
    print(f"{'Test Loss':<20} | {'-':<10} | {test_metrics['loss']:<10.4f}")
    print(f"{'AUROC':<20} | {test_metrics['macro_auroc']:<10.4f} | {test_metrics['micro_auroc']:<10.4f}")
    print(f"{'AUPRC':<20} | {test_metrics['macro_auprc']:<10.4f} | {test_metrics['micro_auprc']:<10.4f}")
    print(f"{'F1 Score':<20} | {test_metrics['macro_f1']:<10.4f} | {test_metrics['micro_f1']:<10.4f}")
    print(f"{'-'*46}")
    print(f"{'Optimal Threshold':<20} | {test_metrics['optimal_threshold']:<21.3f}")

def save_results(test_metrics, config):
    """Save phenotype results to CSV"""
    results_df = pd.DataFrame({
        'metric': ['macro_auroc', 'micro_auroc', 'macro_auprc', 'micro_auprc', 
                   'macro_f1', 'micro_f1', 'optimal_threshold', 'loss'],
        'value': [test_metrics['macro_auroc'], test_metrics['micro_auroc'],
                  test_metrics['macro_auprc'], test_metrics['micro_auprc'],
                  test_metrics['macro_f1'], test_metrics['micro_f1'],
                  test_metrics['optimal_threshold'], test_metrics['loss']]
    })
    
    results_df.to_csv(config.RESULTS_PATH, index=False)
    print(f"  Results saved to {config.RESULTS_PATH}")