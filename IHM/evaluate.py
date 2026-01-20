import torch
import numpy as np
from utils import evaluate_ihm

def test_model(config, test_loader, device, fusion_model, ehr_model, cxr_model, text_model, criterion, best_threshold):
    """Test the trained model"""
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
    print(f"  Validation AUROC: {checkpoint['val_metrics']['auroc']:.4f}")
    print(f"  Optimal threshold: {checkpoint['threshold']:.3f}")
    print(f"  Learnable social threshold: {checkpoint['learnable_threshold']:.4f}")
    
    # Test evaluation
    test_metrics, _ = evaluate_ihm(
        fusion_model, test_loader, criterion, 
        ehr_model, cxr_model, text_model, device, 
        optimal_threshold=best_threshold
    )
    
    print_results(test_metrics)
    
    return test_metrics

def print_results(test_metrics):
    """Print test results in a formatted way"""
    print(f"\n  [TEST RESULTS]")
    print(f"{'-'*46}")
    print(f"{'METRIC':<20} | {'VALUE':<10}")
    print(f"{'-'*46}")
    print(f"{'Test Loss':<20} | {test_metrics['loss']:<10.4f}")
    print(f"{'AUROC':<20} | {test_metrics['auroc']:<10.4f}")
    print(f"{'AUPRC':<20} | {test_metrics['auprc']:<10.4f}")
    print(f"{'F1-Score':<20} | {test_metrics['f1_score']:<10.4f}")
    print(f"{'Accuracy':<20} | {test_metrics['accuracy']:<10.4f}")
    print(f"{'Precision':<20} | {test_metrics['precision']:<10.4f}")
    print(f"{'Recall':<20} | {test_metrics['recall']:<10.4f}")
    print(f"{'Specificity':<20} | {test_metrics['specificity']:<10.4f}")
    print(f"{'-'*46}")
    
    # Confusion Matrix Summary
    print(f"\n  [CONFUSION MATRIX]")
    print(f"    True Positives:  {test_metrics['true_positives']}")
    print(f"    False Positives: {test_metrics['false_positives']}")
    print(f"    False Negatives: {test_metrics['false_negatives']}")
    print(f"    True Negatives:  {test_metrics['true_negatives']}")