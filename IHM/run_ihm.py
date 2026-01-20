#!/usr/bin/env python3
"""
Main script for training and testing the IHM multimodal model.
Usage: python run_ihm.py [--mode train|test|both] [--gpu GPU_ID] [--config CONFIG_PATH]
"""

import argparse
import torch
import torch.optim as optim
from transformers import BertTokenizer
import os

from config import Config
from train import setup_dataloaders, setup_models, setup_optimizer, train_model
from evaluate import test_model
from losses import RobustFocalLoss
from utils import calculate_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Train/Test IHM Multimodal Model")
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['train', 'test', 'both'], 
                        help='Mode: train, test, or both')
    parser.add_argument('--gpu', type=int, default=3, 
                        help='GPU ID to use')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for testing')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update config with command line arguments"""
    if args.gpu is not None:
        config.GPU_ID = args.gpu
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.model_path is not None:
        config.MODEL_SAVE_PATH = args.model_path
    return config

def main():
    args = parse_args()
    
    # Load configuration
    config = Config()
    config = update_config_from_args(config, args)
    
    # Setup device
    device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[SYS] Using Device: {device}")
    print(f"[TASK] In-Hospital Mortality (IHM) Prediction - Binary Classification")
    print(f"[ARCHITECTURE] TripleFiLM Network with 3-Level Fusion: Modality + Temporal + Social")
    
    # Create log directory if it doesn't exist
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Setup dataloaders
    train_loader, val_loader, test_loader, train_ds = setup_dataloaders(config, tokenizer)
    
    # Setup models
    ehr_model, cxr_model, text_model, fusion_model = setup_models(config, train_ds, device)
    
    # Model summary
    print("\n[ARCHITECTURE SUMMARY]")
    print("  TripleFiLM Network with 3-Level Fusion:")
    print("  Level 1: Modality Fusion (EHR + CXR + Text)")
    print("  Level 2: Temporal Fusion (Intra-patient visit history)")
    print("  Level 3: Social Fusion (Inter-patient similarity graph)")
    print(f"  Hidden Dim: {config.HIDDEN_DIM}, FiLM Hidden Dim: {config.FILM_HIDDEN_DIM}")
    
    if args.mode in ['train', 'both']:
        # Calculate class weights
        print("\n[DATA] Calculating class weights for IHM from training data...")
        pos_weight, positive_ratio = calculate_class_weights(train_loader)
        pos_weight = pos_weight.to(device)
        
        # Setup loss function
        criterion = RobustFocalLoss(alpha=config.FOCAL_LOSS_ALPHA, 
                                    gamma=config.FOCAL_LOSS_GAMMA, 
                                    pos_weight=pos_weight)
        
        # Setup optimizer and scheduler
        optimizer = setup_optimizer(config, fusion_model, ehr_model, cxr_model, text_model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
        
        # Train model
        print("\n[TRAINING] Starting training...")
        best_auroc, best_threshold = train_model(
            config, train_loader, val_loader, device,
            fusion_model, ehr_model, cxr_model, text_model,
            criterion, optimizer, scheduler
        )
    else:
        # For test-only mode, load best model
        checkpoint = torch.load(config.MODEL_SAVE_PATH)
        best_threshold = checkpoint.get('threshold', 0.5)
        
        # Setup loss function with default pos_weight
        criterion = RobustFocalLoss(alpha=config.FOCAL_LOSS_ALPHA, 
                                    gamma=config.FOCAL_LOSS_GAMMA)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test' and args.model_path is None:
            print("[WARNING] No model path specified for testing. Using default path.")
        
        # Test model
        test_metrics = test_model(
            config, test_loader, device, 
            fusion_model, ehr_model, cxr_model, text_model,
            criterion, best_threshold
        )
        
        # Final Summary
        print(f"\n{'='*80}")
        print(f"PROCESS COMPLETE")
        print(f"{'='*80}")
        if args.mode == 'both':
            print(f"Best Validation AUROC: {best_auroc:.4f}")
        print(f"Best Threshold: {best_threshold:.3f}")
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()