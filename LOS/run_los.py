#!/usr/bin/env python3
"""
Main script for training and testing the LOS multimodal model.
Usage: python run_los.py [--mode train|test|both] [--gpu GPU_ID] [--epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
"""

import argparse
import torch
import torch.optim as optim
from transformers import BertTokenizer
import os

from config import Config
from train import setup_dataloaders, setup_models, setup_optimizer, train_model
from evaluate import test_model, print_results, save_history
from losses import RobustFocalLoss
from utils import calculate_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Train/Test LOS Multimodal Model")
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['train', 'test', 'both'], 
                        help='Mode: train, test, or both')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for testing')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update config with command line arguments"""
    if args.gpu is not None:
        config.GPU_ID = args.gpu
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
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
    print(f"[TASK] Length of Stay (LOS) Prediction - 9-Class Classification")
    print(f"[ARCHITECTURE] TripleFiLM Network with 3-Level Fusion: Modality + Temporal + Social")
    print(f"[CLASSES] {config.NUM_CLASSES} classes: {', '.join(config.CLASS_NAMES)}")
    
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
    print(f"  Classifier: {config.NUM_CLASSES} classes for LOS prediction")
    
    if args.mode in ['train', 'both']:
        # Calculate class weights
        class_weights = calculate_class_weights(train_ds, config, device)
        
        # Setup loss function
        criterion = RobustFocalLoss(alpha=config.FOCAL_LOSS_ALPHA, 
                                    gamma=config.FOCAL_LOSS_GAMMA, 
                                    class_weights=class_weights)
        
        # Setup optimizer and scheduler
        optimizer = setup_optimizer(config, fusion_model, ehr_model, cxr_model, text_model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
        
        # Train model
        print("\n[TRAINING] Starting training...")
        best_auroc, history = train_model(
            config, train_loader, val_loader, device,
            fusion_model, ehr_model, cxr_model, text_model,
            criterion, optimizer, scheduler
        )
        
        # Save training history
        save_history(history, config)
    else:
        # For test-only mode, load best model and setup loss function
        criterion = RobustFocalLoss(alpha=config.FOCAL_LOSS_ALPHA, 
                                    gamma=config.FOCAL_LOSS_GAMMA)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test' and args.model_path is None:
            print("[WARNING] No model path specified for testing. Using default path.")
        
        # Test model
        test_metrics = test_model(
            config, test_loader, device, 
            fusion_model, ehr_model, cxr_model, text_model,
            criterion
        )
        
        # Final Summary
        print(f"\n{'='*80}")
        print(f"PROCESS COMPLETE")
        print(f"{'='*80}")
        if args.mode == 'both':
            print(f"Best Validation Macro AUROC: {best_auroc:.4f}")
        print(f"Test Macro AUROC: {test_metrics['macro_auroc']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"LOS Prediction Performance Summary:")
        print(f"  Architecture: TripleFiLM with 3-Level Fusion")
        print(f"  Temporal Modeling: Enabled (Intra-patient visit connections)")
        print(f"  Social Graph: Enabled")
        print(f"  Multi-class Classification: {config.NUM_CLASSES} LOS categories")

if __name__ == "__main__":
    main()