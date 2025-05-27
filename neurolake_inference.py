#!/usr/bin/env python
# coding: utf-8

"""
TabM Inference Script for Custom CSV Datasets

Loads a trained TabM model and preprocessing pipeline to make predictions on new data.
Optionally calculates KS statistics from inference outputs for feature analysis.

Usage:
    # Basic inference
    python neurolake_inference.py --data_path new_data.csv --model_path saved_model/model.pth --preprocessing_path saved_model/preprocessing.pkl --output_path predictions.csv
    
    # With KS statistic calculation (binary classification only, requires target column in data)
    python neurolake_inference.py --data_path new_data.csv --model_path saved_model/model.pth --preprocessing_path saved_model/preprocessing.pkl --output_path predictions.csv --calculate_ks --target_column alvo

Requirements:
    - CSV file with same structure as training data (excluding target column)
    - Trained model file (.pth)
    - Preprocessing pipeline file (.pkl)
    - ks_utils.py (optional, for KS statistics calculation)
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.special
import sklearn.preprocessing
import torch
from torch import Tensor

warnings.simplefilter('ignore')
from tabm_reference import Model
warnings.resetwarnings()

# KS statistic calculation uses scipy.stats (imported in function when needed)


def load_model_and_preprocessing(model_path: str, preprocessing_path: str, device: torch.device):
    """Load trained model and preprocessing pipeline."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Create model with saved configuration
    model = Model(
        n_num_features=model_config['n_num_features'],
        cat_cardinalities=model_config['cat_cardinalities'],
        n_classes=model_config['n_classes'],
        backbone=model_config['backbone'],
        bins=model_config['bins'],
        num_embeddings=model_config['num_embeddings'],
        arch_type=model_config['arch_type'],
        k=model_config['k'],
        share_training_batches=model_config['share_training_batches'],
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load preprocessing pipeline and metadata
    with open(preprocessing_path, 'rb') as f:
        preprocessing_data = pickle.load(f)
    
    return model, model_config, preprocessing_data


def preprocess_inference_data(
    df: pd.DataFrame,
    feature_info: dict,
    preprocessing_pipeline,
    sep: str = ','
):
    """Preprocess new data using the saved preprocessing pipeline."""
    
    print(f"Loaded inference data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Remove identifier columns if they exist
    identifier_columns = feature_info['identifier_columns']
    target_column = feature_info['target_column']
    
    # Remove identifier columns and target column if present
    columns_to_remove = []
    for col in identifier_columns:
        if col in df.columns:
            columns_to_remove.append(col)
    
    if target_column in df.columns:
        print(f"Warning: Target column '{target_column}' found in inference data. It will be ignored.")
        columns_to_remove.append(target_column)
    
    if columns_to_remove:
        print(f"Removing columns: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove)
    
    # Separate numerical and categorical features
    numerical_columns = feature_info['numerical_columns']
    categorical_columns = feature_info['categorical_columns']
    
    # Process numerical features
    X_num = None
    if numerical_columns:
        # Check if all numerical columns are present
        missing_num_cols = [col for col in numerical_columns if col not in df.columns]
        if missing_num_cols:
            raise ValueError(f"Missing numerical columns in inference data: {missing_num_cols}")
        
        X_num_df = df[numerical_columns].copy()
        
        # Handle missing and infinite values (same as training)
        print(f"Checking for missing/infinite values in numerical features...")
        
        # Check for missing values
        missing_counts = X_num_df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found missing values in {(missing_counts > 0).sum()} columns")
            for col in missing_counts[missing_counts > 0].index:
                print(f"  {col}: {missing_counts[col]} missing values")
        
        # Fill missing values with median (computed from training data would be better)
        for col in X_num_df.columns:
            if X_num_df[col].isnull().any():
                median_val = X_num_df[col].median()
                X_num_df[col].fillna(median_val, inplace=True)
        
        # Convert to float32
        X_num_df = X_num_df.astype(np.float32)
        
        # Check for infinite values
        inf_mask = np.isinf(X_num_df.values)
        if inf_mask.any():
            print(f"Found {inf_mask.sum()} infinite values, replacing with finite values...")
            X_num_df = X_num_df.replace([np.inf, -np.inf], [np.finfo(np.float32).max/2, np.finfo(np.float32).min/2])
        
        # Final check for any remaining NaN/inf values
        final_check = np.isnan(X_num_df.values) | np.isinf(X_num_df.values)
        if final_check.any():
            print(f"Warning: Still found {final_check.sum()} NaN/inf values after cleaning")
            X_num_df = X_num_df.fillna(0)
            X_num_df = X_num_df.replace([np.inf, -np.inf], 0)
        
        X_num = X_num_df.values
        
        # Apply preprocessing pipeline
        if preprocessing_pipeline is not None:
            X_num = preprocessing_pipeline.transform(X_num)
        
        print(f"Numerical features processed: {X_num.shape}")
    
    # Process categorical features
    X_cat = None
    if categorical_columns:
        # Check if all categorical columns are present
        missing_cat_cols = [col for col in categorical_columns if col not in df.columns]
        if missing_cat_cols:
            raise ValueError(f"Missing categorical columns in inference data: {missing_cat_cols}")
        
        X_cat_df = df[categorical_columns].copy()
        label_encoders = feature_info['label_encoders']
        
        if label_encoders is None:
            raise ValueError("No label encoders found for categorical features")
        
        # Apply label encoders
        for col in categorical_columns:
            if col not in label_encoders:
                raise ValueError(f"No label encoder found for column: {col}")
            
            le = label_encoders[col]
            
            # Handle unseen categories
            X_cat_df[col] = X_cat_df[col].astype(str)
            unseen_mask = ~X_cat_df[col].isin(le.classes_)
            
            if unseen_mask.any():
                print(f"Warning: Found {unseen_mask.sum()} unseen categories in column '{col}'. Setting to first class.")
                X_cat_df.loc[unseen_mask, col] = le.classes_[0]
            
            X_cat_df[col] = le.transform(X_cat_df[col])
        
        X_cat = X_cat_df.values.astype(np.int64)
        print(f"Categorical features processed: {X_cat.shape}")
    
    return X_num, X_cat


def make_predictions(
    model,
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    model_config: dict,
    preprocessing_data: dict,
    device: torch.device,
    batch_size: int = 1024
):
    """Make predictions using the trained model."""
    
    n_samples = X_num.shape[0] if X_num is not None else X_cat.shape[0]
    task_type = model_config['task_type']
    regression_label_stats = preprocessing_data['regression_label_stats']
    
    # Convert to tensors
    X_num_tensor = torch.as_tensor(X_num, device=device) if X_num is not None else None
    X_cat_tensor = torch.as_tensor(X_cat, device=device) if X_cat is not None else None
    
    # Make predictions in batches
    predictions = []
    
    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            batch_x_num = X_num_tensor[i:end_idx] if X_num_tensor is not None else None
            batch_x_cat = X_cat_tensor[i:end_idx] if X_cat_tensor is not None else None
            
            # Get model predictions
            batch_pred = model(batch_x_num, batch_x_cat).squeeze(-1).float()
            predictions.append(batch_pred.cpu().numpy())
    
    # Concatenate all predictions
    y_pred = np.concatenate(predictions, axis=0)
    
    # Post-process predictions based on task type
    if task_type == 'regression':
        if regression_label_stats is not None:
            # Denormalize regression predictions
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
        
        # For regression, average across ensemble members
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(axis=1)
        
        return y_pred, None
    
    else:
        # For classification, apply softmax and average across ensemble members
        if y_pred.ndim > 1:
            y_pred = scipy.special.softmax(y_pred, axis=-1)
            y_pred_proba = y_pred.mean(axis=1)
        else:
            y_pred_proba = scipy.special.softmax(y_pred.reshape(-1, 1), axis=-1)
        
        # Get class predictions
        y_pred_class = y_pred_proba.argmax(axis=1)
        
        return y_pred_class, y_pred_proba


def calculate_ks_from_inference(
    original_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_config: dict,
    feature_info: dict,
    output_path: str
):
    """
    Calculate KS statistic from inference outputs using predicted probabilities vs true labels.
    This matches the KS calculation used during training validation/test.
    
    Args:
        original_df: Original dataframe with features
        predictions_df: DataFrame with predictions
        model_config: Model configuration
        feature_info: Feature information from preprocessing
        output_path: Base path for saving KS results
    """
    task_type = model_config['task_type']
    target_column = feature_info['target_column']
    
    # Only calculate KS for binary classification tasks
    if task_type != 'binclass':
        print(f"⚠️  KS statistics only available for binary classification. Current task: {task_type}")
        return
    
    # Check if we have the target column in the original data
    if target_column not in original_df.columns:
        print(f"⚠️  Target column '{target_column}' not found in data. Cannot calculate KS statistic.")
        print(f"Available columns: {list(original_df.columns)}")
        return
    
    # Check if we have probability predictions
    if 'class_1_probability' not in predictions_df.columns:
        print("⚠️  No class_1_probability column found for KS calculation")
        return
    
    print(f"\n🔄 Calculating KS statistic from inference outputs...")
    print(f"📊 Using predicted probabilities vs true labels (like training validation)")
    
    # Get true labels and predicted probabilities
    y_true = original_df[target_column].values
    y_pred_proba = predictions_df['class_1_probability'].values
    
    # Convert target to binary if needed (same as training script)
    unique_targets = pd.Series(y_true).dropna().unique()
    if len(unique_targets) != 2:
        print(f"⚠️  Target must be binary. Found {len(unique_targets)} unique values: {unique_targets}")
        return
    
    # Convert to 0/1 if not already
    if not set(unique_targets).issubset({0, 1}):
        target_mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
        y_true = pd.Series(y_true).map(target_mapping).values
    
    # Remove any NaN values
    valid_mask = ~pd.isna(y_true) & ~pd.isna(y_pred_proba)
    y_true_clean = y_true[valid_mask]
    y_pred_proba_clean = y_pred_proba[valid_mask]
    
    if len(y_true_clean) == 0:
        print("⚠️  No valid samples for KS calculation")
        return
    
    try:
        # Calculate KS statistic exactly like in training script
        from scipy import stats
        
        # Separate predictions for each class
        pos_scores = y_pred_proba_clean[y_true_clean == 1]
        neg_scores = y_pred_proba_clean[y_true_clean == 0]
        
        # Calculate KS statistic (Kolmogorov-Smirnov test)
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            ks_stat, ks_p_value = stats.ks_2samp(pos_scores, neg_scores)
            
            print(f"\n📊 KS Statistic Results:")
            print(f"   - KS Statistic: {ks_stat:.4f}")
            print(f"   - P-value: {ks_p_value:.2e}")
            print(f"   - Samples analyzed: {len(y_true_clean)}")
            print(f"   - Positive class samples: {len(pos_scores)}")
            print(f"   - Negative class samples: {len(neg_scores)}")
            
            # Interpretation
            if ks_stat > 0.3:
                interpretation = "Very strong discriminative power"
            elif ks_stat > 0.2:
                interpretation = "Strong discriminative power"
            elif ks_stat > 0.1:
                interpretation = "Moderate discriminative power"
            else:
                interpretation = "Weak discriminative power"
            
            print(f"   - Interpretation: {interpretation}")
            
            # Additional statistics
            print(f"\n📈 Additional Statistics:")
            print(f"   - Mean probability (positive class): {pos_scores.mean():.4f}")
            print(f"   - Mean probability (negative class): {neg_scores.mean():.4f}")
            print(f"   - Std probability (positive class): {pos_scores.std():.4f}")
            print(f"   - Std probability (negative class): {neg_scores.std():.4f}")
            
            # Save KS results
            ks_results = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'n_samples': len(y_true_clean),
                'n_positive': len(pos_scores),
                'n_negative': len(neg_scores),
                'mean_prob_positive': pos_scores.mean(),
                'mean_prob_negative': neg_scores.mean(),
                'std_prob_positive': pos_scores.std(),
                'std_prob_negative': neg_scores.std(),
                'interpretation': interpretation,
                'target_column': target_column
            }
            
            ks_output_path = output_path.replace('.csv', '_ks_statistic.csv')
            ks_df = pd.DataFrame([ks_results])
            ks_df.to_csv(ks_output_path, index=False)
            
            print(f"\n💾 KS statistic saved to: {ks_output_path}")
            
            # Add KS summary to predictions metadata
            if hasattr(predictions_df, 'attrs'):
                predictions_df.attrs['ks_statistic'] = ks_stat
                predictions_df.attrs['ks_p_value'] = ks_p_value
                predictions_df.attrs['ks_interpretation'] = interpretation
            
        else:
            print("⚠️  Cannot calculate KS statistic: one class is missing")
            
    except Exception as e:
        print(f"⚠️  Error calculating KS statistic: {e}")


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained TabM model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.pth file)')
    parser.add_argument('--preprocessing_path', type=str, required=True, help='Path to saved preprocessing (.pkl file)')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--sep', type=str, default=',',
                       help='CSV separator (default: comma). Use "\\t" for tab-separated files')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--calculate_ks', action='store_true', 
                       help='Calculate KS statistic between predicted probabilities and true labels (binary classification only)')
    parser.add_argument('--target_column', type=str, default='alvo',
                       help='Name of target column in data for KS calculation (default: alvo)')
    
    args = parser.parse_args()
    
    # Handle separator
    if args.sep == '\\t':
        args.sep = '\t'
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model and preprocessing
    print(f'Loading model from {args.model_path}...')
    print(f'Loading preprocessing from {args.preprocessing_path}...')
    
    model, model_config, preprocessing_data = load_model_and_preprocessing(
        args.model_path, args.preprocessing_path, device
    )
    
    feature_info = preprocessing_data['feature_info']
    preprocessing_pipeline = preprocessing_data['preprocessing_pipeline']
    
    print(f'Model loaded: {model_config["task_type"]} task')
    print(f'Architecture: {model_config["arch_type"]}')
    
    # Load and preprocess data
    print(f'\nLoading data from {args.data_path}...')
    df = pd.read_csv(args.data_path, sep=args.sep)
    
    X_num, X_cat = preprocess_inference_data(
        df, feature_info, preprocessing_pipeline, args.sep
    )
    
    # Make predictions
    print(f'\nMaking predictions...')
    y_pred, y_pred_proba = make_predictions(
        model, X_num, X_cat, model_config, preprocessing_data, device, args.batch_size
    )
    
    # Prepare output
    output_df = df.copy()
    
    if model_config['task_type'] == 'regression':
        output_df['prediction'] = y_pred
        print(f'Regression predictions: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}')
    else:
        output_df['predicted_class'] = y_pred
        
        # Add probability columns
        n_classes = model_config['n_classes']
        for i in range(n_classes):
            output_df[f'class_{i}_probability'] = y_pred_proba[:, i]
        
        print(f'Classification predictions: {len(np.unique(y_pred))} unique classes')
        print(f'Class distribution: {np.bincount(y_pred)}')
    
    # Save predictions
    output_df.to_csv(args.output_path, index=False, sep=args.sep)
    
    print(f'\n✅ Predictions saved to: {args.output_path}')
    print(f'📊 Processed {len(output_df)} samples')
    
    # Calculate KS statistics if requested
    if args.calculate_ks:
        # Store original target column name in feature_info for KS analysis
        feature_info['target_column'] = args.target_column
        
        calculate_ks_from_inference(
            original_df=df,
            predictions_df=output_df,
            model_config=model_config,
            feature_info=feature_info,
            output_path=args.output_path
        )


if __name__ == '__main__':
    main() 