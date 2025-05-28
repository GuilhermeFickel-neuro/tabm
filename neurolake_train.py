#!/usr/bin/env python
# coding: utf-8

"""
TabM Training Script for Custom CSV Datasets

Uses TabM-mini with piecewise-linear embeddings by default for optimal performance.
For binary classification, uses KS statistic instead of accuracy for evaluation.

Usage:
    # For comma-separated CSV with hyperparameter tuning
    python neurolake_train.py --data_path dataset.csv --target_column target --task_type regression --tune_hyperparams
    
    # For tab-separated CSV (like your training file) with fixed hyperparams
    python neurolake_train.py --data_path train.csv --target_column alvo --task_type binclass --sep "\\t"

Requirements:
    - CSV file with headers (comma or tab separated)
    - Target column specified
    - Task type: 'regression', 'binclass', or 'multiclass'
    - Automatically excludes identifier columns (cpf, ref_date, etc.)
    
Evaluation Metrics:
    - Regression: Negative RMSE (higher is better)
    - Binary Classification: KS statistic (higher is better)
    - Multiclass Classification: Accuracy (higher is better)
"""

import argparse
import math
import pickle
import random
import warnings
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import scipy.special
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

warnings.simplefilter('ignore')
from tabm_reference import Model, make_parameter_groups
warnings.resetwarnings()

# Optional Optuna import for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


def save_model_and_preprocessing(
    model,
    preprocessing_pipeline,
    regression_label_stats,
    feature_info,
    model_config,
    save_dir: str = "saved_model"
):
    """Save trained model and preprocessing components for inference."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = save_path / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
    }, model_path)
    
    # Save preprocessing pipeline and metadata
    preprocessing_path = save_path / "preprocessing.pkl"
    preprocessing_data = {
        'preprocessing_pipeline': preprocessing_pipeline,
        'regression_label_stats': regression_label_stats,
        'feature_info': feature_info,
    }
    
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_data, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessing saved to: {preprocessing_path}")
    
    return model_path, preprocessing_path


def load_csv_dataset(
    data_path: str,
    target_column: str,
    task_type: Literal['regression', 'binclass', 'multiclass'],
    categorical_columns: Optional[list] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    sep: str = ','
):
    """Load and preprocess CSV dataset for TabM training."""
    
    # Load data with specified separator
    df = pd.read_csv(data_path, sep=sep)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Identify and exclude identifier columns (common patterns)
    identifier_patterns = ['ref_date', 'date', 'timestamp']
    identifier_columns = []
    for col in df.columns:
        if col.lower() == 'cpf':
            identifier_columns.append(col)
            continue
        if col.lower() != target_column.lower():  # Don't exclude target
            for pattern in identifier_patterns:
                if pattern in col.lower():
                    identifier_columns.append(col)
                    break
    
    if identifier_columns:
        print(f"Excluding identifier columns: {identifier_columns}")
        X = df.drop(columns=[target_column] + identifier_columns)
    else:
        X = df.drop(columns=[target_column])
    
    y = df[target_column].values
    
    # Handle categorical columns
    if categorical_columns is None:
        # Auto-detect categorical columns (object/string types)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Separate numerical and categorical features
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    X_num = None
    if numerical_columns:
        X_num_df = X[numerical_columns].copy()
        
        # Handle missing and infinite values in numerical features
        print(f"Checking for missing/infinite values in numerical features...")
        
        # Check for missing values
        missing_counts = X_num_df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found missing values in {(missing_counts > 0).sum()} columns")
            for col in missing_counts[missing_counts > 0].index:
                print(f"  {col}: {missing_counts[col]} missing values")
        
        # Fill missing values with median
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
            # Replace inf with very large finite values, -inf with very small finite values
            X_num_df = X_num_df.replace([np.inf, -np.inf], [np.finfo(np.float32).max/2, np.finfo(np.float32).min/2])
        
        # Final check for any remaining NaN/inf values
        final_check = np.isnan(X_num_df.values) | np.isinf(X_num_df.values)
        if final_check.any():
            print(f"Warning: Still found {final_check.sum()} NaN/inf values after cleaning")
            # Replace any remaining problematic values with 0
            X_num_df = X_num_df.fillna(0)
            X_num_df = X_num_df.replace([np.inf, -np.inf], 0)
        
        # Filter out numerical features with only one unique value
        constant_features = []
        for col in X_num_df.columns:
            if X_num_df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing {len(constant_features)} numerical features with constant values")
            X_num_df = X_num_df.drop(columns=constant_features)
            numerical_columns = [col for col in numerical_columns if col not in constant_features]
        
        X_num = X_num_df.values if len(X_num_df.columns) > 0 else None
        print(f"Numerical features cleaned: {X_num.shape if X_num is not None else 'None'}")
    
    X_cat = None
    cat_cardinalities = []
    label_encoders = {}
    
    if categorical_columns:
        # Encode categorical features
        X_cat_df = X[categorical_columns].copy()
        
        for col in categorical_columns:
            le = sklearn.preprocessing.LabelEncoder()
            X_cat_df[col] = le.fit_transform(X_cat_df[col].astype(str))
            label_encoders[col] = le
            cat_cardinalities.append(len(le.classes_))
        
        # Filter out features with cardinality 1
        filtered_indices = [i for i, card in enumerate(cat_cardinalities) if card > 1]
        if len(filtered_indices) < len(cat_cardinalities):
            n_removed = len(cat_cardinalities) - len(filtered_indices)
            print(f"Removing {n_removed} categorical features with cardinality 1")
            
            # Keep only features with cardinality > 1
            filtered_categorical_columns = [categorical_columns[i] for i in filtered_indices]
            filtered_cat_cardinalities = [cat_cardinalities[i] for i in filtered_indices]
            filtered_label_encoders = {col: label_encoders[col] for col in filtered_categorical_columns}
            
            # Update the data
            X_cat_df = X_cat_df[filtered_categorical_columns]
            categorical_columns = filtered_categorical_columns
            cat_cardinalities = filtered_cat_cardinalities
            label_encoders = filtered_label_encoders
        
        X_cat = X_cat_df.values.astype(np.int64) if len(categorical_columns) > 0 else None
        print(f"Categorical features: {len(categorical_columns)}, cardinalities: {cat_cardinalities}")
    
    # Process target variable
    if task_type == 'regression':
        y = y.astype(np.float32)
    else:
        if task_type == 'binclass':
            # Ensure binary classification has labels 0 and 1
            le = sklearn.preprocessing.LabelEncoder()
            y = le.fit_transform(y).astype(np.int64)
            n_classes = 2
        else:  # multiclass
            le = sklearn.preprocessing.LabelEncoder()
            y = le.fit_transform(y).astype(np.int64)
            n_classes = len(np.unique(y))
        
        print(f"Classification task: {n_classes} classes")
    
    # Split dataset
    indices = np.arange(len(y))
    train_val_idx, test_idx = sklearn.model_selection.train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y if task_type != 'regression' else None
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        train_val_idx, test_size=val_size, random_state=random_state, 
        stratify=y[train_val_idx] if task_type != 'regression' else None
    )
    
    # Create data splits
    data_splits = {
        'train': {'y': y[train_idx]},
        'val': {'y': y[val_idx]},
        'test': {'y': y[test_idx]}
    }
    
    if X_num is not None:
        data_splits['train']['x_cont'] = X_num[train_idx]
        data_splits['val']['x_cont'] = X_num[val_idx]
        data_splits['test']['x_cont'] = X_num[test_idx]
    
    if X_cat is not None:
        data_splits['train']['x_cat'] = X_cat[train_idx]
        data_splits['val']['x_cat'] = X_cat[val_idx]
        data_splits['test']['x_cat'] = X_cat[test_idx]
    
    n_num_features = X_num.shape[1] if X_num is not None else 0
    n_classes = len(np.unique(y)) if task_type != 'regression' else None
    
    print(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Numerical features: {n_num_features}")
    
    # Create feature info for inference
    feature_info = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'cat_cardinalities': cat_cardinalities,
        'identifier_columns': identifier_columns,
        'target_column': target_column,
        'task_type': task_type,
        'n_classes': n_classes,
        'label_encoders': label_encoders if categorical_columns else None,
    }
    
    return data_splits, n_num_features, cat_cardinalities, n_classes, feature_info


def preprocess_features(data_splits, n_num_features):
    """Preprocess numerical features using quantile transformation."""
    
    if n_num_features == 0:
        return data_splits, None, n_num_features
    
    # Advanced preprocessing with quantile transformation
    X_train = data_splits['train']['x_cont']
    noise = np.random.default_rng(0).normal(0.0, 1e-5, X_train.shape).astype(X_train.dtype)
    
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(X_train) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(X_train + noise)
    
    # Apply preprocessing to all splits
    for split in data_splits:
        if 'x_cont' in data_splits[split]:
            data_splits[split]['x_cont'] = preprocessing.transform(data_splits[split]['x_cont'])
    
    # Check for constant features after preprocessing
    X_train_processed = data_splits['train']['x_cont']
    constant_feature_indices = []
    
    for i in range(X_train_processed.shape[1]):
        feature_values = X_train_processed[:, i]
        # Check if all values are the same (within a small tolerance for floating point)
        if np.allclose(feature_values, feature_values[0], rtol=1e-10, atol=1e-10):
            constant_feature_indices.append(i)
    
    if constant_feature_indices:
        print(f"Removing {len(constant_feature_indices)} numerical features that became constant after preprocessing")
        
        # Create mask for features to keep
        keep_indices = [i for i in range(X_train_processed.shape[1]) if i not in constant_feature_indices]
        
        if len(keep_indices) == 0:
            # All features became constant
            print("Warning: All numerical features became constant after preprocessing")
            for split in data_splits:
                if 'x_cont' in data_splits[split]:
                    del data_splits[split]['x_cont']
            return data_splits, preprocessing, 0
        
        # Filter out constant features from all splits
        for split in data_splits:
            if 'x_cont' in data_splits[split]:
                data_splits[split]['x_cont'] = data_splits[split]['x_cont'][:, keep_indices]
        
        n_num_features = len(keep_indices)
        print(f"Numerical features after preprocessing: {n_num_features}")
    
    return data_splits, preprocessing, n_num_features


def setup_model_and_training(
    n_num_features: int,
    cat_cardinalities: list,
    n_classes: Optional[int],
    task_type: str,
    device: torch.device,
    data_splits: dict,
    use_embeddings: bool = False,
    hyperparams: Optional[dict] = None
):
    """Setup TabM model and training components."""
    
    # Use provided hyperparams or defaults
    if hyperparams is None:
        hyperparams = {
            'n_blocks': 2 if use_embeddings else 3,
            'd_block': 512,
            'dropout': 0.1,
            'lr': 2e-3,
            'weight_decay': 3e-4,
            'd_embedding': 16,
        }
    
    # Configure model architecture
    bins = None
    num_embeddings = None
    
    if use_embeddings and n_num_features > 0:
        # Use TabM-mini with piecewise-linear embeddings for numerical features
        arch_type = 'tabm-mini'
        
        # Compute bins from training data (CRITICAL for piecewise-linear embeddings)
        train_x_cont = torch.as_tensor(data_splits['train']['x_cont'], device=device)
        
        # Check if dataset is large enough for bin computation
        min_samples_for_bins = 50  # rtdl_num_embeddings typically needs at least 48 bins
        if len(train_x_cont) < min_samples_for_bins:
            print(f"Warning: Dataset too small ({len(train_x_cont)} samples) for optimal piecewise-linear embeddings.")
            print(f"Falling back to standard TabM architecture.")
            arch_type = 'tabm'
            bins = None
            num_embeddings = None
        else:
            # Final safety check for NaN/inf values before computing bins
            if torch.isnan(train_x_cont).any() or torch.isinf(train_x_cont).any():
                print(f"Warning: Found NaN/inf values in training data after preprocessing.")
                print(f"Falling back to standard TabM architecture.")
                arch_type = 'tabm'
                bins = None
                num_embeddings = None
            else:
                try:
                    bins = rtdl_num_embeddings.compute_bins(train_x_cont)
                    num_embeddings = {
                        'type': 'PiecewiseLinearEmbeddings',
                        'd_embedding': hyperparams['d_embedding'],
                        'activation': False,
                        'version': 'B',
                    }
                    print(f"Using TabM-mini with piecewise-linear embeddings (bins computed from {train_x_cont.shape[0]} training samples)")
                except Exception as e:
                    print(f"Warning: Failed to compute bins for embeddings: {e}")
                    print(f"Falling back to standard TabM architecture.")
                    arch_type = 'tabm'
                    bins = None
                    num_embeddings = None
    else:
        arch_type = 'tabm'
        print(f"Using standard TabM architecture")
    
    # Create model
    model = Model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        backbone={
            'type': 'MLP',
            'n_blocks': hyperparams['n_blocks'],
            'd_block': hyperparams['d_block'],
            'dropout': hyperparams['dropout'],
        },
        bins=bins,
        num_embeddings=num_embeddings,
        arch_type=arch_type,
        k=32,
        share_training_batches=True,
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        make_parameter_groups(model), 
        lr=hyperparams['lr'], 
        weight_decay=hyperparams['weight_decay']
    )
    
    # Create model configuration for saving
    model_config = {
        'n_num_features': n_num_features,
        'cat_cardinalities': cat_cardinalities,
        'n_classes': n_classes,
        'task_type': task_type,
        'backbone': {
            'type': 'MLP',
            'n_blocks': hyperparams['n_blocks'],
            'd_block': hyperparams['d_block'],
            'dropout': hyperparams['dropout'],
        },
        'bins': bins,
        'num_embeddings': num_embeddings,
        'arch_type': arch_type,
        'k': 32,
        'share_training_batches': True,
        'use_embeddings': use_embeddings,
        'hyperparams': hyperparams,
    }
    
    return model, optimizer, model_config


def tune_hyperparameters(
    data_splits, n_num_features, cat_cardinalities, n_classes, task_type, 
    device, use_embeddings, n_trials=50, timeout=3600
):
    """Tune hyperparameters using Optuna with TabM paper specifications."""
    
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Install with: pip install optuna")
        return None
    
    def objective(trial):
        # Sample hyperparameters according to TabM paper
        if use_embeddings:
            # TabM with embeddings ranges
            n_blocks = trial.suggest_int('n_blocks', 1, 4)
            lr = trial.suggest_float('lr', 5e-5, 3e-3, log=True)
            d_embedding = trial.suggest_int('d_embedding', 8, 32)
        else:
            # Standard TabM ranges  
            n_blocks = trial.suggest_int('n_blocks', 1, 5)
            lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
            d_embedding = 16  # Not used without embeddings
        
        d_block = trial.suggest_int('d_block', 64, 1024)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Weight decay: either 0 or log-uniform range (as in paper)
        if trial.suggest_categorical('use_weight_decay', [True, False]):
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        else:
            weight_decay = 0.0
        
        hyperparams = {
            'n_blocks': n_blocks,
            'd_block': d_block,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay,
            'd_embedding': d_embedding,
        }
        
        try:
            # Setup model with these hyperparams
            model, optimizer, _ = setup_model_and_training(
                n_num_features, cat_cardinalities, n_classes, task_type, 
                device, data_splits, use_embeddings, hyperparams
            )
            
            # Longer training for better hyperparameter evaluation
            best_result = train_model(
                model, optimizer, data_splits, task_type, device,
                n_epochs=1000, patience=50, batch_size=256, verbose=False
            )
            
            return best_result['val']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')
    
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best validation score: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    
    # Convert best params to our format
    best_params = study.best_params.copy()
    if not best_params.get('use_weight_decay', True):
        best_params['weight_decay'] = 0.0
    best_params.pop('use_weight_decay', None)
    
    return best_params


def train_model(
    model, optimizer, data_splits, task_type, device,
    n_epochs: int = 10000, patience: int = 1000, batch_size: int = 256, verbose: bool = True
):
    """Train the TabM model."""
    
    # Handle label preprocessing for regression
    regression_label_stats = None
    Y_train = data_splits['train']['y'].copy()
    
    if task_type == 'regression':
        regression_label_stats = RegressionLabelStats(
            Y_train.mean().item(), Y_train.std().item()
        )
        Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    
    # Convert to tensors
    data = {}
    for split in data_splits:
        data[split] = {}
        for key, value in data_splits[split].items():
            data[split][key] = torch.as_tensor(value, device=device)
        
        if task_type == 'regression':
            data[split]['y'] = data[split]['y'].float()
    
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        Y_train = Y_train.float()
    
    @torch.autocast(device.type, enabled=False)
    def apply_model(part: str, idx: Tensor) -> Tensor:
        x_cont = data[part].get('x_cont')
        x_cat = data[part].get('x_cat')
        
        return model(
            x_cont[idx] if x_cont is not None else None,
            x_cat[idx] if x_cat is not None else None,
        ).squeeze(-1).float()
    
    base_loss_fn = F.mse_loss if task_type == 'regression' else F.cross_entropy
    
    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        k = y_pred.shape[-1 if task_type == 'regression' else -2]
        return base_loss_fn(
            y_pred.flatten(0, 1),
            y_true.repeat_interleave(k) if model.share_training_batches else y_true,
        )
    
    @torch.inference_mode()
    def evaluate(part: str) -> float:
        model.eval()
        eval_batch_size = 8096
        
        y_pred = torch.cat([
            apply_model(part, idx)
            for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
        ]).cpu().numpy()
        
        if task_type == 'regression' and regression_label_stats is not None:
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
        
        if task_type != 'regression':
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        
        y_pred = y_pred.mean(1)
        y_true = data[part]['y'].cpu().numpy()
        
        if task_type == 'regression':
            score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        elif task_type == 'binclass':
            # For binary classification, calculate KS statistic
            # Get probability of positive class (class 1)
            y_pred_proba = y_pred[:, 1] if y_pred.ndim > 1 else y_pred
            
            # Calculate KS statistic using scipy.stats
            from scipy import stats
            
            # Separate predictions for each class
            pos_scores = y_pred_proba[y_true == 1]
            neg_scores = y_pred_proba[y_true == 0]
            
            # Calculate KS statistic (Kolmogorov-Smirnov test)
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
                score = ks_stat
            else:
                # Fallback to accuracy if one class is missing
                score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int))
        else:
            # For multiclass, keep using accuracy
            score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        
        return float(score)
    
    # Training loop
    train_size = len(Y_train)
    best = {'val': -math.inf, 'test': -math.inf, 'epoch': -1}
    remaining_patience = patience
    
    if verbose:
        # Clarify what the score represents based on task type
        if task_type == 'regression':
            metric_name = "Negative RMSE"
        elif task_type == 'binclass':
            metric_name = "KS Statistic"
        else:
            metric_name = "Accuracy"
        
        print(f'Initial test {metric_name}: {evaluate("test"):.4f}')
        print('-' * 80)
    
    for epoch in range(n_epochs):
        # Create batches
        batches = torch.randperm(train_size, device=device).split(batch_size)
        
        # Training step
        for batch_idx in (tqdm(batches, desc=f'Epoch {epoch}') if verbose else batches):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            loss.backward()
            optimizer.step()
        
        # Evaluation
        val_score = evaluate('val')
        test_score = evaluate('test')
        
        if verbose:
            print(f'Epoch {epoch}: (val) {val_score:.4f} (test) {test_score:.4f}')
        
        # Early stopping
        if val_score > best['val']:
            if verbose:
                print('🌸 New best epoch! 🌸')
            best = {'val': val_score, 'test': test_score, 'epoch': epoch}
            remaining_patience = patience
        else:
            remaining_patience -= 1
        
        if remaining_patience < 0:
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            break
    
    return best


def main():
    parser = argparse.ArgumentParser(description='Train TabM on custom CSV dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target_column', type=str, required=True, help='Name of target column')
    parser.add_argument('--task_type', type=str, choices=['regression', 'binclass', 'multiclass'], 
                       required=True, help='Type of task')
    parser.add_argument('--categorical_columns', type=str, nargs='*', default=None,
                       help='List of categorical column names (auto-detected if not specified)')
    parser.add_argument('--sep', type=str, default=',',
                       help='CSV separator (default: comma). Use "\\t" for tab-separated files')
    parser.add_argument('--use_embeddings', action='store_true', default=True,
                       help='Use TabM-mini with piecewise-linear embeddings for numerical features (default: True)')
    parser.add_argument('--no_embeddings', action='store_true', 
                       help='Use standard TabM instead of TabM-mini with embeddings')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Enable hyperparameter tuning with Optuna (requires: pip install optuna)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of hyperparameter tuning trials (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10000, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=1000, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Directory to save model and preprocessing (default: auto-generated based on dataset name)')
    
    args = parser.parse_args()
    
    # Handle embedding flags
    if args.no_embeddings:
        args.use_embeddings = False
    
    # Handle separator (convert \t string to actual tab character)
    if args.sep == '\\t':
        args.sep = '\t'
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess dataset
    print(f'Loading dataset from {args.data_path}...')
    data_splits, n_num_features, cat_cardinalities, n_classes, feature_info = load_csv_dataset(
        args.data_path, args.target_column, args.task_type, args.categorical_columns, sep=args.sep
    )
    
    # Preprocess features
    data_splits, preprocessing, n_num_features = preprocess_features(data_splits, n_num_features)
    
    # Validate embeddings usage
    if args.use_embeddings and n_num_features == 0:
        print("Warning: --use_embeddings specified but no numerical features found. Using standard TabM.")
        args.use_embeddings = False
    
    # Hyperparameter tuning or use defaults
    hyperparams = None
    if args.tune_hyperparams:
        hyperparams = tune_hyperparameters(
            data_splits, n_num_features, cat_cardinalities, n_classes, 
            args.task_type, device, args.use_embeddings, args.n_trials
        )
        if hyperparams is None:
            print("Hyperparameter tuning failed, using default parameters")
    
    # Setup model and training
    model, optimizer, model_config = setup_model_and_training(
        n_num_features, cat_cardinalities, n_classes, args.task_type, 
        device, data_splits, args.use_embeddings, hyperparams
    )
    
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    print(f'Architecture: {"TabM-mini with piecewise-linear embeddings" if model.arch_type == "tabm-mini" and model.num_module is not None else "Standard TabM"}')
    if hyperparams:
        print(f'Tuned hyperparameters: {hyperparams}')
    
    # Train model
    best_result = train_model(
        model, optimizer, data_splits, args.task_type, device,
        args.n_epochs, args.patience, args.batch_size
    )
    
    print('\n' + '='*80)
    print('Training completed!')
    
    # Clarify what the score represents based on task type
    if args.task_type == 'regression':
        metric_name = "Negative RMSE"
    elif args.task_type == 'binclass':
        metric_name = "KS Statistic"
    else:
        metric_name = "Accuracy"
    
    print(f'Best validation {metric_name}: {best_result["val"]:.4f}')
    print(f'Best test {metric_name}: {best_result["test"]:.4f}')
    print(f'Best epoch: {best_result["epoch"]}')
    
    # Save model and preprocessing
    print('\n' + '-'*80)
    print('Saving model and preprocessing...')
    
    # Create save directory name based on dataset and task
    if args.save_dir:
        save_dir = args.save_dir
    else:
        dataset_name = Path(args.data_path).stem
        save_dir = f"saved_model_{dataset_name}_{args.task_type}"
    
    # Handle regression label stats for saving
    regression_label_stats = None
    if args.task_type == 'regression':
        Y_train = data_splits['train']['y']
        regression_label_stats = RegressionLabelStats(
            Y_train.mean().item(), Y_train.std().item()
        )
    
    model_path, preprocessing_path = save_model_and_preprocessing(
        model=model,
        preprocessing_pipeline=preprocessing,
        regression_label_stats=regression_label_stats,
        feature_info=feature_info,
        model_config=model_config,
        save_dir=save_dir
    )
    
    print(f'\n✅ Training and saving completed!')
    print(f'📁 Model saved to: {model_path}')
    print(f'📁 Preprocessing saved to: {preprocessing_path}')
    print(f'\nTo use for inference, you will need both files:')
    print(f'  - Model: {model_path}')
    print(f'  - Preprocessing: {preprocessing_path}')


if __name__ == '__main__':
    main() 