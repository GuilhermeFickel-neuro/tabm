#!/usr/bin/env python
# coding: utf-8

"""
TabM Training Script for Custom CSV Datasets

Uses TabM-mini with piecewise-linear embeddings by default for optimal performance.

Usage:
    python neurolake_train.py --data_path dataset.csv --target_column target --task_type regression

Requirements:
    - CSV file with headers
    - Target column specified
    - Task type: 'regression', 'binclass', or 'multiclass'
"""

import argparse
import math
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


class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


def load_csv_dataset(
    data_path: str,
    target_column: str,
    task_type: Literal['regression', 'binclass', 'multiclass'],
    categorical_columns: Optional[list] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
):
    """Load and preprocess CSV dataset for TabM training."""
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Handle categorical columns
    if categorical_columns is None:
        # Auto-detect categorical columns (object/string types)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Separate numerical and categorical features
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    X_num = X[numerical_columns].values.astype(np.float32) if numerical_columns else None
    X_cat = None
    cat_cardinalities = []
    
    if categorical_columns:
        # Encode categorical features
        X_cat_df = X[categorical_columns].copy()
        label_encoders = {}
        
        for col in categorical_columns:
            le = sklearn.preprocessing.LabelEncoder()
            X_cat_df[col] = le.fit_transform(X_cat_df[col].astype(str))
            label_encoders[col] = le
            cat_cardinalities.append(len(le.classes_))
        
        X_cat = X_cat_df.values.astype(np.int64)
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
    
    return data_splits, n_num_features, cat_cardinalities, n_classes


def preprocess_features(data_splits, n_num_features):
    """Preprocess numerical features using quantile transformation."""
    
    if n_num_features == 0:
        return data_splits, None
    
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
    
    return data_splits, preprocessing


def setup_model_and_training(
    n_num_features: int,
    cat_cardinalities: list,
    n_classes: Optional[int],
    task_type: str,
    device: torch.device,
    data_splits: dict,
    use_embeddings: bool = False
):
    """Setup TabM model and training components."""
    
    # Configure model architecture
    bins = None
    num_embeddings = None
    
    if use_embeddings and n_num_features > 0:
        # Use TabM-mini with piecewise-linear embeddings for numerical features
        arch_type = 'tabm-mini'
        
        # Compute bins from training data (CRITICAL for piecewise-linear embeddings)
        train_x_cont = torch.as_tensor(data_splits['train']['x_cont'], device=device)
        bins = rtdl_num_embeddings.compute_bins(train_x_cont)
        
        num_embeddings = {
            'type': 'PiecewiseLinearEmbeddings',
            'd_embedding': 16,
            'activation': False,
            'version': 'B',
        }
        print(f"Using TabM-mini with piecewise-linear embeddings (bins computed from {train_x_cont.shape[0]} training samples)")
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
            'n_blocks': 2 if use_embeddings else 3,
            'd_block': 512,
            'dropout': 0.1,
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
        lr=2e-3, 
        weight_decay=3e-4
    )
    
    return model, optimizer


def train_model(
    model,
    optimizer,
    data,
    Y_train,
    task_type: str,
    regression_label_stats: Optional[RegressionLabelStats],
    device: torch.device,
    n_epochs: int = 1000,
    patience: int = 16,
    batch_size: int = 256
):
    """Train the TabM model."""
    
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
        else:
            score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1) if task_type != 'regression' else y_pred > 0.5)
        
        return float(score)
    
    # Training loop
    train_size = len(Y_train)
    best = {'val': -math.inf, 'test': -math.inf, 'epoch': -1}
    remaining_patience = patience
    
    print(f'Initial test score: {evaluate("test"):.4f}')
    print('-' * 80)
    
    for epoch in range(n_epochs):
        # Create batches
        batches = torch.randperm(train_size, device=device).split(batch_size)
        
        # Training step
        for batch_idx in tqdm(batches, desc=f'Epoch {epoch}'):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            loss.backward()
            optimizer.step()
        
        # Evaluation
        val_score = evaluate('val')
        test_score = evaluate('test')
        print(f'Epoch {epoch}: (val) {val_score:.4f} (test) {test_score:.4f}')
        
        # Early stopping
        if val_score > best['val']:
            print('🌸 New best epoch! 🌸')
            best = {'val': val_score, 'test': test_score, 'epoch': epoch}
            remaining_patience = patience
        else:
            remaining_patience -= 1
        
        if remaining_patience < 0:
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
    parser.add_argument('--use_embeddings', action='store_true', default=True,
                       help='Use TabM-mini with piecewise-linear embeddings for numerical features (default: True)')
    parser.add_argument('--no_embeddings', action='store_true', 
                       help='Use standard TabM instead of TabM-mini with embeddings')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=16, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle embedding flags
    if args.no_embeddings:
        args.use_embeddings = False
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess dataset
    print(f'Loading dataset from {args.data_path}...')
    data_splits, n_num_features, cat_cardinalities, n_classes = load_csv_dataset(
        args.data_path, args.target_column, args.task_type, args.categorical_columns
    )
    
    # Preprocess features
    data_splits, preprocessing = preprocess_features(data_splits, n_num_features)
    
    # Validate embeddings usage
    if args.use_embeddings and n_num_features == 0:
        print("Warning: --use_embeddings specified but no numerical features found. Using standard TabM.")
        args.use_embeddings = False
    
    # Handle label preprocessing for regression
    regression_label_stats = None
    Y_train = data_splits['train']['y'].copy()
    
    if args.task_type == 'regression':
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
        
        if args.task_type == 'regression':
            data[split]['y'] = data[split]['y'].float()
    
    Y_train = torch.as_tensor(Y_train, device=device)
    if args.task_type == 'regression':
        Y_train = Y_train.float()
    
    # Setup model and training
    model, optimizer = setup_model_and_training(
        n_num_features, cat_cardinalities, n_classes, args.task_type, device, data_splits, args.use_embeddings
    )
    
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    print(f'Architecture: {"TabM-mini with piecewise-linear embeddings" if args.use_embeddings and n_num_features > 0 else "Standard TabM"}')
    
    # Train model
    best_result = train_model(
        model, optimizer, data, Y_train, args.task_type, regression_label_stats, device,
        args.n_epochs, args.patience, args.batch_size
    )
    
    print('\n' + '='*80)
    print('Training completed!')
    print(f'Best validation score: {best_result["val"]:.4f}')
    print(f'Best test score: {best_result["test"]:.4f}')
    print(f'Best epoch: {best_result["epoch"]}')


if __name__ == '__main__':
    main() 