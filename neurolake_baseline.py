#!/usr/bin/env python
# coding: utf-8

"""
Baseline Models Training Script for Binary Classification

Trains and evaluates baseline models (XGBoost, CatBoost, LightGBM) with Optuna hyperparameter tuning.
Uses the same data loading and preprocessing as neurolake_train.py and neurolake_inference.py.
Evaluates using KS statistic for binary classification.

Usage:
    # For tab-separated CSV (default)
    python neurolake_baseline.py --train_path train.csv --test_path test.csv --target_column target_value
    
    # For comma-separated CSV
    python neurolake_baseline.py --train_path train.csv --test_path test.csv --target_column target --sep ","

Requirements:
    - CSV files with headers (tab or comma separated)
    - Target column specified for binary classification
    - Automatically excludes identifier columns (cpf, ref_date, etc.)
    
Evaluation Metrics:
    - Binary Classification: KS statistic (higher is better)
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna


def load_and_preprocess_data(
    train_path: str,
    test_path: str,
    target_column: str,
    sep: str = '\t'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess train and test datasets."""
    
    # Load datasets
    train_df = pd.read_csv(train_path, sep=sep)
    test_df = pd.read_csv(test_path, sep=sep)
    
    print(f"Train dataset: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test dataset: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Check if target column exists
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train dataset")
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test dataset")
    
    # Identify and exclude identifier columns (same logic as neurolake_train.py)
    identifier_patterns = ['ref_date', 'date', 'timestamp']
    identifier_columns = []
    for col in train_df.columns:
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
        X_train = train_df.drop(columns=[target_column] + identifier_columns)
        X_test = test_df.drop(columns=[target_column] + identifier_columns)
    else:
        X_train = train_df.drop(columns=[target_column])
        X_test = test_df.drop(columns=[target_column])
    
    y_train = train_df[target_column].values
    y_test = test_df[target_column].values
    
    # Ensure both datasets have the same columns
    common_columns = list(set(X_train.columns) & set(X_test.columns))
    if len(common_columns) != len(X_train.columns) or len(common_columns) != len(X_test.columns):
        print(f"Warning: Train and test datasets have different columns. Using common columns only.")
        print(f"Train columns: {len(X_train.columns)}, Test columns: {len(X_test.columns)}, Common: {len(common_columns)}")
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
    
    # Auto-detect categorical columns (object/string types) - EXACT MATCH to neurolake_train.py
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = [col for col in X_train.columns if col not in categorical_columns]
    
    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")
    
    # Process numerical features - EXACT MATCH to neurolake_train.py
    X_num = None
    if numerical_columns:
        X_train_num = X_train[numerical_columns].copy()
        X_test_num = X_test[numerical_columns].copy()
        
        # Handle missing and infinite values in numerical features
        print(f"Checking for missing/infinite values in numerical features...")
        
        # Check for missing values
        missing_counts = X_train_num.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found missing values in {(missing_counts > 0).sum()} columns")
            for col in missing_counts[missing_counts > 0].index:
                print(f"  {col}: {missing_counts[col]} missing values")
        
        # Fill missing values with median (using train median for both train and test)
        for col in X_train_num.columns:
            if X_train_num[col].isnull().any():
                median_val = X_train_num[col].median()
                X_train_num[col].fillna(median_val, inplace=True)
                X_test_num[col].fillna(median_val, inplace=True)
            elif X_test_num[col].isnull().any():
                median_val = X_train_num[col].median()
                X_test_num[col].fillna(median_val, inplace=True)
        
        # Convert to float32
        X_train_num = X_train_num.astype(np.float32)
        X_test_num = X_test_num.astype(np.float32)
        
        # Check for infinite values
        inf_mask = np.isinf(X_train_num.values)
        if inf_mask.any():
            print(f"Found {inf_mask.sum()} infinite values in train, replacing with finite values...")
            X_train_num = X_train_num.replace([np.inf, -np.inf], [np.finfo(np.float32).max/2, np.finfo(np.float32).min/2])
        
        inf_mask_test = np.isinf(X_test_num.values)
        if inf_mask_test.any():
            print(f"Found {inf_mask_test.sum()} infinite values in test, replacing with finite values...")
            X_test_num = X_test_num.replace([np.inf, -np.inf], [np.finfo(np.float32).max/2, np.finfo(np.float32).min/2])
        
        # Final check for any remaining NaN/inf values
        final_check_train = np.isnan(X_train_num.values) | np.isinf(X_train_num.values)
        if final_check_train.any():
            print(f"Warning: Still found {final_check_train.sum()} NaN/inf values in train after cleaning")
            X_train_num = X_train_num.fillna(0)
            X_train_num = X_train_num.replace([np.inf, -np.inf], 0)
        
        final_check_test = np.isnan(X_test_num.values) | np.isinf(X_test_num.values)
        if final_check_test.any():
            print(f"Warning: Still found {final_check_test.sum()} NaN/inf values in test after cleaning")
            X_test_num = X_test_num.fillna(0)
            X_test_num = X_test_num.replace([np.inf, -np.inf], 0)
        
        # Filter out numerical features with only one unique value (based on train data)
        constant_features = []
        for col in X_train_num.columns:
            if X_train_num[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Removing {len(constant_features)} numerical features with constant values")
            X_train_num = X_train_num.drop(columns=constant_features)
            X_test_num = X_test_num.drop(columns=constant_features)
            numerical_columns = [col for col in numerical_columns if col not in constant_features]
        
        X_num_train = X_train_num.values if len(X_train_num.columns) > 0 else None
        X_num_test = X_test_num.values if len(X_test_num.columns) > 0 else None
        print(f"Numerical features cleaned: {X_num_train.shape if X_num_train is not None else 'None'}")
    else:
        X_num_train = None
        X_num_test = None
    
    # Process categorical features - EXACT MATCH to neurolake_train.py
    X_cat_train = None
    X_cat_test = None
    cat_cardinalities = []
    label_encoders = {}
    
    if categorical_columns:
        # Encode categorical features
        X_train_cat = X_train[categorical_columns].copy()
        X_test_cat = X_test[categorical_columns].copy()
        
        for col in categorical_columns:
            le = sklearn.preprocessing.LabelEncoder()
            # Fit only on train data (like neurolake_train.py)
            X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))
            
            # Transform test data, handling unseen categories
            test_values = X_test_cat[col].astype(str)
            # Map unseen categories to a new label (len(classes))
            test_encoded = []
            for val in test_values:
                if val in le.classes_:
                    test_encoded.append(le.transform([val])[0])
                else:
                    # Assign unseen categories to class 0 (or could be len(le.classes_))
                    test_encoded.append(0)
            X_test_cat[col] = test_encoded
            
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
            X_train_cat = X_train_cat[filtered_categorical_columns]
            X_test_cat = X_test_cat[filtered_categorical_columns]
            categorical_columns = filtered_categorical_columns
            cat_cardinalities = filtered_cat_cardinalities
            label_encoders = filtered_label_encoders
        
        X_cat_train = X_train_cat.values.astype(np.int64) if len(categorical_columns) > 0 else None
        X_cat_test = X_test_cat.values.astype(np.int64) if len(categorical_columns) > 0 else None
        print(f"Categorical features: {len(categorical_columns)}, cardinalities: {cat_cardinalities}")
    
    # Apply quantile transformation with noise (EXACT MATCH to neurolake_train.py)
    if X_num_train is not None:
        # Add noise to training data before fitting scaler
        noise = np.random.default_rng(0).normal(0.0, 1e-5, X_num_train.shape).astype(X_num_train.dtype)
        
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(X_num_train) // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        )
        
        X_num_train_scaled = scaler.fit_transform(X_num_train + noise)
        X_num_test_scaled = scaler.transform(X_num_test)
        
        # Check for constant features after scaling (EXACT MATCH to neurolake_train.py)
        constant_feature_indices = []
        for i in range(X_num_train_scaled.shape[1]):
            feature_values = X_num_train_scaled[:, i]
            # Check if all values are the same (within a small tolerance for floating point)
            if np.allclose(feature_values, feature_values[0], rtol=1e-10, atol=1e-10):
                constant_feature_indices.append(i)
        
        if constant_feature_indices:
            print(f"Removing {len(constant_feature_indices)} numerical features that became constant after preprocessing")
            
            # Create mask for features to keep
            keep_indices = [i for i in range(X_num_train_scaled.shape[1]) if i not in constant_feature_indices]
            
            if len(keep_indices) == 0:
                # All features became constant
                print("Warning: All numerical features became constant after preprocessing")
                X_num_train_scaled = None
                X_num_test_scaled = None
                numerical_columns = []
            else:
                # Filter out constant features
                X_num_train_scaled = X_num_train_scaled[:, keep_indices]
                X_num_test_scaled = X_num_test_scaled[:, keep_indices]
                numerical_columns = [numerical_columns[i] for i in keep_indices]
                print(f"Numerical features after preprocessing: {len(keep_indices)}")
    else:
        X_num_train_scaled = None
        X_num_test_scaled = None
    
    # Combine numerical and categorical features
    X_train_combined = []
    X_test_combined = []
    
    if X_num_train_scaled is not None:
        X_train_combined.append(X_num_train_scaled)
        X_test_combined.append(X_num_test_scaled)
    
    if X_cat_train is not None:
        X_train_combined.append(X_cat_train.astype(np.float32))  # Convert to float32 for consistency
        X_test_combined.append(X_cat_test.astype(np.float32))
    
    if X_train_combined:
        X_train_final = np.concatenate(X_train_combined, axis=1)
        X_test_final = np.concatenate(X_test_combined, axis=1)
    else:
        raise ValueError("No features remaining after preprocessing")
    
    # Process target variable for binary classification (EXACT MATCH to neurolake_train.py)
    le_target = sklearn.preprocessing.LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train).astype(np.int64)
    y_test_encoded = le_target.transform(y_test).astype(np.int64)
    
    print(f"Final feature matrix: {X_train_final.shape[1]} features")
    print(f"Target classes: {le_target.classes_}")
    print(f"Binary classification: 2 classes")
    
    # Create feature info
    feature_info = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'cat_cardinalities': cat_cardinalities,
        'label_encoders': label_encoders,
        'target_encoder': le_target,
        'n_features': X_train_final.shape[1]
    }
    
    return X_train_final, X_test_final, y_train_encoded, y_test_encoded, feature_info


def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate KS statistic for binary classification."""
    # Get probability of positive class
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Separate predictions for each class
    pos_scores = y_pred_proba[y_true == 1]
    neg_scores = y_pred_proba[y_true == 0]
    
    # Calculate KS statistic
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
        return ks_stat
    else:
        return 0.0


def tune_and_train_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    categorical_features: list = None,
    n_trials: int = 100
) -> Dict[str, Any]:
    """Tune hyperparameters and train a model using Optuna."""
    
    model_class = model_config['class']
    base_params = model_config['base_params']
    trial_params = model_config['trial_params']
    
    def objective(trial):
        # Sample hyperparameters
        params = base_params.copy()
        
        for param_name, param_config in trial_params.items():
            param_type, *param_args = param_config
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, *param_args)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, *param_args)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_args[0])
        
        try:
            # Create and train model
            if model_name == 'catboost' and categorical_features:
                # CatBoost can handle categorical features directly
                model = model_class(**params, cat_features=categorical_features)
            else:
                model = model_class(**params)
            
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate KS statistic
            ks_score = calculate_ks_statistic(y_test, y_pred_proba)
            
            return ks_score
            
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return 0.0
    
    print(f"Tuning {model_name} with {n_trials} trials...")
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final model with best parameters
    best_params = {**base_params, **study.best_params}
    
    if model_name == 'catboost' and categorical_features:
        final_model = model_class(**best_params, cat_features=categorical_features)
    else:
        final_model = model_class(**best_params)
    
    final_model.fit(X_train, y_train)
    
    # Final predictions
    y_pred_proba = final_model.predict_proba(X_test)
    final_ks = calculate_ks_statistic(y_test, y_pred_proba)
    
    return {
        'model': final_model,
        'best_params': best_params,
        'best_trial_score': study.best_value,
        'final_ks': final_ks,
        'study': study
    }


def main():
    parser = argparse.ArgumentParser(description='Train baseline models for binary classification')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--target_column', type=str, default='target_value', help='Name of target column')
    parser.add_argument('--sep', type=str, default='\t',
                       help='CSV separator (default: tab). Use "," for comma-separated files')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of hyperparameter tuning trials (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle separator
    if args.sep == '\\t':
        args.sep = '\t'
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load and preprocess data
    print(f'Loading datasets...')
    print(f'Train: {args.train_path}')
    print(f'Test: {args.test_path}')
    print(f'Target column: {args.target_column}')
    print(f'Separator: {repr(args.sep)}')
    print('-' * 80)
    
    X_train, X_test, y_train, y_test, feature_info = load_and_preprocess_data(
        args.train_path, args.test_path, args.target_column, args.sep
    )
    
    print(f'Preprocessing completed!')
    print(f'Train set: {X_train.shape}')
    print(f'Test set: {X_test.shape}')
    print(f'Features: {feature_info["n_features"]}')
    print('=' * 80)
    
    # Define model configurations
    model_configs = {
        'xgboost': {
            'class': XGBClassifier,
            'base_params': {'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'auc'},
            'trial_params': {
                'n_estimators': ['int', 50, 500],
                'max_depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.3, 1.0],
                'min_child_weight': ['int', 1, 10],
                'gamma': ['float', 0, 5],
                'reg_alpha': ['float', 0, 10],
                'reg_lambda': ['float', 0, 15]
            }
        },
        'catboost': {
            'class': CatBoostClassifier,
            'base_params': {'random_seed': 42, 'verbose': False, 'eval_metric': 'AUC'},
            'trial_params': {
                'iterations': ['int', 50, 500],
                'depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'l2_leaf_reg': ['float', 1, 10],
                'random_strength': ['float', 0, 10],
                'bagging_temperature': ['float', 0, 10]
            }
        },
        'lightgbm': {
            'class': LGBMClassifier,
            'base_params': {'random_state': 42, 'verbosity': -1},
            'trial_params': {
                'n_estimators': ['int', 50, 500],
                'max_depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'num_leaves': ['int', 20, 100],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.3, 1.0],
                'reg_alpha': ['float', 0, 10],
                'reg_lambda': ['float', 0, 15]
            }
        }
    }
    
    # Train and evaluate each model
    results = {}
    
    # Prepare categorical features for CatBoost
    categorical_features = None
    if feature_info['categorical_columns']:
        # Calculate indices of categorical features in the combined feature matrix
        n_numerical = len(feature_info['numerical_columns'])
        categorical_features = list(range(n_numerical, n_numerical + len(feature_info['categorical_columns'])))
    
    for model_name, model_config in model_configs.items():
        print(f'\nTraining {model_name.upper()}...')
        print('-' * 40)
        
        try:
            result = tune_and_train_model(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                categorical_features=categorical_features if model_name == 'catboost' else None,
                n_trials=args.n_trials
            )
            
            results[model_name] = result
            
            print(f'Best trial KS: {result["best_trial_score"]:.4f}')
            print(f'Final test KS: {result["final_ks"]:.4f}')
            print(f'Best parameters: {result["best_params"]}')
            
        except Exception as e:
            print(f'Failed to train {model_name}: {e}')
            continue
    
    # Print final results
    print('\n' + '=' * 80)
    print('FINAL RESULTS')
    print('=' * 80)
    
    if results:
        # Sort by final KS score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['final_ks'], reverse=True)
        
        print(f'{"Model":<12} {"Test KS":<10} {"Best Trial KS":<15}')
        print('-' * 40)
        
        for model_name, result in sorted_results:
            print(f'{model_name.upper():<12} {result["final_ks"]:<10.4f} {result["best_trial_score"]:<15.4f}')
        
        best_model_name, best_result = sorted_results[0]
        print(f'\n🏆 Best model: {best_model_name.upper()} with KS = {best_result["final_ks"]:.4f}')
        
    else:
        print('No models were successfully trained.')
    
    print('\n✅ Baseline evaluation completed!')


if __name__ == '__main__':
    main() 