#!/usr/bin/env python
# coding: utf-8

"""
KS Statistics Utility Functions

Provides utility functions for calculating Kolmogorov-Smirnov statistics
that can be used by other scripts in the TabM pipeline.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List, Optional


def calculate_ks_for_feature(feature_values: np.ndarray, target_values: np.ndarray) -> Tuple[float, float]:
    """
    Calculate KS statistic for a single feature.
    
    Args:
        feature_values: Array of feature values
        target_values: Array of binary target values (0/1)
    
    Returns:
        Tuple of (ks_statistic, p_value)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(feature_values) & ~np.isnan(target_values)
    if valid_mask.sum() == 0:
        return 0.0, 1.0
    
    feature_clean = feature_values[valid_mask]
    target_clean = target_values[valid_mask]
    
    # Separate feature values by class
    positive_class = feature_clean[target_clean == 1]
    negative_class = feature_clean[target_clean == 0]
    
    if len(positive_class) == 0 or len(negative_class) == 0:
        return 0.0, 1.0
    
    # Calculate KS statistic
    ks_stat, p_value = ks_2samp(positive_class, negative_class)
    
    return ks_stat, p_value


def calculate_ks_for_dataframe(df: pd.DataFrame, target_column: str = 'alvo', 
                              exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate KS statistics for all numerical columns in a dataframe.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        exclude_columns: List of columns to exclude from analysis
    
    Returns:
        DataFrame with KS statistics for each feature
    """
    if exclude_columns is None:
        exclude_columns = ['cpf', 'ref_date', 'id', 'customer_id']
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Get target values and convert to binary if needed
    target_values = df[target_column].values
    unique_targets = pd.Series(target_values).dropna().unique()
    
    if len(unique_targets) == 2:
        # Convert to 0/1 if not already
        if not set(unique_targets).issubset({0, 1}):
            target_mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
            target_values = pd.Series(target_values).map(target_mapping).values
    else:
        raise ValueError(f"Target must be binary. Found {len(unique_targets)} unique values")
    
    # Find numerical columns
    numerical_columns = []
    exclude_set = set(exclude_columns + [target_column])
    
    for col in df.columns:
        if col in exclude_set:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
        else:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numerical_columns.append(col)
            except (ValueError, TypeError):
                continue
    
    # Calculate KS statistics
    results = []
    for col in numerical_columns:
        feature_values = pd.to_numeric(df[col], errors='coerce').values
        ks_stat, p_value = calculate_ks_for_feature(feature_values, target_values)
        
        results.append({
            'feature': col,
            'ks_statistic': ks_stat,
            'ks_p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ks_statistic', ascending=False).reset_index(drop=True)
    results_df['ks_rank'] = range(1, len(results_df) + 1)
    
    return results_df


def get_top_ks_features(df: pd.DataFrame, target_column: str = 'alvo', 
                       top_n: int = 20, min_ks: float = 0.0) -> List[str]:
    """
    Get list of top features by KS statistic.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        top_n: Number of top features to return
        min_ks: Minimum KS statistic threshold
    
    Returns:
        List of feature names sorted by KS statistic
    """
    ks_results = calculate_ks_for_dataframe(df, target_column)
    
    # Filter by minimum KS threshold
    filtered_results = ks_results[ks_results['ks_statistic'] >= min_ks]
    
    # Get top N features
    top_features = filtered_results.head(top_n)['feature'].tolist()
    
    return top_features


def print_ks_summary(df: pd.DataFrame, target_column: str = 'alvo', top_n: int = 10):
    """
    Print a summary of KS statistics for the dataset.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        top_n: Number of top features to display
    """
    try:
        ks_results = calculate_ks_for_dataframe(df, target_column)
        
        print(f"\n📊 KS Statistics Summary (Target: {target_column})")
        print("=" * 60)
        print(f"Total numerical features: {len(ks_results)}")
        print(f"Features with KS > 0.1: {(ks_results['ks_statistic'] > 0.1).sum()}")
        print(f"Features with KS > 0.2: {(ks_results['ks_statistic'] > 0.2).sum()}")
        print(f"Features with KS > 0.3: {(ks_results['ks_statistic'] > 0.3).sum()}")
        
        print(f"\nTop {top_n} features by KS statistic:")
        print("-" * 50)
        for _, row in ks_results.head(top_n).iterrows():
            print(f"{row['ks_rank']:2d}. {row['feature']:<25} KS: {row['ks_statistic']:.4f}")
        
        return ks_results
        
    except Exception as e:
        print(f"Error calculating KS statistics: {e}")
        return None


def add_ks_to_inference_results(predictions_df: pd.DataFrame, original_df: pd.DataFrame, 
                               target_column: str = 'alvo') -> pd.DataFrame:
    """
    Add KS statistics information to inference results.
    
    Args:
        predictions_df: DataFrame with model predictions
        original_df: Original dataframe with features and target
        target_column: Name of target column
    
    Returns:
        Enhanced predictions dataframe with KS statistics
    """
    try:
        # Calculate KS statistics for the original data
        ks_results = calculate_ks_for_dataframe(original_df, target_column)
        
        # Create a mapping of feature to KS statistic
        ks_mapping = dict(zip(ks_results['feature'], ks_results['ks_statistic']))
        
        # Add KS information to predictions
        enhanced_df = predictions_df.copy()
        
        # Add overall KS statistics summary
        enhanced_df.attrs['ks_summary'] = {
            'total_features': len(ks_results),
            'mean_ks': ks_results['ks_statistic'].mean(),
            'max_ks': ks_results['ks_statistic'].max(),
            'top_features': ks_results.head(5)['feature'].tolist()
        }
        
        return enhanced_df
        
    except Exception as e:
        print(f"Warning: Could not add KS statistics to results: {e}")
        return predictions_df


if __name__ == '__main__':
    # Example usage
    print("KS Statistics Utility Functions")
    print("Import this module to use KS calculation functions in other scripts.")
    print("\nAvailable functions:")
    print("- calculate_ks_for_feature()")
    print("- calculate_ks_for_dataframe()")
    print("- get_top_ks_features()")
    print("- print_ks_summary()")
    print("- add_ks_to_inference_results()") 