"""
Utility functions for BPM prediction project.

This module contains helper functions for data loading, model evaluation,
visualization, and submission generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict, Any, Optional
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_competition_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load competition datasets consistently.
    
    Args:
        data_dir: Directory containing the competition data files
        
    Returns:
        Tuple of (train_df, test_df, sample_submission_df)
        
    Raises:
        FileNotFoundError: If data files are not found in the specified directory
    """
    data_path = Path(data_dir)
    
    try:
        train_df = pd.read_csv(data_path / "train.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        sample_submission_df = pd.read_csv(data_path / "sample_submission.csv")
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Training set: {train_df.shape}")
        print(f"   - Test set: {test_df.shape}")
        print(f"   - Sample submission: {sample_submission_df.shape}")
        
        return train_df, test_df, sample_submission_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data files in {data_path}. Error: {e}")


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def setup_cross_validation(n_splits: int = 5, shuffle: bool = True, 
                          random_state: int = RANDOM_STATE) -> KFold:
    """
    Set up cross-validation strategy for regression.
    
    Args:
        n_splits: Number of folds
        shuffle: Whether to shuffle the data before splitting
        random_state: Random state for reproducibility
        
    Returns:
        Configured KFold cross-validator
    """
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series, 
                     cv_folds: int = 5, scoring: str = 'neg_root_mean_squared_error') -> Dict[str, float]:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with mean and std of CV scores
    """
    cv = setup_cross_validation(n_splits=cv_folds)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Convert to positive RMSE scores
    scores = -scores
    
    return {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'cv_scores': scores
    }


def plot_target_distribution(y: pd.Series, title: str = "BPM Distribution") -> None:
    """
    Plot distribution of target variable (BPM).
    
    Args:
        y: Target variable series
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Beats Per Minute')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(y, vert=True)
    axes[1].set_ylabel('Beats Per Minute')
    axes[1].set_title(f'{title} - Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n📊 {title} Statistics:")
    print(f"   - Count: {len(y):,}")
    print(f"   - Mean: {y.mean():.2f}")
    print(f"   - Median: {y.median():.2f}")
    print(f"   - Std: {y.std():.2f}")
    print(f"   - Min: {y.min():.2f}")
    print(f"   - Max: {y.max():.2f}")
    print(f"   - Range: {y.max() - y.min():.2f}")


def plot_feature_correlations(df: pd.DataFrame, target_col: str = 'BeatsPerMinute', 
                            figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix focusing on target variable.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        figsize: Figure size tuple
    """
    # Calculate correlations
    corr_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show correlations with target
    target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\n🎯 Features most correlated with {target_col}:")
    for feature, corr in target_corrs.head(10).items():
        if feature != target_col:
            print(f"   - {feature}: {corr:.3f}")


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple model results in a formatted table.
    
    Args:
        results: Dictionary with model names as keys and evaluation results as values
        
    Returns:
        DataFrame with comparison results
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values('cv_mean')
    
    print("🏆 Model Comparison (sorted by CV RMSE):")
    print("=" * 50)
    print(comparison_df[['cv_mean', 'cv_std']].to_string())
    
    return comparison_df


def save_model(model, model_name: str, models_dir: str = "models") -> str:
    """
    Save trained model using joblib.
    
    Args:
        model: Trained model object
        model_name: Name for the saved model
        models_dir: Directory to save models
        
    Returns:
        Path to saved model file
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Model saved: {model_path}")
    return model_path


def load_model(model_path: str):
    """
    Load saved model using joblib.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded model object
    """
    model = joblib.load(model_path)
    print(f"📂 Model loaded: {model_path}")
    return model


def generate_submission(test_ids: pd.Series, predictions: np.ndarray, 
                       filename: str = "submission.csv", 
                       submissions_dir: str = "submissions") -> str:
    """
    Generate submission file in the correct format.
    
    Args:
        test_ids: Test set IDs
        predictions: Model predictions
        filename: Name of submission file
        submissions_dir: Directory to save submission
        
    Returns:
        Path to generated submission file
    """
    os.makedirs(submissions_dir, exist_ok=True)
    
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'BeatsPerMinute': predictions
    })
    
    submission_path = os.path.join(submissions_dir, filename)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"📄 Submission file generated: {submission_path}")
    print(f"   - Shape: {submission_df.shape}")
    print(f"   - BPM range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"   - Sample predictions:")
    print(submission_df.head().to_string(index=False))
    
    return submission_path


def validate_submission(submission_path: str, expected_test_size: int) -> bool:
    """
    Validate submission file format and content.
    
    Args:
        submission_path: Path to submission file
        expected_test_size: Expected number of test samples
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        sub_df = pd.read_csv(submission_path)
        
        # Check columns
        expected_cols = ['ID', 'BeatsPerMinute']
        if list(sub_df.columns) != expected_cols:
            print(f"❌ Column mismatch. Expected: {expected_cols}, Got: {list(sub_df.columns)}")
            return False
        
        # Check shape
        if len(sub_df) != expected_test_size:
            print(f"❌ Size mismatch. Expected: {expected_test_size}, Got: {len(sub_df)}")
            return False
        
        # Check for missing values
        if sub_df.isnull().any().any():
            print("❌ Missing values found in submission")
            return False
        
        # Check BPM values are numeric and reasonable
        bpm_col = sub_df['BeatsPerMinute']
        if not pd.api.types.is_numeric_dtype(bpm_col):
            print("❌ BPM values are not numeric")
            return False
        
        if (bpm_col < 0).any() or (bpm_col > 1000).any():
            print("❌ BPM values outside reasonable range (0-1000)")
            return False
        
        print("✅ Submission validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def print_data_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print comprehensive dataset information.
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset
    """
    print(f"\n📋 {name} Information:")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📊 Column Information:")
    print(df.info())
    
    print(f"\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print(f"\n📈 Numeric Columns Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print(f"\n🏷️ Categorical Columns:")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"   - {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"     Values: {df[col].unique()}")
    else:
        print("No categorical columns found!")
