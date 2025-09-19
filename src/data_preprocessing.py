"""
Data preprocessing module for BPM prediction project.

This module contains functions for data cleaning, validation, feature scaling,
and preprocessing pipeline management.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, List, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Random state for reproducibility
RANDOM_STATE = 42


def validate_data_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive data quality validation.
    
    Args:
        df: DataFrame to validate
        target_col: Name of target column (if present)
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        'shape': df.shape,
        'missing_values': {},
        'duplicates': 0,
        'data_types': {},
        'outliers': {},
        'recommendations': []
    }
    
    # Check missing values
    missing = df.isnull().sum()
    validation_results['missing_values'] = missing[missing > 0].to_dict()
    
    # Check duplicates
    validation_results['duplicates'] = df.duplicated().sum()
    
    # Check data types
    validation_results['data_types'] = df.dtypes.to_dict()
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            validation_results['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'bounds': (lower_bound, upper_bound)
            }
    
    # Generate recommendations
    if validation_results['missing_values']:
        validation_results['recommendations'].append("Handle missing values")
    
    if validation_results['duplicates'] > 0:
        validation_results['recommendations'].append("Remove duplicate rows")
    
    if validation_results['outliers']:
        validation_results['recommendations'].append("Investigate and handle outliers")
    
    # Target-specific validation
    if target_col and target_col in df.columns:
        target_stats = {
            'min': df[target_col].min(),
            'max': df[target_col].max(),
            'mean': df[target_col].mean(),
            'std': df[target_col].std(),
            'skewness': df[target_col].skew()
        }
        validation_results['target_stats'] = target_stats
        
        # Check for reasonable BPM range
        if target_stats['min'] < 30 or target_stats['max'] > 300:
            validation_results['recommendations'].append("Check BPM values - some may be outside typical music range (30-300)")
    
    return validation_results


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median', 
                         categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy for numeric columns ('mean', 'median', 'constant', 'knn')
        categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    # Handle numeric columns
    if len(numeric_cols) > 0 and df_clean[numeric_cols].isnull().any().any():
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        else:
            imputer = SimpleImputer(strategy=strategy)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
        print(f"✅ Handled missing values in numeric columns using {strategy} strategy")
    
    # Handle categorical columns
    if len(categorical_cols) > 0 and df_clean[categorical_cols].isnull().any().any():
        imputer = SimpleImputer(strategy=categorical_strategy)
        df_clean[categorical_cols] = imputer.fit_transform(df_clean[categorical_cols])
        print(f"✅ Handled missing values in categorical columns using {categorical_strategy} strategy")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> Dict[str, pd.Index]:
    """
    Detect outliers in specified columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check (if None, check all numeric columns)
        method: Method to use ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary mapping column names to indices of outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
            outlier_pred = iso_forest.fit_predict(df[[col]].dropna())
            outlier_mask = outlier_pred == -1
        
        outliers[col] = df[outlier_mask].index
    
    return outliers


def handle_outliers(df: pd.DataFrame, outliers: Dict[str, pd.Index], 
                   method: str = 'cap') -> pd.DataFrame:
    """
    Handle detected outliers.
    
    Args:
        df: DataFrame with outliers
        outliers: Dictionary of outliers from detect_outliers()
        method: Method to handle outliers ('remove', 'cap', 'transform')
        
    Returns:
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    
    if method == 'remove':
        # Remove rows with outliers in any column
        all_outlier_indices = set()
        for indices in outliers.values():
            all_outlier_indices.update(indices)
        df_clean = df_clean.drop(index=list(all_outlier_indices))
        print(f"✅ Removed {len(all_outlier_indices)} outlier rows")
        
    elif method == 'cap':
        # Cap outliers to reasonable bounds
        for col, indices in outliers.items():
            if len(indices) > 0:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"✅ Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    elif method == 'transform':
        # Apply log transformation to reduce outlier impact
        for col in outliers.keys():
            if df_clean[col].min() > 0:  # Only if all values are positive
                df_clean[f'{col}_log'] = np.log1p(df_clean[col])
                print(f"✅ Applied log transformation to {col}")
    
    return df_clean


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Scale features using specified scaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, fitted_scaler)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit on training data and transform both sets
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✅ Applied {scaler_type} scaling to features")
    
    return X_train_scaled, X_test_scaled, scaler


def create_train_val_split(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          stratify: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train-validation split.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for validation
        stratify: Series for stratified splitting (optional)
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=stratify
    )
    
    print(f"✅ Created train-validation split:")
    print(f"   - Training: {X_train.shape[0]} samples")
    print(f"   - Validation: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val


def encode_categorical_features(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                               categorical_cols: List[str], 
                               encoding_type: str = 'onehot') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        categorical_cols: List of categorical columns to encode
        encoding_type: Type of encoding ('onehot', 'label', 'target')
        
    Returns:
        Tuple of (encoded_train_df, encoded_test_df)
    """
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    
    if encoding_type == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Fit on training data
        train_encoded = encoder.fit_transform(df_train[categorical_cols])
        test_encoded = encoder.transform(df_test[categorical_cols])
        
        # Create column names
        feature_names = encoder.get_feature_names_out(categorical_cols)
        
        # Add encoded features
        train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=df_train.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=df_test.index)
        
        # Drop original categorical columns and add encoded ones
        df_train_encoded = df_train_encoded.drop(columns=categorical_cols)
        df_test_encoded = df_test_encoded.drop(columns=categorical_cols)
        
        df_train_encoded = pd.concat([df_train_encoded, train_encoded_df], axis=1)
        df_test_encoded = pd.concat([df_test_encoded, test_encoded_df], axis=1)
        
    elif encoding_type == 'label':
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            encoder = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined_data = pd.concat([df_train[col], df_test[col]], axis=0)
            encoder.fit(combined_data.astype(str))
            
            df_train_encoded[col] = encoder.transform(df_train[col].astype(str))
            df_test_encoded[col] = encoder.transform(df_test[col].astype(str))
    
    print(f"✅ Applied {encoding_type} encoding to categorical features")
    
    return df_train_encoded, df_test_encoded


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for BPM prediction.
    """
    
    def __init__(self, 
                 missing_strategy: str = 'median',
                 outlier_method: str = 'cap',
                 scaler_type: str = 'standard',
                 encoding_type: str = 'onehot'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for handling outliers
            scaler_type: Type of feature scaling
            encoding_type: Type of categorical encoding
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.scaler_type = scaler_type
        self.encoding_type = encoding_type
        
        # Store fitted components
        self.scaler = None
        self.categorical_cols = None
        self.numeric_cols = None
        
    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None):
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X_train: Training features
            y_train: Training target (optional)
        """
        # Identify column types
        self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"✅ Pipeline fitted on training data")
        print(f"   - Numeric columns: {len(self.numeric_cols)}")
        print(f"   - Categorical columns: {len(self.categorical_cols)}")
        
        return self
    
    def transform(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply preprocessing transformations.
        
        Args:
            X: Features to transform
            is_training: Whether this is training data
            
        Returns:
            Transformed features
        """
        X_processed = X.copy()
        
        # Handle missing values
        if X_processed.isnull().any().any():
            X_processed = handle_missing_values(X_processed, self.missing_strategy)
        
        # Handle outliers (only on training data)
        if is_training and self.outlier_method != 'none':
            outliers = detect_outliers(X_processed, self.numeric_cols)
            if any(len(indices) > 0 for indices in outliers.values()):
                X_processed = handle_outliers(X_processed, outliers, self.outlier_method)
        
        return X_processed
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit pipeline and transform training data.
        
        Args:
            X_train: Training features
            y_train: Training target (optional)
            
        Returns:
            Transformed training features
        """
        return self.fit(X_train, y_train).transform(X_train, is_training=True)


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature summary.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary DataFrame with feature statistics
    """
    summary_data = []
    
    for col in df.columns:
        col_info = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().sum() / len(df) * 100,
            'unique_count': df[col].nunique(),
            'unique_pct': df[col].nunique() / len(df) * 100
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew()
            })
        else:
            col_info.update({
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'skewness': None
            })
        
        summary_data.append(col_info)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df.round(3)
