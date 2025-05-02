"""
Bank Marketing Term Deposit Prediction - Data Preprocessing

This module contains utilities for preprocessing and transforming
the bank marketing dataset.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_and_clean_data(file_path):
    """
    Load and perform initial cleaning of the bank marketing dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    # Check for and report missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    
    # Check for 'unknown' values in categorical columns
    unknown_values = {}
    for col in df.select_dtypes(include=['object']).columns:
        n_unknown = (df[col] == 'unknown').sum()
        if n_unknown > 0:
            unknown_values[col] = n_unknown
            print(f"'{col}' has {n_unknown} unknown values ({n_unknown/len(df):.2%})")
    
    # Convert target to binary
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    return df


def handle_unknown_values(df, strategy='most_frequent'):
    """
    Handle 'unknown' values in categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling unknown values:
        - 'most_frequent': Replace with most frequent value
        - 'keep': Keep as a separate category
        - 'drop': Drop rows with unknown values
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled unknown values
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        # Identify columns with 'unknown' values
        cols_with_unknown = []
        for col in df_copy.select_dtypes(include=['object']).columns:
            if (df_copy[col] == 'unknown').any():
                cols_with_unknown.append(col)
        
        # Drop rows with 'unknown' values
        for col in cols_with_unknown:
            df_copy = df_copy[df_copy[col] != 'unknown']
        
        print(f"Dropped {len(df) - len(df_copy)} rows with unknown values")
        
    elif strategy == 'most_frequent':
        # Replace with most frequent value
        for col in df_copy.select_dtypes(include=['object']).columns:
            if (df_copy[col] == 'unknown').any():
                # Get most frequent value excluding 'unknown'
                most_frequent = df_copy[df_copy[col] != 'unknown'][col].mode()[0]
                # Replace 'unknown' with most frequent
                df_copy.loc[df_copy[col] == 'unknown', col] = most_frequent
                print(f"Replaced 'unknown' in '{col}' with '{most_frequent}'")
    
    # If strategy is 'keep', do nothing
    
    return df_copy


def create_engineered_features(df):
    """
    Create engineered features to improve model performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional engineered features
    """
    df_copy = df.copy()
    
    # Age groups
    df_copy['age_group'] = pd.cut(
        df_copy['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '60+']
    )
    
    # Binary feature for whether client was contacted before
    df_copy['previously_contacted'] = (df_copy['previous'] > 0).astype(int)
    
    # Binary feature for pdays (999 means client was not previously contacted)
    df_copy['recent_contact'] = (df_copy['pdays'] < 999).astype(int)
    
    # Campaign intensity (number of contacts divided by average)
    avg_campaign = df_copy['campaign'].mean()
    df_copy['campaign_intensity'] = df_copy['campaign'] / avg_campaign
    
    # Economic indicator features
    # Combine economic indicators into a single score
    economic_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    # Normalize economic indicators
    economic_indicators = df_copy[economic_cols].copy()
    for col in economic_cols:
        economic_indicators[col] = (economic_indicators[col] - economic_indicators[col].mean()) / economic_indicators[col].std()
    
    # Create principal economic indicator
    df_copy['economic_indicator'] = economic_indicators.mean(axis=1)
    
    # Interaction between age and job
    df_copy['student_youth'] = ((df_copy['job'] == 'student') & (df_copy['age'] <= 30)).astype(int)
    df_copy['retired_senior'] = ((df_copy['job'] == 'retired') & (df_copy['age'] >= 60)).astype(int)
    
    # Month seasonality (grouped by quarter)
    season_map = {
        'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
        'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
        'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
        'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
    }
    df_copy['season'] = df_copy['month'].map(season_map)
    
    # Communication channel effectiveness
    df_copy['cellular_contact'] = (df_copy['contact'] == 'cellular').astype(int)
    
    return df_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding categorical variables
    with special handling for 'unknown' values.
    """
    
    def __init__(self, columns, handle_unknown='value'):
        """
        Initialize the encoder.
        
        Parameters:
        -----------
        columns : list
            List of column names to encode
        handle_unknown : str
            Strategy for handling unknown values:
            - 'value': Treat as a separate category
            - 'impute': Replace with most frequent value
            - 'error': Raise an error
        """
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.encodings = {}
        self.most_frequent = {}
        
    def fit(self, X, y=None):
        """
        Fit the encoder.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
        y : pd.Series, optional
            Target variable
        
        Returns:
        --------
        self
        """
        X_temp = X.copy()
        
        for col in self.columns:
            if col in X_temp.columns:
                # Store most frequent value
                self.most_frequent[col] = X_temp[col].value_counts().index[0]
                
                # Create mapping for each value
                unique_values = X_temp[col].unique()
                self.encodings[col] = {value: idx for idx, value in enumerate(unique_values)}
        
        return self
    
    def transform(self, X):
        """
        Transform the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        X_temp = X.copy()
        
        for col in self.columns:
            if col in X_temp.columns:
                if self.handle_unknown == 'value':
                    # Use existing encodings, assign new value for unknown
                    X_temp[col] = X_temp[col].map(lambda x: self.encodings[col].get(x, -1))
                
                elif self.handle_unknown == 'impute':
                    # Replace unknown values with most frequent
                    mask = ~X_temp[col].isin(self.encodings[col].keys())
                    X_temp.loc[mask, col] = self.most_frequent[col]
                    # Then encode
                    X_temp[col] = X_temp[col].map(self.encodings[col])
                
                elif self.handle_unknown == 'error':
                    # Check for unknown values
                    mask = ~X_temp[col].isin(self.encodings[col].keys())
                    if mask.any():
                        raise ValueError(f"Found unknown categories in column {col} during transform")
                    # Encode
                    X_temp[col] = X_temp[col].map(self.encodings[col])
        
        return X_temp


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for selecting features.
    """
    
    def __init__(self, features_to_select=None, features_to_drop=None):
        """
        Initialize the selector.
        
        Parameters:
        -----------
        features_to_select : list, optional
            List of features to select. If None, all features except 
            features_to_drop will be selected.
        features_to_drop : list, optional
            List of features to drop. Only used if features_to_select is None.
        """
        self.features_to_select = features_to_select
        self.features_to_drop = features_to_drop
        
    def fit(self, X, y=None):
        """
        Fit the selector.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
        y : pd.Series, optional
            Target variable
        
        Returns:
        --------
        self
        """
        return self
    
    def transform(self, X):
        """
        Transform the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe with selected features
        """
        if self.features_to_select is not None:
            # Select only specified features
            cols_to_select = [col for col in self.features_to_select if col in X.columns]
            return X[cols_to_select]
        
        elif self.features_to_drop is not None:
            # Drop specified features
            cols_to_drop = [col for col in self.features_to_drop if col in X.columns]
            return X.drop(columns=cols_to_drop)
        
        else:
            # Return all features
            return X


def prepare_data_for_modeling(df, target_col='y', test_size=0.2, random_state=42):
    """
    Prepare the data for modeling by splitting into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test
