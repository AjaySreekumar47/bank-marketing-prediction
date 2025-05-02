"""
Bank Marketing Term Deposit Prediction Model

This module contains the implementation of machine learning models
for predicting whether clients will subscribe to term deposits.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


class BankMarketingModel:
    """
    A class that encapsulates the bank marketing prediction model pipeline,
    including preprocessing, model training, and evaluation.
    """
    
    def __init__(self, model_type='gradient_boosting', handle_imbalance=True):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options: 'logistic_regression', 'random_forest', 
            'gradient_boosting'.
        handle_imbalance : bool
            Whether to handle class imbalance using SMOTE.
        """
        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.model = None
        self.preprocessor = None
        self.categorical_features = [
            'job', 'marital', 'education', 'default', 'housing', 
            'loan', 'contact', 'month', 'day_of_week', 'poutcome'
        ]
        self.numerical_features = [
            'age', 'duration', 'campaign', 'pdays', 'previous', 
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
            'euribor3m', 'nr.employed'
        ]
        
    def _create_preprocessor(self):
        """
        Create the preprocessing pipeline for categorical and numerical features.
        
        Returns:
        --------
        sklearn.compose.ColumnTransformer
            Preprocessor pipeline
        """
        # Transformers for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Transformers for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def _create_model(self):
        """
        Create the model based on the selected model type.
        
        Returns:
        --------
        sklearn.base.BaseEstimator
            Classifier model
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, class_weight='balanced')
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def build_pipeline(self):
        """
        Build the complete model pipeline.
        
        Returns:
        --------
        sklearn.pipeline.Pipeline
            Complete pipeline including preprocessing and model
        """
        self.preprocessor = self._create_preprocessor()
        classifier = self._create_model()
        
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target data
        
        Returns:
        --------
        self
        """
        # Build the pipeline
        pipeline = self.build_pipeline()
        
        if self.handle_imbalance:
            # Preprocess data
            X_preprocessed = self.preprocessor.fit_transform(X)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
            
            # Fit classifier on resampled data
            classifier = self._create_model()
            classifier.fit(X_resampled, y_resampled)
            
            # Create final pipeline with fitted components
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', classifier)
            ])
        else:
            # Fit the entire pipeline
            self.model = pipeline.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        
        Returns:
        --------
        np.ndarray
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probability of the positive class.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target data
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Generate classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target data
        param_grid : dict, optional
            Parameter grid for GridSearchCV. If None, a default grid is used.
        cv : int, optional
            Number of cross-validation folds
        
        Returns:
        --------
        self
        """
        # Build the pipeline
        pipeline = self.build_pipeline()
        
        # Set default parameter grid if none provided
        if param_grid is None:
            if self.model_type == 'logistic_regression':
                param_grid = {
                    'classifier__C': [0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['liblinear', 'saga']
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Set model to the best estimator
        self.model = grid_search.best_estimator_
        
        return self, grid_search.best_params_, grid_search.best_score_
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing feature names and their importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Extract model from pipeline
        classifier = self.model.named_steps['classifier']
        
        # Check if model has feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            raise ValueError("Model does not provide feature importances")
        
        # Get feature names from preprocessor
        preprocessor = self.model.named_steps['preprocessor']
        cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(self.categorical_features)
        feature_names = np.concatenate([self.numerical_features, cat_features])
        
        # Create DataFrame with feature names and importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance


def predict_subscription(client_data, model_path=None):
    """
    Predict the probability of a client subscribing to a term deposit.
    
    Parameters:
    -----------
    client_data : pd.DataFrame
        DataFrame containing client information
    model_path : str, optional
        Path to the saved model file. If None, a new model will be trained.
    
    Returns:
    --------
    float
        Probability of subscription
    """
    import joblib
    
    if model_path:
        # Load the saved model
        model = joblib.load(model_path)
    else:
        raise ValueError("Model path must be provided")
    
    # Make prediction
    prob = model.predict_proba(client_data)[:, 1]
    return prob
