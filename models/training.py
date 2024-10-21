# =============================================================================
# Machine Learning Model Training Module
# =============================================================================

import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

logger = logging.getLogger()

def train_models(data, numerical_columns, additional_numerical_features):
    """
    Trains multiple machine learning models with hyperparameter tuning and evaluates their performance.

    Parameters:
    - data (pd.DataFrame): The preprocessed dataset.
    - numerical_columns (list): List of numerical feature names.
    - additional_numerical_features (list): List of additional numerical features.

    Returns:
    - model_performance (dict): Dictionary containing performance metrics for each model.
    - trained_models (dict): Dictionary of trained model objects.
    - feature_cols (list): List of feature column names used in training.
    """
    try:
        logger.info("Starting model training with hyperparameter tuning.")

        # Suppress ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', message='Solver terminated early.*')  # For early stopping warnings

        # Define features and target
        feature_cols = numerical_columns + additional_numerical_features + [
            col for col in data.columns if col not in
            ['income', 'income_numeric'] + numerical_columns + additional_numerical_features + [col + '_raw' for col in numerical_columns + additional_numerical_features]
        ]

        X = data[feature_cols]
        y = data['income_numeric']

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        logger.info("Encoded categorical variables.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("Split data into training and testing sets.")

        # Initialize models and hyperparameter grids
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        }

        param_grids = {
            'Logistic Regression': [
                {
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'C': [0.1, 1, 10],
                    'max_iter': [1000]
                },
                {
                    'penalty': ['elasticnet'],
                    'solver': ['saga'],
                    'l1_ratio': [0.5],
                    'C': [0.1, 1, 10],
                    'max_iter': [1000]
                }
            ],
            'Decision Tree': {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            },
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            }
        }

        model_performance = {}
        trained_models = {}

        for model_name, model in models.items():
            logger.info(f"Training model with hyperparameter tuning: {model_name}")

            param_grid = param_grids[model_name]

            if model_name == 'Logistic Regression':
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=5,  # Adjust as needed
                    scoring='accuracy',
                    cv=3,
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(X_train, y_train)
            elif model_name == 'XGBoost':
                # Convert data to NumPy arrays with appropriate data types
                X_train_mod = X_train.values.astype(np.float32)
                y_train_mod = y_train.values.astype(np.int32)
                X_test_mod = X_test.values.astype(np.float32)

                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=3,
                    n_jobs=-1
                )
                search.fit(X_train_mod, y_train_mod)
            else:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=3,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)

            best_model = search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {search.best_params_}")

            if model_name == 'XGBoost':
                y_pred = best_model.predict(X_test_mod)
            else:
                y_pred = best_model.predict(X_test)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            model_performance[model_name] = {
                'Accuracy': round(accuracy, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            }

            trained_models[model_name] = best_model

            logger.info(f"Model {model_name} performance after tuning: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")

        logger.info("Model training and evaluation with hyperparameter tuning completed.")
        return model_performance, trained_models, feature_cols
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise
