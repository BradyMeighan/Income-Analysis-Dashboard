# =============================================================================
# Data Preprocessing Module
# =============================================================================

import pandas as pd
import numpy as np
import logging
import time
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()

def load_default_data(file_path='adult.xlsx'):
    try:
        data = pd.read_excel(file_path)
        logger.info(f"Default data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        logger.error(f"Failed to load default data from {file_path}: {e}")
        raise

def preprocess_data(data, is_prediction=False, numerical_means=None, numerical_stds=None):
    """
    Preprocesses the census dataset by performing data cleaning, handling missing values,
    removing duplicates and outliers, encoding categorical variables, and standardizing
    numerical features.

    Parameters:
    - data (pd.DataFrame): The raw dataset.
    - is_prediction (bool): Flag indicating if the data is for prediction.
                             If True, 'income' column is not expected.
    - numerical_means (dict): Dictionary of means for numerical features from training data.
    - numerical_stds (dict): Dictionary of standard deviations for numerical features from training data.

    Returns:
    - data (pd.DataFrame): The cleaned and preprocessed dataset.
    - metrics (dict): Dictionary containing preprocessing metrics.
    - numerical_columns (list): List of numerical feature names.
    - categorical_columns (list): List of categorical feature names.
    - additional_numerical_features (list): List of additional numerical features.
    - numerical_means (dict): Updated numerical means.
    - numerical_stds (dict): Updated numerical standard deviations.
    """
    try:
        start_time = time.time()
        logger.info("Starting data preprocessing.")

        # Initialize metrics dictionary to track preprocessing steps
        metrics = {
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'missing_values_before': 0,
            'missing_values_after': 0,
            'outliers_detected': 0,
            'outliers_removed': 0
        }

        # Data Cleanup: Mapping numerical codes back to categorical labels if necessary
        mappings = {
            'workclass': dict(enumerate(['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                                         'Self-emp-inc', 'Without-pay', 'Never-worked'])),
            'education': dict(enumerate(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
                                         'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th',
                                         '1st-4th', 'Preschool', '12th'])),
            'marital-status': dict(enumerate(['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                                              'Separated', 'Married-AF-spouse', 'Widowed'])),
            'occupation': dict(enumerate(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
                                          'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing',
                                          'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv',
                                          'Armed-Forces', 'Priv-house-serv'])),
            'relationship': dict(enumerate(['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])),
            'race': dict(enumerate(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])),
            'sex': dict(enumerate(['Male', 'Female'])),
            'native-country': dict(enumerate(['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico',
                                              'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines',
                                              'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos',
                                              'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador',
                                              'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia',
                                              'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                                              'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                                              'Holand-Netherlands'])),
            'income': dict(enumerate(['<=50K', '>50K']))
        }

        def should_map(column):
            """
            Determines if a column should be mapped from numerical codes to categorical labels.

            Parameters:
            - column (str): The column name.

            Returns:
            - bool: True if the column is numerical and needs mapping, False otherwise.
            """
            return pd.api.types.is_numeric_dtype(data[column])

        # Apply mappings to relevant columns
        for col, mapping in mappings.items():
            if col in data.columns and should_map(col):
                data[col] = data[col].map(mapping)
                logger.info(f"Applied mapping to column: {col}.")

        if not is_prediction:
            # Handle duplicate rows
            initial_row_count = data.shape[0]
            data = data.drop_duplicates()
            final_row_count = data.shape[0]
            metrics['duplicates_found'] = initial_row_count - final_row_count
            metrics['duplicates_removed'] = metrics['duplicates_found']  # All duplicates are removed
            logger.info(f"Duplicates found: {metrics['duplicates_found']}. Duplicates removed: {metrics['duplicates_removed']}.")

            # Handle missing values
            metrics['missing_values_before'] = data.isnull().sum().sum()
            logger.info(f"Missing values before handling: {metrics['missing_values_before']}.")

            # Define categorical and numerical columns
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                                   'relationship', 'race', 'sex', 'native-country', 'income']
            numerical_columns = ['age', 'hours-per-week']
            additional_numerical_features = ['capital-gain', 'capital-loss']  # Exclude 'education-num'

            # Exclude 'fnlwgt' as per recommendations
            if 'fnlwgt' in data.columns:
                data = data.drop(columns=['fnlwgt'])
                logger.info("Dropped 'fnlwgt' column.")

            # Update additional numerical features after excluding 'fnlwgt'
            additional_numerical_features = [feat for feat in additional_numerical_features
                                             if feat in data.columns and pd.api.types.is_numeric_dtype(data[feat])]
            logger.info(f"Additional numerical features: {additional_numerical_features}.")

            # Preserve raw numerical data for visualization and prediction
            for col in numerical_columns + additional_numerical_features:
                data[col + '_raw'] = data[col]
            logger.info("Preserved raw numerical data for visualization and prediction.")

            # Clean the 'income' column to ensure consistent mapping
            data['income'] = data['income'].astype(str).str.strip().str.rstrip('.')
            logger.info("Cleaned 'income' column for consistent mapping.")

            # Impute missing values instead of dropping rows
            # Categorical columns: impute with mode
            for col in categorical_columns:
                if col in data.columns:
                    mode_value = data[col].mode()[0]
                    data[col].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed missing values in '{col}' with mode: {mode_value}.")

            # Numerical columns: impute with median
            for col in numerical_columns + additional_numerical_features:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    median_value = data[col].median()
                    data[col].fillna(median_value, inplace=True)
                    logger.info(f"Imputed missing values in '{col}' with median: {median_value}.")

            metrics['missing_values_after'] = data.isnull().sum().sum()
            logger.info(f"Missing values after handling: {metrics['missing_values_after']}.")

            # Detect and handle outliers using Z-score method for numerical columns
            z_scores = np.abs((data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std())
            outliers = (z_scores > 3).any(axis=1)
            metrics['outliers_detected'] = outliers.sum()
            data = data[~outliers]
            metrics['outliers_removed'] = metrics['outliers_detected']  # Assuming removal of all detected outliers
            logger.info(f"Outliers detected: {metrics['outliers_detected']}. Outliers removed: {metrics['outliers_removed']}.")

            # Standardizing numerical columns
            numerical_means = {}
            numerical_stds = {}
            for col in numerical_columns + additional_numerical_features:
                mean = data[col].mean()
                std = data[col].std()
                numerical_means[col] = mean
                numerical_stds[col] = std
                data[col] = (data[col] - mean) / std
                logger.info(f"Standardized numerical column: {col}.")

            # Create 'income_numeric' for correlation analysis
            data['income_numeric'] = data['income'].map({'<=50K': 0, '>50K': 1})
            logger.info("Created 'income_numeric' column for correlation analysis.")

            # Verify if mapping was successful
            if data['income_numeric'].isnull().any():
                num_missing = data['income_numeric'].isnull().sum()
                logger.warning(f"{num_missing} entries in 'income_numeric' could not be mapped and are set to NaN.")
                # Drop rows with unmapped 'income_numeric' values
                data = data.dropna(subset=['income_numeric'])
                metrics['missing_values_after'] = data.isnull().sum().sum()
                logger.info(f"After dropping unmapped 'income_numeric' rows, missing values: {metrics['missing_values_after']}.")

            # Ensure 'income_numeric' is integer type
            data['income_numeric'] = data['income_numeric'].astype(int)
            logger.info("'income_numeric' column converted to integer type.")

        else:
            # Prediction preprocessing: 'income' column is not expected
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                                   'relationship', 'race', 'sex', 'native-country']
            numerical_columns = ['age', 'hours-per-week']
            additional_numerical_features = ['capital-gain', 'capital-loss']  # Exclude 'education-num'

            # Exclude 'fnlwgt' as per recommendations
            if 'fnlwgt' in data.columns:
                data = data.drop(columns=['fnlwgt'])
                logger.info("Dropped 'fnlwgt' column.")

            # Update additional numerical features after excluding 'fnlwgt'
            additional_numerical_features = [feat for feat in additional_numerical_features
                                             if feat in data.columns and pd.api.types.is_numeric_dtype(data[feat])]
            logger.info(f"Additional numerical features: {additional_numerical_features}.")

            # Preserve raw numerical data for visualization and prediction
            for col in numerical_columns + additional_numerical_features:
                data[col + '_raw'] = data[col]
            logger.info("Preserved raw numerical data for visualization and prediction.")

            # Impute missing values instead of dropping rows
            # Categorical columns: impute with mode
            for col in categorical_columns:
                if col in data.columns:
                    mode_value = data[col].mode()[0]
                    data[col].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed missing values in '{col}' with mode: {mode_value}.")

            # Numerical columns: impute with median
            for col in numerical_columns + additional_numerical_features:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    median_value = data[col].median()
                    data[col].fillna(median_value, inplace=True)
                    logger.info(f"Imputed missing values in '{col}' with median: {median_value}.")

            metrics['missing_values_before'] = data.isnull().sum().sum()
            metrics['missing_values_after'] = data.isnull().sum().sum()
            logger.info(f"Missing values after handling: {metrics['missing_values_after']}.")

            # Detect and handle outliers using Z-score method for numerical columns
            z_scores = np.abs((data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std())
            outliers = (z_scores > 3).any(axis=1)
            metrics['outliers_detected'] = outliers.sum()
            data = data[~outliers]
            metrics['outliers_removed'] = metrics['outliers_detected']  # Assuming removal of all detected outliers
            logger.info(f"Outliers detected: {metrics['outliers_detected']}. Outliers removed: {metrics['outliers_removed']}.")

            # Standardizing numerical columns using training data means and stds
            if numerical_means is None or numerical_stds is None:
                error_message = "numerical_means and numerical_stds must be provided for prediction."
                logger.error(error_message)
                raise ValueError(error_message)
            
            for col in numerical_columns + additional_numerical_features:
                mean = numerical_means.get(col, 0)
                std = numerical_stds.get(col, 1)
                data[col] = (data[col] - mean) / std
                logger.info(f"Standardized numerical column: {col}.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds.")

        # Return all necessary variables
        return data, metrics, numerical_columns, categorical_columns, additional_numerical_features, numerical_means, numerical_stds
    except Exception as e:
        logger.error(f"An error occurred during pre-processing: {e}")
        raise