# =============================================================================
# Fairness Analysis Module
# =============================================================================

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import recall_score

logger = logging.getLogger()

def fairness_analysis(data, models, feature_cols):
    """
    Performs fairness analysis on the trained models across different demographic groups.

    Parameters:
    - data (pd.DataFrame): The preprocessed dataset.
    - models (dict): Dictionary of trained machine learning models.
    - feature_cols (list): List of feature column names used in training.

    Returns:
    - fairness_metrics (dict): Dictionary containing fairness metrics for each model and demographic group.
    """
    try:
        logger.info("Starting fairness analysis.")

        # Define demographic groups (e.g., race and sex)
        demographic_groups = data[['race', 'sex']].drop_duplicates().to_dict('records')

        fairness_metrics = {}

        # Define features and target
        X = data[feature_cols]
        y = data['income_numeric']

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        for model_name, model in models.items():
            logger.info(f"Performing fairness analysis for model: {model_name}")

            if model_name == 'XGBoost':
                # Convert X to NumPy array of type float32
                X_mod = X.values.astype(np.float32)
                y_pred = model.predict(X_mod)
            else:
                y_pred = model.predict(X)

            data_copy = data.copy()
            data_copy['prediction'] = y_pred

            fairness_metrics[model_name] = {}

            for group in demographic_groups:
                race = group['race']
                sex = group['sex']
                subset = data_copy[(data_copy['race'] == race) & (data_copy['sex'] == sex)]

                if subset.empty:
                    continue

                # Calculate True Positive Rate (Recall)
                true_positive_rate = recall_score(subset['income_numeric'], subset['prediction'])

                # Calculate base rate (overall recall)
                overall_recall = recall_score(y, y_pred)
                equal_opportunity_difference = true_positive_rate - overall_recall

                fairness_metrics[model_name][f"{race} - {sex}"] = {
                    'True Positive Rate': round(true_positive_rate, 4),
                    'Equal Opportunity Difference': round(equal_opportunity_difference, 4)
                }

                logger.info(f"Fairness metrics for {race} - {sex} in {model_name}: True Positive Rate={true_positive_rate}, Equal Opportunity Difference={equal_opportunity_difference}")

        logger.info("Fairness analysis completed.")
        return fairness_metrics
    except Exception as e:
        logger.error(f"An error occurred during fairness analysis: {e}")
        raise
