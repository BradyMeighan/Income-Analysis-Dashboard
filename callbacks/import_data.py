# =============================================================================
# Import Data Callback
# =============================================================================

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import base64
import io
from data.preprocessing import preprocess_data
import json
import logging
from models.training import train_models

logger = logging.getLogger()

def register_import_data_callbacks(app):
    @app.callback(
        [Output('import-prediction-output', 'children'),
         Output('prediction-results', 'children'),
         Output('import-visual-content', 'children'),
         Output('uploaded-data-store', 'data')],
        [Input('run-predictions', 'n_clicks'),
         Input('import-visual-tabs', 'active_tab')],
        [State('upload-data', 'contents'),
         State('upload-data', 'filename'),
         State('import-model-dropdown', 'value'),
         State('stored-data', 'data'),
         State('uploaded-data-store', 'data')],
        prevent_initial_call=True
    )
    def handle_import_interaction(n_clicks, active_tab, contents, filename, model_name, store_data, uploaded_data_store):
        ctx = dash.callback_context
    
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
    
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
        if trigger_id == 'run-predictions':
            if contents is None:
                return dbc.Alert("No file uploaded yet.", color="warning"), no_update, no_update, no_update
    
            # Decode the uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            # Read the CSV file
            try:
                uploaded_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                logger.info(f"Uploaded file {filename} successfully.")
            except Exception as e:
                error_message = f"Failed to parse CSV file: {e}"
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Check if required columns are present
            required_columns = ['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'native-country', 'age',
                                'capital-gain', 'capital-loss', 'hours-per-week']
            missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
            if missing_columns:
                error_message = f"The following required columns are missing in the uploaded file: {', '.join(missing_columns)}."
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Retrieve numerical_means and numerical_stds from store_data
            numerical_means = store_data.get('numerical_means', {})
            numerical_stds = store_data.get('numerical_stds', {})
    
            if not numerical_means or not numerical_stds:
                error_message = "Numerical means and standard deviations are missing from stored data."
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Preprocess the uploaded data for prediction
            try:
                preprocessed_uploaded_data, upload_metrics, num_cols, cat_cols, add_num_feats, _, _ = preprocess_data(
                    uploaded_df, is_prediction=True, numerical_means=numerical_means, numerical_stds=numerical_stds
                )
            except Exception as e:
                error_message = f"An error occurred during preprocessing: {str(e)}"
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Prepare features for prediction
            X_input = preprocessed_uploaded_data.drop(columns=['income_numeric'], errors='ignore')
            X_input = pd.get_dummies(X_input, drop_first=True)
            logger.info("Encoded categorical variables for input data.")
    
            # Ensure the input data has the same columns as the training data
            X_train = pd.read_json(store_data['data'], orient='split')[feature_cols]
            X_train = pd.get_dummies(X_train, drop_first=True)
            X_input = X_input.reindex(columns=X_train.columns, fill_value=0)
            logger.info("Reindexed input data to match training dummy variables.")
    
            # Retrieve the selected model
            trained_models = {model: eval(model) for model in store_data.get('trained_models', {})}
            model = trained_models.get(model_name, None)
            if model is None:
                error_message = "Selected model is not available."
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Make predictions
            try:
                if model_name == 'XGBoost':
                    X_input_mod = X_input.values.astype(np.float32)
                    predictions = model.predict(X_input_mod)
                else:
                    predictions = model.predict(X_input)
            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"
                logger.error(error_message)
                return dbc.Alert(error_message, color="danger"), no_update, no_update, no_update
    
            # Map predictions to income categories
            income_map = {0: '<=50K', 1: '>50K'}
            predicted_incomes = [income_map.get(int(pred), "Unknown") for pred in predictions]
    
            # Add the predictions to the preprocessed_uploaded_data
            preprocessed_uploaded_data['Predicted Income'] = predicted_incomes
    
            # Update the uploaded data store
            uploaded_data_store = preprocessed_uploaded_data.to_json(date_format='iso', orient='split')
    
            # Generate visual content for the default tab
            visual_content = generate_import_visualization('pred-dist-tab', preprocessed_uploaded_data)
    
            # Prepare prediction results with full data and row numbers
            # Reset index to include row numbers
            preprocessed_uploaded_data.reset_index(inplace=True)
            preprocessed_uploaded_data.rename(columns={'index': 'Row Number'}, inplace=True)
    
            # Create the table with all columns
            prediction_results = dbc.Table.from_dataframe(
                preprocessed_uploaded_data.head(10),  # Display the first 10 rows or remove .head(10) to display all
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
    
            # Prepare prediction output message
            prediction_output = dbc.Alert(
                f"Predictions completed successfully using the {model_name} model.",
                color="success",
                style={'font-size': '20px'}
            )
    
            return prediction_output, prediction_results, visual_content, uploaded_data_store
    
    
        elif trigger_id == 'import-visual-tabs':
            if uploaded_data_store is None:
                return no_update, no_update, dbc.Alert("No predictions to visualize. Please upload a file and run predictions.", color="warning"), no_update
    
            preprocessed_uploaded_data = pd.read_json(uploaded_data_store, orient='split')
            visual_content = generate_import_visualization(active_tab, preprocessed_uploaded_data)
    
            return no_update, no_update, visual_content, no_update
    
        else:
            raise dash.exceptions.PreventUpdate
    
    # -----------------------------------------------------------------------------
    # Function to Generate Import Visualizations
    # -----------------------------------------------------------------------------
    def generate_import_visualization(active_tab, preprocessed_uploaded_data):
        if active_tab == 'pred-dist-tab':
            # Histogram of Predicted Income
            fig = px.histogram(
                preprocessed_uploaded_data, x='Predicted Income',
                color='Predicted Income',
                title='Distribution of Predicted Income',
                color_discrete_map={'<=50K': '#636EFA', '>50K': '#EF553B'}
            )
        elif active_tab == 'feature-pred-tab':
            # Box plots of numerical features by Predicted Income
            numerical_features = [col for col in preprocessed_uploaded_data.columns if col.endswith('_raw')]
            fig = make_subplots(rows=1, cols=len(numerical_features),
                                subplot_titles=[col.replace('_raw', '').replace('-', ' ').title() for col in numerical_features])
            for i, feature in enumerate(numerical_features):
                box = px.box(
                    preprocessed_uploaded_data, x='Predicted Income', y=feature,
                    color='Predicted Income',
                    color_discrete_map={'<=50K': '#636EFA', '>50K': '#EF553B'}
                )
                for trace in box.data:
                    fig.add_trace(trace, row=1, col=i+1)
            fig.update_layout(title='Feature Distribution by Predicted Income', showlegend=False)
        else:
            fig = go.Figure()
        return dcc.Graph(figure=fig)
        
    # -----------------------------------------------------------------------------
    # Callback to Update Prediction History Table
    # -----------------------------------------------------------------------------
    @app.callback(
        Output('prediction-history-table', 'children'),
        Input('prediction-history', 'data')
    )
    def update_history_table(history_data):
        if not history_data:
            return html.P("No predictions made yet.")
    
        # Convert history data to DataFrame
        history_df = pd.DataFrame(history_data)
    
        # Reorder columns to place 'Predicted Income' and 'Model' at the end
        cols = [col for col in history_df.columns if col not in ['Predicted Income', 'Model']]
        cols.extend(['Model', 'Predicted Income'])
        history_df = history_df[cols]
    
        # Display only the last 5 predictions
        history_df = history_df.tail(5)
    
        # Create Table
        table = dbc.Table.from_dataframe(
            history_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True
        )
    
        return table
