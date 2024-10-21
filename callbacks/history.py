# =============================================================================
# Prediction History Callback
# =============================================================================

from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import logging

logger = logging.getLogger()

def register_history_callbacks(app):
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
