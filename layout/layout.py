# =============================================================================
# Dashboard Layout Module
# =============================================================================

import dash_bootstrap_components as dbc
from dash import html, dcc

def serve_layout():
    return dbc.Container([
        # Store for processed data, metrics, models, and fairness
        dcc.Store(id='stored-data', data={}),
    
        dcc.Store(id='prediction-history', data=[]),  # Store for Prediction History
        
        dcc.Store(id='uploaded-data-store', data=None),  # Store for Uploaded Data
    
        # Header Section with Navbar
        dbc.NavbarSimple(
            brand="Income Analysis Dashboard",
            color="primary",
            dark=True,
            fluid=True,
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="#")),
                dbc.NavItem(dbc.NavLink("About", href="#")),
                # Add more nav items if needed
            ]
        ),
    
        # Spacer
        html.Br(),
    
        # Navigation Tabs
        dbc.Tabs([
            dbc.Tab(label='Overview & Findings', tab_id='findings-tab'),
            dbc.Tab(label='Data Preprocessing', tab_id='preprocessing-tab'),
            dbc.Tab(label='Visualizations', tab_id='visualizations-tab'),
            dbc.Tab(label='Modeling', tab_id='modeling-tab'),
            dbc.Tab(label='Fairness Analysis', tab_id='fairness-tab'),
            dbc.Tab(label='Quality Assurance', tab_id='qa-tab'),
            dbc.Tab(label='Prediction', tab_id='prediction-tab'),
            dbc.Tab(label='Import Data', tab_id='import-data-tab')  # New Import Data Tab
        ], id='tabs', active_tab='findings-tab', className="mb-3"),
    
        # Content Area for Selected Tab
        html.Div(id='tabs-content', className="p-4")
    ], fluid=True)
