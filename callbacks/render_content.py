# =============================================================================
# Render Content Callback
# =============================================================================

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fairness.analysis import fairness_analysis
import json
from data.preprocessing import preprocess_data
from models.training import train_models
import logging

logger = logging.getLogger()

def register_render_content_callbacks(app, trained_models):
    @app.callback(
        Output('tabs-content', 'children'),
        Input('tabs', 'active_tab'),
        State('stored-data', 'data')
    )
    def render_content(tab, store_data):
        # Retrieve necessary variables from store_data
        numerical_columns = store_data.get('numerical_columns', [])
        additional_numerical_features = store_data.get('additional_numerical_features', [])
        categorical_columns = store_data.get('categorical_columns', [])
        numerical_means = store_data.get('numerical_means', {})
        numerical_stds = store_data.get('numerical_stds', {})
        feature_cols = store_data.get('feature_cols', [])
        trained_models_names = store_data.get('trained_models_names', {})
        model_performance_json = store_data.get('model_performance_json', '{}')
        fairness_metrics_json = store_data.get('fairness_metrics_json', '{}')
        data_json = store_data.get('data', None)
        metrics = store_data.get('metrics', {})

        if tab == 'findings-tab':
            # Sprint 1 Content
            sprint1_goal = """
        ### **Sprint 1 Goal**

        Our primary objective for **Sprint 1** was to lay the foundation for the Income Prediction System by focusing on data preprocessing and exploratory data analysis (EDA). We aimed to ensure data integrity, handle missing values, and perform initial visualizations to understand the dataset's structure and key characteristics.
        """

            sprint1_backlog = [
                "- **Data Collection and Loading:** Successfully loaded the `adult.xlsx` dataset for analysis.",
                "- **Data Cleaning:** Handled missing values, duplicates, and outliers to prepare the data for modeling.",
                "- **Exploratory Data Analysis:** Performed initial visualizations and statistical summaries to understand data distributions.",
                "- **Documentation and QA:** Established initial documentation and quality assurance processes.",
            ]

            # Sprint 2 Content
            sprint2_goal = """
        ### **Sprint 2 Goal**

        In **Sprint 2**, our goal was to develop and evaluate machine learning models for our Income Prediction System. We aimed to create a baseline logistic regression model, compare it with other model types, and perform initial fairness checks across demographic groups. This set the stage for fine-tuning and user interface development in the next sprint.
        """

            sprint2_backlog = [
                "- **Baseline Model Development:** Trained a logistic regression model as a baseline for income prediction.",
                "- **Model Comparison:** Implemented and compared multiple models (logistic regression, decision tree, random forest, XGBoost) to find the best-performing model.",
                "- **Fairness Analysis:** Conducted initial fairness checks across different demographic groups to identify and address potential biases in the model predictions.",
            ]

            # Sprint 3 Content
            sprint3_goal = """
        ### **Sprint 3 Goal**

        The focus of **Sprint 3** was to enhance the user interface and experience of the Income Prediction System. We aimed to integrate the models into a user-friendly dashboard, allowing users to input data, view predictions, and analyze results through interactive visualizations.
        """

            sprint3_backlog = [
                "- **Dashboard Development:** Built an interactive dashboard using Dash and Plotly.",
                "- **Model Integration:** Integrated the trained models into the dashboard for real-time predictions.",
                "- **User Input Forms:** Developed forms for users to input data and receive income predictions.",
                "- **Interactive Visualizations:** Added charts and graphs to visualize data distributions and model results.",
                "- **Export Functionality:** Enabled users to export prediction results for further analysis.",
            ]

            # Overall Findings and Conclusions
            overall_findings = """
        ## **Overall Findings and Conclusions**

        Throughout **Sprints 1 to 3**, the Income Prediction System evolved from data preprocessing to a fully functional interactive dashboard. Key accomplishments and insights include:

        - **Data Preprocessing:** Ensured data quality by handling missing values, duplicates, and outliers. This foundational work was critical for building reliable models.
        - **Model Performance:** The XGBoost model consistently outperformed others in terms of accuracy and F1-score, indicating its effectiveness for this classification task.
        - **Fairness Considerations:** Fairness analysis revealed that while the XGBoost model performed best overall, the Random Forest model demonstrated more consistent performance across different demographic groups. This highlights the trade-off between model accuracy and fairness.
        - **User Interface Development:** The dashboard provides an accessible platform for users to interact with the models, input data, and visualize results, enhancing the system's usability.
        - **Challenges Overcome:** Addressed challenges related to data imbalance, model bias, and integrating complex models into a web application.
        """

            # Retrospective and Next Steps
            retrospective = """
        ## **Retrospective and Next Steps**

        - **Strengths:**
          - Effective collaboration and communication among team members.
          - Successful implementation of machine learning models with robust performance.
          - Development of an intuitive user interface that meets user needs.

        - **Areas for Improvement:**
          - Enhance model fairness without significantly compromising accuracy.
          - Implement more advanced techniques for bias mitigation.
          - Conduct user testing to gather feedback for further UI/UX improvements.

        - **Future Work:**
          - Explore additional models and techniques to improve both performance and fairness.
          - Incorporate more features or external data sources to enrich the model.
          - Implement continuous integration and deployment for streamlined updates and maintenance.
        """

            # Assemble the content into the layout
            return dbc.Container([
                # Sprint 1 Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Sprint 1: Data Preprocessing & EDA")),
                            dbc.CardBody([
                                dcc.Markdown(sprint1_goal, style={'font-size': '18px'}),
                                dcc.Markdown('\n'.join(sprint1_backlog), style={'font-size': '18px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),

                # Sprint 2 Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Sprint 2: Model Development & Fairness Analysis")),
                            dbc.CardBody([
                                dcc.Markdown(sprint2_goal, style={'font-size': '18px'}),
                                dcc.Markdown('\n'.join(sprint2_backlog), style={'font-size': '18px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),

                # Sprint 3 Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Sprint 3: Dashboard Implementation & User Experience")),
                            dbc.CardBody([
                                dcc.Markdown(sprint3_goal, style={'font-size': '18px'}),
                                dcc.Markdown('\n'.join(sprint3_backlog), style={'font-size': '18px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),

                # Overall Findings Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Overall Findings and Conclusions")),
                            dbc.CardBody([
                                dcc.Markdown(overall_findings, style={'font-size': '18px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),

                # Retrospective Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Retrospective and Next Steps")),
                            dbc.CardBody([
                                dcc.Markdown(retrospective, style={'font-size': '18px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),

                # Retrospective Image (Optional)
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(src="/assets/retrospective.png", style={'width': '100%', 'margin-top': '20px'}),
                            ])
                        ], className="mb-4")
                    ], width=12)
                ])
            ], fluid=True)

        elif tab == 'preprocessing-tab':
            # Retrieve preprocessing metrics
            metrics = store_data.get('metrics', {})
            pre_processing_metrics = [
                {"label": "Duplicate Rows Found", "value": metrics.get('duplicates_found', 0)},
                {"label": "Duplicate Rows Removed", "value": metrics.get('duplicates_removed', 0)},
                {"label": "Missing Values Before Handling", "value": metrics.get('missing_values_before', 0)},
                {"label": "Missing Values After Handling", "value": metrics.get('missing_values_after', 0)},
                {"label": "Outliers Detected", "value": metrics.get('outliers_detected', 0)},
                {"label": "Outliers Removed", "value": metrics.get('outliers_removed', 0)},
            ]
    
            # Methodology Explanation Markdown
            methodology_explanation = """
    ### **Data Preprocessing Methodology**
    
    The following steps were undertaken to preprocess and clean the dataset:
    
    1. **Imported Raw Data:**
       - Loaded the adult.xlsx dataset into Python using pandas for further analysis.
    
    2. **Data Cleanup:**
       - **Encoding Categorical Variables:** Converted numerical codes back to meaningful categorical labels for features such as workclass, education, marital-status, occupation, relationship, race, sex, native-country, and income using predefined mapping dictionaries.
    
    3. **Handling Duplicates:**
       - **Detection:** Identified duplicate rows in the dataset.
       - **Removal:** Removed all duplicate entries to ensure data integrity.
    
    4. **Handling Missing Values:**
       - **Detection:** Counted the total number of missing values before handling.
       - **Handling:** 
         - **Categorical Features:** Imputed missing values with the mode (most frequent value).
         - **Numerical Features:** Imputed missing values with the median to preserve data distribution.
    
    5. **Outlier Detection and Handling:**
       - **Detection:** Utilized the Z-score method to identify outliers in numerical columns (age and hours-per-week). Specifically, data points with a Z-score greater than 3 in any numerical feature were considered outliers.
       - **Handling:** Removed all detected outliers to prevent skewing the analysis and ensure a more accurate representation of the data distribution.
    
    6. **Normalization:**
       - **Standardizing Numerical Features:** Applied Z-score normalization to numerical columns (age and hours-per-week) to standardize the data. This process ensures that each feature has a mean of 0 and a standard deviation of 1, facilitating comparison and further analysis.
    
    7. **Additional Feature Standardization:**
       - **Standardizing Additional Numerical Features:** Standardized other numerical features (capital-gain, capital-loss) to prepare them for correlation analysis.
    
    8. **Creation of income_numeric:**
       - **Encoding Income:** Created a numeric encoding of the income column to facilitate correlation analysis, where <=50K is mapped to 0 and >50K is mapped to 1.
    
    These preprocessing steps collectively ensured that the dataset is clean, free from duplicates and outliers, imputed for missing values, and standardized for accurate and meaningful analysis.
            """
    
            return dbc.Container([
                # Methodology Explanation
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Data Preprocessing Methodology")),
                            dbc.CardBody([
                                dcc.Markdown(methodology_explanation, style={'font-size': '18px'})
                            ])
                        ], className="mb-4")
                    ], width=10)
                ], justify='center'),
    
                # Preprocessing Metrics Table
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Preprocessing Summary")),
                            dbc.CardBody([
                                dbc.Table(
                                    [
                                        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
                                    ] + [
                                        html.Tr([html.Td(metric['label']), html.Td(metric['value'])]) 
                                        for metric in pre_processing_metrics
                                    ],
                                    bordered=True,
                                    hover=True,
                                    responsive=True,
                                    striped=True
                                )
                            ])
                        ], className="mb-4")
                    ], width=8)
                ], justify='center')
            ], fluid=True)
    
        elif tab == 'visualizations-tab':
            raw_data_json = store_data.get('raw_data', None)
            if raw_data_json is None:
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Raw data not found. Please ensure the data is loaded.", color="danger")
                        ], width=12)
                    ])
                ])
            raw_data = pd.read_json(raw_data_json, orient='split')

            data_json = store_data.get('data', None)
            if data_json is None:
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Data not found. Please ensure preprocessing is completed.", color="danger")
                        ], width=12)
                    ])
                ])
            
            # Load data from JSON
            data = pd.read_json(data_json, orient='split')
    
            # Define numerical and categorical columns
            numerical_columns_vis = ['age', 'hours-per-week']
            categorical_columns_vis = ['workclass', 'education', 'marital-status', 'occupation', 
                                       'relationship', 'race', 'sex', 'native-country', 'income']
            additional_numerical_features_vis = [col for col in ['capital-gain', 'capital-loss'] 
                                             if col in data.columns]
            
            # Define categorical options excluding 'income'
            categorical_options = [col for col in categorical_columns_vis if col != 'income']
    
            # -----------------------------------------------------------------------------
            # Numerical Distributions (Raw Data)
            # -----------------------------------------------------------------------------
            fig_age = px.histogram(
                data_frame=data, x='age_raw',
                nbins=30, 
                title='Age Distribution',
                labels={'age_raw': 'Age'},
                opacity=0.7,
                color_discrete_sequence=['#636EFA']
            )
            fig_hours = px.histogram(
                data_frame=data, x='hours-per-week_raw',
                nbins=30, 
                title='Hours Per Week Distribution',
                labels={'hours-per-week_raw': 'Hours Per Week'},
                opacity=0.7,
                color_discrete_sequence=['#EF553B']
            )
    
            # -----------------------------------------------------------------------------
            # Correlation Matrix
            # -----------------------------------------------------------------------------
            numeric_columns_for_corr = additional_numerical_features_vis + numerical_columns_vis + ['income_numeric']
            numeric_columns_for_corr = [col for col in numeric_columns_for_corr if not col.endswith('_raw')]
            corr_matrix = data[numeric_columns_for_corr].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title='Correlation Matrix'
            )
            fig_corr.update_layout(
                title={'x':0.5},
                coloraxis_colorbar=dict(title="Correlation Coefficient")
            )
            
            # -----------------------------------------------------------------------------
            # Bivariate Analysis: Box Plots using Raw Data
            # -----------------------------------------------------------------------------
            fig_box_age = px.box(
                data_frame=data, x='income', y='age_raw',
                title='Age by Income Level',
                labels={'income': 'Income', 'age_raw': 'Age'},
                color='income',
                color_discrete_map={'<=50K':'#636EFA', '>50K':'#EF553B'}
            )
            fig_box_hours = px.box(
                data_frame=data, x='income', y='hours-per-week_raw',
                title='Hours Per Week by Income Level',
                labels={'income': 'Income', 'hours-per-week_raw': 'Hours Per Week'},
                color='income',
                color_discrete_map={'<=50K':'#636EFA', '>50K':'#EF553B'}
            )
    
            return dbc.Container([
                # Numerical Distributions Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Age Distribution", className="card-title"),
                                dcc.Graph(figure=fig_age)
                            ])
                        ], className="mb-4")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Hours Per Week Distribution", className="card-title"),
                                dcc.Graph(figure=fig_hours)
                            ])
                        ], className="mb-4")
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown("""
    **Interpretation:**
    
    The histograms above display the distributions of **Age** and **Hours Per Week**. These features reflect the actual values in the dataset, providing insights into the demographic and work patterns of the individuals.
                        """, style={'font-size': '18px'})
                    ], width=12)
                ], className="mb-4"),
    
                # Categorical Distributions Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Categorical Distributions", className="card-title"),
                                html.Label("Select Categorical Variable:", style={'font-size': '18px'}),
                                dcc.Dropdown(
                                    id='categorical-dropdown',
                                    options=[{'label': col.replace('-', ' ').title(), 'value': col} for col in categorical_options],
                                    value='education',
                                    clearable=False
                                )
                            ])
                        ], className="mb-4")
                    ], width=4)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='categorical-graph')
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown("""
    ### **Understanding the Categorical Distributions**
    
    Use the dropdown above to select a categorical variable. The resulting bar chart illustrates the distribution of income levels across the selected category.
                            """, style={'font-size': '18px'})
                    ], width=12)
                ], className="mb-4"),
    
    
                # Correlation Matrix Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Correlation Matrix", className="card-title"),
                                dcc.Graph(figure=fig_corr),
                                html.Div([
                                    dcc.Markdown("""
    ### **Understanding the Correlation Matrix**
    
    The **Correlation Matrix** visualizes the pairwise correlation coefficients between numerical features in the dataset. Correlation coefficients range from -1 to 1, indicating the strength and direction of the linear relationship between two variables.
    
    - **Positive Correlation:** A value close to 1 implies a strong positive relationship; as one feature increases, the other tends to increase.
    - **Negative Correlation:** A value close to -1 implies a strong negative relationship; as one feature increases, the other tends to decrease.
    - **No Correlation:** A value around 0 indicates no linear relationship.
    
    **Key Insights:**
    - Features with high absolute correlation values are potential predictors for income levels.
    - Understanding these relationships helps in feature selection and building predictive models.
                                    """)
                                ], style={'padding': '20px'})
                            ])
                        ], className="mb-4")
                    ], md=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown("""
    **Interpretation:**
    
    The correlation matrix showcases the relationships between numerical features. Strong positive or negative correlations can indicate potential predictors for income levels.
                            """, style={'font-size': '18px'})
                    ], width=12)
                ], className="mb-4"),
    
                # Bivariate Analysis Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Age by Income Level", className="card-title"),
                                dcc.Graph(figure=fig_box_age)
                            ])
                        ], className="mb-4")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Hours Per Week by Income Level", className="card-title"),
                                dcc.Graph(figure=fig_box_hours)
                            ])
                        ], className="mb-4")
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown("""
    **Interpretation:**
    
    Box plots compare the distributions of **Age** and **Hours Per Week** across different income levels. These visualizations help identify differences and patterns between the two income groups.
                            """, style={'font-size': '18px'})
                    ], width=12)
                ], className="mb-4")
            ])
    
        elif tab == 'modeling-tab':
            # Retrieve model performance
            model_performance_json = store_data.get('model_performance_json', '{}')
            model_performance = json.loads(model_performance_json)
    
            # Convert to DataFrame for visualization
            df_performance = pd.DataFrame(model_performance).T.reset_index().rename(columns={'index': 'Model'})
            df_melt = df_performance.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
            # Bar chart for model performance
            fig_model_perf = px.bar(
                df_melt, x='Model', y='Value', color='Metric',
                barmode='group',
                title='Model Performance Comparison',
                labels={'Value': 'Score'},
                height=600
            )
    
            # Explanation of Metrics
            metrics_explanation = """
    ### **Understanding Performance Metrics**
    
    - **Accuracy:** Measures the proportion of correct predictions out of all predictions made. It indicates how often the model is correct overall.
      
    - **Precision:** Indicates the proportion of positive identifications that were actually correct. High precision means that when the model predicts a positive class, it's usually right.
      
    - **Recall (True Positive Rate):** Measures the proportion of actual positives that were correctly identified by the model. High recall means that the model captures most of the positive cases.
      
    - **F1-Score:** The harmonic mean of Precision and Recall. It provides a balance between the two, especially useful when you need to take both metrics into account.
            """
    
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Modeling & Evaluation", className="card-title"),
                                dcc.Markdown("""
    ### **Machine Learning Models**
    
    In Sprint 2, we developed and evaluated multiple machine learning models to predict income levels. The models trained include:
    
    - **Logistic Regression:** A baseline model for binary classification.
    - **Decision Tree Classifier:** Captures non-linear relationships.
    - **Random Forest Classifier:** An ensemble method that improves prediction accuracy and controls overfitting.
    - **XGBoost Classifier:** An advanced ensemble method that often provides superior performance through gradient boosting.
    
    Below is a comparison of their performance metrics.
                                """, style={'font-size': '18px'}),
                                
                                dcc.Graph(figure=fig_model_perf),
                                
                                dcc.Markdown(metrics_explanation, style={'font-size': '18px'}),
                                
                                dcc.Markdown("""
    **Interpretation:**
    
    The bar chart above compares the performance of different models based on key metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**. This comparison aids in selecting the most effective model for income prediction.
                            """, style={'font-size': '18px'})
                            ])
                        ], className="mb-4")
                    ], width=12)
                ])
            ], fluid=True)
    
        elif tab == 'fairness-tab':
            # Retrieve fairness metrics
            fairness_metrics_json = store_data.get('fairness_metrics_json', '{}')
            fairness_metrics = json.loads(fairness_metrics_json)
    
            # Convert to DataFrame for visualization
            df_fairness = pd.DataFrame()
            for model_name, groups in fairness_metrics.items():
                temp_df = pd.DataFrame(groups).T.reset_index().rename(columns={'index': 'Group'})
                temp_df['Model'] = model_name
                df_fairness = pd.concat([df_fairness, temp_df], ignore_index=True)
            df_fairness_melt = df_fairness.melt(id_vars=['Model', 'Group'], var_name='Metric', value_name='Value')
    
            # Bar chart for fairness metrics
            fig_fairness = px.bar(
                df_fairness_melt, x='Group', y='Value', color='Metric',
                facet_col='Model',
                barmode='group',
                title='Fairness Analysis Across Demographic Groups',
                labels={'Value': 'Metric Value'},
                height=800
            )
    
            # Simplified Explanation of Fairness Analysis
            fairness_explanation = """
    ### **Understanding Fairness Analysis**
    
    **What is Fairness Analysis?**
    
    Fairness analysis checks if our models treat different groups of people equally. For example, we want to ensure that the model doesn't favor one race or gender over another when predicting income levels.
    
    **Why is it Important?**
    
    Ensuring fairness is crucial to avoid biased decisions that could negatively impact certain groups. It helps in building trustworthy and ethical AI systems.
    
    **How Does it Affect the Models?**
    
    If a model is unfair, it might perform well overall but perform poorly for specific groups. Identifying and addressing these biases ensures that the model is reliable and equitable for everyone.
            """
    
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Fairness Analysis", className="card-title"),
                                dcc.Graph(figure=fig_fairness),
                                dcc.Markdown("""
    **Interpretation:**
    
    The bar chart illustrates the **True Positive Rate** and **Equal Opportunity Difference** for each demographic group across different models. 
    
    - **True Positive Rate (Recall):** Shows how well the model identifies individuals who earn more than 50K within each group.
      
    - **Equal Opportunity Difference:** Indicates the difference in True Positive Rates between each group and the overall model performance. A smaller difference means the model is more fair.
        
    **Observations:**
    - **Logistic Regression**: Has the most variation in both metrics across groups.
    - **Decision Tree**: More consistent True Positive Rate, but some variation in Equal Opportunity Difference.
    - **Random Forest**: Very consistent True Positive Rate, minimal variation in Equal Opportunity Difference.
    - **XGBoost**: Some variation in both metrics, but less extreme than Logistic Regression.
    
    The **Random Forest** model appears to be the best choice because:
    - It has the most consistent True Positive Rate across all groups. 
    - Its Equal Opportunity Difference is very close to zero for most groups.
    
    This suggests the Random Forest model is the most fair and consistent across different demographic groups, making it likely the best option for reducing bias in decision-making.
                        """, style={'font-size': '18px'})
                            ])
                        ], className="mb-4")
                    ], width=12)
                ])
            ], fluid=True)
    
        elif tab == 'qa-tab':
            # Retrieve preprocessed data
            data_json = store_data.get('data', None)
            if data_json is None:
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Data not found. Please ensure preprocessing is completed.", color="danger")
                        ], width=12)
                    ])
                ])
            
            # Load data from JSON
            data = pd.read_json(data_json, orient='split')
    
            # Quality Assurance Section
            qa_content = [
                "### **Quality Assurance**",
                "Ensuring the accuracy and reliability of our Python script is paramount. Below are the comprehensive QA measures implemented during Sprint 1:",
                "",
                "#### **1. Data Validation**",
                "- **Step-by-Step Verification:** After each preprocessing step (e.g., handling missing values, removing duplicates), data integrity was validated by:",
                "  - Checking the number of missing values.",
                "  - Ensuring no unexpected data types or values exist.",
                "- **Automated Checks:** Incorporated assertions within the script to automatically validate data at each stage.",
                "",
                "#### **2. Code Reviews**",
                "- **Peer Review Process:** Conducted thorough code reviews:",
                "  - Identify and rectify potential bugs.",
                "  - Ensure adherence to coding standards and best practices.",
                "- **Outcome:** Enhanced code quality and consistency across the project.",
                "",
                "#### **3. Error Handling**",
                "- **Robust Mechanisms:** Implemented try-except blocks to gracefully handle unexpected data and runtime errors.",
                "- **User Feedback:** Provided informative error messages to aid in troubleshooting without exposing sensitive information.",
                "",
                "#### **4. Logging**",
                "- **Integration:** Employed Python's logging module to track the script's execution flow and record anomalies.",
                "- **Log Details:** Captured information such as:",
                "  - Start and completion of data preprocessing steps.",
                "  - Number of duplicates removed.",
                "  - Outliers detected and handled.",
                "  - Errors encountered during execution.",
                "- **Usage:** Facilitates debugging and provides a historical record of script operations.",
                "",
                "#### **5. Documentation**",
                "- **Comprehensive Guides:** Maintained detailed documentation outlining:",
                "  - Function purposes and usage.",
                "  - Data preprocessing methodologies.",
                "  - QA processes and testing procedures.",
                "- **Accessibility:** Ensured that documentation is easily accessible for current and future team members.",
                "",
                "#### **Concrete Examples and Outcomes**",
                "",
                "- **Outlier Detection Bug:**",
                "  - **Scenario:** An error was encountered where the outlier detection was not correctly identifying outliers in the hours-per-week column.",
                "  - **Resolution:** Revised the Z-score calculation method and revalidated with test data, ensuring accurate outlier detection and removal.",
                "",
                "- **Performance Bottleneck:**",
                "  - **Scenario:** The data loading process was significantly slow due to inefficient file reading methods.",
                "  - **Resolution:** Optimized the data loading by specifying data types and utilizing more efficient pandas functions, reducing load time by 30%."
            ]
    
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Markdown('\n'.join(qa_content), style={'font-size': '18px'})
                            ])
                        ], className="mb-4")
                    ], width=10)
                ], justify='center')
            ], fluid=True)
    
        elif tab == 'prediction-tab':
            # Retrieve data
            data_json = store_data.get('data', None)
            if data_json is None:
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Data not found. Please ensure preprocessing is completed.", color="danger")
                        ], width=12)
                    ])
                ])
    
            # Load data from JSON
            data = pd.read_json(data_json, orient='split')
    
            # Build input components with tooltips
            input_components = []
    
            # Model selection dropdown
            model_options = [{'label': model_name, 'value': model_name} for model_name in trained_models.keys()]
            model_dropdown = dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Model", html_for='model-dropdown')
                        ], md=6, lg=4),
                        dbc.Col([
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=model_options,
                                value='XGBoost',  # Default model
                                clearable=False
                            )
                        ], md=6, lg=4)
                    ], className="mb-3")
                ], width=12)
            ])
            input_components.append(model_dropdown)
    
            # Function to create tooltip
            def create_tooltip(target_id, message):
                return dbc.Tooltip(
                    message,
                    target=target_id,
                    placement='right',
                    style={'font-size': '14px'}
                )
    
            # For each feature, create an input with tooltip
            for feature in feature_cols:
                tooltip_id = f"tooltip-{feature}"
                if feature in numerical_columns + additional_numerical_features:
                    # Numerical input
                    mean_value = numerical_means.get(feature, 0)
                    input_field = dbc.Row([
                        dbc.Col([
                            dbc.Label([
                                feature.replace('-', ' ').title(),
                                " ",
                                html.Span(
                                    "ⓘ",
                                    id=tooltip_id,
                                    style={"cursor": "pointer", "color": "#17a2b8"}
                                )
                            ], html_for={'type': 'input', 'index': feature})
                        ], md=6, lg=4),
                        dbc.Col([
                            dbc.Input(
                                id={'type': 'input', 'index': feature},
                                type='number',
                                value=mean_value,
                                placeholder=f"e.g., {mean_value:.2f}"
                            ),
                            create_tooltip(
                                tooltip_id,
                                f"{feature.replace('-', ' ').title()}: Enter a numerical value. Example: {mean_value:.2f}"
                            )
                        ], md=6, lg=4)
                    ], className="mb-3")
                else:
                    # Categorical input
                    options = [{'label': val, 'value': val} for val in sorted(data[feature].unique())]
                    input_field = dbc.Row([
                        dbc.Col([
                            dbc.Label([
                                feature.replace('-', ' ').title(),
                                " ",
                                html.Span(
                                    "ⓘ",
                                    id=tooltip_id,
                                    style={"cursor": "pointer", "color": "#17a2b8"}
                                )
                            ], html_for={'type': 'input', 'index': feature})
                        ], md=6, lg=4),
                        dbc.Col([
                            dcc.Dropdown(
                                id={'type': 'input', 'index': feature},
                                options=options,
                                value=options[0]['value'],  # Default value
                                clearable=False
                            ),
                            create_tooltip(
                                tooltip_id,
                                f"{feature.replace('-', ' ').title()}: Select one option. Example: {options[0]['value']}"
                            )
                        ], md=6, lg=4)
                    ], className="mb-3")
                input_components.append(input_field)
    
            # Arrange inputs in rows and columns
            form_layout = dbc.Row(input_components, justify='center')
    
            # Add a submit button
            submit_button = dbc.Button('Predict', id='predict-button', color='primary', className="mt-2")
    
            # Div to display the prediction result
            prediction_output = html.Div(id='prediction-output')
    
            # Prediction History Section
            prediction_history_section = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Prediction History")),
                        dbc.CardBody([
                            html.Div(id='prediction-history-table')  # Updated to display prediction history
                        ])
                    ], className="mb-4")
                ], width=10)
            ], justify='center')
    
            # Assemble the layout
            layout = dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Income Prediction", className="mb-4"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Form([
                                    form_layout,
                                    submit_button
                                ])
                            ])
                        ], className="mb-4")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Prediction Result")),
                            dbc.CardBody([
                                prediction_output
                            ])
                        ], className="mb-4")
                    ], width=6)
                ], justify='center'),
                prediction_history_section  # Added Prediction History Section
            ], fluid=True)
    
            return layout
    
        elif tab == 'import-data-tab':
            # Retrieve data
            data_json = store_data.get('data', None)
            if data_json is None:
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Data not found. Please ensure preprocessing is completed.", color="danger")
                        ], width=12)
                    ])
                ])
            
            # Load data from JSON
            data = pd.read_json(data_json, orient='split')
    
            # Build layout for Import Data Tab
            import_layout = dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Import Your Data for Income Prediction", className="mb-4")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Upload CSV File")),
                            dbc.CardBody([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin-bottom': '20px'
                                    },
                                    multiple=False
                                ),
                                dbc.Alert(
                                    "Ensure your CSV file has the following columns: workclass, education, marital-status, occupation, relationship, race, native-country, age, capital-gain, capital-loss, hours-per-week.",
                                    color="info"
                                )
                            ])
                        ], className="mb-4")
                    ], width=6)
                ], justify='center'),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Select Model for Prediction")),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id='import-model-dropdown',
                                    options=[{'label': model, 'value': model} for model in trained_models.keys()],
                                    value='XGBoost',
                                    clearable=False
                                )
                            ])
                        ], className="mb-4")
                    ], width=6)
                ], justify='center'),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Run Predictions", id='run-predictions', color='success', className="mb-4")
                    ], width=4)
                ], justify='center'),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='import-prediction-output')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Prediction Results"),
                        dcc.Loading(
                            id="loading-predictions",
                            type="default",
                            children=html.Div(id='prediction-results')
                        )
                    ], width=12)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Visualizations on Predictions"),
                        dbc.Tabs([
                            dbc.Tab(label='Prediction Distribution', tab_id='pred-dist-tab'),
                            dbc.Tab(label='Feature vs Prediction', tab_id='feature-pred-tab')
                        ], id='import-visual-tabs', active_tab='pred-dist-tab', className="mb-3")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='import-visual-content')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Export Results", id='export-results-button', color='primary', className="mb-4"),
                        dcc.Download(id='download-results')
                    ], width=2)
                ], justify='center')
            ], fluid=True)
    
            return import_layout
    
        else:
            return dbc.Jumbotron([
                html.H1("404: Page not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {tab} was not recognized."),
            ])
            
    @app.callback(
        Output('categorical-graph', 'figure'),
        Input('categorical-dropdown', 'value'),
        State('stored-data', 'data')
    )
    def update_categorical_graph(selected_categorical, store_data):
        data_json = store_data.get('data', None)
        if data_json is None:
            return {}  # Return an empty figure or some default figure
        data = pd.read_json(data_json, orient='split')

        fig = px.histogram(
            data,
            x=selected_categorical,
            color='income',
            barmode='group',
            title=f"Distribution of Income Levels by {selected_categorical.replace('-', ' ').title()}",
            labels={selected_categorical: selected_categorical.replace('-', ' ').title(), 'count': 'Count'},
            color_discrete_map={'<=50K':'#636EFA', '>50K':'#EF553B'}
        )
        return fig


    @app.callback(
    Output('download-results', 'data'),
    Input('export-results-button', 'n_clicks'),
    State('uploaded-data-store', 'data'),
    prevent_initial_call=True
    )
    def export_results(n_clicks, uploaded_data_store):
        if uploaded_data_store is None:
            return None  # No data to export
        data = pd.read_json(uploaded_data_store, orient='split')
        return dcc.send_data_frame(data.to_csv, "prediction_results.csv", index=False)

