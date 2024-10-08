import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title="SMV Prediction App",
    # page_icon="favicon.ico",  # Ensure this is the correct path to your favicon
    # layout="wide"
)
# Add custom CSS for styling with enhancements and animations
st.markdown(
    """
    <style>
    /* Set the background color for the entire page to a light grey */
    section.main {
        background-color: #f0f0f5; /* Updated to a light grey */
        color: #333333;
        padding: 20px;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .stApp {
        background-color: #f0f0f5;;  /* Updated to a light grey */
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    /* Customize sidebar background and text color */
    .css-1d391kg {
        background-color: #f0f0f5;  /* Sidebar background color */
        color: white;  /* Sidebar text color */
    }
    .css-1d391kg .stSidebar {
        background-color: #f0f0f5;; /* Sidebar background color */
        color: white; /* Sidebar text color */
    }
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        margin: 20px 0;
        font-weight: bold;
        text-align: center;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-size: 1.2em;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s, transform 0.2s, box-shadow 0.2s ease;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .button:hover {
        background-color: #45a049;
        transform: translateY(-3px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    .input-field {
        margin: 10px 0;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        width: 100%;
        max-width: 400px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .metrics {
        margin: 20px 0;
        padding: 15px;
        border-radius: 5px;
        background-color: #BEBEBE;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .metrics:hover {
        background-color: #A9A9A9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved models
model_rf = joblib.load('trained_model_rf.pkl')
model_xgboost = joblib.load('trained_model_xgboost.pkl')

# Load the dataset
file_path = 'SMV-12&7GG.xlsx'
df = pd.read_excel(file_path)

# One-hot encode the dataset
df_encoded = pd.get_dummies(df, columns=['Operation', 'Operation Position', 'Yarn Type', 'GG', 'Operation Description', 'Knit Construction'])

# Get the feature names used in training (excluding target column 'SMV')
feature_columns = df_encoded.drop('SMV', axis=1).columns.tolist()

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["SMV Prediction App", "🚀Overview: The SMV Prediction Project", 
                                      "📊Data Preparation: Getting Ready for Modeling", 
                                      "💻Modeling: Random Forest & XGBoost", 
                                      "📈Results: Error Analysis & Model Performance"])

# Display logo and title with changes
st.image("IND Logo PNG +.png", use_column_width=True, width=700)  # Logo
st.markdown('<h1 class="title">SMV Prediction App</h1>', unsafe_allow_html=True)  # Title

if page == "SMV Prediction App":
    # Main App content for SMV prediction
    st.header("Predict SMV using RandomForest & XGBoost")

    # Input fields for the categories (Only once for both models)
    GG = st.radio('Select GG', df['GG'].unique().tolist())
    Operation = st.selectbox('Select Operation', df['Operation'].unique().tolist())
    Operation_Position = st.selectbox('Select Operation Position', df['Operation Position'].unique().tolist())
    Operation_Description = st.selectbox('Select Operation Description', df['Operation Description'].unique().tolist())
    Yarn_Type = st.selectbox('Select Yarn Type', df['Yarn Type'].unique().tolist())
    Knit_Construction = st.selectbox('Select Knit Construction', df['Knit Construction'].unique().tolist())
    MC_Speed = st.selectbox('Select MC Speed', df['MC Speed'].unique().tolist())
    Length = st.number_input('Enter Length (cm)', min_value=0.0, max_value=300.0, step=1.0)

    if st.button('Predict SMV'):
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'GG': [GG],
            'Operation': [Operation],
            'Operation Position': [Operation_Position],
            'Operation Description': [Operation_Description],
            'Yarn Type': [Yarn_Type],
            'Knit Construction': [Knit_Construction],
            'MC Speed': [MC_Speed],
            'Length (cm)': [Length]
        })

        # Apply the same one-hot encoding to the input data
        input_encoded = pd.get_dummies(input_data, columns=['Operation', 'Operation Position', 'Yarn Type', 'GG', 'Knit Construction', 'Operation Description'])

        # Ensure the input has the same columns as the training data
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Check if the input values match an existing row in the original DataFrame
        existing_row = df[(df['GG'] == GG) &
                          (df['MC Speed'] == MC_Speed) &
                          (df['Operation'] == Operation) &
                          (df['Knit Construction'] == Knit_Construction) &
                          (df['Yarn Type'] == Yarn_Type) &
                          (df['Operation Position'] == Operation_Position) &
                          (df['Operation Description'] == Operation_Description) &
                          (df['Length (cm)'] == Length)]
        actual_smv = existing_row['SMV'].values[0] if not existing_row.empty else None
        with st.spinner('Processing your prediction...'):

            # Model Predictions
            try:
                # Random Forest Prediction
                prediction_rf = model_rf.predict(input_encoded)[0]

                # XGBoost Prediction
                prediction_xgboost = model_xgboost.predict(input_encoded)[0]

                st.write(f"**Random Forest Predicted SMV:** {prediction_rf:.2f}")
                st.write(f"**XGBoost Predicted SMV:** {prediction_xgboost:.2f}")
                if actual_smv is not None:
                    st.write(f"**Exact match found!** Actual SMV: {actual_smv:.2f}")

                    # Calculate errors for both models
                    error_rf = abs(prediction_rf - actual_smv)
                    error_xgboost = abs(prediction_xgboost - actual_smv)

                    relative_error_rf = (error_rf / actual_smv) * 100
                    relative_error_xgboost = (error_xgboost / actual_smv) * 100

                    # Display error metrics for both models
                    st.markdown(f"<div class='metrics'><strong>Random Forest:</strong><br>Point Difference: {error_rf:.2f}<br>Relative Error: {relative_error_rf:.2f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metrics'><strong>XGBoost:</strong><br>Point Difference: {error_xgboost:.2f}<br>Relative Error: {relative_error_xgboost:.2f}%</div>", unsafe_allow_html=True)

                    # Determine better model
                    if error_rf < error_xgboost:
                        st.success("Random Forest is the better fit for this prediction.")
                    else:
                        st.success("XGBoost is the better fit for this prediction.")
                else:
                    st.write("**New combination detected!** No actual SMV available.")
                    # Example: simple average
                    combined_prediction = (prediction_rf + prediction_xgboost) / 2
                    st.write(f"**On average, the SMV is estimated to be around** {combined_prediction:.2f}")


            except ValueError as e:
                st.error(f"An error occurred: {e}")

            # Save prediction to Excel
            if st.button("Save Prediction"):
                predictions_df = pd.DataFrame({
                    'GG': [GG],
                    'Operation': [Operation],
                    'Operation Position': [Operation_Position],
                    'Operation Description': [Operation_Description],
                    'Yarn Type': [Yarn_Type],
                    'Knit Construction': [Knit_Construction],
                    'MC Speed': [MC_Speed],
                    'Length (cm)': [Length],
                    'RF_Predicted_SMV': [prediction_rf],
                    'XGBoost_Predicted_SMV': [prediction_xgboost]
                })

                # Load or create the Excel file for saving predictions
                try:
                    history_df = pd.read_excel('prediction_history.xlsx')
                    combined_df = pd.concat([history_df, predictions_df], ignore_index=True)
                except FileNotFoundError:
                    combined_df = predictions_df  # Create new DataFrame if file does not exist

                # Save to Excel
                combined_df.to_excel('prediction_history.xlsx', index=False)
                st.success("Prediction saved successfully!")


elif page == "🚀Overview: The SMV Prediction Project":
    # Overview page content
    st.header("🚀 The Journey of Predicting SMV: From Data to Results")
    st.write("""
        **What is SMV?**  
        Standard Minute Value (SMV) is a critical measure in the textile industry that determines the time required to complete a specific operation. By quantifying SMV, industries can better estimate labor costs, optimize operations, and improve overall productivity. It’s essential for production planning, efficiency measurement, and cost control.
        
        **Problem Statement:**  
        Our goal is to predict SMV using various factors such as operation type, yarn type, knit construction, and machine settings. With a focus on optimizing labor productivity and cost estimation, we employ machine learning models like **Random Forest** and **XGBoost** to deliver high-accuracy predictions. These predictions help decision-makers streamline processes, improve time management, and reduce production costs.

        **Approach:**

        - **Data Collection:**  
        We gathered real-world data from textile operations, capturing a variety of critical factors:
            - Operation Type (e.g., sewing, knitting, dyeing)
            - Yarn Type (cotton, polyester, blends, etc.)
            - Knit Construction (patterns and techniques used)
            - Machine Parameters (speed, settings, needle types)
        These variables were selected based on their direct impact on the SMV and operational efficiency.

        - **Data Preprocessing:**  
        Raw data is rarely perfect. To prepare the data for machine learning models, we followed these essential steps:
            - **Cleaning:** Removing irrelevant or redundant data, and addressing missing values.
            - **Handling Outliers:** Identifying extreme values and deciding whether to cap or remove them.
            - **Encoding Categorical Variables:** Transforming categorical features into numerical values through **one-hot encoding** for compatibility with the machine learning models.
            - **Feature Scaling:** Normalizing numerical features like length, speed, and material width to prevent bias during training.

        - **Feature Engineering:**  
        In this step, we enhanced the dataset by creating new, insightful features. Some of the engineered features include:
            - Interaction terms between machine speed and operation type.
            - Transformations (e.g., log, polynomial) of skewed data to improve model fit.
            These new features help the models learn complex relationships between different parameters and improve prediction accuracy.

        - **Model Training and Evaluation:**  
        We used two advanced machine learning algorithms: **Random Forest** and **XGBoost**. Both models are highly effective at handling structured data and offer robust, accurate predictions.
            - **Random Forest:** Builds multiple decision trees, averages their predictions, and prevents overfitting by bagging.
            - **XGBoost:** A highly optimized gradient boosting model that’s efficient, fast, and works well with sparse datasets.
        
        After training, the models are evaluated based on how well they predict SMV values compared to the actual data. Metrics like **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-Squared** are used to measure performance.
        
        - **Model Comparison:**  
        Once trained, we compared both models using evaluation metrics and identified the model that performed best. This model is deployed to provide predictions in real-time in this application.
    """)


elif page == "📊Data Preparation: Getting Ready for Modeling":
    # Data preparation page content
    st.header("📊 Data Preparation for SMV Prediction")
    st.write("""
        Data preparation is a crucial part of any machine learning project. Here's how we processed our dataset for the best possible results.

        - **Data Collection:**  
        We worked with a rich dataset consisting of multiple parameters that influence SMV. Some key features include:
            - Operation Type: Different tasks performed in textile production.
            - Yarn Type: The materials used in production (e.g., cotton, wool, polyester).
            - Machine Speed: The speed at which the machine operates during the task.
            - Knit Construction: The technique or pattern used in knitting.
            - Material Length: The length of material processed during operations.

        - **Preprocessing Steps:**  
        This phase focuses on cleaning and organizing the raw data:
            - **Data Cleaning:** Eliminating irrelevant or redundant data, filling missing values, and correcting inconsistencies.
            - **Handling Missing Values:** For numerical features, missing values were imputed using statistical methods like the median or mean. For categorical data, the mode was used or the rows were dropped based on the extent of missingness.
            - **Encoding Categorical Variables:** Categorical variables such as operation type, yarn type, and knit construction were converted into numerical format using **one-hot encoding** to allow machine learning models to interpret them.
            - **Scaling Numerical Features:** Features such as material length and machine speed were scaled using **Min-Max Scaling** to ensure that no feature dominates the others due to its range.

        - **Feature Engineering:**  
        To improve the model's predictive power, we engineered additional features:
            - Created interaction terms to capture relationships between features (e.g., operation type × machine speed).
            - Applied transformations like logarithmic scaling to deal with skewed distributions.
            - Generated new features from existing ones to enhance model learning.
    """)


elif page == "💻Modeling: Random Forest & XGBoost":
    # Modeling page content
    st.header("💻 Modeling Techniques: Random Forest & XGBoost")
    st.write("""
        We used two powerful machine learning models to predict SMV:

        **Random Forest:**
        - An ensemble method that constructs multiple decision trees and aggregates their predictions.
        - It’s highly accurate for datasets with non-linear relationships.
        - Random Forest also reduces the risk of overfitting by averaging the outcomes of multiple trees.
        - Suitable for datasets with both numerical and categorical variables.

        **XGBoost:**
        - A high-performance implementation of gradient boosting.
        - It combines the predictions of weak learners in an iterative fashion, improving the final prediction step by step.
        - XGBoost is efficient, scalable, and often outperforms other algorithms in structured data scenarios.
        - Incorporates regularization techniques to control model complexity and improve generalization.

        Both models were trained on the same dataset, allowing us to compare their performance on key metrics like **MAE**, **RMSE**, and **R-Squared**. Each model's strengths were assessed, and the most effective one was chosen for deployment.
    """)


elif page == "📈Results: Error Analysis & Model Performance":
    # Results page content
    st.header("📈 Results: Error Analysis & Model Performance")
    st.write("""
        After training and testing the models, we performed a detailed evaluation to determine how well each model predicted SMV. The key performance metrics include:

        - **Mean Absolute Error (MAE):**  
        Measures the average absolute difference between predicted and actual SMV values.

        - **Mean Squared Error (MSE):**  
        Squares the errors to penalize larger differences between predictions and actual values.

        - **R-Squared:**  
        A statistical measure that represents the proportion of variance in the dependent variable that can be explained by the independent variables. It helps to evaluate the goodness-of-fit for the model.

        **Conclusion:**  
        After comparing the performance of both models on these metrics, we selected the one with the best accuracy and least error for deployment. This model is now used for real-time SMV predictions, improving the overall efficiency of the textile manufacturing process.
    """)


# Footer with acknowledgments
st.markdown("---")
st.write("Developed by [INDESORE](https://www.indesore.com/) - All Rights Reserved © 2024")
