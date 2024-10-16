import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

model_xgboost = joblib.load('trained_xgboost_model.pkl')
model_rf = joblib.load('trained_random_forest_model.pkl')

# Load dataset for reference
file_path = 'SMV.xlsx'
data = pd.read_excel(file_path)

# Define categorical and numerical features
categorical_features = ['GG', 'Operation', 'Operation Position', 'Operation Description',
                        'Fiber 1', 'Fiber 2', 'Fiber 3', 'Knit Construction',
                        'Count 1', 'Count 2', 'Count 3']
numerical_features = ['Percentage 1', 'Percentage 2', 'Percentage 3',
                      'Ply 1', 'Ply 2', 'Ply 3', 'MC Speed', 'Length (cm)']

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ["SMV Prediction App", "🚀Overview: The SMV Prediction Project", 
                                      "📊Data Preparation: Getting Ready for Modeling", 
                                      "💻Modeling: Random Forest & XGBoost", 
                                      "📈Results: Error Analysis & Model Performance"])

# Main App content
st.image("IND Logo PNG +.png", use_column_width=True, width=700)
st.markdown('<h1 class="title">SMV Prediction App</h1>', unsafe_allow_html=True)

# Input fields
GG = st.radio('Select GG', data['GG'].unique().tolist())
Operation = st.selectbox('Select Operation', data['Operation'].unique().tolist())
Operation_Position = st.selectbox('Select Operation Position', data['Operation Position'].unique().tolist())
Operation_Description = st.selectbox('Select Operation Description', data['Operation Description'].unique().tolist())
Knit_Construction = st.selectbox('Select Knit Construction', data['Knit Construction'].unique().tolist())

# Dynamic selection of number of fibers
num_fibers = st.selectbox('Select Number of Fibers', [1, 2, 3])

# Fiber 1 inputs
Percentage_1, Fiber_1, Count_1, Ply_1 = 0.0, 'None', 0, 0
if num_fibers >= 1:
    Percentage_1 = st.number_input('Enter Percentage 1', min_value=0.0, max_value=100.0, step=0.1)
    Fiber_1 = st.selectbox('Select Fiber 1', data['Fiber 1'].unique().tolist())
    Count_1 = st.selectbox('Select Count 1', data['Count 1'].unique().tolist())  # Dynamically select from data
    Ply_1 = st.number_input('Enter Ply 1', min_value=0)

# Fiber 2 inputs
Percentage_2, Fiber_2, Count_2, Ply_2 = 0.0, 'None', 0, 0
if num_fibers >= 2:
    Percentage_2 = st.number_input('Enter Percentage 2', min_value=0.0, max_value=100.0, step=0.1)
    Fiber_2 = st.selectbox('Select Fiber 2', data['Fiber 2'].unique().tolist())
    Count_2 = st.selectbox('Select Count 2', data['Count 2'].unique().tolist())  # Dynamically select from data
    Ply_2 = st.number_input('Enter Ply 2', min_value=0)

# Fiber 3 inputs
Percentage_3, Fiber_3, Count_3, Ply_3 = 0.0, 'None', 0, 0
if num_fibers == 3:
    Percentage_3 = st.number_input('Enter Percentage 3', min_value=0.0, max_value=100.0, step=0.1)
    Fiber_3 = st.selectbox('Select Fiber 3', data['Fiber 3'].unique().tolist())
    Count_3 = st.selectbox('Select Count 3', data['Count 3'].unique().tolist())  # Dynamically select from data
    Ply_3 = st.number_input('Enter Ply 3', min_value=0)

MC_Speed = st.selectbox('Select MC Speed', data['MC Speed'].unique().tolist())
Length = st.number_input('Enter Length (cm)', min_value=0.0, max_value=300.0, step=1.0)

# Prediction button
if st.button('Predict SMV'):
    # Create DataFrame from input
    input_data = pd.DataFrame({
        'GG': [GG], 'Operation': [Operation], 'Operation Position': [Operation_Position],
        'Operation Description': [Operation_Description], 'Knit Construction': [Knit_Construction],
        'Percentage 1': [Percentage_1], 'Fiber 1': [Fiber_1], 'Count 1': [Count_1], 'Ply 1': [Ply_1],
        'Percentage 2': [Percentage_2], 'Fiber 2': [Fiber_2], 'Count 2': [Count_2], 'Ply 2': [Ply_2],
        'Percentage 3': [Percentage_3], 'Fiber 3': [Fiber_3], 'Count 3': [Count_3], 'Ply 3': [Ply_3],
        'MC Speed': [MC_Speed], 'Length (cm)': [Length]
    })

    # One-hot encode the input
    input_encoded = pd.get_dummies(input_data, columns=categorical_features)

    # Match the features with the model's training set
    X_train_columns = pd.get_dummies(data[categorical_features + numerical_features]).columns
    input_encoded = input_encoded.reindex(columns=X_train_columns, fill_value=0)

    # Convert to NumPy for prediction
    input_encoded_np = input_encoded.values.astype(np.float32)

    # Model predictions
    with st.spinner('Processing your prediction...'):
        try:
            # Random Forest prediction
            prediction_rf = model_rf.predict(input_encoded_np)[0]
            # XGBoost prediction
            prediction_xgboost = model_xgboost.predict(input_encoded_np)[0]

            st.write(f"**Random Forest Predicted SMV:** {prediction_rf:.2f}")
            st.write(f"**XGBoost Predicted SMV:** {prediction_xgboost:.2f}")
            combined_prediction = (prediction_rf + prediction_xgboost) / 2
            st.write(f"**On average, the SMV is estimated to be around** {combined_prediction:.2f}")  

            # Find actual SMV if available
            existing_row = data[
                (data['GG'] == GG) & (data['Operation'] == Operation) & 
                (data['Operation Position'] == Operation_Position) &
                (data['Operation Description'] == Operation_Description) &
                (data['Percentage 1'] == Percentage_1) & (data['Fiber 1'] == Fiber_1) &
                (data['Count 1'] == Count_1) & (data['Ply 1'] == Ply_1) & 
                (data['Percentage 2'] == Percentage_2) & (data['Fiber 2'] == Fiber_2) &
                (data['Count 2'] == Count_2) & (data['Ply 2'] == Ply_2) & 
                (data['Percentage 3'] == Percentage_3) & (data['Fiber 3'] == Fiber_3) & 
                (data['Count 3'] == Count_3) & (data['Ply 3'] == Ply_3) & 
                (data['Knit Construction'] == Knit_Construction) & 
                (data['MC Speed'] == MC_Speed) & (data['Length (cm)'] == Length)
            ]

            if not existing_row.empty:
                actual_smv = existing_row['SMV'].values[0]
                st.write(f"**Exact match found! Actual SMV:** {actual_smv:.2f}")

                # Calculate errors
                error_rf = abs(prediction_rf - actual_smv)
                error_xgboost = abs(prediction_xgboost - actual_smv)
                st.write(f"**Random Forest Error:** {error_rf:.2f}")
                st.write(f"**XGBoost Error:** {error_xgboost:.2f}")

            else:
                st.write("**New combination detected! No actual SMV available.**")

        except ValueError as e:
            st.error(f"An error occurred: {e}")

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
