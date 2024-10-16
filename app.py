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

# Load the saved models


# Load the saved models
model_xgboost = joblib.load('trained_xgboost_model.pkl')
model_rf = joblib.load('trained_random_forest_model.pkl')

# Load the dataset (used for feature reference)
file_path = 'SMV.xlsx'
data = pd.read_excel(file_path)

# Define the categorical and numerical features
categorical_features = ['GG', 'Operation', 'Operation Position', 'Operation Description',
                        'Fiber 1', 'Fiber 2', 'Fiber 3', 'Knit Construction',
                        'Count 1', 'Count 2', 'Count 3']

numerical_features = ['Percentage 1', 'Percentage 2', 'Percentage 3',
                      'Ply 1', 'Ply 2', 'Ply 3', 'MC Speed', 'Length (cm)']

# Sidebar for page navigation
st.sidebar.title("Navigation")

# Display logo and title with changes
st.image("IND Logo PNG +.png", use_column_width=True, width=700)  # Logo
st.markdown('<h1 class="title">SMV Prediction App</h1>', unsafe_allow_html=True)  # Title

# Main App content for SMV prediction
st.header("Predict SMV using RandomForest & XGBoost")

# Input fields for the categories (Only once for both models)
GG = st.radio('Select GG', data['GG'].unique().tolist())
Operation = st.selectbox('Select Operation', data['Operation'].unique().tolist())
Operation_Position = st.selectbox('Select Operation Position', data['Operation Position'].unique().tolist())
Operation_Description = st.selectbox('Select Operation Description', data['Operation Description'].unique().tolist())
Knit_Construction = st.selectbox('Select Knit Construction', data['Knit Construction'].unique().tolist())

# Inputs for Fiber 1, Fiber 2, Fiber 3 and their corresponding counts and ply
Percentage_1 = st.number_input('Enter Percentage 1', min_value=0.0, max_value=100.0, step=0.1)
Fiber_1 = st.selectbox('Select Fiber 1', data['Fiber 1'].unique().tolist())
Count_1 = st.number_input('Enter Count 1', min_value=0)
Ply_1 = st.number_input('Enter Ply 1', min_value=0)

Percentage_2 = st.number_input('Enter Percentage 2', min_value=0.0, max_value=100.0, step=0.1)
Fiber_2 = st.selectbox('Select Fiber 2', data['Fiber 2'].unique().tolist())
Count_2 = st.number_input('Enter Count 2', min_value=0)
Ply_2 = st.number_input('Enter Ply 2', min_value=0)

Percentage_3 = st.number_input('Enter Percentage 3', min_value=0.0, max_value=100.0, step=0.1)
Fiber_3 = st.selectbox('Select Fiber 3', data['Fiber 3'].unique().tolist())
Count_3 = st.number_input('Enter Count 3', min_value=0)
Ply_3 = st.number_input('Enter Ply 3', min_value=0)

MC_Speed = st.selectbox('Select MC Speed', data['MC Speed'].unique().tolist())
Length = st.number_input('Enter Length (cm)', min_value=0.0, max_value=300.0, step=1.0)

if st.button('Predict SMV'):
    # Create a DataFrame from the input
    input_data = pd.DataFrame({
        'GG': [GG],
        'Operation': [Operation],
        'Operation Position': [Operation_Position],
        'Operation Description': [Operation_Description],
        'Knit Construction': [Knit_Construction],
        'Percentage 1': [Percentage_1],
        'Fiber 1': [Fiber_1],
        'Count 1': [Count_1],
        'Ply 1': [Ply_1],
        'Percentage 2': [Percentage_2],
        'Fiber 2': [Fiber_2],
        'Count 2': [Count_2],
        'Ply 2': [Ply_2],
        'Percentage 3': [Percentage_3],
        'Fiber 3': [Fiber_3],
        'Count 3': [Count_3],
        'Ply 3': [Ply_3],
        'MC Speed': [MC_Speed],
        'Length (cm)': [Length]
    })

    # One-hot encode the input data using the same columns as in the training data
    input_encoded = pd.get_dummies(input_data, columns=categorical_features)

    # Get the list of feature names from the original dataset used for training (X_train columns)
    # This assumes that you have already defined X_train.columns earlier during model training
    X_train_columns = pd.get_dummies(data[categorical_features + numerical_features]).columns

    # Reindex the input data to match the training features
    input_encoded = input_encoded.reindex(columns=X_train_columns, fill_value=0)

    # Convert input data to NumPy arrays
    input_encoded_np = input_encoded.values.astype(np.float32)

    # Model Predictions
    with st.spinner('Processing your prediction...'):
        try:
            # Random Forest Prediction
            prediction_rf = model_rf.predict(input_encoded_np)[0]

            # XGBoost Prediction
            prediction_xgboost = model_xgboost.predict(input_encoded_np)[0]

            st.write(f"**Random Forest Predicted SMV:** {prediction_rf:.2f}")
            st.write(f"**XGBoost Predicted SMV:** {prediction_xgboost:.2f}")

        except ValueError as e:
            st.error(f"An error occurred: {e}")




