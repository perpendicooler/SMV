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
    
    /* Style for title */
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        margin: 20px 0;
        font-weight: bold;
        text-align: center;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }

    /* Style for buttons */
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
    
    /* Style for input fields */
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
    
    /* Style for metrics box */
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
    
    /* Center align the tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center; /* Center align the tabs */
        gap: 30px; /* Adjust the spacing between tabs */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained models
model_xgboost = joblib.load('trained_xgboost_model.pkl')
model_rf = joblib.load('trained_random_forest_model.pkl')

# Load the dataset for reference
file_path = 'SMV.xlsx'
data = pd.read_excel(file_path)

# Define categorical and numerical features
categorical_features = ['GG', 'Operation', 'Operation Position', 'Operation Description',
                        'Fiber 1', 'Fiber 2', 'Fiber 3', 'Knit Construction',
                        'Count 1', 'Count 2', 'Count 3']
numerical_features = ['Percentage 1', 'Percentage 2', 'Percentage 3',
                      'Ply 1', 'Ply 2', 'Ply 3', 'MC Speed', 'Length (cm)']

# Create tabs for different sections of the app
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠SMV Prediction App", "🚀Overview", 
                                        "📊Data Preparation", "💻Modeling", "📈Results"])

with tab1:
    # Main Prediction App content
    st.image("IND Logo PNG +.png", use_column_width=True)
    st.markdown('<h1 class="title">SMV Prediction App</h1>', unsafe_allow_html=True)

    # Input fields for predictions
    GG = st.radio('Select GG', data['GG'].unique().tolist())
    Operation = st.selectbox('Select Operation', data['Operation'].unique().tolist())
    Operation_Position = st.selectbox('Select Operation Position', data['Operation Position'].unique().tolist())
    Operation_Description = st.selectbox('Select Operation Description', data['Operation Description'].unique().tolist())
    Knit_Construction = st.selectbox('Select Knit Construction', data['Knit Construction'].unique().tolist())

    # Dynamic selection of number of fibers
    num_fibers = st.selectbox('Select Number of Fibers', [1, 2, 3])

    # Fiber 1 inputs
    if num_fibers >= 1:
        Percentage_1 = st.number_input('Enter Percentage 1', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_1 = st.selectbox('Select Fiber 1', data['Fiber 1'].unique().tolist())
        Count_1 = st.selectbox('Select Count 1', data['Count 1'].unique().tolist())
        Ply_1 = st.number_input('Enter Ply 1', min_value=0)

    # Fiber 2 inputs
    if num_fibers >= 2:
        Percentage_2 = st.number_input('Enter Percentage 2', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_2 = st.selectbox('Select Fiber 2', data['Fiber 2'].unique().tolist())
        Count_2 = st.selectbox('Select Count 2', data['Count 2'].unique().tolist())
        Ply_2 = st.number_input('Enter Ply 2', min_value=0)

    # Fiber 3 inputs
    if num_fibers == 3:
        Percentage_3 = st.number_input('Enter Percentage 3', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_3 = st.selectbox('Select Fiber 3', data['Fiber 3'].unique().tolist())
        Count_3 = st.selectbox('Select Count 3', data['Count 3'].unique().tolist())
        Ply_3 = st.number_input('Enter Ply 3', min_value=0)

    MC_Speed = st.selectbox('Select MC Speed', data['MC Speed'].unique().tolist())
    Length = st.number_input('Enter Length (cm)', min_value=0.0, max_value=3000.0, step=1.0)

    # Prediction button
    if st.button('Predict SMV'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'GG': [GG], 'Operation': [Operation], 'Operation Position': [Operation_Position],
            'Operation Description': [Operation_Description], 'Knit Construction': [Knit_Construction],
            'Percentage 1': [Percentage_1], 'Fiber 1': [Fiber_1], 'Count 1': [Count_1], 'Ply 1': [Ply_1],
            'Percentage 2': [Percentage_2] if num_fibers >= 2 else [0], 'Fiber 2': [Fiber_2] if num_fibers >= 2 else [None],
            'Count 2': [Count_2] if num_fibers >= 2 else [None], 'Ply 2': [Ply_2] if num_fibers >= 2 else [0],
            'Percentage 3': [Percentage_3] if num_fibers == 3 else [0], 'Fiber 3': [Fiber_3] if num_fibers == 3 else [None],
            'Count 3': [Count_3] if num_fibers == 3 else [None], 'Ply 3': [Ply_3] if num_fibers == 3 else [0],
            'MC Speed': [MC_Speed], 'Length (cm)': [Length]
        })

        # One-hot encode the input data
        input_encoded = pd.get_dummies(input_data, columns=data.select_dtypes(include=['object']).columns)

        # Ensure that all model columns are present
        model_columns = pd.get_dummies(data.drop(columns=['SMV'])).columns
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Convert to numpy array for prediction
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

                # Search for actual SMV in existing data
                matching_row = data[
                    (data['GG'] == GG) & (data['Operation'] == Operation) &
                    (data['Operation Position'] == Operation_Position) &
                    (data['Operation Description'] == Operation_Description) &
                    (data['Knit Construction'] == Knit_Construction) &
                    (data['MC Speed'] == MC_Speed) & (data['Length (cm)'] == Length) &
                    (data['Percentage 1'] == Percentage_1) & (data['Fiber 1'] == Fiber_1) &
                    (data['Count 1'] == Count_1) & (data['Ply 1'] == Ply_1) &
                    (data['Percentage 2'] == Percentage_2) & (data['Fiber 2'] == Fiber_2) &
                    (data['Count 2'] == Count_2) & (data['Ply 2'] == Ply_2) &
                    (data['Percentage 3'] == Percentage_3) & (data['Fiber 3'] == Fiber_3) &
                    (data['Count 3'] == Count_3) & (data['Ply 3'] == Ply_3)
                ]

                if not matching_row.empty:
                    actual_smv = matching_row['SMV'].values[0]
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

with tab2:
    st.markdown("## Overview of the SMV Prediction Project")
    st.write("This section gives an overview of the SMV prediction project, including key objectives, challenges, and methodologies used.")

with tab3:
    st.markdown("## Data Preparation: Getting Ready for Modeling")
    st.write("This section explains how the data was prepared for the SMV prediction model, including cleaning, encoding, and feature engineering.")

with tab4:
    st.markdown("## Modeling: Random Forest & XGBoost")
    st.write("This section discusses the models used for SMV prediction: Random Forest and XGBoost, with details on model training and evaluation.")

with tab5:
    st.markdown("## Results: Error Analysis & Model Performance")
    st.write("This section provides insights into model performance, error analysis, and the comparison between Random Forest and XGBoost predictions.")
