import streamlit as st
import pandas as pd
import joblib

# Set up the Streamlit app configuration
st.set_page_config(
    page_title="SMV Prediction App",
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
        background-color: #f0f0f5;  /* Updated to a light grey */
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    /* Customize sidebar background and text color */
    .css-1d391kg {
        background-color: #f0f0f5;  /* Sidebar background color */
        color: white;  /* Sidebar text color */
    }
    .css-1d391kg .stSidebar {
        background-color: #f0f0f5; /* Sidebar background color */
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

# Load the trained models
model_rf = joblib.load('trained_random_forest_model.pkl')
model_xgboost = joblib.load('trained_xgboost_model.pkl')

# Load the original dataset for existing row checks
df = pd.read_excel('SMV-12&7GG.xlsx')  # Update with your actual dataset path

# Sidebar inputs
st.sidebar.title("Input Parameters")

GG = st.sidebar.selectbox('Select GG:', df['GG'].unique())
Operation = st.sidebar.selectbox('Select Operation:', df['Operation'].unique())
Operation_Position = st.sidebar.selectbox('Select Operation Position:', df['Operation Position'].unique())
Operation_Description = st.sidebar.selectbox('Select Operation Description:', df['Operation Description'].unique())

# New parameters
Percentage_1 = st.sidebar.number_input('Percentage 1', min_value=0.0, max_value=100.0, value=0.0)
Fiber_1 = st.sidebar.selectbox('Select Fiber 1:', df['Fiber 1'].unique())
Count_1 = st.sidebar.number_input('Count 1', min_value=0)
Ply_1 = st.sidebar.number_input('Ply 1', min_value=0)

Percentage_2 = st.sidebar.number_input('Percentage 2', min_value=0.0, max_value=100.0, value=0.0)
Fiber_2 = st.sidebar.selectbox('Select Fiber 2:', df['Fiber 2'].unique())
Count_2 = st.sidebar.number_input('Count 2', min_value=0)
Ply_2 = st.sidebar.number_input('Ply 2', min_value=0)

Percentage_3 = st.sidebar.number_input('Percentage 3', min_value=0.0, max_value=100.0, value=0.0)
Fiber_3 = st.sidebar.selectbox('Select Fiber 3:', df['Fiber 3'].unique())
Count_3 = st.sidebar.number_input('Count 3', min_value=0)
Ply_3 = st.sidebar.number_input('Ply 3', min_value=0)

Knit_Construction = st.sidebar.selectbox('Select Knit Construction:', df['Knit Construction'].unique())
MC_Speed = st.sidebar.selectbox('Select MC Speed:', df['MC Speed'].unique())
Length = st.sidebar.number_input('Length (cm)', min_value=0)

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'GG': [GG],
    'Operation': [Operation],
    'Operation Position': [Operation_Position],
    'Operation Description': [Operation_Description],
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
    'Knit Construction': [Knit_Construction],
    'MC Speed': [MC_Speed],
    'Length (cm)': [Length]
})

# One-hot encode input data
input_encoded = pd.get_dummies(input_data, columns=['Operation', 'Operation Position', 
                                                     'Fiber 1', 'Fiber 2', 'Fiber 3', 
                                                     'GG', 'Operation Description', 
                                                     'Knit Construction'])

# Get feature columns from the original DataFrame for consistency
feature_columns = model_rf.feature_names_in_

# Ensure the input has the same columns as the training data
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Check if the input values match an existing row in the original DataFrame
existing_row = df[
    (df['GG'] == GG) &
    (df['Operation'] == Operation) &
    (df['Operation Position'] == Operation_Position) &
    (df['Operation Description'] == Operation_Description) &
    (df['Percentage 1'] == Percentage_1) &
    (df['Fiber 1'] == Fiber_1) &
    (df['Count 1'] == Count_1) &
    (df['Ply 1'] == Ply_1) &
    (df['Percentage 2'] == Percentage_2) &
    (df['Fiber 2'] == Fiber_2) &
    (df['Count 2'] == Count_2) &
    (df['Ply 2'] == Ply_2) &
    (df['Percentage 3'] == Percentage_3) &
    (df['Fiber 3'] == Fiber_3) &
    (df['Count 3'] == Count_3) &
    (df['Ply 3'] == Ply_3) &
    (df['Knit Construction'] == Knit_Construction) &
    (df['MC Speed'] == MC_Speed) &
    (df['Length (cm)'] == Length)
]

# Check if an actual SMV exists for the input
actual_smv = existing_row['SMV'].values[0] if not existing_row.empty else None

# Prediction button
if st.button('Predict SMV'):
    with st.spinner('Processing your prediction...'):
        try:
            # Random Forest Prediction
            prediction_rf = model_rf.predict(input_encoded)[0]

            # XGBoost Prediction
            prediction_xgboost = model_xgboost.predict(input_encoded)[0]

            st.write(f"**Random Forest Predicted SMV:** {prediction_rf:.2f}")
            st.write(f"**XGBoost Predicted SMV:** {prediction_xgboost:.2f}")

            # Display Actual SMV if exists
            if actual_smv is not None:
                st.write(f"**Actual SMV:** {actual_smv:.2f}")
            else:
                st.write("**Actual SMV:** Not available for this input.")

            # Save predictions to Excel file
            predictions = pd.DataFrame({
                'Input Data': [input_data.to_dict(orient='records')],
                'Predicted RF SMV': [prediction_rf],
                'Predicted XGBoost SMV': [prediction_xgboost],
                'Actual SMV': [actual_smv if actual_smv is not None else 'N/A']
            })
            with pd.ExcelWriter('Prediction_History.xlsx', mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                predictions.to_excel(writer, index=False, header=not writer.sheets)
                st.success("Predictions saved to Prediction_History.xlsx")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Overview page content
st.title("SMV Prediction App")
st.image('your_logo.png', width=700)  # Replace with your actual logo path

st.header("Overview")
st.write("""
This application predicts the Standard Minute Value (SMV) based on various inputs related to garment operations. It uses two different machine learning models - Random Forest and XGBoost - to provide predictions. You can input parameters related to the production process, and the app will return the predicted SMV along with any actual SMV data if available.
""")

# Add more informative sections as necessary
st.header("Objective")
st.write("""
The objective of this project is to enhance the decision-making process in garment manufacturing by providing accurate SMV predictions. The app serves as a tool for production managers and analysts to optimize operations and improve efficiency.
""")

st.header("How It Works")
st.write("""
1. Input relevant parameters in the sidebar.
2. Click 'Predict SMV' to generate predictions.
3. Review the predicted SMV values and compare them with actual data.
4. Predictions are automatically saved for future reference.
""")

st.header("Data Source")
st.write("""
The data used for training the models comes from internal records and includes various features that affect the SMV in garment production.
""")

st.header("Machine Learning Models")
st.write("""
- **Random Forest**: A robust model that uses multiple decision trees to provide an average prediction.
- **XGBoost**: An optimized gradient boosting model that performs exceptionally well with structured data.
""")

st.header("Contact")
st.write("""
For any questions or feedback regarding this application, please contact the development team at [email@example.com](mailto:email@example.com).
""")
