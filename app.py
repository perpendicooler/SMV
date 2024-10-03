import streamlit as st
import pandas as pd
import joblib

# Add custom CSS for styling with enhancements and animations
st.markdown(
    """
    <style>
    /* Set the background color for the entire page to a light grey */
    section.main {
        background: #eaeaea;  /* Updated to a light grey */
        color: #333333;
        padding: 20px;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .stApp {
        background: #eaeaea;  /* Updated to a light grey */
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    /* Customize sidebar background and text color */
    .css-1d391kg {
        background-color: #4CAF50;  /* Sidebar background color */
        color: white;  /* Sidebar text color */
    }
    .css-1d391kg .stSidebar {
        background-color: #4CAF50; /* Sidebar background color */
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
page = st.sidebar.selectbox("Go to", ["SMV Prediction App", "Overview: The SMV Prediction Project", 
                                      "Data Preparation: Getting Ready for Modeling", 
                                      "Modeling: Random Forest & XGBoost", 
                                      "Results: Error Analysis & Model Performance"])

# Display logo and title with changes
st.image("IND Logo PNG +.png", use_column_width=True, width=700)  # Logo
st.markdown('<h1 class="title">SMV Prediction App</h1>', unsafe_allow_html=True)  # Title

if page == "SMV Prediction App":
    # Main App content for SMV prediction
    st.header("Predict SMV using RandomForest & XGBoost")

    # Input fields for the categories (Only once for both models)
    GG = st.selectbox('Select GG', df['GG'].unique().tolist())
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

        except ValueError as e:
            st.error(f"Error during prediction: {e}")

    # Download the dataset option
    if st.button('Download Original Data'):
        df.to_excel('SMV_original_data.xlsx', index=False)
        with open('SMV_original_data.xlsx', 'rb') as f:
            st.download_button('Download Original Data', f, file_name='SMV_original_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

elif page == "Overview: The SMV Prediction Project":
    # Overview page content
    st.header("The Journey of Predicting SMV: From Data to Results")
    st.write("""
        **What is SMV?**
        Standard Minute Value (SMV) is a key metric in the textile industry that measures the time it takes to complete a specific operation. 
        Accurate prediction of SMV helps optimize labor productivity and cost estimation.
        
        **Problem Statement:**
        The task is to predict the SMV based on various features like operation type, yarn type, knit construction, and more. 
        Using machine learning models like Random Forest and XGBoost, we aim to deliver highly accurate predictions.

        **Approach:**
        - Data Collection
        - Data Preprocessing
        - Feature Engineering
        - Model Training and Evaluation
    """)

elif page == "Data Preparation: Getting Ready for Modeling":
    # Data preparation page content
    st.header("Data Preparation for SMV Prediction")
    st.write("""
        - **Data Collection:** The data used consists of various parameters affecting SMV. 
        - **Preprocessing Steps:** This includes cleaning the data, handling missing values, and encoding categorical variables.
        - **Feature Engineering:** Creating new features based on existing ones to enhance model performance.
    """)

elif page == "Modeling: Random Forest & XGBoost":
    # Modeling page content
    st.header("Modeling Techniques Used")
    st.write("""
        **Random Forest:**
        - An ensemble learning method that constructs multiple decision trees.
        - It combines their predictions to improve accuracy and control overfitting.

        **XGBoost:**
        - An optimized gradient boosting framework.
        - It is effective in handling sparse data and uses advanced regularization techniques.

        Both models were trained and tested on the same dataset, allowing us to compare their performances accurately.
    """)

elif page == "Results: Error Analysis & Model Performance":
    # Results page content
    st.header("Results and Model Performance Analysis")
    st.write("""
        - **Model Evaluation Metrics:**
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - R-Squared

        **Conclusion:**
        A comparison of both models based on these metrics helped us choose the best performing model for predicting SMV accurately.
    """)

# Footer with acknowledgments
st.markdown("---")
st.write("Developed by [MD ARIF HOSSAIN](https://github.com/perpendicooler) - All Rights Reserved © 2024")
