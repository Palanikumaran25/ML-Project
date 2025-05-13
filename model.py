import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="End Semester Marks Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide the sidebar toggle button using CSS
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_FILENAME = "Linear_regression_model.pkl"

def train_and_save_model():
    """Train and save the Linear Regression model if it doesn't exist."""
    data = {
        "MSE": [60, 70, 80, 90, 50, 40, 85, 75],
        "Attendance": [90, 85, 80, 95, 70, 65, 88, 92],
        "ESE": [65, 75, 85, 95, 55, 45, 88, 82]
    }
    df = pd.DataFrame(data)
    X = df[["MSE", "Attendance"]]
    y = df["ESE"]

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    return model

@st.cache_resource
def load_model():
    """Load the trained model, or train a new one if not found."""
    if os.path.exists(MODEL_FILENAME):
        with open(MODEL_FILENAME, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.warning("Model not found. Training a new one using sample data...")
        return train_and_save_model()

model = load_model()

st.title("📚 End Semester Marks Prediction")
st.markdown("This app predicts **End Semester Marks** based on Mid-Semester Marks and Attendance.")

st.header("📥 Enter Student Data")
mse = st.number_input("Mid-Semester Marks", min_value=0.0, max_value=100.0, step=1.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("🎯 Predict "):
    input_data = np.array([[mse, attendance]])
    prediction = model.predict(input_data)
    st.success(f"📈 Predict: **{round(prediction[0], 2)}**")

st.markdown("---")

st.header("📤 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload your CSV file (with 'MSE' and 'Attendance' columns)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ["MSE", "Attendance"]

        if not all(col in df.columns for col in required_columns):
            st.error(f"❗ CSV must contain the following columns: {required_columns}")
        else:
            input_data = df[required_columns].values
            predictions = model.predict(input_data)
            df["Predicted ESE"] = predictions

            st.success("✅ Batch Prediction Complete!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predicted_marks.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}") 
        
st.markdown("---")

st.markdown( 
   """
   <style>
   .stApp {
       background-color: #FFA07A;
   }
   </style>
   """,
   unsafe_allow_html=True
)
