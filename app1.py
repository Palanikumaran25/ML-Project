import streamlit as st
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier

st.title("🌸 Iris Flower Prediction App") 

#Load Model
model = joblib.load("model.pkl")  

#Define models
models = { 
    "Select a model": None,
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

#Sidebar: Model Selection
st.sidebar.header("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
selected_model = models[selected_model_name]

#Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.number_input(
        'Sepal Length (cm)', min_value=4.3, max_value=7.9, value=5.4, step=0.1, key="sepal_length")
    
    sepal_width = st.sidebar.number_input(
        'Sepal Width (cm)', min_value=2.0, max_value=4.4, value=3.4, step=0.1, key="sepal_width")
    
    petal_length = st.sidebar.number_input(
        'Petal Length (cm)', min_value=1.0, max_value=6.9, value=1.3, step=0.1, key="petal_length")
    
    petal_width = st.sidebar.number_input(
        'Petal Width (cm)', min_value=0.1, max_value=2.5, value=0.2, step=0.1, key="petal_width")

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

#Get user input
input_df = user_input_features() 

#Show input data
st.subheader("Input Features")
st.write(input_df)

#Prediction button
if st.sidebar.button("🧠 Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    iris_species = ['Setosa', 'Versicolor', 'Virginica']
    st.subheader("Prediction")
    st.write(f"**Predicted Species:** {iris_species[prediction[0]]}")

    st.subheader("Prediction Probabilities")
    st.write(pd.DataFrame(prediction_proba, columns=iris_species))  
    
st.markdown("---")
    
st.subheader("📁 Upload CSV File")
uploaded_file = st.file_uploader("Upload your iris data CSV", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_upload)
    
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F0F0;
    }
    </style>
    """,
    unsafe_allow_html=True
)  

st.markdown(
      """
      <style>
      [data-testid="stSidebar"] {
          background-color: #FFA421;
      }
      </style>
      """,
      unsafe_allow_html=True
) 

st.markdown("---")