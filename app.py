import streamlit as st
import numpy as np
import pickle

# Custom page config
st.set_page_config(page_title="Iris Flower Predictor 🌸", layout="centered")

# Load model
try:
    with open("iris_dataset.pkl", 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"🚫 Failed to load model: {e}")
    st.stop()

# Sidebar info
with st.sidebar:
    st.title("🔍 About the App")
    st.markdown("""
    This app predicts the **Iris flower species** 🌺 based on:
    - Sepal Length & Width
    - Petal Length & Width
    
    **Model**: Logistic Regression  
    **Dataset**: Classic Iris Dataset  
    """)
    st.markdown("---")
    st.write("👨‍💻 Made with ❤️ by Varenya Visen")

# Main Title
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🌸 Iris Flower Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter flower measurements to identify the species</h4>", unsafe_allow_html=True)
st.markdown("---")

# Input sliders with proper range
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3, 0.1)
with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# Predict button
if st.button("🌼 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    try:
        prediction = model.predict(input_data)

        if isinstance(prediction[0], (int, np.integer)):
            species = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
            pred_label = species[prediction[0]]
        else:
            pred_label = prediction[0]

        st.markdown(f"<h2 style='text-align: center; color: #28a745;'>🌿 Predicted Species: <u>{pred_label}</u></h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
