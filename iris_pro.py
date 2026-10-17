# importing liabraries 

import streamlit as st
import joblib 
import pandas as pd 
import numpy as np 
import sklearn 

# loading the model 
with open("D:/CODING/ML using Python/Venv/Code/projects/iris_model.joblib", "rb") as f:
    model = joblib.load(f)

st.title("Iris Flower Classification")
st.write("_This is a simple web application for Machine learning model which can classify iris flower species based on the input features._")
st.subheader("This will classify the iris flower among 3 species :- Setosa, Versicolor and Virginica")

def user_input():
    sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 5.0) # it contain min, max and default value
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0)
    petal_length = st.slider("Petal lenght (cm)", 1.0, 6.9, 4.0)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

    return pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

input_df = user_input()
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'probablity' not in st.session_state:
    st.session_state.probablity = None


# button for prediction 
if st.button("Predict", help = 'Click to predict the species of iris flower based on your given input features'):
    st.session_state.prediction = model.predict(input_df)

if st.session_state.prediction is not None:
    if st.session_state.prediction[0] == 0:
        st.write(f"The predicted species of iris flower is:- **Setosa**")
        st.image(f"https://daylily-phlox.eu/wp-content/uploads/2021/10/Iris-setosa-dwarf-form.jpg", caption='Iris Setosa')

    elif st.session_state.prediction[0] == 1:
        st.write(f"The predicted species of iris flower is:- **Versicolor**")
        st.image(f"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOOQ-r6ZWBnqYWAcGJXLBflT-vpt9BShdJJSa2Ln0WC6wI6_vq2T2cz6U&s=10",
                 caption = "Iris Veriscolor")

    else:
        st.write(f"The predicted species of iris flower is:- **Virginica**")
        st.image(f"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQc_JbGX9iLXRT7NQ7w1GM-_P4nF5MUAc5Rz3hqzHTmRfMPFwNuNJthaC0&s=10",
                 caption = 'Iris Virginica')

# button for probablity
if st.button("Probablity", help='Click to see the probablity of each predicted species'):
    st.session_state.probablity = model.predict_proba(input_df)

if st.session_state.probablity is not None:
    st.write(f"The probablity of each predicted species is:-")
    st.write(f"Setosa: {st.session_state.probablity[0][0]*100:.2f}%")
    st.write(f"Versicolor: {st.session_state.probablity[0][1]*100:.2f}%")
    st.write(f"Virginica: {st.session_state.probablity[0][2]*100:.2f}%")

if st.button("Clear", help='click to clear the previous prediction and probablity'):
    st.session_state.prediction = None
    st.session_state.probablity = None
    st.rerun()

