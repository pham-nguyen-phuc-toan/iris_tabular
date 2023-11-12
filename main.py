import streamlit as st
import pickle as pkl
import numpy as np

class_list = {'0': 'Setosa', '1': 'Versicolor', '2': 'Virginica'}

st.title('Iris classification based on sepal and petal size')

input = open('lrc_iris.pkl', 'rb')
model = pkl.load(input)

st.header('Choose the size of iris flower sepal and petal')

sepal_length = st.slider('Sepal length (cm)', 0.0, 20.0, 0.0, 0.1)
sepal_width = st.slider('Sepal width (cm)', 0.0, 20.0, 0.0, 0.1)
petal_length = st.slider('Petal length (cm)', 0.0, 20.0, 0.0, 0.1)
petal_width = st.slider('Petal width (cm)', 0.0, 20.0, 0.0, 0.1)

if sepal_length >= 0 and sepal_width >= 0 and petal_length >= 0 and petal_width >= 0:
    if st.button('Predict'):
        feature_vector = np.array([sepal_length, sepal_width, petal_length, petal_width])
        feature_vector = feature_vector.reshape(-1, 1)
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
