import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
**This is a Machine Learning App**

Created by: Matt Fiedler

Adjust the parameters to see real time **Iris flower** type predictions using a Random Forest Classifier!
""")

st.sidebar.header('Adjust Flower Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 3.1, 6.4, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.8, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.9, 4.1, 3.1)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.7, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris_df = datasets.load_iris()
X = iris_df.data
Y = iris_df.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Flower Type Labels')
st.write(iris_df.target_names)

st.subheader('Flower Type Prediction')
st.write(iris_df.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)