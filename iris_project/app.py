import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Page Title
st.title("🌸 Iris Flower Classification App")

# Load Dataset
columns = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth',
    'Species'
]

df = pd.read_csv("iris.data", names=columns)

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 2.0, 5.0, 3.5
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)", 1.0, 7.0, 1.4
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)", 0.1, 3.0, 0.2
)

# Prediction Button
if st.button("Predict Species"):

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Species: {prediction}")

# Display Dataset
if st.checkbox("Show Dataset"):
    st.subheader("Iris Dataset")
    st.dataframe(df.head())

# Display Accuracy
accuracy = model.score(X_test, y_test)

st.write(f"### Model Accuracy: {accuracy:.2f}")