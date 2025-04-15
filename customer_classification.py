import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Title and header
st.title("Customer Classification App")
st.header("Classify Customers Based on Their Profile")

# Step 1: Create sample data
data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Age': np.random.randint(18, 70, 200),
    'Annual Income (k$)': np.random.randint(15, 150, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200),
    'Category': np.random.choice(['Low-Value', 'Mid-Value', 'High-Value'], 200)
})

# Step 2: Preprocessing
label_gender = LabelEncoder()
data['Gender'] = label_gender.fit_transform(data['Gender'])

label_category = LabelEncoder()
data['Category'] = label_category.fit_transform(data['Category'])

X = data.drop('Category', axis=1)
y = data['Category']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split & Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluation (optional to display)
acc = accuracy_score(y_test, model.predict(X_test))

# Sidebar for user input
st.sidebar.title("Input Customer Details")

name = st.sidebar.text_input("Customer Name", value="John")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 15, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

if st.sidebar.button("Predict Category"):
    gender_num = 1 if gender == "Male" else 0
    user_data = pd.DataFrame([[gender_num, age, income, score]],
                             columns=["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
    
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    category_name = label_category.inverse_transform([prediction])[0]

    st.subheader(f"Hello, {name}")
    st.success(f"Predicted Customer Category: **{category_name}**")
    st.caption(f"(Model Accuracy: {round(acc * 100, 2)}%)")


if st.checkbox("Show Sample Training Data"):
    st.write(data.head())