# import libraries
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# load the model
model = tf.keras.models.load_model('model.keras')

# load the encoders and the scaler 
with open('label_en_gender.pkl', 'rb') as file:
    label_en_gender = pickle.load(file)

with open('onehot_en_geo.pkl', 'rb') as file:
    onehot_en_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', onehot_en_geo.categories_[0])
gender = st.selectbox('Gender', label_en_gender.classes_)
age = st.slider('age', 18, 90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# preprocess the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_en_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# one hot encode the geography
geo_encoded = onehot_en_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_en_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with the input data
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input data
input_scaled_df = scaler.transform(input_df)

# predict the churn
pred = model.predict(input_scaled_df)
prediction = pred[0][0]

st.write(f'Churn Probability: {prediction:.2f}')

if prediction > 0.5:
    st.write('The customer is leave the bank')
else:
    st.write('The customer is not leave the bank')