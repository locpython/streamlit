import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
st.title("Revenue Prediction")
pred = st.number_input('Input Temperature', step = 1)
# The correct direct download URL
file_id = '1wZETgqcnl0dQJ8FV3hI_Y64LK1KNfGyE'
direct_link = f'https://drive.google.com/uc?export=download&id={file_id}'

# Now use this URL to read the CSV file into a pandas DataFrame
df = pd.read_csv(direct_link)
x = df['Temperature'].values.reshape(-1, 1)
y = df['Revenue'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict([[pred]])
k = y_pred[0][0]
if st.button('Predict', type="primary"):
    st.balloons()
    st.snow()
    st.write(f"Revenue Prediction: {round(k,1)}")
