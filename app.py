import streamlit as st
from model import scaler,model
import pandas as pd

st.title("HOUSE PREDICTION MODEL😊😊")
# st.write("dataset sariggga lekhapovadam valla prediction sarigga ravatla 😢😢")

square_feet = st.number_input("Square Feet", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

year_built = st.number_input("Year Built", min_value=1900)

neighborhood = st.selectbox(
    "Neighborhood",
    ["Urban", "Rural"]
)

button=st.button("Predict Price")

if button:
        input_data = pd.DataFrame({
        'SquareFeet': [square_feet],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'YearBuilt': [year_built],
        'Neighborhood_Suburb': [1 if neighborhood == "Suburb" else 0],
        'Neighborhood_Urban': [1 if neighborhood == "Urban" else 0]
    })
        scaled=scaler.transform(input_data)

        prediction=model.predict(scaled)

        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")



