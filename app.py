import streamlit as st
import pandas as pd
import joblib

# 1. Load model
model = joblib.load("nyc_taxi_random_forest.pkl")

# 2. Page title
st.title("NYC Taxi Fare Predictor")

# 3. User inputs
trip_distance = st.number_input(
    "Trip Distance",
    min_value=0.0
)

trip_duration = st.number_input(
    "Trip Duration (minutes)",
    min_value=0.0
)

passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    step=1
)

pickup_hour = st.number_input(
    "Pickup Hour",
    min_value=0,
    max_value=23,
    step=1
)

pickup_dayofweek = st.number_input(
    "Pickup Day of Week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6,
    step=1
)

is_weekend = st.selectbox(
    "Weekend?",
    [0, 1]
)

pickup_location = st.number_input(
    "Pickup Location ID",
    min_value=1,
    step=1
)

dropoff_location = st.number_input(
    "Dropoff Location ID",
    min_value=1,
    step=1
)

ratecode = st.number_input(
    "Ratecode ID",
    min_value=1,
    step=1
)

# 4. Prediction button
if st.button("Predict Fare"):

    if trip_distance <= 0:
        st.error("Trip distance must be greater than 0.")

    elif trip_duration <= 0:
        st.error("Trip duration must be greater than 0.")

    elif passenger_count < 1:
        st.error("Passenger count must be at least 1.")

    else:
        new_trip = pd.DataFrame({
            "trip_distance": [trip_distance],
            "trip_duration_minutes": [trip_duration],
            "passenger_count": [passenger_count],
            "pickup_hour": [pickup_hour],
            "pickup_dayofweek": [pickup_dayofweek],
            "is_weekend": [is_weekend],
            "PULocationID": [pickup_location],
            "DOLocationID": [dropoff_location],
            "RatecodeID": [ratecode]
        })

        prediction = model.predict(new_trip)[0]

        st.success(
            f"Predicted Total Amount: ${prediction:.2f}"
        )