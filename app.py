import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory."
        )
        return None, None


model, scaler = load_model_and_scaler()

# Title and description
st.title("✈️ Flight Price Prediction System")
st.markdown(
    """
This application predicts flight prices based on various factors such as airline, route, departure time, and more.
Fill in the details below to get an estimated price for your flight.
"""
)

# Sidebar for information
with st.sidebar:
    st.header("📊 About")
    st.info(
        """
    This model uses machine learning to predict flight prices based on:
    - Airline
    - Source and Destination Cities
    - Departure Time
    - Number of Stops
    - Flight Duration
    - Travel Class
    - Days Until Departure
    """
    )

    st.header("🎯 Model Performance")
    st.metric("R² Score", "98.5%")
    st.metric("Model", "Random Forest")

    st.header("💡 Tips")
    st.markdown(
        """
    - Book early for better prices
    - Non-stop flights cost more
    - Late night flights are cheaper
    - Business class costs 7-8x more
    - Prices increase closer to departure
    """
    )

# Main input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Flight Details")

    airline = st.selectbox(
        "Airline", ["SpiceJet", "AirAsia", "Vistara", "Air_India", "Indigo", "GO_FIRST"]
    )

    source_city = st.selectbox(
        "Source City",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
    )

    destination_city = st.selectbox(
        "Destination City",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
    )

    stops = st.selectbox("Number of Stops", ["zero", "one", "two_or_more"])

    travel_class = st.selectbox("Travel Class", ["Economy", "Business"])

with col2:
    st.subheader("Time & Duration Details")

    departure_time = st.selectbox(
        "Departure Time",
        ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
    )

    duration = st.number_input(
        "Flight Duration (hours)",
        min_value=0.5,
        max_value=24.0,
        value=2.5,
        step=0.25,
        help="Enter flight duration in hours (e.g., 2.5 for 2 hours 30 minutes)",
    )

    days_left = st.number_input(
        "Days Until Departure",
        min_value=0,
        max_value=365,
        value=15,
        step=1,
        help="Number of days from today until departure",
    )


# Feature mapping for encoding
def encode_features(
    airline,
    source_city,
    departure_time,
    stops,
    destination_city,
    travel_class,
    duration,
    days_left,
):
    # Create mappings (these match the LabelEncoder alphabetical sorting from training data)
    airline_map = {
        "AirAsia": 0,
        "Air_India": 1,
        "GO_FIRST": 2,
        "Indigo": 3,
        "SpiceJet": 4,
        "Vistara": 5,
    }
    source_map = {
        "Bangalore": 0,
        "Chennai": 1,
        "Delhi": 2,
        "Hyderabad": 3,
        "Kolkata": 4,
        "Mumbai": 5,
    }
    departure_map = {
        "Afternoon": 0,
        "Early_Morning": 1,
        "Evening": 2,
        "Late_Night": 3,
        "Morning": 4,
        "Night": 5,
    }
    stops_map = {"one": 0, "two_or_more": 1, "zero": 2}
    destination_map = {
        "Bangalore": 0,
        "Chennai": 1,
        "Delhi": 2,
        "Hyderabad": 3,
        "Kolkata": 4,
        "Mumbai": 5,
    }
    class_map = {"Business": 0, "Economy": 1}

    features = [
        airline_map.get(airline, 0),
        source_map.get(source_city, 0),
        departure_map.get(departure_time, 0),
        stops_map.get(stops, 0),
        destination_map.get(destination_city, 0),
        class_map.get(travel_class, 0),
        duration,
        days_left,
    ]

    return np.array(features).reshape(1, -1)


# Prediction button
st.markdown("---")
if st.button("🔮 Predict Flight Price", use_container_width=True):
    if model is None or scaler is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        if source_city == destination_city:
            st.error("⚠️ Source and destination cities cannot be the same!")
        else:
            # Prepare features
            input_features = encode_features(
                airline,
                source_city,
                departure_time,
                stops,
                destination_city,
                travel_class,
                duration,
                days_left,
            )

            # Scale features
            try:
                input_scaled = scaler.transform(input_features)

                # Make prediction
                prediction = model.predict(input_scaled)[0]

                # Convert to INR (from USD)
                prediction_inr = prediction * 87.04

                # Display result
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")

                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h2>Estimated Flight Price</h2>
                        <h1 style="color: #FF4B4B; font-size: 48px;">₹ {prediction_inr:,.0f}</h1>
                        <p style="color: #008080; font-size: 20px;">${prediction:,.0f} USD</p>
                        <p style="color: #666;">This is an estimated price based on historical data</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Additional insights
                st.markdown("---")
                st.markdown("### 💡 Price Insights")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if stops == "zero":
                        st.success(
                            "✅ Non-stop flight - Faster but typically costs more"
                        )
                    elif stops == "one":
                        st.info("ℹ️ One stop - Balanced price and time")
                    else:
                        st.warning("⚠️ Multiple stops - Cheaper but longer travel time")

                with col2:
                    if departure_time in ["Early_Morning", "Late_Night"]:
                        st.success("💰 Off-peak timing - Usually cheaper")
                    elif departure_time in ["Morning", "Evening"]:
                        st.warning("📈 Peak hours - Premium pricing")
                    else:
                        st.info("⏰ Mid-day flight - Moderate pricing")

                with col3:
                    if days_left < 7:
                        st.error("⚡ Last minute booking - Higher prices expected")
                    elif days_left < 30:
                        st.warning("📅 Book soon - Prices may increase")
                    else:
                        st.success("🎯 Early booking - Best prices!")

                # Price breakdown
                st.markdown("---")
                st.markdown("### 📊 Price Factors")

                factors_col1, factors_col2 = st.columns(2)

                with factors_col1:
                    st.markdown(
                        f"""
                    **Route:** {source_city} → {destination_city}  
                    **Airline:** {airline}  
                    **Class:** {travel_class}  
                    **Stops:** {stops.replace('_', ' ').title()}
                    """
                    )

                with factors_col2:
                    st.markdown(
                        f"""
                    **Departure:** {departure_time.replace('_', ' ')}  
                    **Duration:** {duration} hours  
                    **Days Left:** {days_left} days  
                    **Base Price:** ${prediction:,.2f}
                    """
                    )

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info(
                    "Please ensure all fields are filled correctly and the model files are compatible."
                )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with Streamlit • Machine Learning Model • Flight Price Prediction</p>
    <p>⚠️ Prices are estimates based on historical data and may vary from actual booking prices</p>
    <p>Model trained on 300K+ flight records from Indian airlines</p>
</div>
""",
    unsafe_allow_html=True,
)
