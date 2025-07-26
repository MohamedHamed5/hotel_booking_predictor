

# Project Name: Hotel Booking Predictor
# Date: 21/5/25
# Created By: Strangers Team
# - Alaa Salah
# - Hagar Mahmoud
# - Mohamed Hamed


import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from ml_models1 import load_and_preprocess_data, train_cancellation_model, train_adr_model

# Define paths for saved models
MODELS_DIR = "trained_models"
CANCEL_MODEL_PATH = os.path.join(MODELS_DIR, "cancellation_model.joblib")
CANCEL_FEATURES_PATH = os.path.join(MODELS_DIR, "cancellation_features.joblib")
ADR_MODEL_PATH = os.path.join(MODELS_DIR, "adr_model.joblib")
ADR_FEATURES_PATH = os.path.join(MODELS_DIR, "adr_features.joblib")

# --- Helper function to ensure models are trained/loaded ---
@st.cache_resource # Caches the resource (models, features) across sessions
def initialize_models(data_path="hotel_bookings.csv"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    c_model, c_features, a_model, a_features = None, None, None, None
    models_ready = False

    try:
        if os.path.exists(CANCEL_MODEL_PATH) and os.path.exists(CANCEL_FEATURES_PATH):
            c_model = joblib.load(CANCEL_MODEL_PATH)
            c_features = joblib.load(CANCEL_FEATURES_PATH)
            print("Cancellation model loaded from disk.")
        if os.path.exists(ADR_MODEL_PATH) and os.path.exists(ADR_FEATURES_PATH):
            a_model = joblib.load(ADR_MODEL_PATH)
            a_features = joblib.load(ADR_FEATURES_PATH)
            print("ADR model loaded from disk.")
        if c_model and a_model:
            models_ready = True
    except Exception as e:
        st.error(f"Error loading models: {e}. Will attempt to retrain.")
        c_model, a_model = None, None # Reset to ensure retraining

    if not models_ready:
        st.warning("Models not found or failed to load. Training new models...")
        if not os.path.exists(data_path):
            st.error(f"Data file '{data_path}' not found. Cannot train models.")
            return None, None, None, None, False
        try:
            with st.spinner("Loading and processing data for model training... This may take a few minutes."):
                raw_df = pd.read_csv(data_path)
                df_processed = load_and_preprocess_data(dataframe=raw_df.copy())

            if df_processed is not None and not df_processed.empty:
                with st.spinner("Training cancellation prediction model..."):
                    c_model, c_features = train_cancellation_model(df_processed.copy())
                if c_model:
                    joblib.dump(c_model, CANCEL_MODEL_PATH)
                    joblib.dump(c_features, CANCEL_FEATURES_PATH)
                    st.success("Cancellation model trained and saved successfully.")
                else:
                    st.error("Failed to train cancellation model.")
                    return None, None, None, None, False

                with st.spinner("Training ADR prediction model..."):
                    a_model, a_features = train_adr_model(df_processed.copy())
                if a_model:
                    joblib.dump(a_model, ADR_MODEL_PATH)
                    joblib.dump(a_features, ADR_FEATURES_PATH)
                    st.success("ADR model trained and saved successfully.")
                else:
                    st.error("Failed to train ADR model.")
                    return None, None, None, None, False
                models_ready = True
            else:
                st.error("Data processing failed. Cannot train models.")
                return None, None, None, None, False
        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            return None, None, None, None, False
            
    return c_model, c_features, a_model, a_features, models_ready

# --- Streamlit App Layout ---
st.set_page_config(page_title="Hotel Booking Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("🏨 Hotel Booking Predictor")
st.markdown("""
    <style>
        .stApp {
            background-color: #F5F5DC; # Beige background
        }
        .stButton>button {
            background-color: #8B4513; # Brown color
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 5px;
            border: 1px solid #D3D3D3; # Light Gray
        }
        h1, h2, h3, .stMarkdown p {
            color: #800000 !important; # Maroon color for titles
        }
        .result-cancel-yes, .result-cancel-no, .result-adr {
            color: #8B4513;
            font-weight: bold;
        }
        .custom-notification {
            background-color: #F0E6CC;
            border-left: 5px solid #800000;
            color: #800000;
            font-weight: bold;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-notification">
    Please enter booking details in the sidebar and click <b>'Predict Now'</b> to see results.
</div>
""", unsafe_allow_html=True)

# Initialize models
cancellation_model, cancellation_features, adr_model, adr_features, models_initialized = initialize_models()

if not models_initialized:
    st.error("Failed to initialize models. Please check the console for errors and try refreshing the page.")
    st.stop() # Stop the app if models aren't ready

st.sidebar.header("📝 Booking Information Input")

# Define input fields
fields_definition = [
    ("Hotel Type", "hotel", "dropdown", ["City Hotel", "Resort Hotel"], "Hotel Type:"),
    ("Lead Time (days)", "lead_time", "number_input", 100, "Waiting Time:"),
    ("Arrival Year", "arrival_date_year", "dropdown", [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "Year of Arrival:"),
    ("Arrival Month", "arrival_date_month", "dropdown", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], " Month of Arrival:"),
    ("Weekend Nights", "stays_in_weekend_nights", "number_input", 1, "Weekend Nights:"),
    ("Weekday Nights", "stays_in_week_nights", "number_input", 2, "Weekday Nights:"),
    ("Number of Adults", "adults", "number_input", 2, "Number of adults:"),
    ("Number of Children", "children", "number_input", 0, "Number of children:"),
    ("Number of Babies", "babies", "number_input", 0, "Number of babies:"),
    ("Meal Type", "meal", "dropdown", ["BB", "FB", "HB", "SC", "Undefined"], "Meal Type:"),
    ("Market Segment", "market_segment", "dropdown", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"], "Market segment:"),
    ("Distribution Channel", "distribution_channel", "dropdown", ["TA/TO", "Direct", "Corporate", "GDS"], "Distribution channel:"),
    ("Repeated Guest?", "is_repeated_guest", "dropdown", [0, 1], "Is the guest returning? (0 = No, 1 = Yes)"),
    ("Previous Cancellations", "previous_cancellations", "number_input", 0, "Number of previous cancellations:"),
    ("Reserved Room Type", "reserved_room_type", "dropdown", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"], "Type of room booked:"),
    ("Assigned Room Type", "assigned_room_type", "dropdown", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"], "Type of assigned room:"),
    ("Booking Changes", "booking_changes", "number_input", 0, "Booking Changes:"),
    ("Deposit Type", "deposit_type", "dropdown", ["No Deposit", "Non Refund", "Refundable"], "Deposit Type:"),
    ("Customer Type", "customer_type", "dropdown", ["Transient", "Transient-Party", "Contract", "Group"], "Customer Type:"),
    ("Car Parking Required?", "required_car_parking_spaces", "dropdown", [0, 1], "Is parking required? (0 = No, 1 = Yes)"),
    ("Total Requests", "total_of_special_requests", "number_input", 0, "Total Special Requests:"),
    ("Average Daily Rate (if known)", "adr", "number_input", 100.0, "Average daily rate (if known)", "float"),
]

input_data = {}
for label_en, key, field_type, options_or_default, help_text, *data_type_hint in fields_definition:
    if field_type == "number_input":
        val_type = float if len(data_type_hint) > 0 and data_type_hint[0] == "float" else int
        input_data[key] = st.sidebar.number_input(
        label_en, 
        value=val_type(options_or_default), 
        min_value=(0.0 if val_type == float else 0),
        help=help_text, 
        step=(0.1 if val_type == float else 1)
)
    elif field_type == "dropdown":
        # Ensure options are strings for display if they are not already
        str_options = [str(opt) for opt in options_or_default]
        selected_option = st.sidebar.selectbox(label_en, options=str_options, index=str_options.index(str(options_or_default[0])), help=help_text)
        # Convert back to original type if necessary (e.g., int for 0/1)
        if all(isinstance(opt, int) for opt in options_or_default):
            input_data[key] = int(selected_option)
        else:
            input_data[key] = selected_option

if st.sidebar.button("🔮 Predict Now!"):
    if not cancellation_model or not adr_model:
        st.error("Models are not ready. Please try again or refresh the page.")
    else:
        # Create DataFrame for prediction
        predict_df = pd.DataFrame([input_data])
        
        # Add total_guests based on inputs
        if all(col in predict_df.columns for col in ["adults", "children", "babies"]):
             predict_df["total_guests"] = predict_df["adults"] + predict_df["children"] + predict_df["babies"]
        else: # if any of these are missing, we can't compute total_guests, model might fail
            st.warning("Insufficient data to calculate total guests. This may affect prediction accuracy.")
            if "total_guests" in cancellation_features or "total_guests" in adr_features:
                 predict_df["total_guests"] = 0 # Default or handle as NaN if preprocessor can

        st.subheader("📈 Prediction Results:")
        col1, col2 = st.columns(2)

        try:
            # Cancellation Prediction
            input_df_cancel_processed = pd.DataFrame(columns=cancellation_features)
            for col_feature in cancellation_features:
                if col_feature in predict_df.columns:
                    input_df_cancel_processed[col_feature] = predict_df[col_feature]
                else:
                    # This is a fallback. Ideally, the preprocessor handles missing columns if trained to do so.
                    input_df_cancel_processed[col_feature] = np.nan 
            
            cancel_pred_proba = cancellation_model.predict_proba(input_df_cancel_processed[cancellation_features])
            cancel_prediction = (cancel_pred_proba[:, 1] >= 0.5).astype(int)[0]
            cancel_probability = cancel_pred_proba[0][1]

            with col1:
                st.markdown("**Cancellation Prediction:**")
                if cancel_prediction == 1:
                    st.markdown(f"<p class='result-cancel-yes'>Yes, this booking is likely to be canceled.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='result-cancel-no'>No, this booking is unlikely to be canceled.</p>", unsafe_allow_html=True)
                st.markdown(f"*Cancellation probability: {cancel_probability:.2%}*", unsafe_allow_html=True)

        except Exception as e:
            with col1:
                st.error(f"Error in cancellation prediction: {e}")
            print(f"Cancellation prediction error: {e}")
            import traceback
            traceback.print_exc()

        try:
            # ADR Prediction
            input_df_adr_processed = pd.DataFrame(columns=adr_features)
            for col_feature in adr_features:
                if col_feature in predict_df.columns:
                    input_df_adr_processed[col_feature] = predict_df[col_feature]
                else:
                    input_df_adr_processed[col_feature] = np.nan

            adr_prediction = adr_model.predict(input_df_adr_processed[adr_features])[0]
            with col2:
                st.markdown("**Average Daily Rate (ADR) Prediction:**")
                st.markdown(f"<p class='result-adr'>Predicted price: ${adr_prediction:.2f}</p>", unsafe_allow_html=True)
        
        except Exception as e:
            with col2:
                st.error(f"Error in ADR prediction: {e}")
            print(f"ADR prediction error: {e}")
            import traceback
            traceback.print_exc()

st.markdown("--- ")
st.markdown("Developed by Strangers Team")
st.markdown("IEEE Kafrelsheikh Student Branch Python Committee '25.")
