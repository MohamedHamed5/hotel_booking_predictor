

# Project Name: Hotel Booking Predictor
# Date: 5/21/25
# Created By: Strangers Team
# - Alaa Salah
# - Hagar Mahmoud
# - Mohamed Hamed

import pandas as pd
import os
import joblib
from ml_models1 import load_and_preprocess_data, train_cancellation_model, train_adr_model

# Define paths for saved models
MODELS_DIR = "trained_models"
CANCEL_MODEL_PATH = os.path.join(MODELS_DIR, "cancellation_model.joblib")
CANCEL_FEATURES_PATH = os.path.join(MODELS_DIR, "cancellation_features.joblib")
ADR_MODEL_PATH = os.path.join(MODELS_DIR, "adr_model.joblib")
ADR_FEATURES_PATH = os.path.join(MODELS_DIR, "adr_features.joblib")

DATA_PATH = "hotel_bookings.csv" # Path to the raw data file

def pretrain_and_save_models():
    print("Starting model pre-training process...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Data file 	'{DATA_PATH}	' not found. Cannot train models.")
        return False

    try:
        print(f"Loading and preprocessing data from {DATA_PATH}...")
        raw_df = pd.read_csv(DATA_PATH)
        df_processed = load_and_preprocess_data(dataframe=raw_df.copy())
        print("Data loaded and preprocessed successfully.")

        if df_processed is not None and not df_processed.empty:
            print("Training cancellation model...")
            cancellation_model, cancellation_features = train_cancellation_model(df_processed.copy())
            if cancellation_model and cancellation_features is not None:
                joblib.dump(cancellation_model, CANCEL_MODEL_PATH)
                joblib.dump(cancellation_features, CANCEL_FEATURES_PATH)
                print(f"Cancellation model trained and saved to {CANCEL_MODEL_PATH}")
            else:
                print("Failed to train cancellation model.")
                return False

            print("Training ADR model...")
            adr_model, adr_features = train_adr_model(df_processed.copy())
            if adr_model and adr_features is not None:
                joblib.dump(adr_model, ADR_MODEL_PATH)
                joblib.dump(adr_features, ADR_FEATURES_PATH)
                print(f"ADR model trained and saved to {ADR_MODEL_PATH}")
            else:
                print("Failed to train ADR model.")
                return False
            
            print("All models pre-trained and saved successfully!")
            return True
        else:
            print("Data processing failed. Cannot train models.")
            return False
    except Exception as e:
        print(f"An error occurred during model pre-training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    pretrain_and_save_models()

