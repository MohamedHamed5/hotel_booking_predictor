

# Project Name: Hotel Booking Predictor
# Date: 5/21/25
# Created By: Strangers Team
# - Alaa Salah
# - Hagar Mahmoud
# - Mohamed Hamed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_and_preprocess_data(df_path=None, dataframe=None):

    if dataframe is not None:
        df = dataframe.copy()
    elif df_path:
        df = pd.read_csv(df_path)
    else:
        raise ValueError("Either df_path or dataframe must be provided.")


    df.drop_duplicates(inplace=True)

    # Fill/Convert specific columns (based on notebook cells 14, 19, 22, 26)
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0).astype('int64')
    if 'company' in df.columns:
        df.drop(columns=['company'], inplace=True)
    
    if 'agent' in df.columns and not df['agent'].empty:
        df['agent'] = df['agent'].fillna(df['agent'].mode()[0])
    
    if 'country' in df.columns and 'hotel' in df.columns and not df['country'].empty:
        df['country'] = df.groupby('hotel')['country'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else (df['country'].mode()[0] if not df['country'].mode().empty else 'Unknown')))
    elif 'country' in df.columns and not df['country'].empty: # Fallback if 'hotel' column is not present for grouping
        df['country'] = df['country'].fillna(df['country'].mode()[0] if not df['country'].mode().empty else 'Unknown')


    # Convert data types (based on notebook cell 14)
    for col in ['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type', 'deposit_type', 'agent']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
    
    if 'is_canceled' in df.columns:
        df['is_canceled'] = df['is_canceled'].astype('bool')
    if 'is_repeated_guest' in df.columns:
        df['is_repeated_guest'] = df['is_repeated_guest'].astype('bool')
    if 'required_car_parking_spaces' in df.columns:
        df['required_car_parking_spaces'] = df['required_car_parking_spaces'].astype('bool')

    # Create 'total_guests' (based on notebook cell 38)
    if all(col in df.columns for col in ['adults', 'children', 'babies']):
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        # Remove rows where adults, children, and babies are all zero, as these are likely invalid bookings.
        df = df[~((df['adults']==0) & (df['children']==0) & (df['babies']==0))]

    if 'adr' in df.columns:
        df.dropna(subset=['adr'], inplace=True) # Ensure ADR is not NaN before filtering
        df = df[df['adr'] >= 0] # Keep non-negative ADR values, including 0 for now, will be filtered in train_adr_model

    if 'is_canceled' in df.columns:
        df.dropna(subset=['is_canceled'], inplace=True)
        
    return df

# -----------------------------------------------------------------------------
# Model 1: Predict Booking Cancellation (is_canceled)
# -----------------------------------------------------------------------------
def train_cancellation_model(df):
    print("\n--- Training Cancellation Prediction Model ---")
    
    target = 'is_canceled'
    if target not in df.columns:
        print(f"Target column '{target}' not found in DataFrame.")
        return None, None

    features = [
        'hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month', 
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
        'babies', 'meal', 'market_segment', 'distribution_channel', 
        'is_repeated_guest', 'previous_cancellations',
        'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
        'agent', 'customer_type', 'adr', 'required_car_parking_spaces', 
        'total_of_special_requests', 'total_guests'
    ]
    
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        print("No suitable features found for cancellation model.")
        return None, None
        
    X = df[available_features].copy() # Use .copy() to avoid SettingWithCopyWarning
    y = df[target].astype(int) 

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')) 
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training cancellation model with {len(available_features)} features: {', '.join(available_features)}")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCancellation Model Accuracy (RandomForest): {accuracy:.4f}")
    print("Classification Report (RandomForest):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (RandomForest):")
    print(confusion_matrix(y_test, y_pred))
    
    return model_pipeline, available_features

# -----------------------------------------------------------------------------
# Model 2: Predict Average Daily Rate (ADR)
# -----------------------------------------------------------------------------
def train_adr_model(df):
    print("\n--- Training ADR Prediction Model ---")
    target = 'adr'

    if target not in df.columns:
        print(f"Target column '{target}' not found in DataFrame.")
        return None, None
    
    df_adr = df[df[target] > 0].copy() # Filter out ADR <= 0
    if df_adr.empty:
        print("No valid ADR data (ADR > 0) found for training.")
        return None, None

    features = [
        'hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'meal', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'reserved_room_type', 'assigned_room_type',
        'booking_changes', 'deposit_type', 'customer_type', 
        'required_car_parking_spaces', 'total_of_special_requests', 'total_guests'
    ]
    
    available_features = [f for f in features if f in df_adr.columns]
    if not available_features:
        print("No suitable features found for ADR model.")
        return None, None

    X = df_adr[available_features].copy()
    y = df_adr[target]

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training ADR model with {len(available_features)} features: {', '.join(available_features)}")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nADR Model Mean Squared Error: {mse:.4f}")
    print(f"ADR Model R-squared: {r2:.4f}")
    
    return model_pipeline, available_features

# -----------------------------------------------------------------------------
# Main execution block (for script-based training and evaluation)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting ML model training process (script-based)...")
    
    # Define the path to the data file
    # Assuming 'hotel_bookings.csv' is in the '/home/ubuntu/upload/' directory as per context
    data_file_path = r"F:\trained_models\hotel_bookings.csv"

    try:
        # Load the original dataframe
        original_df = pd.read_csv(data_file_path)
        print(f"Successfully loaded data from {data_file_path}")
        
        # Preprocess the data using the defined function
        print("Preprocessing data...")
        df_processed = load_and_preprocess_data(dataframe=original_df)
        print("Data preprocessing complete.")

        if df_processed is not None and not df_processed.empty:
            # Train and evaluate the cancellation model
            print("Training cancellation model...")
            cancel_model, cancel_features = train_cancellation_model(df_processed.copy()) # Pass a copy
            if cancel_model:
                print("Cancellation model training and evaluation complete.")
            else:
                print("Cancellation model training failed.")
            

        else:
            print("Data processing resulted in an empty DataFrame. Cannot train models.")

    except FileNotFoundError:
        print(f"Error: The data file was not found at {data_file_path}")
    except Exception as e:
        print(f"An error occurred during the script execution: {e}")
    
    print("\nML model script execution finished.")

