import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

def load_pickle(filename):
    """Load data from a pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_mpu6050_data(csv_file, window_size=200, overlap=0.5):
    """
    Preprocess MPU6050 data for fall detection model
    """
    data = pd.read_csv(csv_file)
    
    if 'timestamp' in data.columns:
        data = data.drop('timestamp', axis=1)
    
    expected_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    if not all(col in data.columns for col in expected_columns):
        if len(data.columns) >= 6:
            column_mapping = {data.columns[i]: expected_columns[i] for i in range(6)}
            data = data.rename(columns=column_mapping)[expected_columns]
        else:
            raise ValueError("CSV file must have at least 6 columns for sensor data.")

    data['acc_magnitude'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
    data['gyro_magnitude'] = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)

    step = int(window_size * (1 - overlap))
    num_windows = (len(data) - window_size) // step + 1
    
    if num_windows <= 0:
        raise ValueError(f"Data too short ({len(data)} samples) for window size {window_size}")
    
    X = np.zeros((num_windows, window_size, len(data.columns)), dtype='float32')
    for i in range(num_windows):
        start_idx = i * step
        X[i] = data.iloc[start_idx:start_idx + window_size].values

    return X

def load_scaler(data_path):
    """Load the saved scaler or create a new one"""
    scaler_path = os.path.join(data_path, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    
    X_train = load_pickle(os.path.join(data_path, 'X_train.pkl'))
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler

def normalize_windows(X, scaler):
    """Normalize the data windows"""
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

def test_fall_detection(model_path, data_path, csv_file, window_size=200, overlap=0.5, threshold=0.5):
    """Test fall detection model"""
    model = load_model(model_path)
    X = preprocess_mpu6050_data(csv_file, window_size, overlap)
    X_normalized = normalize_windows(X, load_scaler(data_path))
    predictions_prob = model.predict(X_normalized)
    predictions = (predictions_prob > threshold).astype(int)
    return predictions_prob, predictions, X

import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(predictions_prob, predictions, windows, file_name, save_dir="resultsplot"):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename based on the uploaded file
    plot_filename = f"fall_detection_{os.path.splitext(file_name)[0]}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    num_windows = len(predictions_prob)

    # Convert to NumPy arrays if needed
    predictions_prob = np.array(predictions_prob).flatten()
    predictions = np.array(predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(range(num_windows), predictions_prob, label='Fall Probability', color='b')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')

    # Ensure fall detections are correctly extracted
    fall_indices = np.where(predictions == 1)[0]
    fall_probs = predictions_prob[fall_indices] if len(fall_indices) > 0 else []

    plt.scatter(fall_indices, fall_probs, color='red', marker='o', label='Detected Falls')

    plt.xlabel('Window Index')
    plt.ylabel('Fall Probability')
    plt.title(f'Fall Probability - {file_name}')
    plt.legend()

    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved at: {plot_path}")  # Debug print
        return plot_path
    except Exception as e:
        print(f"Error saving plot: {e}")  # Debug error
        return None  # Return None if saving fails
