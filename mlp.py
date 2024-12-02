import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Function to preprocess data
def preprocess_user_data(data, relevant_keys, sample_size=1000, overlap=0.75):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    filtered_data = data[data['keyPressed'].isin(relevant_keys)].reset_index(drop=True)
    step = int(sample_size * (1 - overlap))
    samples = []
    for start_idx in range(0, len(filtered_data) - sample_size + 1, step):
        samples.append(filtered_data.iloc[start_idx:start_idx + sample_size])
    return samples

# Function to calculate key hold times
def calculate_hold_times(sample, relevant_keys):
    hold_times = {key: [] for key in relevant_keys}
    for key in relevant_keys:
        key_events = sample[sample['keyPressed'] == key]
        for _, down_event in key_events[key_events['keyState'] == 'DOWN'].iterrows():
            up_event = key_events[
                (key_events['keyState'] == 'UP') & (key_events['timestamp'] > down_event['timestamp'])
            ]
            if not up_event.empty:
                hold_time = (up_event.iloc[0]['timestamp'] - down_event['timestamp']).total_seconds()
                hold_times[key].append(hold_time)
    return hold_times

# Function to remove outliers
def remove_outliers(hold_times):
    clean_hold_times = {}
    for key, times in hold_times.items():
        if times:
            q1 = pd.Series(times).quantile(0.25)
            q3 = pd.Series(times).quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            clean_hold_times[key] = [time for time in times if lower_bound <= time <= upper_bound]
        else:
            clean_hold_times[key] = []
    return clean_hold_times

# Function to compute features
def compute_features(hold_times):
    features = {}
    for key, times in hold_times.items():
        if times:
            features[f"{key}_mean"] = sum(times) / len(times)
            features[f"{key}_std"] = pd.Series(times).std(ddof=0)
        else:
            features[f"{key}_mean"] = 0
            features[f"{key}_std"] = 0
    return features

# Load and preprocess the dataset
dataset_dir = '/Users/aniketmali/Desktop/sem 1/Topics in Information Security/Assignement/Assignment 2/finalAttempt/desktopKeyboardDataset'  # Replace with your dataset folder path
relevant_keys = {"t", "a", "e", "i", "h"}
all_features = []

# Process all users' data
user_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
for user_file in user_files:
    user_data_path = os.path.join(dataset_dir, user_file)
    user_data = pd.read_csv(user_data_path)
    samples = preprocess_user_data(user_data, relevant_keys)
    for sample in samples:
        hold_times = calculate_hold_times(sample, relevant_keys)
        hold_times_cleaned = remove_outliers(hold_times)
        features = compute_features(hold_times_cleaned)
        features['userID'] = user_file.split('.')[0]
        all_features.append(features)

# Create DataFrame from features
features_df = pd.DataFrame(all_features)

# Prepare features and labels
X = features_df.drop(columns=["userID"]).values
y = features_df["userID"].values

# Encode labels and normalize features
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define 3-fold cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

# Define the MLP model
def create_mlp(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    return model

# Updated MLP model with Dropout
def create_mlp_with_dropout(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer with 20% rate
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer with 20% rate
    model.add(Dense(output_dim, activation='softmax'))
    return model

# Train and evaluate the model using cross-validation directly
fold_accuracies = []
all_histories = []
# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


for train_idx, test_idx in skf.split(X_scaled, y_encoded):
    # Split the data
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
    
    # Create and compile the model
    model = create_mlp_with_dropout(input_dim=X_scaled.shape[1], output_dim=y_categorical.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\nStarting training for the next fold...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),  # Use the test set as the validation set for this fold
        epochs=200,
        batch_size=16,
        verbose=1,
        callbacks=[early_stopping]  # Include early stopping
    )
    all_histories.append(history.history)  # Save the history for plotting
    
    # Evaluate the model
    print("\nEvaluating the model...")
    _, accuracy = model.evaluate(X_test, y_test, verbose=1)
    fold_accuracies.append(accuracy)

# Calculate and print average accuracy
average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Accuracy across folds: {average_accuracy:.2f}")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(all_histories[0]['loss'], label='Training Loss')
plt.plot(all_histories[0]['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(all_histories[0]['accuracy'], label='Training Accuracy')
plt.plot(all_histories[0]['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
