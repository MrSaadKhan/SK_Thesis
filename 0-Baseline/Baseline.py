import os
import json
import ipaddress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set your folder path
folder_path = '/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix/'

# Fields to exclude
exclude_fields = ['sourceMacAddress', 'destinationMacAddress', 'sourceIPv4Address']

# 1. List all JSON files and get the five smallest ones
files = os.listdir(folder_path)
files_with_size = [(file, os.path.getsize(os.path.join(folder_path, file))) for file in files if file.endswith('.json')]
smallest_files = sorted(files_with_size, key=lambda x: x[1])[:5]  # Get the five smallest files

# Initialize a list to hold filtered data and corresponding labels
all_filtered_data = []
device_labels = []

# Process each smallest file
for smallest_file in smallest_files:
    smallest_file_name = smallest_file[0]
    smallest_file_path = os.path.join(folder_path, smallest_file_name)
    print(f"Inspecting the smallest file: {smallest_file_name}\n")

    filtered_data = []

    def is_local_ip(ip):
        """Check if the given IP address is a private/local IP address."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False  # If it's not a valid IP, consider it not local

    # Extract device name from the filename (without extension)
    device_name = os.path.splitext(smallest_file_name)[0]
    device_labels.append(device_name)  # Collect device names for labeling

    with open(smallest_file_path, 'r') as f:
        for line in f:
            try:
                # Load each JSON object from the file
                json_data = json.loads(line)
                
                if "flows" in json_data:
                    flows_data = json_data["flows"]
                    
                    source_ip = flows_data.get('sourceIPv4Address')
                    destination_ip = flows_data.get('destinationIPv4Address')

                    # Check if both source and destination IPs are local
                    if not (is_local_ip(source_ip) and is_local_ip(destination_ip)):
                        # Remove the unwanted fields
                        filtered_flows = {key: value for key, value in flows_data.items() if key not in exclude_fields}
                        
                        # Append filtered data to the list
                        filtered_data.append(filtered_flows)

            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    # Append filtered DataFrame and corresponding labels for this device
    filtered_df = pd.DataFrame(filtered_data)
    if not filtered_df.empty:
        # Add device labels based on the number of rows in the filtered DataFrame
        device_name_series = pd.Series([device_name] * len(filtered_df))
        filtered_df['device_label'] = device_name_series  # Add the device label column
        all_filtered_data.append(filtered_df)  # Append filtered DataFrame to list

# Combine all filtered data into a single DataFrame
combined_df = pd.concat(all_filtered_data, ignore_index=True)

# Drop non-numeric columns
combined_numeric_df = combined_df.select_dtypes(include=[np.number])

# Check if there are any numeric fields left after filtering
if combined_numeric_df.empty:
    print("No numeric fields left for classification after filtering.")
else:
    # Drop rows with missing values
    combined_numeric_df = combined_numeric_df.dropna()

    # Check if there are any rows left after dropping NaNs
    if combined_numeric_df.empty:
        print("All data points dropped due to missing values after filtering.")
    else:
        # Prepare features and labels
        feature_columns = combined_numeric_df.columns.tolist()
        X = combined_numeric_df[feature_columns]

        # Create labels from the 'device_label' column in combined_df
        y = combined_df['device_label'].dropna()  # Get device labels ensuring they match

        # Ensure y matches the number of rows in X
        y = y[:len(X)]  # Trim y if it's longer than X

        # Encode the labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Create and train the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display classification report
        print("Classification report for combined files:\n")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Save confusion matrix as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')

        # Set transparent background
        plt.gcf().patch.set_facecolor('none')

        # Create a folder for saving the confusion matrix
        output_dir = os.path.join(os.getcwd(), 'confusion_matrices')
        os.makedirs(output_dir, exist_ok=True)

        # Save the normalized confusion matrix plot as SVG and PDF with 300 DPI
        plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.svg'), format='svg', bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.pdf'), format='pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

        print("Normalized confusion matrix saved successfully as SVG and PDF.")
