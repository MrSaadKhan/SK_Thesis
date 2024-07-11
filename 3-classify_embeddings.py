import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
import time
from memory_profiler import memory_usage
import gc
import create_plots

# Mapping device names to their indices
def map_device_name(file_paths):
    device_names = [os.path.basename(fp).split('_')[0] for fp in file_paths]
    unique_devices = sorted(set(device_names))
    device_to_index = {device: idx for idx, device in enumerate(unique_devices)}
    return device_to_index

def classify_embeddings_random_forest(folder_path, output_name, vector_size):
    def load_embeddings(file_path):
        embeddings = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Loading embeddings from {os.path.basename(file_path)}', unit=' vectors'):
                vector = np.array([float(x) for x in line.strip().split()])
                embeddings.append(vector)
        return embeddings

    # List of file paths in the folder
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.txt')]

    # Map device names to indices
    device_to_index = map_device_name(file_paths)

    # Load embeddings and labels
    all_embeddings = []
    all_labels = []
    for file_path in sorted(file_paths):  # Sorted to ensure seen and unseen pairs are together
        device_name = os.path.basename(file_path).split('_')[0]
        device_index = device_to_index[device_name]
        device_embeddings = load_embeddings(file_path)
        labels = [device_index] * len(device_embeddings)
        all_embeddings.extend(device_embeddings)
        all_labels.extend(labels)

    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    print(f'Training set size: {len(X_train)}')
    print(f'Testing set size: {len(X_test)}')

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Training with progress bar
    clf.fit(X_train, y_train)
    print('Training completed.')

    # Print lengths of training and testing data used for classification
    print(f'Training data length for classification: {len(X_train)}')
    print(f'Testing data length for classification: {len(X_test)}')

    # Make predictions with progress bar
    y_pred = []
    for batch in tqdm(np.array_split(X_test, 10), desc='Classifying', unit=' batches'):
        y_pred.extend(clf.predict(batch))
    y_pred = np.array(y_pred)

    print(f"Evaluation of RF classifier at a vector size of {vector_size}")
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)

    # Confusion Matrix with device names as labels
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Convert to percentage
    device_names = sorted(device_to_index, key=device_to_index.get)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=device_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")  # Format values as floats with two decimal places
    # plt.title(f'Confusion Matrix - {output_name}')
    plt.title(f'Confusion Matrix - {" ".join(word.capitalize() for word in output_name.split("_"))}')
    plt.savefig(f'plots/{output_name}_confusion_matrix_{vector_size}.png')  # Save figure with appropriate filename
    plt.show()
    return accuracy

def main():
    file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b'
    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

    vector_list = [128]  # Vector sizes to evaluate
    embed_options = ["bert_embeddings", "fast_text_embeddings"]  # Embedding options

    accuracy_list = []  # List to store accuracies

    for vector_size in vector_list:
        print(f"Classifying embeddings at vector size: {vector_size}")

        for option in embed_options:
            embed_name = f"{option}_{vector_size}"
            folder_path = os.path.join(file_path, embed_name)

            gc.collect()
            start_time = time.time()

            if os.path.exists(folder_path):
                accuracy = classify_embeddings_random_forest(folder_path, embed_name, vector_size)
                accuracy_list.append((vector_size, option, accuracy))
                print(f"Accuracy for {embed_name}: {accuracy}")

            else:
                print(f"{embed_name} does not exist!")

            print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print("Accuracies:")
    print(accuracy_list)

if __name__ == "__main__":
    main()
