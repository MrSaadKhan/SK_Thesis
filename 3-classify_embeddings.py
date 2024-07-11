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
    device_names = []
    for fp in file_paths:
        # Extract device name from the file name before ".json"
        device_name = os.path.basename(fp).split('.json')[0].replace('_', ' ')
        device_name = ' '.join(word.capitalize() for word in device_name.split())
        device_names.append(device_name)
        
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
        # Extract device name from the file name before ".json"
        device_name = os.path.basename(file_path).split('.json')[0].replace('_', ' ')
        device_name = ' '.join(word.capitalize() for word in device_name.split())
        
        if device_name not in device_to_index:
            print(f"Device name '{device_name}' not found in device_to_index dictionary.")
            continue
        
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
    
    # Plotting the confusion matrix with labels and transparent background
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
    ax.set_title(f'Confusion Matrix - {" ".join(word.capitalize() for word in output_name.split("_"))}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(device_names, rotation=45)
    ax.set_yticklabels(device_names)
    plt.tight_layout()
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_{vector_size}.png', dpi=300, transparent=True)
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_{vector_size}.svg', dpi=300, transparent=True)
    # plt.show()

    return accuracy

def plot_accuracy_vs_vector_size(data):
    bert_data = [item for item in data if item[1] == 'bert_embeddings']
    fasttext_data = [item for item in data if item[1] == 'fast_text_embeddings']

    plt.figure(figsize=(8, 6))
    plt.plot([item[0] for item in bert_data], [item[2] for item in bert_data], marker='x', linestyle='-', color='b', label='BERT')
    plt.plot([item[0] for item in fasttext_data], [item[2] for item in fasttext_data], marker='o', linestyle='--', color='r', label='FastText')

    plt.xlabel('Vector Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Vector Size')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    # plt.show()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig('plots/classifier_accuracy.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/classifier_accuracy.svg', format='svg', dpi=300, transparent=True)

def main(vector_list):
    file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b'
    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

    stats_list = []

    time_descriptions = [
        "FastText Embeddings Classification Total Time",
        "BERT Embeddings Classification Total Time"
    ]
    memory_descriptions = [
        "FastText Embeddings Classification Total Memory Usage",
        "BERT Embeddings Classification Total Memory Usage"
    ]

    embed_options = ["bert_embeddings", "fast_text_embeddings"]  # Embedding options

    accuracy_list = []  # List to store accuracies

    for vector_size in vector_list:
        print(f"Classifying embeddings at vector size: {vector_size}")

        bert_embeddings_classification_time = 0
        bert_embeddings_classification_mem_usage = 0
        fast_text_embeddings_classification_time = 0
        fast_text_embeddings_classification_mem_usage = 0

        for option in embed_options:
            embed_name = f"{option}_{vector_size}"
            folder_path = os.path.join(file_path, embed_name)

            gc.collect()
            start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
            start_time = time.time()

            if os.path.exists(folder_path):
                accuracy = classify_embeddings_random_forest(folder_path, embed_name, vector_size)
                accuracy_list.append((vector_size, option, accuracy))
                print(f"Accuracy for {embed_name}: {accuracy}")

                if option.startswith("bert_embeddings"):
                    bert_embeddings_classification_time = time.time() - start_time
                    bert_embeddings_classification_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

                if option.startswith("fast_text_embeddings"):
                    fast_text_embeddings_classification_time = time.time() - start_time
                    fast_text_embeddings_classification_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

            else:
                print(f"{embed_name} does not exist!")

            print(f"Time taken: {time.time() - start_time:.2f} seconds")

    stats_list.append((
    (fast_text_embeddings_classification_time, bert_embeddings_classification_time),
    (fast_text_embeddings_classification_mem_usage, bert_embeddings_classification_mem_usage)
    ))
    print(stats_list)
    plot_accuracy_vs_vector_size(accuracy_list)
    create_plots.plot_graphs_classifier(stats_list, vector_list, time_descriptions, memory_descriptions)


if __name__ == "__main__":
    vector_list = [128, 768]
    main(vector_list)
