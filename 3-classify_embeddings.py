import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

def classify_embeddings_random_forest(folder_path, output_name, vector_size):
    def load_embeddings(file_path):
        embeddings = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Loading embeddings from {os.path.basename(file_path)}', unit=' vectors'):
                vector = np.array([float(x) for x in line.strip().split()])
                embeddings.append(vector)
        return embeddings

    def read_specific_line(file_path, line_number):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for current_line_number, line in enumerate(file, start=1):
                    if current_line_number == line_number:
                        vector = line.strip().split()  # Split the line into a list of values
                        vector = [float(value) for value in vector]  # Convert values to float
                        return np.array(vector)  # Convert to numpy array and return
            print(f"Line {line_number} does not exist in the file.")
        except FileNotFoundError:
            print(f"The file at {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    # Test read specific line from the first file
    test_file_path = os.path.join(folder_path, sorted(os.listdir(folder_path))[0])
    test_line_number = 1
    # test_vector = read_specific_line(test_file_path, test_line_number)
    # if test_vector is not None:
    #     print(f"Test vector from line {test_line_number} of {test_file_path}: {test_vector}")

    # List of file paths in the folder
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.txt')]

    # Load embeddings and labels
    all_embeddings = []
    all_labels = []
    for i, file_path in enumerate(sorted(file_paths)):  # Sorted to ensure seen and unseen pairs are together
        device_embeddings = load_embeddings(file_path)
        labels = [i // 2] * len(device_embeddings)
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
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(all_labels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {output_name}')
    plt.savefig(f'{output_name}_confusion_matrix.png')  # Save figure with appropriate filename
    plt.show()


if __name__ == "__main__":
    file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b' 
    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

    embed_option = ["bert_embeddings", "fast_text_embeddings"]
    vector_size = 768

    embed_option = [f"{option}_{vector_size}" for option in embed_option]

    for option in embed_option:
        folder_path = os.path.join(file_path, option)
        classify_embeddings_random_forest(folder_path, option, vector_size)
