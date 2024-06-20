import os
import create_fasttext_embeddings
import create_bert_embeddings

def main(vector_size = 768):
    # Directory path to read files from
    file_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    # List of files to exclude
    exclusion_list = ['sony_network_camera.json', 'mouse_computer_room_hub.json', 'planex_camera_one_shot!.json']

    # Get a list of all devices in the directory
    all_devices = os.listdir(file_path)
    # Filter out devices that are in the exclusion list
    filtered_devices = [device for device in all_devices if device not in exclusion_list]
    # Sort devices by file size
    devices_sorted = sorted(filtered_devices, key=lambda device: os.path.getsize(os.path.join(file_path, device)))
    # Select the five smallest devices from the sorted list
    device_list = devices_sorted[:5]

    print(device_list)

    # Train the FastText model and create it's embeddings
    model_filename = create_fasttext_embeddings.train_fasttext_model(file_path, device_list, 1, vector_size)
    create_fasttext_embeddings.create_embeddings(model_filename, file_path, device_list, vector_size)

    # Create BERT embeddings using pretrained model
    create_bert_embeddings.create_embeddings(file_path, device_list, vector_size)

if __name__ == "__main__":
    vector_list = [128, 64, 32, 15, 5]
    for vector in vector_list:
        print(f"Creating embeddings at vector size: {vector}")
        main(vector)