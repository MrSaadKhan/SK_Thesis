import os
import time
import create_fasttext_embeddings
import create_bert_embeddings
from memory_profiler import profile, memory_usage

@profile
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
    start_memory = memory_usage(-1, interval=0.1)[0]
    start_time = time.time()

    model_filename = create_fasttext_embeddings.train_fasttext_model(file_path, device_list, 1, vector_size)
    fast_text_training_time = time.time() - start_time
    fast_text_training_mem_usage = memory_usage(-1, interval=0.1)[0] - start_memory

    start_memory = memory_usage(-1, interval=0.1)[0]
    start_time = time.time()

    create_fasttext_embeddings.create_embeddings(model_filename, file_path, device_list, vector_size)
    fast_text_embeddings_creation_time = time.time() - start_time
    fast_text_embeddings_creation_mem_usage = memory_usage(-1, interval=0.1)[0] - start_memory

    start_memory = memory_usage(-1, interval=0.1)[0]
    start_time = time.time()

    # Create BERT embeddings using pretrained model
    # devices_lengths = [seen, unseen]

    devices_lengths, temp = create_bert_embeddings.create_embeddings(file_path, device_list, vector_size)
    if temp is not None:
        bert_embeddings_creation_time = time.time() - fast_text_embeddings_creation_time
        bert_embeddings_creation_mem_usage = memory_usage(-1, interval=0.1)[0] - start_memory
    else:
        bert_embeddings_creation_time = 0
        bert_embeddings_creation_mem_usage = 0

    seen, unseen = devices_lengths
    total = seen + unseen

    # Per flow!
    times = (fast_text_training_time/unseen, fast_text_embeddings_creation_time/total, bert_embeddings_creation_time/total)
    memories = (fast_text_training_mem_usage/unseen, fast_text_embeddings_creation_mem_usage/total, bert_embeddings_creation_mem_usage/total)

    return times, memories

def print_stats(stats_list, vector_list):
    print("-----------------------")

    # Define descriptions for each item in times and memories
    time_descriptions = ["FastText Training Time per Flow",
                         "FastText Embeddings Creation Time per Total",
                         "BERT Embeddings Creation Time per Total"]

    memory_descriptions = ["FastText Training Memory Usage per Flow",
                           "FastText Embeddings Creation Memory Usage per Total",
                           "BERT Embeddings Creation Memory Usage per Total"]

    # Printing the stats
    for vector, (times, memories) in zip(vector_list, stats_list):
        print(f"Stats for category: {vector}")

        # Print times with descriptions
        print("Times:")
        for desc, item in zip(time_descriptions, times):
            print(f"{desc}: {item}")

        # Print memories with descriptions
        print("Memories:")
        for desc, item in zip(memory_descriptions, memories):
            print(f"{desc}: {item}")

        print("-----------------------")

if __name__ == "__main__":
    vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
    stats_list = []
    for vector in vector_list:
        print(f"Creating embeddings at vector size: {vector}")
        times, memories = main(vector)
        stats_list.append((times, memories))

    print_stats(stats_list, vector_list)