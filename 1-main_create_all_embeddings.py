import os
import time
import create_fasttext_embeddings
import create_bert_embeddings
import create_plots
from memory_profiler import profile, memory_usage
import gc

@profile
def main(device_low, device_high, save_dir, vector_size = 768):
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
    device_list = devices_sorted[device_low:device_high]
    print(device_list)

    # Train the FastText model and create it's embeddings
    gc.collect()
    start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
    start_time = time.time()

    new_dir = os.path.join(save_dir, 'FastText')
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    model_filename = create_fasttext_embeddings.train_fasttext_model(file_path, device_list, new_dir, 1, vector_size)
    fast_text_training_time = time.time() - start_time
    fast_text_training_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

    gc.collect()
    start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
    start_time = time.time()

    seen_ft, unseen_ft = create_fasttext_embeddings.create_embeddings(model_filename, file_path, device_list, vector_size)
    fast_text_embeddings_creation_time = time.time() - start_time
    fast_text_embeddings_creation_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

    gc.collect()
    start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
    start_time = time.time()

    # Create BERT embeddings using pretrained model
    # devices_lengths = [seen, unseen]

    seen, unseen, temp = create_bert_embeddings.create_embeddings(file_path, device_list, vector_size)
    if temp is not None:
        bert_embeddings_creation_time = time.time() - start_time
        bert_embeddings_creation_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory
    else:
        bert_embeddings_creation_time = 0
        bert_embeddings_creation_mem_usage = 0

    total = seen + unseen
    if total == 0:
        total = seen_ft + unseen_ft
        unseen = unseen_ft
        seen = seen_ft

    # Per flow!
    if total != 0:
        times = (fast_text_training_time/unseen, fast_text_embeddings_creation_time/total, bert_embeddings_creation_time/total)
        memories = (fast_text_training_mem_usage/unseen, fast_text_embeddings_creation_mem_usage/total, bert_embeddings_creation_mem_usage/total)

    else:
        times = (0, 0, 0)
        memories = times

    return times, memories

def print_stats(stats_list, vector_list):
    print("-----------------------")

    # Define descriptions for each item in times and memories
    time_descriptions = ["FastText Training",
                         "FastText",
                         "BERT"]

    memory_descriptions = ["FastText Training",
                           "FastText",
                           "BERT"]

    # Printing the stats
    for vector, (times, memories) in zip(vector_list, stats_list):
        print(f"Stats for category: {vector}")

        # Print times with descriptions
        print("Times (sec):")
        for desc, item in zip(time_descriptions, times):
            print(f"{desc}: {item}")

        # Print memories with descriptions
        print("Memories (MB):")
        for desc, item in zip(memory_descriptions, memories):
            print(f"{desc}: {item}")

        print("-----------------------")

if __name__ == "__main__":
    # vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
    # vector_list = [128, 256, 512, 768]
    vector_list = [128]
    stats_list = []

    time_descriptions = ["FastText Training",
                         "FastText",
                         "BERT"]

    memory_descriptions = ["FastText Training",
                           "FastText",
                           "BERT"]

    # Analyzes devices device_low - device_high
    device_high = 5
    device_low = 0

    cwd = os.getcwd()

    for vector in vector_list:
        print(f"Creating embeddings at vector size: {vector}")
        
        save_dir = str(device_low) + "-" + str(device_high)
        save_dir = os.path.join(cwd, save_dir, str(vector))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        times, memories = main(device_low, device_high, save_dir, vector)
        stats_list.append((times, memories))
    print(stats_list)
    print_stats(stats_list, vector_list)
    create_plots.plot_graphs_embedder(stats_list, vector_list, time_descriptions, memory_descriptions)