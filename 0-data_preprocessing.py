import os
import prepare_data

def save_individual_file(list_of_lists, name, path):
    path = os.path.join(path, f'{name}.txt')
    with open(path, 'w') as file:
        for inner_list in list_of_lists:
            line = ','.join(map(str, inner_list))
            file.write(line + '\n')
    
def save_files(seen, unseen, name, path):
    save_individual_file(seen, name + '_seen', path)
    save_individual_file(unseen, name + '_unseen', path)

def get_data(save_path, device):
    seen = read_file_as_list_of_lists(os.path.join(save_path, device + "_seen.txt"))
    unseen = read_file_as_list_of_lists(os.path.join(save_path, device + "_unseen.txt"))
    return seen, unseen

def read_file_as_list_of_lists(file_path):
    list_of_lists = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip newline characters and split the line into a list of words
                list_of_lists.append(line.strip().split())
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return list_of_lists

if __name__ == "__main__":
    file_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    cwd = os.getcwd()
    
    group_option=0
    time_group=0
    num2word_option=0   # Unlikely to be implemented
    
    if group_option == 0:
        group_path = 'ungrouped'
        save_path = os.path.join(cwd, 'preprocessed_data', group_path)

    else:
        group_path = 'grouped'
        time_path = str(time_group)
        save_path = os.path.join(cwd, 'preprocessed_data', group_path, time_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # List of files to exclude
    exclusion_list = ['sony_network_camera.json', 'mouse_computer_room_hub.json', 'planex_camera_one_shot!.json']

    device_high = 5
    device_low = 0

    all_devices = os.listdir(file_path)
    filtered_devices = [device for device in all_devices if device not in exclusion_list]
    devices_sorted = sorted(filtered_devices, key=lambda device: os.path.getsize(os.path.join(file_path, device)))
    device_list = devices_sorted[device_low:device_high]
    print(device_list)

    lengths = []
    for device in device_list:
        print(os.path.join(file_path, device))

        seen, unseen = prepare_data.prepare_data(os.path.join(file_path, device))

        length = ((len(seen), len(unseen)))
        print(f"Device: {device} \nSeen Length: {length[0]}\nUnseen Length: {length[1]}")
        save_files(seen, unseen, device, save_path)

        # lengths.append(length)

    for device in device_list:
        print(len(read_file_as_list_of_lists(os.path.join(save_path, device + "_seen.txt"))))

