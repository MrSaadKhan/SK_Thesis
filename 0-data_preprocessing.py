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

    for device in device_list:
        print(os.path.join(file_path, device))
        seen, unseen = prepare_data.prepare_data(os.path.join(file_path, device))
        save_files(seen, unseen, device, save_path)


#  Requirements:
#  A proper file structure for all preprocessed data to exist.
#  Call prepare_data.prepare_data(filepath) for every device to process