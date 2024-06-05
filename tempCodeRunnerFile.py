
    # Get a list of all devices in the directory
    all_devices = os.listdir(file_path)

    # Filter out devices that are in the exclusion list
    filtered_devices = [device for device in all_devices if device not in exclusion_list]

    # Sort devices by file size
    devices_sorted = sorted(filtered_devices, key=lambda device: os.path.getsize(os.path.join(file_path, device)))

    # Select the five smallest devices from the sorted list
    device_list = devices_sorted[:5]