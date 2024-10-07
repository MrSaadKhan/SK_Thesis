import os

def print_files_with_mapping(directory, input_output_map):
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        
        # Print the mapped output for each file
        for file in files:
            if os.path.isfile(os.path.join(directory, file)):  # Check if it is a file
                # Remove the .json extension and format the name
                file_name = file[:-5].replace('_', ' ').title()
                # Get the mapped output if it exists
                mapped_output = input_output_map.get(file_name, file_name)  # Default to original name if not found
                print(mapped_output)
    except Exception as e:
        print(f"An error occurred: {e}")

# Mapping from the input file names to desired output
input_output_map = {
    "Amazon Echo Gen2"                      : "Amazon Echo Gen2",
    "Au Network Camera"                     : "Network Camera",
    "Au Wireless Adapter"                   : "Wireless Adapter",
    "Bitfinder Awair Breathe Easy"          : "Bitfinder Smart Air Monitor",
    "Candy House Sesami Wi-fi Access Point" : "Candy House Wi-Fi AP",
    "Google Home Gen1"                      : "Google Home Gen1",
    "I-o Data Qwatch"                       : "IO Data Camera",
    "Irobot Roomba"                         : "iRobot Roomba",
    "Jvc Kenwood Cu-hb1"                    : "JVC Smart Home Hub",
    "Jvc Kenwood Hdtv Ip Camera"            : "JVC Camera",
    "Line Clova Wave"                       : "Line Smart Speaker",
    "Link Japan Eremote"                    : "Link eRemote",
    "Mouse Computer Room Hub"               : "Mouse Computer Room Hub",
    "Nature Remo"                           : "Nature Smart Remote",
    "Panasonic Doorphone"                   : "Panasonic Doorphone",
    "Philips Hue Bridge"                    : "Philips Hue Light",
    "Planex Camera One Shot!"               : "Planex Camera",
    "Planex Smacam Outdoor"                 : "Planex Outdoor Camera",
    "Planex Smacam Pantilt"                 : "Planex PanTilt Camera",
    "Powerelectric Wi-fi Plug"              : "PowerElectric Wi-Fi Plug",
    "Qrio Hub"                              : "Qrio Hub",
    "Sony Bravia"                           : "Sony Bravia",
    "Sony Network Camera"                   : "Sony Network Camera",
    "Sony Smart Speaker"                    : "Sony Smart Speaker",
    "Xiaomi Mijia Led"                      : "Xiaomi Mijia LED"
}

# Specify the directory you want to list
directory_path = '/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix/'
print_files_with_mapping(directory_path, input_output_map)
