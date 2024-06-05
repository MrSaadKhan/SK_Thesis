import random
import math
import clean_data, group_data, number_to_words

def prepare_data(file_path, group_option=0, time_group=0, num2word_option=0):
    num_elements = []
    data = []

    # Clean the data for the single file
    temp, temp1 = clean_data.clean_data(file_path)
    data.append(temp)  # Append the cleaned data to the list
    num_elements.append(temp1)

    def split_list(lst, split_index):
        return lst[:split_index], lst[split_index:]

    def flatten(data):
        flattened_data = []
        for sublist in data:
            flattened_data.extend(sublist)
        return flattened_data

    flattened_data = flatten(data)
    dev1, _ = split_list(flattened_data, num_elements[0])  # Split the list (second part is unused)

    del data, flattened_data

    def random_split(lst, n):
        selected_indices = set(random.sample(range(len(lst)), n))
        selected_items = [lst[i] for i in selected_indices]
        remaining_items = [lst[i] for i in range(len(lst)) if i not in selected_indices]
        return selected_items, remaining_items

    def apply_group_data(dataset):
        if len(dataset) == 0:
            return [], []
        else:
            unseen, seen = random_split(dataset, math.floor(0.3 * len(dataset)))
            return group_data.group_data(unseen, time_group), group_data.group_data(seen, time_group)

    if group_option == 1:
        dev1_unseen, dev1_seen = apply_group_data(dev1)
    else:
        dev1_unseen, dev1_seen = random_split(dev1, math.floor(0.3 * num_elements[0]))

    del dev1

    if num2word_option == 1:
        dev1_seen = number_to_words.convert_numericals_to_words(dev1_seen)
        dev1_unseen = number_to_words.convert_numericals_to_words(dev1_unseen)

    print('\033[92mData prepared successfully ✔\033[0m')
    return dev1_seen, dev1_unseen
