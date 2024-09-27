import main_create_all_embeddings
import classify_embeddings


vector_list = [128, 256, 512, 768]
# vector_list      = [128]

device_low       = 0
# device_high      = 5
device_high      = [5]

group_option     = 0

time_group       = 0
num2word_option  = 0   # Unlikely to be implemented

window_group     = 1
window_size      = 10
slide_length     = 1

for device_high_option in device_high:

    main_create_all_embeddings.main_ext(vector_list, device_low, device_high_option, group_option, time_group, num2word_option, window_group, window_size, slide_length)
    classify_embeddings.main_ext(vector_list, device_low, device_high_option, group_option, time_group, num2word_option, window_group, window_size, slide_length)
