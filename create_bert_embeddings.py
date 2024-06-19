import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import prepare_data
from tqdm import tqdm

def create_device_embedding(model, tokenizer, file_path, device, vector_size=768):
    embeddings_folder = "bert_embeddings"
    seen_embeddings_filename = os.path.join(embeddings_folder, device + "_seen_bert_embeddings.txt")
    unseen_embeddings_filename = os.path.join(embeddings_folder, device + "_unseen_bert_embeddings.txt")
    
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        return
    
    # seen_embeddings = []
    # unseen_embeddings = []

    def get_sentence_embedding(sentence, model, tokenizer, vector_size):
        sentence_str = ' '.join(sentence)
        inputs = tokenizer(sentence_str, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
        return cls_embedding

    device_file_path = os.path.join(file_path, device)
    seen, unseen = prepare_data.prepare_data(device_file_path)

    # for sentence in seen:
    #     embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
    #     print(embedding)
    #     seen_embeddings.append(embedding)

    # for sentence in unseen:
    #     embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
    #     unseen_embeddings.append(embedding)

    # with open(seen_embeddings_filename, 'w') as f:
    #     for embedding in seen_embeddings:
    #         f.write(' '.join(map(str, embedding.tolist())) + '\n')

    # with open(unseen_embeddings_filename, 'w') as f:
    #     for embedding in unseen_embeddings:
    #         f.write(' '.join(map(str, embedding.tolist())) + '\n')

    # with open(seen_embeddings_filename, 'w') as f_seen:
    #     for sentence in seen:
    #         embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
    #         f_seen.write(' '.join(map(str, embedding.tolist())) + '\n')

    # with open(unseen_embeddings_filename, 'w') as f_unseen:
    #     for sentence in unseen:
    #         embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
    #         f_unseen.write(' '.join(map(str, embedding.tolist())) + '\n')
    total_sentences = len(seen) + len(unseen)
    with open(seen_embeddings_filename, 'w') as f_seen, tqdm(total=total_sentences, desc=f'Processing {device}', unit='sentence') as pbar:
        for sentence in seen:
            embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
            f_seen.write(' '.join(map(str, embedding.tolist())) + '\n')
            pbar.update(1)

    with open(unseen_embeddings_filename, 'w') as f_unseen, tqdm(total=total_sentences, desc=f'Processing {device}', unit='sentence', initial=len(seen)) as pbar:
        for sentence in unseen:
            embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
            f_unseen.write(' '.join(map(str, embedding.tolist())) + '\n')
            pbar.update(1)
    print(f'Number of seen embeddings created: {len(seen)}')
    print(f'Number of unseen embeddings created: {len(unseen)}')

def create_embeddings(file_path, device_list):
    # model_name = 'bert-base-uncased'
    code_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

    if not os.path.exists(file_path):
        file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b'

    model_name = os.path.join(code_path, "bert_tiny")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    for device in device_list:
        create_device_embedding(model, tokenizer, file_path, device, 128)