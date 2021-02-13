# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle

class FinalModel(nn.Module):
    def __init__(self, char_emb_dim, word_emb_dim, hidden_dim, vocab_size, charset_size, tagset_size, window, l):
        super(FinalModel, self).__init__()
        self.char_embeddings = nn.Embedding(charset_size, char_emb_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim)
        self.conv1 = nn.Conv1d(char_emb_dim, l, window, padding=(window-1)//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(word_emb_dim+l, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, batch_sentence, batch_words):
        # Pass each window through CNN, max_pool the results for each word
        
        B, S = batch_sentence.shape
        
        # [B*S, W, c_emb]
        chars_batch = self.char_embeddings(batch_words)
        
        # [B*S, c_emb, W]
        chars_batch = chars_batch.permute(0,2,1)
        
        # [B*S, l, W]
        conv_out = self.conv1(chars_batch)
        conv_out = self.relu(conv_out)
        
        # [B*S, l]
        pool_out, _ = torch.max(conv_out, dim=2)
        
        # [B, S, l]
        cnn_word_vecs = pool_out.reshape((B, S, -1))
        
        # [B, S, w_emb]
        word_embeds = self.word_embeddings(batch_sentence)
        
        # [B, S, w_emb+l]
        concated = torch.cat((word_embeds, cnn_word_vecs), dim=2)
        
        # [B, S, hidden]
        lstm_out, _ = self.lstm(concated)
        
        # [B, S, T]
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores  

def read_dataset(file_path, labeled=True):
    """
    Read the dataset from file
    Args:
        file_path (str): path to the file to read from
        with_tags (bool): flag that indicates the presence of tags in data.
                          Use False to read test data.
    Returns:
        If with_tags is true, the list of tuples, one for each sentence
            One tuple contains list of lowercase words and corresponding list of tags
        Othervise the list of lowercase word lists, one fo each sentence
    """
    
    dataset = []
    with open(file_path, "r") as data_file:
        for line in data_file.readlines():
            # Split each sentence into items
            items = line[:-1].split(" ")
            if labeled:
                # If tags are present, create separate lists of words and tags
                words = []
                tags = []
                for item in items:
                    [word, tag] = item.rsplit("/", 1)
                    words.append(word.lower())
                    tags.append(tag)
                dataset.append((words, tags))
            else:
                # If tags are not present, append word list to the dataset
                dataset.append([word for word in items])
    return dataset


def prepare_sequence(sequence, 
                     dictionary, 
                     absent_key=None, 
                     pad_key=None, 
                     required_len=50, 
                     random_chance=0):  
    """
    Translate sequence according to dictionary.
    Args:
        sequence (list): list of keys
        dictionary (dict): mapping from key to integer
        absent_key (str): key which will substitute absent keys in sequence.
                            if None, absent keys will be ignored
        random_sub (bool): flag which indicatesthe need to randomly change keys in sequence 
                            with absent key with some chance (10% maybe)
                            if None, random substitution will not be used.
    Returns:
        list of transformed sequence
    """
    translated_seq = []
    for key in sequence:
        # Handle absent keys if absent_key specified
        if key not in dictionary:
            if absent_key is not None:
                translated_seq.append(dictionary[absent_key])
        # Random substitute if random_key specified
        elif absent_key is not None and torch.rand(1)[0]<random_chance:
            translated_seq.append(dictionary[absent_key])
        else:
            translated_seq.append(dictionary[key])
    if pad_key is not None:
        pad_len = required_len - len(translated_seq)
        translated_seq += [dictionary[pad_key]]*pad_len
    return torch.tensor(translated_seq, dtype=torch.long)


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file
    
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    SPECIAL_KEYS = model_data['special_keys']

    model = model_data['model']
    model.zero_grad()
    model.eval()

    test_dataset = read_dataset(test_file, labeled=False)
    with torch.no_grad():
        with open(out_file, "w") as out_f:
            for sent in test_dataset:
                lowered_sent = [w.lower() for w in sent]
                codded_sentence = prepare_sequence(lowered_sent,model_data['word_to_idx'],
                                                absent_key=SPECIAL_KEYS['absent']['word'],
                                                pad_key=None, random_chance=0)
                
                codded_words = []
                max_word_len = max([len(w) for w in sent])
                for word in sent:
                    codded_word = prepare_sequence(word,model_data['char_to_idx'],
                                                absent_key=SPECIAL_KEYS['absent']['char'],
                                                pad_key=SPECIAL_KEYS['padding']['char'],
                                                required_len=max_word_len,
                                                random_chance=0)

                    codded_words.append(codded_word) 
                codded_words = torch.stack(codded_words,dim=0)                         
                codded_sentence = codded_sentence.reshape((1,-1))

                tag_scores = model(codded_sentence, codded_words)
                tag_scores= tag_scores.reshape((-1,tag_scores.shape[-1]))
                pred = torch.argmax(tag_scores, dim=1)

                pred_taggs = [model_data['idx_to_tag'][int(t)] for t in pred]
                pairs = []
                for i in range(len(sent)):
                    pairs.append(sent[i]+"/"+pred_taggs[i])
                
                out_f.write(" ".join(pairs)+"\n")
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
