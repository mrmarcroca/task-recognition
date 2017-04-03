import json
import csv, codecs, io
import numpy as np
import re
import itertools
from itertools import compress
from collections import Counter
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"^teste$", "", string)
    string = re.sub(r"^test$", "", string)
    string = re.sub(r"^prueba$", "", string)
    string = re.sub(r"^prova$", "", string)
    string = re.sub(r'[\s]*https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE) # remove urls
    string = re.sub(r'[a-zA-Z0-9_]*@[a-zA-Z0-9_]*\.com[.[a-z]*]?','', string, flags=re.MULTILINE ) # remove emails
    string = re.sub(r'(\w{1})\1\1\1+',r'\1', string, flags=re.MULTILINE) # repeated letter >x3 goes to x1
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # replaces all characters not in this set with a space
    string = re.sub(r"\'s", " \'s", string) # puts a space before every    's
    string = re.sub(r"\'ve", " \'ve", string) # puts a space before every  've
    string = re.sub(r"n\'t", " n\'t", string) # puts a space before every  n't
    string = re.sub(r"\'re", " \'re", string) # puts a space before every  're
    string = re.sub(r"\'d", " \'d", string) # puts a space before every    'd
    string = re.sub(r"\'ll", " \'ll", string) # puts a space before every  'll
    string = re.sub(r",", " , ", string) # puts a space before and after every ,
    # replace multiple ! with one !
    string = re.sub(r"!", " ! ", string) # puts a space before and after every !
    string = re.sub(r"\(", " \( ", string) # puts a space before and after every (
    string = re.sub(r"\)", " \) ", string) # puts a space before and after every )
    string = re.sub(r"\?", " \? ", string) # puts a space before and after every ?
    string = re.sub(r"\s{2,}", " ", string) # converts 2 or more spaces to 1 space
    return string.strip().lower()


def load_data_and_labels(path_to_json):
    # read in json file
    #file_handle = open(path_to_json)
    #file_content = file_handle.read()
    #data = json.loads(file_content) # list. each element is a dictionary with keys 'message' and 'task'

    with file_io.FileIO(path_to_json, 'r') as f:
        data = json.load(f)

    # extract instant message and task label into separate lists
    ims = [obs['message'] for obs in data]
    ims = [im.strip() for im in ims]
    labs = [int(obs['task']) for obs in data]
    labs = [[1,0] if lab==0 else [0,1] for lab in labs]
    
    # clean ims
    x_text = [clean_str(im) for im in ims]
    
    # remove any examples that are more than 70 and less than 4 tokens long
    doc_lengths = [len(x.split(" ")) for x in x_text]
    fil = [lgth<70 and lgth>3 for lgth in doc_lengths]
    x_text = list(compress(x_text, fil))
    y = list(compress(labs, fil))
    
    # remove any examples that are less than 10 chars long (in case of short words)
    lengths = [len(x) for x in x_text]
    fil = [s>9 for s in lengths]
    x_text = list(compress(x_text, fil))
    y = np.array(list(compress(y, fil)))
    
    return [x_text, y]


def load_data_and_labels2(path_to_json):
    # read in json file
    file_handle = open(path_to_json)
    file_content = file_handle.read()
    file_content = re.sub(r"\sNULL", " null", file_content)
    data = json.loads(file_content) # list. each element is a dictionary with keys 'message' and 'task'
    
    # extract instant message and task label into separate lists
    ims = [obs['recognized_text'] for obs in data]
    ims = [im.strip() for im in ims]
    labs = [int(obs['status']) for obs in data]
    labs = [[0,1] if lab==2 else [1,0] for lab in labs]
    
    # clean ims
    x_text = [clean_str(im) for im in ims]
    
    # remove any examples that are less than 5 chars long
    lengths = [len(x) for x in x_text]
    fil = [s>5 for s in lengths]
    x_text = list(compress(x_text, fil))
    y = list(compress(labs, fil))
    
    # remove any examples that are more than 70 tokens long
    doc_lengths = [len(x.split(" ")) for x in x_text]
    fil2 = [lgth<70 and lgth>2 for lgth in doc_lengths]
    x_text = list(compress(x_text, fil2))
    y = np.array(list(compress(y, fil2)))
    
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
