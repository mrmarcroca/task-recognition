import numpy as np
import re
import itertools
import json
# import csv, codecs, io
from collections import Counter
import itertools
from itertools import compress
# import tensorflow as tf
# from tensorflow.python.lib.io import file_io


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
    file_handle = open(path_to_json)
    file_content = file_handle.read()
    data = json.loads(file_content) # list. each element is a dictionary with keys 'message' and 'task'

    # with file_io.FileIO("data/hibox_tasks_train.json", 'r') as f:
    #     datas = json.load(f)
     

    # datas = list(open("./data/hibox_tasks_train.json", "r").readlines())
    # datas = [s.strip() for s in datas]

    # extract instant message and task label into separate lists
    ims = [obs['message'] for obs in data] #list comprehension for loop
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
    x_text = [word.split(" ") for word in x_text]
    y = np.array(list(compress(y, fil)))
    
    return [x_text, y]

# def load_data_and_labels():
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = list(open("./data/rt-polarity.pos", "r").readlines())
#     positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open("./data/rt-polarity.neg", "r").readlines())
#     negative_examples = [s.strip() for s in negative_examples]
#     # Split by words
#     x_text = positive_examples + negative_examples
#     x_text = [clean_str(sent) for sent in x_text]
#     x_text = [s.split(" ") for s in x_text]
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(path_to_json):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(path_to_json)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
