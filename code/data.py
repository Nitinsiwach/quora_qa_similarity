



import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
import pandas as pd
from tqdm import *
import numpy as np
from os.path import join as pjoin
from spacy.lang.en import English
from nltk.tokenize import word_tokenize
from random import shuffle

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "quora")
    glove_dir = os.path.join("download", "dwr")
    data_dir = os.path.join("download", "quora")
    source_dir = os.path.join("data", "quora")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    parser.add_argument("--split", default = 0.8, type = float)
    parser.add_argument("--tokenizer", default = 'spaCy')
    parser.add_argument("--mode", default = 'train')
    parser.add_argument("--question_lower", default = 5) #this is to discard len(q1) + len(q2) < threshold
    parser.add_argument("--question_upper", default = 70) #this is to discard len(q1) or len(q2) > threshold
    return parser.parse_args()


def create_txt(data_path, train_path, val_path, test_path, split, question_lower, question_upper, mode):
    if mode == 'test':
        split = 1
    xin = pd.read_csv(data_path)
    xin_notnull = xin[~(xin.question1.isnull() | xin.question2.isnull())]
    xin_notnull = xin_notnull.sample(frac = 1).reset_index(drop = True)
    splitid = int(xin_notnull.shape[0]*split)
    xin_train = xin_notnull[:splitid+1]
    xin_val = xin_notnull[splitid:]

    question1_train_list = (xin_train.question1.str.replace("\n", "") + "\n").tolist()
    question2_train_list = (xin_train.question2.str.replace("\n", "") + "\n").tolist()
    question1_val_list = (xin_val.question1.str.replace("\n", "") + "\n").tolist()
    question2_val_list = (xin_val.question2.str.replace("\n", "") + "\n").tolist()

    train_upper_flag = np.array([False if (len(i.split()) > question_upper) or (len(j.split()) > question_upper) else True for i, j in zip(question1_train_list, question2_train_list)])

    val_upper_flag = np.array([False if (len(i.split()) > question_upper) or (len(j.split()) > question_upper) else True for i, j in zip(question1_val_list, question2_val_list)])
    question1_train = pd.Series(question1_train_list)[train_upper_flag].tolist()
    question2_train = pd.Series(question2_train_list)[train_upper_flag].tolist()
    question1_val = pd.Series(question2_val_list)[val_upper_flag].tolist()
    question2_val = pd.Series(question2_val_list)[val_upper_flag].tolist()

    question1_train = "".join(question1_train)
    question2_train = "".join(question2_train)
    question1_val = "".join(question1_val)
    question2_val = "".join(question2_val)
    labels_train = "".join(map(str, xin_train.is_duplicate[train_upper_flag].tolist()))
    labels_val = "".join(map(str, xin_val.is_duplicate[val_upper_flag].tolist()))

    if mode == 'train':
        with open(train_path + "\question1_train.txt", 'w', encoding = 'utf-8') as q1_t, \
            open(train_path + "\question2_train.txt", 'w', encoding = 'utf-8') as q2_t, \
            open(val_path + "\question1_val.txt", 'w', encoding = 'utf-8') as q1_v, \
            open(val_path + "\question2_val.txt", 'w', encoding = 'utf-8') as q2_v, \
            open(train_path + "\labels_train.txt", 'w', encoding = 'utf-8') as l_t, \
            open(val_path + "\labels_val.txt", 'w', encoding = 'utf-8') as l_v:
                q1_t.write(question1_train), \
                q2_t.write(question2_train), \
                q1_v.write(question1_val), \
                q2_v.write(question2_val), \
                l_t.write(labels_train), \
                l_v.write(labels_val)
    if mode == 'test':
        with open(test_path + "\question1_test.txt", 'w', encoding = 'utf-8') as q1_t, \
            open(test_path + "\question2_test.txt", 'w', encoding = 'utf-8') as q2_t, \
            open(test_path + "\labels_test.txt", 'w', encoding = 'utf-8') as l_t:
                q1_t.write(question1_train), \
                q2_t.write(question2_train), \
                l_t.write(labels_train)


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    '''arguments:
            vocabulary_path: vocabulary file with a token in each line
        retuns:
            vocab, rev_vocab
            vocab: a dictionary with signature {'vocab': idx}
            rev_vocab: a list of all the tokens in vocabulary_path
            There is 1 to 1 mapping between rev_vocab and vocab'''
    print('initializing vocabulary')
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    signature: 
    creates a numpy matrix glove with word vectors corresonding to tokens in vocab_list
    word vec for vocab_list[i] = glove[i]
    writes glove to save_path.npz
    
    :param vocab_list: [vocab]. a list of vocab
    :return:
    """
    print('procesing glove')
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        with open(glove_path, 'r', encoding = 'utf-8') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print(("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path)))
        np.savez_compressed(save_path, glove=glove)
        print(("saved trimmed glove matrix at: {}".format(save_path)))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    '''Iterates through all data_paths and creates a vocab of unique tokens 
    sorted according to their frequency in collective of data_paths
    writes it at vocabulary_path'''
    print('creating vocabulary')
    if not gfile.Exists(vocabulary_path):
        print(("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths))))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print(("processing line %d" % counter))
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print(("Vocabulary size: %d" % len(vocab_list)))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    '''converts sentence to a list  of their token ids according to vocabulary provided
    in case a token is not present it is replaced by token id for unk symbol'''
    if tokenizer:
        words = tokenizer(sentence)
        words = [word.orth_ for word in words if not word.orth_.isspace()]
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    '''converts file at data_path to a list of token_ids mapped 1 to 1 according to open(vocabulary_path)'''
    print('converting data to token ids')
    if not gfile.Exists(target_path):
        print(("Tokenizing data in %s" % data_path))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print(("tokenizing line %d" % counter))
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    
def get_tokenizer(name):
    if name == 'spaCy':
        tokenizer = English()
    if name == 'word_tokenize':
        tokenizer = word_tokenize
    return tokenizer

def basic_tokenizer(sentence):
    return sentence.split()

if __name__ == '__main__':
    args = setup_args()
    tokenizer = get_tokenizer(args.tokenizer)
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")
    data_path = pjoin(args.data_dir, "train.csv")
    train_path = pjoin(args.source_dir)
    valid_path = pjoin(args.source_dir)
    test_path = pjoin(args.source_dir)
    
    split = args.split
    
    create_txt(data_path, train_path, valid_path, test_path, split,args.question_lower, args.question_upper, 'train')
    
    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "question1_train.txt"),
                       pjoin(args.source_dir, "question2_train.txt"),
                      pjoin(args.source_dir, "question1_val.txt"),
                      pjoin(args.source_dir, "question2_val.txt")])
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
                  random_init=args.random_init)

    question1_train_ids_path = train_path + "/.ids.train.question1"
    question2_train_ids_path = train_path + "/.ids.train.question2"
    data_to_token_ids(train_path + "/question1_train.txt", question1_train_ids_path, vocab_path, tokenizer = tokenizer)
    data_to_token_ids(train_path + "/question2_train.txt", question2_train_ids_path, vocab_path, tokenizer = tokenizer)

    question1_val_ids_path = valid_path + "/.ids.val.question1"
    question2_val_ids_path = valid_path + "/.ids.val.question2"
    data_to_token_ids(valid_path + "/question1_val.txt", question1_val_ids_path, vocab_path, tokenizer = tokenizer)
    data_to_token_ids(valid_path + "/question2_val.txt", question2_val_ids_path, vocab_path, tokenizer = tokenizer)
    if args.mode == 'test':
        data_path = pjoin(args.data_dir, "test.csv")
        create_txt(data_path, train_path, valid_path, test_path, split,args.question_lower, args.question_upper, args.mode)
        question1_test_ids_path = test_path + "/.ids.test.question1"
        question2_test_ids_path = test_path + "/.ids.test.question2"
        data_to_token_ids(test_path + "/question1_test.txt", question1_test_ids_path, vocab_path, tokenizer = tokenizer)
        data_to_token_ids(test_path + "/question2_test.txt", question2_test_ids_path, vocab_path, tokenizer = tokenizer)