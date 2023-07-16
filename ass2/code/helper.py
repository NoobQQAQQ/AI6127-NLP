import unicodedata
import re
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# construct datasets from files
def read_data(data_dir, src_lang):
    prefix = data_dir + 'news-commentary-v9.'
    src_file = prefix + src_lang + '-en.' + src_lang
    target_file = prefix + src_lang + '-en.en'
    with open(src_file, "rb") as f:
        src_sents = [_.strip().decode("utf-8") for _ in f.readlines()]
    with open(target_file, "rb") as f:
        target_sents = [_.strip().decode("utf-8") for _ in f.readlines()]
    assert (len(src_sents) == len(target_sents))
    return list(zip(src_sents, target_sents))


# a helper class to serve as vocabulary for different languages
class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    # s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# calculate the length of a sentence
def sentence_len(sent):
    return len(sent.split(' '))


def filter_pair(p, max_len):
    return sentence_len(p[0]) <= max_len and \
           sentence_len(p[1]) <= max_len


def filter_pairs(pairs, max_len):
    return [pair for pair in pairs if filter_pair(pair, max_len)]


# preprocess the dataset, including:
# 1. normalize text
# 2. count the max length of sentences in source language
# 3. make 2 vocabularies for source language and target language
def preprocess_data(dataset, src_lang, max_len, target_lang='en', filter=False):
    dataset = [[normalize_string(s) for s in pairs] for pairs in dataset]
    src_vocab, target_vocab = Vocab(src_lang), Vocab(target_lang)
    print("There are %s sentence pairs" % len(dataset))
    if filter:
        dataset = filter_pairs(dataset, max_len)
        print("Trimmed to %s sentence pairs" % len(dataset))
    max_len = 0
    for pair in dataset:
        src_vocab.add_sentence(pair[0])
        if sentence_len(pair[0]) > max_len:
            max_len = sentence_len(pair[0])
        target_vocab.add_sentence(pair[1])
    print("Counted words:")
    print(src_vocab.name, src_vocab.n_words)
    print(target_vocab.name, target_vocab.n_words)
    return src_vocab, target_vocab, dataset, max_len


# Randomly split a dataset into 5 subsets (S1 ~ S5)
def random_split_five(dataset):
    random.shuffle(dataset)
    length = len(dataset) // 5
    s1 = dataset[0:length]
    s2 = dataset[length:2 * length]
    s3 = dataset[2 * length:3 * length]
    s4 = dataset[3 * length:4 * length]
    s5 = dataset[4 * length:len(dataset)]
    return s1, s2, s3, s4, s5


# make train and test sets
def make_sets(indicator, s1, s2, s3, s4, s5):
    if indicator == 1:
        return s2 + s3 + s4 + s5, s1
    elif indicator == 2:
        return s1 + s3 + s4 + s5, s2
    elif indicator == 3:
        return s1 + s2 + s4 + s5, s3
    elif indicator == 4:
        return s1 + s2 + s3 + s5, s4
    elif indicator == 5:
        return s1 + s2 + s3 + s4, s5
    else:
        raise Exception('error in make_sets!')


# convert words in a sentence to their indexes in corresponding vocab
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ') if word in vocab.word2index]


def tensor_from_sentence(vocab, sentence):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, src_vocab, target_vocab):
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(target_vocab, pair[1])
    return input_tensor, target_tensor


# plot training curves
def show_plot(losses):
    x = range(1, len(losses) + 1)
    plt.xlabel('Pairs/50')
    plt.ylabel('Training Loss')
    plt.plot(x, losses, color="blue", linewidth=1, linestyle="-", label="train loss")
    plt.show()


def get_glove_matrix(filename, vocab, emb_dim):
    # get glove matrix from file
    filename = "../pretrain/" + filename + ".txt"
    glove = dict()
    with open(filename, 'rb') as f:
        for line in f.readlines():
            line = line.decode().split()
            if len(line) > 257:
                continue
            word = line[0]
            vec = np.array(line[1:]).astype(np.float)
            glove[word] = vec

    # use glove matrix to generate weights_matrix
    weights_matrix = np.random.rand(vocab.n_words, emb_dim)
    for idx, word in vocab.index2word.items():
        if word in glove:
            weights_matrix[idx] = glove[word]

    return weights_matrix


if __name__ == '__main__':
    # prepare data for glove pretrain
    # with open("../pretrain/news-commentary-v8.ru", "rb") as f:
    #     src_sents = [_.strip().decode("utf-8") for _ in f.readlines()]
    #     src_sents = [normalize_string(s) for s in src_sents]
    # with open("../pretrain/news-commentary-v8.ru", "w+", encoding="utf-8") as f:
    #     # write to file
    #     for s in src_sents:
    #         f.write(s + '\n')
    savename = "../table/" + "src_lang"
    savename += str(0)
    savename += str(0)
    savename += str(0)
    savename += '.csv'
    print(savename)
