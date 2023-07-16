import pycrfsuite
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from itertools import chain


def read_data(filepath):
    # read data line by line
    with open(filepath, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]  # remove white space

    # get split line No. of sentences
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if not _])

    # get information sentence by sentence
    sentences = []
    for j in range(len(index) - 1):
        sent = []
        segment = content[index[j] + 1: index[j + 1]]
        for line in segment:
            sent.append(line.split())
        sentences.append(sent)

    return sentences


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'postag': postag,
        #'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #'-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        # Indicate that it is the 'beginning of a document'
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #'+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        # Features for words that are not at the end of a document
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    # return [label for token, postag, label in sent]
    return [label for token, label in sent]


def get_stat(pred, true, table):
    for i in range(len(pred)):
        # single label
        if len(true[i]) == 1:
            true_label = true[i][0]
            if pred[i] == true_label:
                table[label2id[pred[i]]][0] = table[label2id[pred[i]]][0] + 1  # TP + 1
            else:
                table[label2id[pred[i]]][2] = table[label2id[pred[i]]][2] + 1  # FP + 1
                table[label2id[true_label]][1] = table[label2id[true_label]][1] + 1  # FN + 1
        # multi-label
        else:
            for true_label in true[i]:
                if pred[i] == true_label:
                    table[label2id[pred[i]]][0] = table[label2id[pred[i]]][0] + 1  # TP + 1
                else:
                    table[label2id[pred[i]]][2] = table[label2id[pred[i]]][2] + 1  # FP + 1
                    table[label2id[true_label]][1] = table[label2id[true_label]][1] + 1  # FN + 1


def print_f1(table):
    tp = table[:, 0]
    fn = table[:, 1]
    fp = table[:, 2]
    table[:, 3][tp != 0] = tp[tp != 0] / (tp[tp != 0] + fp[tp != 0])
    table[:, 4][tp != 0] = tp[tp != 0] / (tp[tp != 0] + fn[tp != 0])
    p = table[:, 3]
    r = table[:, 4]
    table[:, 5][(p+r) != 0] = 2 * p[(p+r) != 0] * r[(p+r) != 0] / (p[(p+r) != 0] + r[(p+r) != 0])
    table[:, 6] = tp + fn
    # calculate macro F1 and weighted F1 (remove 'O')
    O_index = label2id['O']
    removed_f1 = np.delete(table[:, 5], O_index)
    removed_support = np.delete(table[:, 6], O_index)
    total = np.sum(removed_support)
    wf1 = np.sum((removed_support / total) * removed_f1)
    print(f'macro/weighted F1 (remove O): {np.mean(removed_f1)}/{wf1}')


if __name__ == '__main__':
    # read datasets from file
    train_sents = read_data('../data/wnut17train.conll')
    test_sents = read_data("../data/emerging.test.conll")

    # get training/testing datasets
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # construct label vocabulary
    vocab_labels = build_vocab_from_iterator(y_train)
    label2id = vocab_labels.get_stoi()

    # training phase
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.train('wnut17.crfsuite')

    # evaluation phase
    tagger = pycrfsuite.Tagger()
    tagger.open('wnut17.crfsuite')
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    # print result
    # generate a table to count f1 scores, shape of table: number of class by 3(TP, FN, FP, P, R, F1, support)
    f1_table = np.zeros((len(vocab_labels), 7), dtype=float)
    y_pred = list(chain.from_iterable(y_pred))
    y_test = list(chain.from_iterable(y_test))
    y_test = [label.split(',') for label in y_test]
    get_stat(y_pred, y_test, f1_table)
    print_f1(f1_table)
    np.savetxt('../table/crf.csv', f1_table, fmt='%.4f', delimiter=",")
