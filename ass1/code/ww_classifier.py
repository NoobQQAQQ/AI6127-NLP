import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from functools import partial
from itertools import chain
from tqdm import tqdm


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


def label_str2int(sent, vocab_labels):
    return [vocab_labels.get_stoi()[label] for label in sent]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def construct_vocab(train_set, min_freq=2):
    vocab = build_vocab_from_iterator(train_set, min_freq=min_freq, specials=["<pad>", "<unk>"])
    vocab.set_default_index(1)
    return vocab


def convert_tokens_to_inds(sentence, word_2_id):
    return [word_2_id.get(t, word_2_id["<unk>"]) for t in sentence]


def pad_sentence_for_window(sentence, window_size, pad_token="<pad>"):
    return [pad_token]*window_size + sentence + [pad_token]*window_size


def my_collate(data, window_size, word_2_id, vocab_labels):
    """
    For some chunk of sentences and labels
        -add window padding
        -pad for lengths using pad_sequence
        -convert our labels to one-hots
        -return padded inputs, one-hot labels, and lengths
    """

    x_s, y_s = zip(*data)

    # deal with input sentences as we've seen
    window_padded = [convert_tokens_to_inds(pad_sentence_for_window(sentence, window_size), word_2_id)
                     for sentence in x_s]
    # append zeros to each list of token ids in batch so that they are all the same length
    padded = nn.utils.rnn.pad_sequence([torch.LongTensor(t) for t in window_padded], batch_first=True)
    # convert labels to one-hots (only index)
    labels = []
    lengths = []
    for y in y_s:
        lengths.append(len(y))
        labels.append(label_str2int(y, vocab_labels))
    # convert labels to 1-d tensor
    labels = list(chain.from_iterable(labels))
    # it means we are training
    if len(vocab_labels) == 13:
        labels = torch.tensor(labels).to(device)
    return padded.long().to(device), labels, torch.LongTensor(lengths).to(device)


class SoftmaxWordWindowClassifier(nn.Module):

    def __init__(self, config, vocab_size, pad_idx=0):
        super(SoftmaxWordWindowClassifier, self).__init__()
        self.window_size = 2 * config["half_window"] + 1
        self.embed_dim = config["embed_dim"]
        self.hidden_dim1 = config["hidden_dim1"]
        self.hidden_dim2 = config["hidden_dim2"]
        self.num_hidden = config["num_hidden"]
        self.num_classes = config["num_classes"]
        self.freeze_embeddings = config["freeze_embeddings"]

        """
        Embedding layer
        -model holds an embedding for each layer in our vocab
        -sets aside a special index in the embedding matrix for padding vector (of zeros)
        -by default, embeddings are parameters (so gradients pass through them)
        """
        self.embed_layer = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_idx)
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        """
        Hidden layer
        -we want to map embedded word windows of dim (window_size+1)*self.embed_dim to a hidden layer.
        -nn.Sequential allows you to efficiently specify sequentially structured models
            -first the linear transformation is evoked on the embedded word windows
            -next the nonlinear transformation tanh is evoked.
        """
        if self.num_hidden == 1:
            self.hidden_layer = nn.Sequential(nn.Linear(self.window_size * self.embed_dim,
                                                        self.hidden_dim1),
                                              nn.Tanh())
        else:
            assert (self.num_hidden == 2)
            self.hidden_layer = nn.Sequential(nn.Linear(self.window_size * self.embed_dim,
                                                        self.hidden_dim1),
                                              nn.Tanh(),
                                              nn.Linear(self.hidden_dim1,
                                                        self.hidden_dim2),
                                              nn.Tanh())

        """
        Output layer
        -we want to map elements of the output layer (of size self.hidden dim) to a number of classes.
        """
        if self.num_hidden == 1:
            self.output_layer = nn.Linear(self.hidden_dim1, self.num_classes)
        else:
            assert (self.num_hidden == 2)
            self.output_layer = nn.Linear(self.hidden_dim2, self.num_classes)

    def forward(self, inputs):
        """
        Let B:= batch_size
            L:= window-padded sentence length
            D:= self.embed_dim
            S:= self.window_size
            H:= self.hidden_dim

        inputs: a (B, L) tensor of token indices
        """
        B, L = inputs.size()

        """
        Reshaping.
        Takes in a (B, L) LongTensor
        Outputs a (B, L~, S) LongTensor
        """
        # Fist, get our word windows for each word in our input.
        token_windows = inputs.unfold(1, self.window_size, 1)
        _, adjusted_length, _ = token_windows.size()

        # Good idea to do internal tensor-size sanity checks, at the least in comments!
        assert token_windows.size() == (B, adjusted_length, self.window_size)

        """
        Embedding.
        Takes in a torch.LongTensor of size (B, L~, S) 
        Outputs a (B, L~, S, D) FloatTensor.
        """
        embedded_windows = self.embed_layer(token_windows)

        """
        Reshaping.
        Takes in a (B, L~, S, D) FloatTensor.
        Resizes it into a (B, L~, S*D) FloatTensor.
        -1 argument "infers" what the last dimension should be based on leftover axes.
        """
        embedded_windows = embedded_windows.view(B, adjusted_length, -1)

        """
        Layer 1.
        Takes in a (B, L~, S*D) FloatTensor.
        Resizes it into a (B, L~, H) FloatTensor
        """
        layer_1 = self.hidden_layer(embedded_windows)

        """
        Layer 2
        Takes in a (B, L~, H) FloatTensor.
        Resizes it into a (B, L~, num_classes) FloatTensor.
        """
        output = self.output_layer(layer_1)
        return output


def loss_function(outputs, labels, lengths):
    """Computes negative LL loss on a batch of model predictions."""
    # remove the paddings
    outputs = [outputs[i, :lengths[i], :] for i in range(int(outputs.shape[0]))]
    # cat a batch of sentences
    outputs = torch.cat(outputs, dim=0)
    loss = nn.CrossEntropyLoss().to(device)
    return loss(outputs, labels)


def train_epoch(loss_function, optimizer, model, train_data):
    ## For each batch, we must reset the gradients
    ## stored by the model.
    total_loss = 0
    for batch, labels, lengths in train_data:
        # clear gradients
        optimizer.zero_grad()
        # evoke model in training mode on batch
        outputs = model.forward(batch)
        # compute loss w.r.t batch
        loss = loss_function(outputs, labels, lengths)
        # pass gradients back, startiing on loss value
        loss.backward()
        # update parameters
        optimizer.step()
        total_loss += loss.item()

    # return the total to keep track of how you did this time around
    return total_loss


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

    # get training/dev/testing datasets
    X_train = [sent2tokens(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2tokens(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # construct token and label vocabulary
    vocab = construct_vocab(X_train, min_freq=1)
    id_2_word, word_2_id = vocab.get_itos(), vocab.get_stoi()
    vocab_labels = build_vocab_from_iterator(y_train)
    label2id, id2label = vocab_labels.get_stoi(), vocab_labels.get_itos()
    test_labels = build_vocab_from_iterator(y_test)
    test_label2id, test_id2label = test_labels.get_stoi(), test_labels.get_itos()

    # training setting
    config = {"batch_size": 128,
              "num_classes": 13,  # count manually
              "half_window": 1,
              "embed_dim": 512,
              "hidden_dim1": 256,
              "hidden_dim2": 64,
              "num_hidden": 1,  # only try 1 or 2 here
              "freeze_embeddings": False,
              "lr": .0005,
              "num_epochs": 300,
              }
    
    # load dataset
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               collate_fn=partial(my_collate,
                                                                  window_size=config['half_window'],
                                                                  word_2_id=word_2_id,
                                                                  vocab_labels=vocab_labels),
                                               )
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)),
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              collate_fn=partial(my_collate,
                                                                 window_size=config['half_window'],
                                                                 word_2_id=word_2_id,
                                                                 vocab_labels=test_labels),
                                              )
    
    # training preparations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SoftmaxWordWindowClassifier(config, len(word_2_id)).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])

    # training phase
    losses = []
    for epoch in tqdm(range(config['num_epochs'])):
        epoch_loss = train_epoch(loss_function, optimizer, model, train_loader)
        losses.append(epoch_loss)
    # print(losses)
    x = np.arange(1, config['num_epochs'] + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.plot(x, losses, color="blue", linewidth=1, linestyle="-", label="train loss")
    plt.show()

    # evaluation phase
    # generate a table to count f1 scores, shape of table: number of class by 7(TP, FN, FP, P, R, F1, support)
    f1_table = np.zeros((len(vocab_labels), 7), dtype=float)
    model.eval()
    for test_instance, labels, lengths in test_loader:
        outputs = model.forward(test_instance)
        batch_size = int(outputs.shape[0])
        outputs = torch.cat([outputs[i, :lengths[i], :] for i in range(batch_size)], dim=0)
        preds = torch.argmax(outputs, dim=1).tolist()
        preds = vocab_labels.lookup_tokens(preds)
        truth = [test_id2label[id].split(',') for id in labels]
        get_stat(preds, truth, f1_table)
    print_f1(f1_table)
    print(id2label)
    np.savetxt('../table/test.csv', f1_table, fmt='%.4f', delimiter=",")
        
