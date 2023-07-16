import numpy as np
import argparse

from helper import *
from model import EncoderRNN, AttnDecoderRNN
from trainer import RNNTrainer
from evaluator import BleuEvaluator


if __name__ == '__main__':
    random.seed(2101985)  # set the seed for reproducibility
    # read data from files, the result dataset is a list of sentence pairs, e.g. [(cs1, en1), ...]
    # there are 4 datasets (cs-en, de-en, fr-en, ru-en), but we can only process one at a time
    # add argparser for processing 4 datasets parallelly
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default="cs", type=str)
    parser.add_argument('--decode_scheme', default=0, type=int)
    parser.add_argument('--use_glove', default=1, type=int)
    parser.add_argument('--attn_type', default=2, type=int)
    args = parser.parse_args()
    print(args)
    # print current setting
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    # src_lang = "cs"
    src_lang = args.src
    dataset = read_data('../data/', src_lang)
    lang_len = {"cs": 40, "de": 42, "fr": 46, "ru": 50}  # used to filter long sentences
    max_len = lang_len[src_lang]

    # preprocess the dataset, including:
    # 1. normalize text and filter sentences that are too long
    # 2. count the max length of sentences in source language
    # 3. make 2 vocabularies for source language and target language
    src_vocab, target_vocab, dataset, max_len = preprocess_data(dataset, src_lang, max_len=max_len, filter=True)

    # Randomly split a dataset into 5 subsets (S1 ~ S5)
    s1, s2, s3, s4, s5 = random_split_five(dataset)

    # training setting, we don't need batchifying here
    config = {  # "batch_size": 128,
              "max_len": max_len + 1,  # add one for EOS_token
              "embed_dim": 256,  # dimension of word embedding
              "hidden_dim": 256,  # dimension of hidden state
              "beam_size": 10,
              "len_norm": 1,  # 0/1 : no use/use
              "decode_scheme": args.decode_scheme,  # 0/1 : greedy/beam search
              "use_glove": args.use_glove,  # 0/1 : no use/use
              "attn_type": args.attn_type,  # 0/1/2: default/multiplicative/additive
              "teacher_forcing_ratio": 0.5,
              "lr": .01,
              "num_epochs": 300,
              }

    # do 5-fold train-test here
    result_table = np.zeros((6, 3), dtype=float)  # store results
    s_mat = None
    t_mat = None
    if config["use_glove"]:
        s_mat = get_glove_matrix(src_lang, src_vocab, config["embed_dim"])
        t_mat = get_glove_matrix("en", target_vocab, config["embed_dim"])

    for i in range(5):
        # make train and test sets
        train_set, test_set = make_sets(i+1, s1, s2, s3, s4, s5)
        # convert data in train set to tensor on GPU
        train_set = [tensors_from_pair(pair, src_vocab, target_vocab) for pair in train_set]
        # define encoder and decoder
        encoder = EncoderRNN(vocab_size=src_vocab.n_words, config=config, mat=s_mat).to(device)
        decoder = AttnDecoderRNN(vocab_size=target_vocab.n_words, config=config, mat=t_mat).to(device)

        print(f"cross validate {i + 1} times")
        # training phase
        trainer = RNNTrainer(config)
        trainer.fit(encoder, decoder, train_set)

        # evaluation phase
        evaluator = BleuEvaluator(config)
        b1, b2, b3 = evaluator.eval(encoder, decoder, test_set, src_vocab, target_vocab)
        result_table[i] = [b1, b2, b3]

    # save result
    result_table[5] = np.average(result_table[0:5, :], axis=0)
    savename = "../table/" + src_lang
    savename += str(config['decode_scheme'])
    savename += str(config['use_glove'])
    savename += str(config["attn_type"])
    savename += '.csv'
    np.savetxt(savename, result_table, fmt='%.4f', delimiter=",")
