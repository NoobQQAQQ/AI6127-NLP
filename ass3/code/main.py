import torch
from torch.backends import cudnn

from parameter import get_parameters
from tools import Corpus, batchify
from model import TransformerLM
from trainer import TransformerLMTrainer


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for fast training
    cudnn.benchmark = True
    torch.set_num_threads(2)
    # Load data
    corpus = Corpus('../data')

    # get train/val/test data set
    train_data = batchify(corpus.train, config.train_bs)
    val_data = batchify(corpus.valid, config.eval_bs)
    test_data = batchify(corpus.test, config.eval_bs)

    ntoken = len(corpus.dictionary)
    model = TransformerLM(config, ntoken).to(device)

    trainer = TransformerLMTrainer(config, ntoken)
    trainer.train(model, train_data, val_data)
    trainer.test(model, test_data)
