import torch
import math
import torch.nn as nn
from tqdm import tqdm
from tools import get_batch


class TransformerLMTrainer:
    def __init__(self, cfg, ntoken):
        self.epoch = cfg.epoch
        self.lr = cfg.lr
        self.ntoken = ntoken
        self.seq_len = cfg.seq_len

    def train_epoch(self, model, train_data, criterion):
        model.train()
        total_loss = 0.
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.seq_len)):
            data, targets = get_batch(self.seq_len, train_data, i)
            model.zero_grad()
            output = model(data)
            output = output.view(-1, self.ntoken)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-self.lr)
            total_loss += loss.item()
        return total_loss / (len(train_data) - 1)

    def evaluate(self, model, test_data, criterion):
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i in range(0, test_data.size(0) - 1, self.seq_len):
                data, targets = get_batch(self.seq_len, test_data, i)
                output = model(data)
                output = output.view(-1, self.ntoken)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(test_data) - 1)

    def train(self, model, train_data, val_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_losses = []
        val_losses = []
        best_val_loss = None
        criterion = nn.NLLLoss().to(device)
        for epoch in tqdm(range(1, self.epoch+1), desc="Training"):
            train_losses.append(self.train_epoch(model, train_data, criterion))
            val_loss = self.evaluate(model, val_data, criterion)
            val_losses.append(val_loss)
            print(f"epoch {epoch}: val_loss {val_losses[epoch-1]} | val_ppl {math.exp(val_losses[epoch-1])}")
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.lr /= 4.0

    def test(self, model, test_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.NLLLoss().to(device)
        test_loss = self.evaluate(model, test_data, criterion)
        print(f"test_loss {test_loss} | test_ppl {math.exp(test_loss)}")
