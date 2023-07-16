import torch.nn as nn
from torch import optim
from tqdm import tqdm

from helper import *


class RNNTrainer:
    def __init__(self, config):
        self.lr = config["lr"]
        self.max_len = config["max_len"]
        self.tf_ratio = config["teacher_forcing_ratio"]

    # train one sentence a step
    def train_step(self, input_tensor, target_tensor, encoder, decoder,
                   encoder_optimizer, decoder_optimizer, criterion):

        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_hidden = encoder.init_hidden()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(self.max_len, encoder.hidden_dim, device=device)

        # feed words in a sentence one by one to encoder
        for ei in range(input_length):
            # output shape is LxBxH
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # decoding phase
        loss = 0
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.tf_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def fit(self, encoder, decoder, train_set, plot_every=50):
        plot_losses = []
        plot_loss_total = 0  # Reset every plot_every

        # training preparations
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=self.lr)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=self.lr)
        criterion = nn.NLLLoss().to(device)
        train_set = random.sample(train_set, len(train_set) // 5)
        # train through dataset once
        for idx, pair in enumerate(tqdm(train_set, desc="Training", unit="pair"), start=1):
            input_tensor = pair[0]
            target_tensor = pair[1]
            loss = self.train_step(input_tensor, target_tensor, encoder,
                                   decoder, encoder_optimizer, decoder_optimizer, criterion)
            plot_loss_total += loss
            if idx % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        show_plot(plot_losses)
