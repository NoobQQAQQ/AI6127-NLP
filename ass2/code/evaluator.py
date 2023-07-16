from nltk.translate.bleu_score import corpus_bleu
from queue import PriorityQueue
from tqdm import tqdm
from helper import *
import torch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, len_norm):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.len_norm = len_norm

    def eval(self, alpha=1.0):
        reward = 0
        if self.len_norm:
            return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        else:
            return self.logp

    def __lt__(self, other):
        return self.eval() < other.eval()


class BleuEvaluator:
    def __init__(self, config):
        self.len_norm = config["len_norm"]
        self.beam_size = config["beam_size"]
        self.decode_scheme = config["decode_scheme"]
        self.max_len = config["max_len"]

    # greedy decode
    def greedy(self, encoder, decoder, sentence, src_vocab, target_vocab):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            input_tensor = tensor_from_sentence(src_vocab, sentence)
            input_length = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch.zeros(self.max_len, encoder.hidden_dim, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            # decoding phase
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(self.max_len, self.max_len)

            for di in range(self.max_len):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                _, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(target_vocab.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    # beam search decode
    def beam_search(self, encoder, decoder, sentence, src_vocab, target_vocab):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            input_tensor = tensor_from_sentence(src_vocab, sentence)
            input_length = input_tensor.size(0)
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch.zeros(self.max_len, encoder.hidden_dim, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            # decoding phase
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden

            # Number of sentence to generate
            endnodes = []
            number_required = 1

            # starting node -  hidden vector, previous node, word id, logp, length, len_norm
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, self.len_norm)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue
                # elif n.leng > max_length:
                #    continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, self.beam_size)
                nextnodes = []

                for new_k in range(self.beam_size):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, self.len_norm)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))

                # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(number_required)]

            _, n = endnodes[0]  # must assume # of sentences required = 1
            utterance = []
            utterance.append(target_vocab.index2word[n.wordid.item()])

            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(target_vocab.index2word[n.wordid.item()])

            utterance = utterance[::-1]

        return utterance, None

    # eval
    def eval(self, encoder, decoder, test_set, src_vocab, target_vocab):
        references, candidates = [], []
        if self.decode_scheme == 1:
            test_set = random.sample(test_set, len(test_set) // 5)
        # test_set = random.sample(test_set, len(test_set) // 10)
        for src_sent, target_sent in tqdm(test_set, desc="Testing", unit="pair"):
            # convert a sentence to a list of lists of words
            target_sent = [target_sent.split(' ')]
            references.append(target_sent)
            if self.decode_scheme == 0:
                output_words, _ = self.greedy(encoder, decoder, src_sent, src_vocab, target_vocab)
            else:
                output_words, _ = self.beam_search(encoder, decoder, src_sent, src_vocab, target_vocab)
            candidates.append(output_words)
        score1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
        score2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
        score3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
        return score1, score2, score3
