import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as nnf
from parsivar import Tokenizer, Normalizer
import torch.nn as nn
import torch.optim as optim
# import matplo/tlib.pyplot as plt
from collections import defaultdict


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = nnf.logsigmoid(score)

        return log_probs

corpus = open('ann.txt', encoding='utf8').read()


def tokenize_corpus(corpus):
    word_normalizer = Normalizer()
    word_tokenizer = Tokenizer()
    tokens = word_tokenizer.tokenize_words(word_normalizer.normalize(corpus))

    return tokens

def tokenize_corpus_sent(corpus):
    sentence_normalizer = Normalizer()
    sentence_tokenizer = Tokenizer()
    tokens = sentence_tokenizer.tokenize_sentences(sentence_normalizer.normalize(corpus))

    return tokens


tokenized_corpus = tokenize_corpus(corpus)
vocabulary = []
for word in tokenized_corpus:
    if word not in vocabulary:
        vocabulary.append(word)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}  # word to index
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}  # index to word

vocabulary_size = len(vocabulary)
embedding_dims = 100  # dimension of hidden layers
num_epochs = 100
learning_rate = 0.001

def create_skipgram_dataset(text):
    import random
    data = []
    # word_normalizer = Normalizer()

    # text = tokenize_corpus_sent(text)
    text = tokenize_corpus(text)
    for i in range(3, len(text) - 3):
        data.append((text[i], text[i - 3], 1))
        data.append((text[i], text[i - 2], 1))
        data.append((text[i], text[i - 1], 1))
        data.append((text[i], text[i + 1], 1))
        data.append((text[i], text[i + 2], 1))
        data.append((text[i], text[i + 3] ,1))
        # negative sampling
        for _ in range(6):
            if random.random() < 0.5 or i >= len(text) - 4:
                rand_id = random.randint(0, i-1)
            else:
                rand_id = random.randint(i + 4, len(text) - 1)
            data.append((text[i], text[rand_id], 0))
    return data

skipgram_train = create_skipgram_dataset(corpus)
def train_skipgram():
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocabulary_size, embedding_dims)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = .0
        for in_w, out_w, target in skipgram_train:
            in_w_var = Variable(torch.LongTensor([word2idx[in_w]]))
            out_w_var = Variable(torch.LongTensor([word2idx[out_w]]))

            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        if(epoch % 10== 0):
          print(total_loss)
        losses.append(total_loss)
    return model, losses


sg_model, sg_losses = train_skipgram()


def test_skipgram(test_data, model):
    print('====Test SkipGram===')
    correct_ct = 0
    for in_w, out_w, target in test_data:
        in_w_var = Variable(torch.LongTensor([word2idx[in_w]]))
        out_w_var = Variable(torch.LongTensor([word2idx[out_w]]))

        model.zero_grad()
        log_probs = model(in_w_var, out_w_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted = predicted[0]
        if predicted == target:
            correct_ct += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct/len(test_data)*100, correct_ct, len(test_data)))

print('------')
test_skipgram(skipgram_train, sg_model)
