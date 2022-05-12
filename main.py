import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as nnf
from parsivar import Tokenizer, Normalizer
import torchtext

corpus = open('shams_corpus.txt', encoding='utf8').read().split("\n")


# incomplete splitting commas
def tokenize_corpus(corpus):
    my_normalizer = Normalizer()
    my_tokenizer = Tokenizer()
    tokens = my_tokenizer.tokenize_sentences(my_normalizer.normalize(corpus))
    return tokens


tokenized_corpus = tokenize_corpus(corpus)
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}  # word to index
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}  # index to word

vocabulary_size = len(vocabulary)


window_size = 5
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, treated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make sure not jump out of the sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            idx_pairs.append((indices[center_word_pos], indices[context_word_pos]))

words = torch.tensor([word2idx[w] for w in vocabulary], dtype=torch.long)  # putting words into a tensor
one_hot_encoding = nnf.one_hot(words)


# input layer needed to be defined


embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(one_hot_encoding).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = nnf.log_softmax(z2, dim=0)

        loss = nnf.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
