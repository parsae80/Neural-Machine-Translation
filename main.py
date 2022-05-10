import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as NNF


corpus = []


# incomplete splitting commas
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
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
            # make sure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

words = torch.tensor([word2idx[w] for w in vocabulary], dtype=torch.long)

one_hot_encoding = NNF.one_hot(words)
print(vocabulary)
print(one_hot_encoding)