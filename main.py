import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as nnf
from parsivar import Tokenizer, Normalizer
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = torch.tanh(score)

        return log_probs

corpus = open('ann.txt', encoding='utf8').read()

persian_stopwords = ['و', 'در', 'به', 'از', 'كه', 'مي', 'اين', 'ز'
                      'است', 'را', 'با', 'هاي', 'براي', 'آن', 'يك', 'شود', 'شده', 'خود', 'ها', 'كرد',
                      'شد', 'اي', 'تا', 'كند', 'بر', 'بود', 'گفت', 'نيز', 'وي', 'هم', 'كنند', 'دارد',
                      'ما', 'كرده', 'يا', 'اما', 'بايد', 'دو', 'اند', 'هر', 'خواهد', 'او', 'مورد', 'آنها',
                      'باشد', 'ديگر', 'مردم', 'نمي', 'بين', 'پيش', 'پس', 'اگر', 'همه', 'صورت', 'يكي',
                      'هستند', 'بي', 'من', 'دهد', 'هزار', 'نيست', 'استفاده', 'داد', 'داشته', 'راه', 'داشت',
                      'چه', 'همچنين', 'كردند', 'داده', 'بوده', 'دارند', 'همين', 'ميليون', 'سوي', 'شوند',
                      'بيشتر', 'بسيار', 'روي', 'گرفته', 'هايي', 'تواند', 'اول', 'نام', 'هيچ', 'چند', 'جديد',
                      'بيش', 'شدن', 'كردن', 'كنيم', 'نشان', 'حتي', 'اينكه', 'ولی', 'توسط', 'چنين', 'برخي',
                      'نه', 'ديروز', 'دوم', 'درباره', 'بعد', 'مختلف', 'گيرد', 'شما', 'گفته', 'آنان', 'بار',
                      'طور', 'گرفت', 'دهند', 'گذاري', 'بسياري', 'طي', 'بودند', 'ميليارد', 'بدون', 'تمام',
                      'كل', 'تر', 'براساس', 'شدند', 'ترين', 'امروز', 'باشند', 'ندارد', 'چون', 'قابل', 'گويد',
                      'ديگري', 'همان', 'خواهند', 'قبل', 'آمده', 'اكنون', 'تحت', 'طريق', 'گيري', 'جاي', 'هنوز',
                      'چرا', 'البته', 'كنيد', 'سازي', 'سوم', 'كنم', 'بلكه', 'زير', 'توانند', 'ضمن', 'فقط', 'بودن',
                      'حق', 'آيد', 'وقتي', 'اش', 'يابد', 'نخستين', 'مقابل', 'خدمات', 'امسال', 'تاكنون', 'مانند',
                      'تازه', 'آورد', 'فكر', 'آنچه', 'نخست', 'نشده', 'شايد', 'چهار', 'جريان', 'پنج', 'ساخته',
                      'زيرا', 'نزديك', 'برداري', 'كسي', 'ريزي', 'رفت', 'گردد', 'مثل', 'آمد', 'ام', 'بهترين',
                      'دانست', 'كمتر', 'دادن', 'تمامي', 'جلوگيري', 'بيشتري', 'ايم', 'ناشي', 'چيزي', 'آنكه', 'بالا',
                      'بنابراين', 'ايشان', 'بعضي', 'دادند', 'داشتند', 'برخوردار', 'نخواهد', 'هنگام', 'نبايد', 'غير', 'نبود',
                      'ديده', 'وگو', 'داريم', 'چگونه', 'بندي', 'خواست', 'فوق', 'ده', 'نوعي', 'هستيم', 'ديگران', 'همچنان',
                      'سراسر', 'ندارند', 'گروهي', 'سعي', 'روزهاي', 'آنجا', 'يكديگر', 'كردم', 'بيست', 'بروز', 'سپس', 'رفته',
                      'آورده', 'نمايد', 'باشيم', 'گويند', 'زياد', 'خويش', 'همواره', 'گذاشته', 'شش', 'نداشته', 'شناسي', 'خواهيم',
                      'آباد', 'داشتن', 'نظير', 'همچون', 'باره', 'نكرده', 'شان', 'سابق', 'هفت', 'دانند', 'جايي', 'بی', 'جز',
                      'زیرِ', 'رویِ', 'سریِ', 'تویِ', 'جلویِ', 'پیشِ', 'عقبِ', 'بالایِ', 'خارجِ', 'وسطِ', 'بیرونِ', 'سویِ', 'کنارِ',
                      'پاعینِ', 'نزدِ', 'نزدیکِ', 'دنبالِ', 'حدودِ', 'برابرِ', 'طبقِ', 'مانندِ', 'ضدِّ', 'هنگامِ', 'برایِ', 'مثلِ', 'بارة',
                      'اثرِ', 'تولِ', 'علّتِ', 'سمتِ', 'عنوانِ', 'قصدِ', 'روب', 'جدا', 'کی', 'که', 'چیست', 'هست', 'کجا',
                      'کجاست', 'کَی', 'چطور', 'کدام', 'آیا', 'مگر', 'چندین', 'یک', 'چیزی', 'دیگر', 'کسی', 'بعری',
                      'هیچ', 'چیز', 'جا', 'کس', 'هرگز', 'یا', 'تنها', 'بلکه', 'خیاه', 'بله', 'بلی', 'آره', 'آری',
                      'مرسی', 'البتّه', 'لطفاً', 'ّه', 'انکه', 'وقتیکه', 'همین', 'پیش', 'مدّتی', 'هنگامی', 'مان', 'تان']


def tokenize_corpus(corpus):
    word_normalizer = Normalizer()
    word_tokenizer = Tokenizer()
    tokens = word_tokenizer.tokenize_words(word_normalizer.normalize(corpus))
    for token in tokens:
        if token in persian_stopwords:
            tokens.remove(token)

    return tokens

tokenized_corpus = tokenize_corpus(corpus)
vocabulary = []
for word in tokenized_corpus:
    if word not in vocabulary:
        vocabulary.append(word)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}  # word to index
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}  # index to word

vocabulary_size = len(vocabulary)
embedding_dims = 5  # dimension of hidden layers
num_epochs = 1
learning_rate = 0.001

def create_skipgram_dataset(text):
    import random
    data = []
    text = tokenize_corpus(text)
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i - 2], 1))
        data.append((text[i], text[i - 1], 1))
        data.append((text[i], text[i + 1], 1))
        data.append((text[i], text[i + 2], 1))
        # negative sampling
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i-1)
            else:
                rand_id = random.randint(i+3, len(text)-1)
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
          print(total_loss/len(skipgram_train))
        losses.append(total_loss)
    return model, losses


sg_model, sg_losses = train_skipgram()
model = SkipGram(vocabulary_size, embedding_dims)
FILE = 'model_weights.pth'
torch.save(model.state_dict(), FILE)

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


test_skipgram(skipgram_train, sg_model)


def get_most_similar(test_word, model, vocabs):
    log_probability = []
    data_prob_list = []
    sim_word = []
    for vocab_word in vocabs:
        in_w = Variable(torch.LongTensor([word2idx[test_word]]))
        out_w = Variable(torch.LongTensor([word2idx[vocab_word]]))
        log_probability.append(model(in_w, out_w).data)
        sim_word.append(vocab_word)
        data_prob_list = list(zip(log_probability, sim_word))

    return data_prob_list


key_word = input("کلمه مورد نظر را وارد کنید\n")
window_size = input("تعداد را وارد کنید\n")
probability = sorted(get_most_similar(key_word, sg_model, vocabulary))
def print_similar(probability, window_size):
    i = 0
    p, w = zip(*probability)
    for word in w:
        print(word)
        i += 1
        if i == (int)(window_size) + 1:
            break

print_similar(probability, window_size)

