#  -*- coding: utf-8 -*-

import unicodedata
import pickle
import numpy as np
import chainer
import chainer.functions as F
from nltk import word_tokenize
from enc_dec import EncoderDecoder


# load data
with open('data/corpus/vocab.pkl', 'br') as f:
    vocab = pickle.load(f)
with open('data/corpus/id2wd.pkl', 'br') as f:
    id2wd = pickle.load(f)
w_num = len(vocab)


# output words
def mt(model, post_line):
    model.H.reset_state()
    for i in range(len(post_line)):
        if vocab.get(post_line[i], None) is not None:
            wid = vocab[post_line[i]]
        else:
            print(post_line[i])
            print('this word is not in vocab.')
            raise ValueError
        x_k = model.embedx(chainer.Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
    x_k = model.embedx(chainer.Variable(np.array([vocab['<eos>']], dtype=np.int32), volatile='on'))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    sentence = ""
    sentence = sentence + id2wd[wid] + " "
    # print(id2wd[wid])
    loop = 0
    while (wid != vocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(chainer.Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        # print(id2wd[wid])
        sentence = sentence + id2wd[wid] + " "
        loop += 1
    print(sentence)


# main part
post_test_lines = open('data/post-test.txt').read().split('\n')

hidden = 100
for epoch in range(100):
    model = EncoderDecoder(w_size=w_num, hidden=hidden)
    filename = "data/conv-" + str(epoch) + ".model"
    chainer.serializers.load_npz(filename, model)
    for i in range(len(post_test_lines) - 1):
        jln = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(post_test_lines[i])]
        jnlr = jln[::-1]
        print(epoch, ":")
        mt(model, jnlr)