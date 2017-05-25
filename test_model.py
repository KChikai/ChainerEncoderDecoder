#  -*- coding: utf-8 -*-

import pickle
import numpy as np
import chainer
import chainer.functions as F
from enc_dec import EncoderDecoder


# load data
with open('data/corpus/post-vocab.pkl', 'br') as f:
    post_vocab = pickle.load(f)
with open('data/corpus/cmnt-vocab.pkl', 'br') as f:
    cmnt_vocab = pickle.load(f)
with open('data/corpus/id2wd.pkl', 'br') as f:
    id2wd = pickle.load(f)
post_w_num = len(post_vocab)
cmnt_w_num = len(cmnt_vocab)


# output words
def mt(model, post_line):
    model.H.reset_state()
    for i in range(len(post_line)):
        # wid = post_vocab[post_line[i]]
        wid = post_vocab.get(post_line[i], len(post_vocab)-1)
        x_k = model.embedx(chainer.Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
    x_k = model.embedx(chainer.Variable(np.array([post_vocab['<eos>']], dtype=np.int32), volatile='on'))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    sentence = ""
    sentence = sentence + id2wd[wid] + " "
    # print(id2wd[wid])
    loop = 0
    while (wid != cmnt_vocab['<eos>']) and (loop <= 30):
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
    model = EncoderDecoder(post_size=post_w_num, cmnt_size=cmnt_w_num, hidden=hidden)
    filename = "data/mt-" + str(epoch) + ".model"
    chainer.serializers.load_npz(filename, model)
    for i in range(len(post_test_lines) - 1):
        jln = post_test_lines[i].split()
        jnlr = jln[::-1]
        print(epoch, ":")
        mt(model, jnlr)