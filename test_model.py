#  -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
from enc_dec import EncoderDecoder


# load data
post_vocab = {}
post_lines = open('post.txt').read().split('\n')
for i in range(len(post_lines)):
    lt = post_lines[i].split()
    for w in lt:
        if w not in post_vocab:
            post_vocab[w] = len(post_vocab)
post_vocab['<eos>'] = len(post_vocab)
post_w_num = len(post_vocab)

cmnt_vocab = {}
id2wd = {}
cmnt_lines = open('cmnt.txt').read().split('\n')
for i in range(len(cmnt_lines)):
    lt = cmnt_lines[i].split()
    for w in lt:
        if w not in cmnt_vocab:
            w_id = len(cmnt_vocab)
            cmnt_vocab[w] = w_id
            id2wd[w_id] = w
w_id = len(cmnt_vocab)
cmnt_vocab['<eos>'] = w_id
id2wd[w_id] = '<eos>'
cmnt_w_num = len(cmnt_vocab)


# output words
def mt(model, post_line):
    model.H.reset_state()
    for i in range(len(post_line)):
        wid = post_vocab[post_line[i]]
        x_k = model.embedx(chainer.Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
    x_k = model.embedx(chainer.Variable(np.array([post_vocab['<eos>']], dtype=np.int32), volatile='on'))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    print(id2wd[wid])
    loop = 0
    while (wid != cmnt_vocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(chainer.Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        print(id2wd[wid])
        loop += 1


# main part
post_lines = open('jp-test.txt').read().split('\n')

hidden = 100
for epoch in range(100):
    model = EncoderDecoder(post_size=post_w_num, cmnt_size=cmnt_w_num, hidden=hidden)
    filename = "mt-" + str(epoch) + ".model"
    chainer.serializers.load_npz(filename, model)
    for i in range(len(post_lines) - 1):
        jln = post_lines[i].split()
        jnlr = jln[::-1]
        print(epoch, ":")
        mt(model, jnlr)