#  -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from enc_dec import EncoderDecoder

post_lines = open('jp-test.txt').read().split('\n')


def mt(model, jline):
    pass


hidden = 100
for epoch in range(100):
    model = EncoderDecoder(post_size=, cmnt_size=, hidden=hidden)
    filename = "mt-" + str(epoch) + ".model"
    chainer.serializers.load_npz(filename, model)
    for i in range(len(post_lines) - 1):
        jln = post_lines[i].split()
        jnlr = jln[::-1]
        print(epoch, ":")
        mt(model, jnlr)