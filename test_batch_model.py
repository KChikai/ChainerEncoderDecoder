#  -*- coding: utf-8 -*-

import argparse
import pickle
import numpy as np
import chainer
import chainer.functions as F
from enc_dec_batch import EncoderDecoder


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# GPU settings
gpu_device = 0
if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(gpu_device).use()
# global variable (initialize)
xp = np


# load data
with open('data/corpus/vocab.pkl', 'br') as f:
    vocab = pickle.load(f)
with open('data/corpus/id2wd.pkl', 'br') as f:
    id2wd = pickle.load(f)
post_w_num = len(vocab)

# main part
post_test_lines = open('data/post-test.txt').read().split('\n')

hidden = 100
for epoch in range(100):
    model = EncoderDecoder(w_size=post_w_num, hidden=hidden, gpu_flg=args.gpu)
    filename = "data/batch-" + str(epoch) + ".model"
    chainer.serializers.load_npz(filename, model)
    for i in range(len(post_test_lines) - 1):
        jln = post_test_lines[i].split()
        jnlr = jln[::-1]
        print(epoch, ":")
        result = model.interpreter(post_line=jnlr, batch=1, id2wd=id2wd)
        print(result)