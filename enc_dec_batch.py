#  -*- coding: utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"


import argparse
import pickle
import unicodedata
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from nltk import word_tokenize

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


# for N step LSTM
class LSTM(L.NStepLSTM):
    def __init__(self, in_size, out_size, dropout=0.1, use_cudnn=True):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, chainer.Variable)
        assert isinstance(hx, chainer.Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == xp:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = chainer.Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = chainer.Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')

        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys


def sequence_embed(embed, xs):
    """
    convert text formatted ID to node data.
    :param embed: F.EmbedID
    :param xs: list of ndarray (np.int32). len(list) == the number of batch size
    :return: string: output sentence
    """
    x_len = [len(x) for x in xs]
    x_section = xp.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs


# definite encoder-decoder model
class EncoderDecoder(chainer.Chain):
    def __init__(self, w_size, hidden, gpu_flg=0):
        super(EncoderDecoder, self).__init__(
            embedx=F.EmbedID(w_size, hidden),
            embedy=F.EmbedID(w_size, hidden),
            H=LSTM(hidden, hidden),
            W=L.Linear(hidden, w_size),
        )
        global xp
        xp = chainer.cuda.cupy if gpu_flg >= 0 else np
        self.hidden = hidden

    def __call__(self, post_lines, cmnt_lines, vocab):
        self.H.reset_state()
        eos = xp.array([vocab['<eos>']], np.int32)
        cmnt_lines_in = [F.concat([eos, line], axis=0) for line in cmnt_lines]
        cmnt_lines_out = [F.concat([line, eos], axis=0) for line in cmnt_lines]

        exs = sequence_embed(self.embedx, post_lines)
        h = self.H(exs, train=True)
        eys = sequence_embed(self.embedy, cmnt_lines_in)
        h = self.H(eys, train=True)
        loss = F.softmax_cross_entropy(self.W(F.concat(h, axis=0)), F.concat(cmnt_lines_out, axis=0))
        return loss

    def interpreter(self, post_line, batch, id2wd, max_length=15):
        self.H.reset_state()
        exs = sequence_embed(self.embedx, [post_line])
        h = self.H(exs, train=False)
        ys = self.xp.zeros(batch, xp.int32)
        result = ""
        for i in range(max_length):
            eys = self.embedy(ys)
            eys = chainer.functions.split_axis(eys, batch, 0, force_tuple=True)
            ys = self.H(eys, train=False)
            cys = chainer.functions.concat(ys, axis=0)
            wy = self.W(cys)
            ys = self.xp.argmax(wy.data, axis=1).astype(xp.int32)
            result = result + id2wd[ys[0]] + " "
            if id2wd[ys[0]] == '<eos>':
                break
        return result


def main():

    # load data
    vocab = {}
    id2wd = {}
    post_lines = open('data/post.txt').read().split('\n')
    for i in range(len(post_lines)):
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in post_lines[i].split()]
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w
    cmnt_lines = open('data/cmnt.txt').read().split('\n')
    for i in range(len(cmnt_lines)):
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in cmnt_lines[i].split()]
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w
    post_test_lines = open('data/post-test.txt').read().split('\n')
    for i in range(len(post_test_lines)):
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in post_test_lines[i].split()]
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    # add <eos> tag
    w_id = len(vocab)
    vocab['<eos>'] = w_id
    id2wd[w_id] = '<eos>'

    # number of all words
    w_num = len(vocab)

    # save each dictionaries
    with open('data/corpus/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('data/corpus/id2wd.pkl', 'wb') as f:
        pickle.dump(id2wd, f)

    # create an instance of encoder-decoder model
    demb = 100
    model = EncoderDecoder(w_size=w_num, hidden=demb, gpu_flg=args.gpu)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # convert corpus from words to ids
    post_text_lines = [post_line.split()[::-1] for post_line in post_lines]
    post_lines = []
    for post in post_text_lines:
        post_lines.append(xp.array([vocab[unicodedata.normalize('NFKC', word.lower())] for word in post], xp.int32))
    cmnt_text_lines = [cmnt_line.split() for cmnt_line in cmnt_lines]
    cmnt_lines = []
    for cmnt in cmnt_text_lines:
        cmnt_lines.append(xp.array([vocab[unicodedata.normalize('NFKC', word.lower())] for word in cmnt], xp.int32))
    eos = xp.array([vocab['<eos>']], np.int32)
    cmnt_lines = [F.concat([line, eos], axis=0) for line in cmnt_lines]

    # start learning a model
    batchsize = 100
    for epoch in range(100):
        for i in range(0, len(post_lines) - batchsize, batchsize):
            jlnr = post_lines[i: i + batchsize]
            eln = cmnt_lines[i: i + batchsize]
            model.H.reset_state()
            model.cleargrads()
            loss = model(jlnr, eln, vocab)
            loss.backward()
            loss.unchain_backward()     # truncate
            optimizer.update()
        outfile = "data/batch-" + str(epoch) + ".model"
        chainer.serializers.save_npz(outfile, model)
        print(epoch, 'epoch')


if __name__ == '__main__':
    main()