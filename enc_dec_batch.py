#  -*- coding: utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


# for N step LSTM
class LSTM(L.NStepLSTM):
    def __init__(self, in_size, out_size, dropout=0.5, use_cudnn=True):
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
        if self.xp == np:
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


# definite encoder-decoder model
class EncoderDecoder(chainer.Chain):
    def __init__(self, w_size, hidden):
        super(EncoderDecoder, self).__init__(
            embedx=F.EmbedID(w_size, hidden),
            embedy=F.EmbedID(w_size, hidden),
            H=LSTM(hidden, hidden),
            W=L.Linear(hidden, w_size),
        )

    def __call__(self, post_lines, cmnt_lines, vocab, batchsize):
        self.H.reset_state()
        wids = [vocab[word] for post_line in post_lines for word in post_line]
        x_k = self.embedx(chainer.Variable(np.array([wids], dtype=np.int32)))
        h = self.H(x_k)
        x_k = self.embedx(chainer.Variable(np.array([vocab['<eos>'] for _ in range(batchsize)], dtype=np.int32)))
        tx = chainer.Variable(np.array([vocab[cmnt_lines[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(cmnt_lines)):
            wid = vocab[cmnt_lines[i]]
            x_k = self.embedy(chainer.Variable(np.array([wid], dtype=np.int32)))
            next_wid = vocab['<eos>'] if (i == len(cmnt_lines) - 1) else vocab[cmnt_lines[i + 1]]
            tx = chainer.Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss


def main():

    # load data
    vocab = {}
    id2wd = {}

    post_lines = open('data/post.txt').read().split('\n')
    for i in range(len(post_lines)):
        lt = post_lines[i].split()
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    cmnt_lines = open('data/cmnt.txt').read().split('\n')
    for i in range(len(cmnt_lines)):
        lt = cmnt_lines[i].split()
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    post_test_lines = open('data/post-test.txt').read().split('\n')
    for i in range(len(post_test_lines)):
        lt = post_test_lines[i].split()
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    w_id = len(vocab)
    vocab['<eos>'] = w_id
    id2wd[w_id] = '<eos>'

    w_num = len(vocab)

    # save each dictionaries
    with open('data/corpus/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('data/corpus/id2wd.pkl', 'wb') as f:
        pickle.dump(id2wd, f)

    # create an instance of encoder-decoder model
    demb = 100
    model = EncoderDecoder(w_num, demb)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # start learning a model
    batchsize = 5
    post_lines = [post_line.split()[::-1] for post_line in post_lines]
    post_lines = [vocab[word] for post_line in post_lines for word in post_line]
    cmnt_lines = [cmnt_line.split() for cmnt_line in cmnt_lines]
    cmnt_lines = [vocab[word] for cmnt_line in cmnt_lines for word in cmnt_line]

    for epoch in range(100):
        for i in range(len(post_lines) - 1):
            jlnr = post_lines[i: i + batchsize]
            eln = cmnt_lines[i: i + batchsize]
            model.H.reset_state()
            model.cleargrads()
            loss = model(jlnr, eln, vocab, batchsize)
            loss.backward()
            loss.unchain_backward()     # truncate
            optimizer.update()
        outfile = "data/batch-" + str(epoch) + ".model"
        chainer.serializers.save_npz(outfile, model)
        print(epoch, 'epoch')


if __name__ == '__main__':
    main()