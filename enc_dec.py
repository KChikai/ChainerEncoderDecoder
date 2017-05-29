#  -*- coding: utf-8 -*-

import pickle
import unicodedata
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from nltk import word_tokenize


# definite encoder-decoder model
class EncoderDecoder(chainer.Chain):
    def __init__(self, w_size, hidden):
        super(EncoderDecoder, self).__init__(
            embedx=F.EmbedID(w_size, hidden),
            embedy=F.EmbedID(w_size, hidden),
            H=L.LSTM(hidden, hidden),
            W=L.Linear(hidden, w_size),
        )

    def __call__(self, post_line, cmnt_line, vocab):
        self.H.reset_state()
        for i in range(len(post_line)):
            wid = vocab[post_line[i]]
            x_k = self.embedx(chainer.Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(chainer.Variable(np.array([vocab['<eos>']], dtype=np.int32)))
        tx = chainer.Variable(np.array([vocab[cmnt_line[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(cmnt_line)):
            wid = vocab[cmnt_line[i]]
            x_k = self.embedy(chainer.Variable(np.array([wid], dtype=np.int32)))
            next_wid = vocab['<eos>'] if (i == len(cmnt_line) - 1) else vocab[cmnt_line[i + 1]]
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
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(post_lines[i])]
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    cmnt_lines = open('data/cmnt.txt').read().split('\n')
    for i in range(len(cmnt_lines)):
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(cmnt_lines[i])]
        for w in lt:
            if w not in vocab:
                w_id = len(vocab)
                vocab[w] = w_id
                id2wd[w_id] = w

    post_test_lines = open('data/post-test.txt').read().split('\n')
    for i in range(len(post_test_lines)):
        lt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(post_test_lines[i])]
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
    for epoch in range(100):
        for i in range(len(post_lines) - 1):
            jln = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(post_lines[i])]
            jlnr = jln[::-1]
            eln = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(cmnt_lines[i])]
            model.H.reset_state()
            model.cleargrads()
            loss = model(jlnr, eln, vocab)
            loss.backward()
            loss.unchain_backward()     # truncate
            optimizer.update()
        outfile = "data/conv-" + str(epoch) + ".model"
        chainer.serializers.save_npz(outfile, model)
        print(epoch, 'epoch')


if __name__ == '__main__':
    main()