#  -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


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


class EncoderDecoder(chainer.Chain):
    def __init__(self, post_size, cmnt_size, hidden):
        super(EncoderDecoder, self).__init__(
            embedx=F.EmbedID(post_size, hidden),
            embedy=F.EmbedID(cmnt_size, hidden),
            H=L.LSTM(hidden, hidden),
            W=L.Linear(hidden, cmnt_size),
        )

    def __call__(self, jline, eline):
        self.H.reset_state()
        for i in range(len(jline)):
            wid = post_vocab[jline[i]]
            x_k = self.embedx(chainer.Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
            x_k = self.embedx(chainer.Variable(np.array([post_vocab['<eos>']], dtype=np.int32)))
            tx = chainer.Variable(np.array([cmnt_vocab[eline[0]]], dtype=np.int32))
            h = self.H(x_k)
            accum_loss = F.softmax_cross_entropy(self.W(h), tx)
            for i in range(len(eline)):
                wid = cmnt_vocab[eline[i]]
                x_k = self.embedy(chainer.Variable(np.array([wid], dtype=np.int32)))
                next_wid = cmnt_vocab['<eos>'] if (i == len(eline) - 1) else cmnt_vocab[eline[i + 1]]
                tx = chainer.Variable(np.array([next_wid], dtype=np.int32))
                h = self.H(x_k)
                loss = F.softmax_cross_entropy(self.W(h), tx)
                accum_loss = loss if accum_loss is None else accum_loss + loss
            return accum_loss

demb = 100
model = EncoderDecoder(post_w_num, cmnt_w_num, demb)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    for i in range(len(post_lines) - 1):
        jln = post_lines[i].split()
        jlnr = jln[::-1]
        eln = cmnt_lines[i].split()
        model.H.reset_state()
        model.cleargrads()
        loss = model(jlnr, eln)
        loss.backward()
        loss.unchain_backward()     # truncate
        optimizer.update()
    outfile = "mt-" + str(epoch) + ".model"
    chainer.serializers.save_npz(outfile, model)



print(epoch, ":", )



def main():
    pass

if __name__ == '__main__':
    main()