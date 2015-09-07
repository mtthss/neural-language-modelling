#!/usr/bin/env python
"""
Sample script of recurrent neural network language model.
Modifying the chainer tutorial code on recurrent networks
The chainer tutorial in turn is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm
"""
import sys
import math
import time
import argparse

import six
import numpy as np
import cPickle as pk

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

# Parsing args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

n_epoch = 10        # number of epochs
vector_size = 250
n_units = 8      # number of units per layer
batchsize = 1       # minibatch size
bprop_len = 15      # length of truncated BPTT
grad_clip = 5       # gradient norm threshold to clip
vocab = {}          # vocab data structure

def load_data(filename):
    global vocab, n_vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('./data/duc-train.txt')
valid_data = load_data('./data/duc-dev.txt')
test_data = load_data('./data/duc-test.txt')
print('#vocab =', len(vocab))

# pickle unpickle
pk.dump(vocab, open("./models_continue_supersmall/vocab.pkl","wb"))
pk.dump(n_units, open("./models_continue_supersmall/units.pkl","wb"))
old_model = pk.load(open("./models_continue/model2699.pkl","r"))

# Prepare RNNLM model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), vector_size),
                            l1_x=F.Linear(vector_size, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))

# initilize params values
count = 0
for param in model.parameters:
    if count==0:
        param[:] = old_model.parameters[0]
    else:
        a = param.shape[0]
        try:
            b = param.shape[1]
        except:
            b = None
        if b!=None:
            param[:] = old_model.parameters[count][0:a, 0:b]
        else:
            param[:] = old_model.parameters[count][0:a]
    count += 1

# initialize gpu
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# Neural net architecture
def forward_one_step(x_data, y_data, state, train=True):
    if args.gpu >= 0:
        x_data = cuda.to_gpu(x_data)
        y_data = cuda.to_gpu(y_data)
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)

    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=True)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=True)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(h2)

    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax_cross_entropy(y, t)

def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# Evaluation routine
def evaluate(dataset):
    sum_log_perp = mod.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = dataset[i:i + 1]
        y_batch = dataset[i + 1:i + 2]
        state, loss = forward_one_step(x_batch, y_batch, state, train=False)
        sum_log_perp += loss.data.reshape(())
    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))

# Learning loop
whole_len = train_data.shape[0]
jump = whole_len // batchsize
cur_log_perp = mod.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
state = make_initial_state()
accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
best_perplexity = 4721800.46

print('going to train {} iterations'.format(jump * n_epoch))
for i in six.moves.range(jump * n_epoch):
    x_batch = np.array([train_data[(jump * j + i) % whole_len]
                        for j in six.moves.range(batchsize)])
    y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                        for j in six.moves.range(batchsize)])
    state, loss_i = forward_one_step(x_batch, y_batch, state)
    accum_loss += loss_i
    cur_log_perp += loss_i.data.reshape(())

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))

        optimizer.clip_grads(grad_clip)
        optimizer.update()

    if (i + 1) % 100 == 0:

        # compute perplexity
        now = time.time()
        throuput = 100. / (now - cur_at)
        perp = math.exp(cuda.to_cpu(cur_log_perp) / 100)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        cur_at = now
        cur_log_perp.fill(0)

        # save model
        if perp < best_perplexity:
            best_perplexity = perp
            pk.dump(model, open("./models_continue_supersmall/model"+str(i)+".pkl","wb"))

    if (i + 1) % 1000 == 0:#(i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        cur_at += time.time() - now  # skip time of evaluation
    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)

#save model
pk.dump(model, open("./models_continue_supersmall/model.pkl","wb"))
pk.dump(vocab, open("./models_continue_supersmall/vocab.pkl","wb"))
pk.dump(n_units, open("./models_continue/units.pkl","wb"))