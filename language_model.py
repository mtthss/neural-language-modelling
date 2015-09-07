import nltk
import random
import math
import cPickle as pk
import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F


__author__ = 'matteo'


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
batchsize = 30

units = pk.load(open("./models_continue/units.pkl","r"))
vocab = pk.load(open("./models_continue/vocab.pkl","r"))
model = pk.load(open("./models_continue/model_seed_112.pkl","r"))

sample1 = "the poll was conducted for the times mirror center for the people \
           and the press in washington audaciously pretenciously pursued"
sample2 = "the poll conducted was times the for mirror for the people \
           the and center press in washington audaciously pretenciously pursued"
original_ref = "International organized crime takes many forms, including trafficking in drugs, laundering of criminal proceeds, and smuggling of human beings, arms, and wildlife. \
       The Columbian drug cartels, based in Medellin and Cali, are the major source of cocaine smuggled into the US. \
       Panama, Mexico, Guatemala, Nigeria, Nigeria, and Cuba have been identified as conduits for drugs into the US; and financial institutions in the US, Austria, Panama, Switzerland, Ecuador, Russia and Nigeria have been accused of money laundering. \
       Officials throughout Central America have been accused of corruption for allowing drug smuggling. \
       Manuel Noriega, the deposed Panamanian ruler, was accused by the US of accepting bribes to allow use of Panama as conduit for cocaine bound for the US and as a money-laundering haven. \
       The Columbian cartels has worked with the Italian Mafia to smuggle drugs into the US and Western Europe. \
       The cartels and the Mafia have also joined with the Chinese Triads, and with gangs from the former Soviet Union and Eastern Europe to engage in drug trafficking and money laundering in Europe and the UK. \
       Pakistani tribal leaders have been smuggling heroin in Western Europe via Russia and Ukraine. \
       Chinese organizations have smuggled illegal aliens into the US, and Italian organizations have smuggled aliens, mainly prostitutes, into Italy. \
       Nigerian, Russian, and Israeli gangs have smuggled arms, and crime organizations have smuggled cigarettes from the US into Canada, where they are highly taxed. \
       Organized networks in Africa have been trafficking in wildlife, including rhinos, chimpanzees, and rare birds."

l = sent_detector.tokenize(original_ref)
random.shuffle(l)
random_ref1 = " ".join(l)
random.shuffle(l)
random_ref2 = " ".join(l)
random.shuffle(l)
random_ref3 = " ".join(l)

def convert(txt):
    words = txt.replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            dataset[i] = vocab["<UKN>"]
        else:
            dataset[i] = vocab[word]
    return dataset

def forward_one_step(x_data, y_data, state, train=True):
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

def make_initial_state(batchsize=batchsize, train=False):
    return {name: chainer.Variable(np.zeros((batchsize, units),dtype=np.float32),volatile=not train) for name in ('c1', 'h1', 'c2', 'h2')}

def evaluate(dataset):
    sum_log_perp = np.zeros(())
    state = make_initial_state(batchsize=30, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = dataset[i:i + 1]
        y_batch = dataset[i + 1:i + 2]
        state, loss = forward_one_step(x_batch, y_batch, state, train=False)
        sum_log_perp += loss.data.reshape(())
    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))

conv_sample1 = convert(sample1)
conv_sample2 = convert(sample2)
conv_orig = convert(original_ref)
conv_rand1 = convert(random_ref1)
conv_rand2 = convert(random_ref2)
conv_rand3 = convert(random_ref3)
print evaluate(conv_sample1)
print evaluate(conv_sample2)
print evaluate(conv_orig)
print evaluate(conv_rand1)
print evaluate(conv_rand2)
print evaluate(conv_rand3)