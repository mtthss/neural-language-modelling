#!/usr/bin/env python
import urllib

host = 'https://raw.githubusercontent.com'
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.train.txt' % host,
    './data/ptb.train.txt')
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.valid.txt' % host,
    './data/ptb.valid.txt')
urllib.urlretrieve(
    '%s/tomsercu/lstm/master/data/ptb.test.txt' % host,
    './data/ptb.test.txt')