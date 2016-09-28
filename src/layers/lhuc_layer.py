
import numpy, time, cPickle, gzip, sys, os, copy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import logging


class SigmoidLayer_LHUC(object):
    def __init__(self, rng, x, n_in, n_out, W = None, b = None, c = None, activation = T.tanh, p=0.0, training=0):

        self.x = x

        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.x = T.switch(srng.binomial(size=x.shape, p=p), x, 0)
            else:
                self.x =  (1-p) * x


        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_value = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_value,
                              name='W', borrow=True)
        if b is None:
            b = theano.shared(value=numpy.zeros((n_out,),
                              dtype=theano.config.floatX),
                              name='b', borrow=True)
        if c is None:
            c_value = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_out),
                      size=(n_out,)), dtype=theano.config.floatX)
            c = theano.shared(value=c_value, name='c', borrow=True)

        self.W = W
        self.b = b
        self.c = c


        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.delta_c = theano.shared(value=numpy.zeros((n_out),
                                     dtype=theano.config.floatX), name='delta_c')

        self.output = T.dot(self.x, self.W) + self.b
        self.output = activation(self.output)
        self.output = 2.* T.nnet.sigmoid(self.c) * self.output

        self.params = [self.W, self.b, self.c]
        self.delta_params = [self.delta_W, self.delta_b, self.delta_c]

    def errors(self, y):
        L = T.sum( (self.output-y)*(self.output-y), axis=1 )
        errors = T.mean(L)
        return (errors)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates
