################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

###THEANO_FLAGS='cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu0,floatX=float32,exception_verbosity=high' python dnn.py
"""
"""
import pickle
import os
import sys
import time

import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.layers import LinearLayer, SigmoidLayer, HiddenLayer
from utils.providers import ListDataProvider

from training_schemes.rprop import compile_RPROP_train_function

import logging

class DNN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 n_outs=10, l1_reg = None, l2_reg = None,
                 hidden_layers_sizes=[500, 500],
                 hidden_activation='tanh', output_activation='linear',
                 use_rprop=0, rprop_init_update=0.001):

        logger = logging.getLogger("DNN initialization")

        self.sigmoid_layers = []
        self.params = []
        self.delta_params   = []
        self.n_layers = len(hidden_layers_sizes)

        self.output_activation = output_activation

        self.use_rprop = use_rprop
        self.rprop_init_update = rprop_init_update

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.matrix('y')

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.tanh)  ##T.nnet.sigmoid)  #
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            self.delta_params.extend(sigmoid_layer.delta_params)

        # add final layer
        if self.output_activation == 'linear':
            self.final_layer = LinearLayer(rng = numpy_rng,
                                           input=self.sigmoid_layers[-1].output,
                                           n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)
        elif self.output_activation == 'sigmoid':
            self.final_layer = SigmoidLayer(
                 rng = numpy_rng,
                 input=self.sigmoid_layers[-1].output,
                 n_in=hidden_layers_sizes[-1],
                 n_out=n_outs, activation=T.nnet.sigmoid)
        else:
            logger.critical("This output activation function: %s is not supported right now!" %(self.output_activation))
            sys.exit(1)

        self.params.extend(self.final_layer.params)
        self.delta_params.extend(self.final_layer.delta_params)

        ### MSE
        self.finetune_cost = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))

        self.errors = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))

        ### L1-norm
        if self.l1_reg is not None:
            for i in range(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        ### L2-norm
        if self.l2_reg is not None:
            for i in range(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()


    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size, \
                                                     return_valid_score_i=False):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        layer_size = len(self.params)
        lr_list = []
        for i in range(layer_size):
            lr_list.append(learning_rate)

        ##top 2 layers use a smaller learning rate
        ##hard-code now, change it later
        if layer_size > 4:
            for i in range(layer_size-4, layer_size):
                lr_list[i] = learning_rate * 0.5

        # compute list of fine-tuning updates
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        if self.use_rprop == 0:

            updates = OrderedDict()
            layer_index = 0
            for dparam, gparam in zip(self.delta_params, gparams):
                updates[dparam] = momentum * dparam - gparam * lr_list[layer_index]
                layer_index += 1

            for dparam, param in zip(self.delta_params, self.params):
                updates[param] = param + updates[dparam]

            on_unused_input_value = 'raise'  ## Theano's default

        elif self.use_rprop:
            updates = compile_RPROP_train_function(self, gparams)
            on_unused_input_value = 'warn'


        ## Retain learning rate and momentum to make interface backwards compatible,
        ## even with RPROP where we don't use them, means we have to use on_unused_input='warn'.

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.125),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              on_unused_input=on_unused_input_value,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        valid_fn = theano.function([],
              outputs=self.errors,
              givens={self.x: valid_set_x,
                      self.y: valid_set_y})

        valid_score_i = theano.function([index],
              outputs=self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        if return_valid_score_i:
            return train_fn, valid_fn, valid_score_i
        else:
            return train_fn, valid_fn

    def parameter_prediction(self, test_set_x):  #, batch_size

        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.output,
              givens={self.x: test_set_x[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter

    ## the function to output activations at a hidden layer
    def generate_top_hidden_layer(self, test_set_x, bn_layer_index):

        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.sigmoid_layers[bn_layer_index].output,
              givens={self.x: test_set_x[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter



if __name__ == '__main__':

    train_scp = '/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/nn_scp/train.scp'
    valid_scp = '/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/nn_scp/gen.scp'

    model_dir = '/afs/inf.ed.ac.uk/group/project/dnn_tts/practice/nnets_model'

    log_dir =  '/afs/inf.ed.ac.uk/group/project/dnn_tts/practice/log'

    finetune_lr=0.01
    pretraining_epochs=100
    pretrain_lr=0.01
    training_epochs=100
    batch_size=32

    n_ins = 898
    n_outs = 229

    hidden_layers_sizes = [512, 512, 512]

#    test_DBN(train_scp, valid_scp, log_dir, model_dir, n_ins, n_outs, hidden_layers_sizes,
#             finetune_lr, pretraining_epochs, pretrain_lr, training_epochs, batch_size)

    dnn_generation()
