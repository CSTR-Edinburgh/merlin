###THEANO_FLAGS='cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu0,floatX=float32,exception_verbosity=high' python dnn.py
"""
"""
import pickle
import os
import sys
import time

import numpy# as np
import gnumpy as gnp

#cudamat

#import theano
#import theano.tensor as T

import logging

class DNN(object):

    def __init__(self, numpy_rng, n_ins=100,
                 n_outs=100, l1_reg = None, l2_reg = None,
                 hidden_layer_sizes=[500, 500],
                 hidden_activation='tanh', output_activation='linear'):

        logger = logging.getLogger("DNN initialization")

        self.n_layers = len(hidden_layer_sizes)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        assert self.n_layers > 0

        self.W_params = []
        self.b_params = []
        self.mW_params = []
        self.mb_params = []

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i-1]
            W_value = gnp.garray(numpy_rng.normal(0.0, 1.0/numpy.sqrt(input_size), size=(input_size, hidden_layer_sizes[i])))
            b_value = gnp.zeros(hidden_layer_sizes[i])
            mW_value = gnp.zeros((input_size, hidden_layer_sizes[i]))
            mb_value = gnp.zeros(hidden_layer_sizes[i])
            self.W_params.append(W_value)
            self.b_params.append(b_value)
            self.mW_params.append(mW_value)
            self.mb_params.append(mb_value)

        #output layer
        input_size = hidden_layer_sizes[self.n_layers-1]
        W_value = gnp.garray(numpy_rng.normal(0.0, 1.0/numpy.sqrt(input_size), size=(input_size, n_outs)))
        b_value = gnp.zeros(n_outs)
        mW_value = gnp.zeros((input_size, n_outs))
        mb_value = gnp.zeros(n_outs)
        self.W_params.append(W_value)
        self.b_params.append(b_value)
        self.mW_params.append(mW_value)
        self.mb_params.append(mb_value)

    def backpropagation(self, train_set_y):
#        (train_set_x, train_set_y) = train_xy

        # assuming linear output and square error cost function
        observation_error = self.final_layer_output - train_set_y

        self.W_grads = []
        self.b_grads = []
        current_error = observation_error
        current_activation = self.activations[-1]
        current_W_grad = gnp.dot(current_activation.T, observation_error)
        current_b_grad = gnp.dot(gnp.ones((1, observation_error.shape[0])), observation_error)
        self.W_grads.append(current_W_grad)
        self.b_grads.append(current_b_grad)

        propagate_error = gnp.dot(observation_error, self.W_params[self.n_layers].T) # final layer is linear output, gradient is one
        for i in reversed(list(range(self.n_layers))):
            current_activation = self.activations[i]
            current_gradient = 1.0 - current_activation ** 2
            current_W_grad = gnp.dot(current_activation.T, propagate_error)
            current_b_grad = gnp.dot(gnp.ones((1, propagate_error.shape[0])), propagate_error)
            propagate_error = gnp.dot(propagate_error, self.W_params[i].T) * current_gradient

            self.W_grads.insert(0, current_W_grad)
            self.b_grads.insert(0, current_b_grad)


    def feedforward(self, train_set_x):
        self.activations = []

        self.activations.append(train_set_x)

        for i in range(self.n_layers):
            current_activations = gnp.tanh(gnp.dot(self.activations[i], self.W_params[i]) + self.b_params[i])
            self.activations.append(current_activations)

        #output layers
        self.final_layer_output = gnp.dot(self.activations[self.n_layers], self.W_params[self.n_layers]) + self.b_params[self.n_layers]

    def gradient_update(self, batch_size, learning_rate, momentum):

        multiplier = learning_rate / batch_size;
        for i in range(len(self.W_grads)):

            if i >= len(self.W_grads) - 2:
                local_multiplier = multiplier * 0.5
            else:
                local_multiplier = multiplier

            self.W_grads[i] = (self.W_grads[i] + self.W_params[i] * self.l2_reg) * local_multiplier
            self.b_grads[i] = self.b_grads[i] * local_multiplier   # + self.b_params[i] * self.l2_reg

            #update weights and record momentum weights
            self.mW_params[i] = (self.mW_params[i] * momentum) - self.W_grads[i]
            self.mb_params[i] = (self.mb_params[i] * momentum) - self.b_grads[i]
            self.W_params[i] += self.mW_params[i]
            self.b_params[i] += self.mb_params[i]
#        print   self.W_params[0].shape, self.W_params[len(self.W_params)-1].shape

    def finetune(self, train_xy, batch_size, learning_rate, momentum):
        (train_set_x, train_set_y) = train_xy

        train_set_x = gnp.as_garray(train_set_x)
        train_set_y = gnp.as_garray(train_set_y)

        self.feedforward(train_set_x)
        self.backpropagation(train_set_y)
        self.gradient_update(batch_size, learning_rate, momentum)

        self.errors = gnp.sum((self.final_layer_output - train_set_y) ** 2, axis=1)

        return  self.errors.as_numpy_array()

    def parameter_prediction(self, test_set_x):
        test_set_x = gnp.as_garray(test_set_x)

        current_activations = test_set_x

        for i in range(self.n_layers):
            current_activations = gnp.tanh(gnp.dot(current_activations, self.W_params[i]) + self.b_params[i])

        final_layer_output = gnp.dot(current_activations, self.W_params[self.n_layers]) + self.b_params[self.n_layers]

        return  final_layer_output.as_numpy_array()

#    def parameter_prediction(self, test_set_x):  #, batch_size

#        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

#        test_out = theano.function([], self.final_layer.output,
#              givens={self.x: test_set_x[0:n_test_set_x]})
#        predict_parameter = test_out()
#        return predict_parameter


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

    hidden_layer_sizes = [512, 512, 512]

#    test_DBN(train_scp, valid_scp, log_dir, model_dir, n_ins, n_outs, hidden_layer_sizes,
#             finetune_lr, pretraining_epochs, pretrain_lr, training_epochs, batch_size)

    dnn_generation()
