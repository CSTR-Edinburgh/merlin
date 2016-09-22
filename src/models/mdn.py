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
import cPickle
import os
import sys
import time
import math

import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.layers import LinearLayer, SigmoidLayer, HiddenLayer, GeneralLayer, MixtureDensityOutputLayer
from utils.providers import ListDataProvider

from training_schemes.rprop import compile_RPROP_train_function

import logging

class MixtureDensityNetwork(object):
    def __init__(self, numpy_rng, n_ins=784, n_outs=24, l1_reg = None, l2_reg = None, 
                 hidden_layers_sizes=[500, 500], 
                 hidden_activation='tanh', output_activation='linear', var_floor=0.01, 
                 n_component=1, beta_opt=False, use_rprop=0, rprop_init_update=0.001, 
                 eff_sample_size=0.8, mean_log_det=-100.0):

        logger = logging.getLogger("Multi-stream DNN initialization")

        self.sigmoid_layers = []
        self.params = []
        self.delta_params   = []

        self.final_layers = []
        
        self.n_outs = n_outs

        self.n_layers = len(hidden_layers_sizes)
        
        self.output_activation = output_activation
        self.var_floor = var_floor

        self.use_rprop = use_rprop
        self.rprop_init_update = rprop_init_update
        
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.beta_opt = beta_opt
        self.eff_sample_size = eff_sample_size
        self.mean_log_det = mean_log_det
        
        assert self.n_layers > 0

        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        self.y = T.matrix('y') 

        for i in xrange(self.n_layers):
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

        hidden_output_size = hidden_layers_sizes[-1]
        
        self.final_layer = MixtureDensityOutputLayer(rng = numpy_rng,
                                                input = sigmoid_layer.output,
                                                n_in = hidden_output_size,
                                                n_out = self.n_outs, 
                                                n_component = n_component, 
                                                var_floor = self.var_floor)
        self.params.extend(self.final_layer.params)
        self.delta_params.extend(self.final_layer.delta_params)        

        ### Maximum likelihood
        self.finetune_cost = 0.0

        self.errors = 0.0

        epsd = self.eff_sample_size**(-2.0/(n_outs + 2.0))
        beta = (epsd - 1.0) + math.sqrt(epsd*(epsd - 1.0))

        if self.beta_opt:
            assert n_component == 1, "beta optimisation only implemented for single-component MDNs"
            for i in xrange(n_component):  #n_component
                sigma = self.final_layer.sigma[:, i*n_outs:(i+1)*n_outs]
                mu    = self.final_layer.mu[:, i*n_outs:(i+1)*n_outs]
                mix_weight = self.final_layer.mix[:, i]
                
                xEx = -0.5 * beta * T.sum(((self.y - mu)**2) * T.inv(sigma), axis=1)
                exponent = (0.5 * (n_outs + 2.0) * T.log(1 + beta)) + xEx
                point_fit = T.exp(exponent) - beta
                
                log_det_mult = -0.5 * beta * T.sum(T.log(sigma), axis=1)
                
                log_det_mult += (0.5 * beta * self.mean_log_det) # normalise by mean_log_det
                
                beta_obj = (mix_weight**2) * point_fit * T.exp(log_det_mult)
                
                self.finetune_cost += -T.mean(beta_obj)
    
            # lines to compute debugging information for later printing
            #self.errors = T.min(T.min(T.log(sigma), axis=1))
            #self.errors = T.mean(T.sum(T.log(sigma), axis=1)) # computes mean_log_det
            #self.errors = -xEx # (vector quantity) should be about 0.5 * beta * n_outs
            #self.errors = point_fit  # (vector quantity) should be about one
            #self.errors = T.mean(T.exp(exponent)) / T.exp(T.max(exponent)) # fraction of the data used, should be about efficiency
            #self.errors = T.mean(point_fit) # should be about one
            #self.errors = log_det_mult # (vector quantity) about zero, or always less if using Rprop
            #self.errors = beta_obj # (vector quantity) objective function terms
            #self.errors = self.finetune_cost # disable this line below when debugging
        else:     

            all_mix_prob = []
            
            print   n_component
            for i in xrange(n_component):  #n_component
                sigma = self.final_layer.sigma[:, i*n_outs:(i+1)*n_outs]
                mu    = self.final_layer.mu[:, i*n_outs:(i+1)*n_outs]
                mix_weight = self.final_layer.mix[:, i]

                xEx = -0.5 * T.sum(((self.y - mu)**2) * T.inv(sigma), axis=1)
                normaliser = 0.5 * ( n_outs * T.log(2 * numpy.pi) + T.sum(T.log(sigma), axis=1))
                exponent = xEx + T.log(mix_weight) - normaliser
                all_mix_prob.append(exponent)

            max_exponent = T.max(all_mix_prob, axis=0, keepdims=True)
            mod_exponent = T.as_tensor_variable(all_mix_prob) - max_exponent
            
            self.finetune_cost = - T.mean(max_exponent + T.log(T.sum(T.exp(mod_exponent), axis=0)))

            #self.errors = self.finetune_cost
            

        if self.l2_reg is not None:
            for i in xrange(self.n_layers-1):
                W = self.params[i * 2]
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            self.finetune_cost += self.l2_reg * T.sqr(self.final_layer.W_mu).sum()
            self.finetune_cost += self.l2_reg * T.sqr(self.final_layer.W_sigma).sum()
            self.finetune_cost += self.l2_reg * T.sqr(self.final_layer.W_mix).sum()

        self.errors = self.finetune_cost # disable this line if debugging beta_opt


    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

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
        for i in xrange(layer_size):
            lr_list.append(learning_rate)

        ##top 2 layers use a smaller learning rate
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

            train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
                  theano.Param(momentum, default = 0.5)],
                  outputs=self.errors,
                  updates=updates,
                  on_unused_input='ignore',
                  givens={self.x: train_set_x[index * batch_size:
                                              (index + 1) * batch_size],
                          self.y: train_set_y[index * batch_size:
                                              (index + 1) * batch_size]})

        elif self.use_rprop:        
            updates = compile_RPROP_train_function(self, gparams)
            
            ## retain learning rate and momentum to make interface backwards compatible,
            ## but we won't use them, means we have to use on_unused_input='warn'.
            ## Otherwise same function for RPROP or otherwise -- can move this block outside if clause.              
            train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
                  theano.Param(momentum, default = 0.5)],
                  outputs=self.errors,
                  updates=updates,
                  on_unused_input='warn',
                  givens={self.x: train_set_x[index * batch_size:
                                              (index + 1) * batch_size],
                          self.y: train_set_y[index * batch_size:
                                              (index + 1) * batch_size]})   
                                                                                
        valid_fn = theano.function([], 
              outputs=self.errors,
              on_unused_input='ignore',              
              givens={self.x: valid_set_x,
                      self.y: valid_set_y})

        valid_score_i = theano.function([index], 
              outputs=self.errors,
              on_unused_input='ignore',              
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        return train_fn, valid_fn


    def parameter_prediction(self, test_set_x):  #, batch_size
    
        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.mu,
              givens={self.x: test_set_x[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_mix(self, test_set_x):  #, batch_size
    
        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.mix,
              givens={self.x: test_set_x[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_sigma(self, test_set_x):  #, batch_size
    
        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.sigma,
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


