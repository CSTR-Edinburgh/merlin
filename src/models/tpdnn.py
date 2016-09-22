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

import cPickle
import os
import sys
import time

import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.layers import LinearLayer, SigmoidLayer, HiddenLayer, TokenProjectionLayer, SplitHiddenLayer
from utils.providers import ListDataProvider

from training_schemes.rprop import compile_RPROP_train_function

import logging

class TokenProjectionDNN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 n_outs=10, l1_reg = None, l2_reg = None, 
                 hidden_layers_sizes=[500, 500], 
                 hidden_activation='tanh', output_activation='linear',
                 projection_insize=100, projection_outsize=10,
                 first_layer_split=True, expand_by_minibatch=False,
                 initial_projection_distrib='gaussian',
		         use_rprop=0, rprop_init_update=0.001):
                        ## beginning at label index 1, 5 blocks of 49 inputs each to be projected to 10 dim.

        logger = logging.getLogger("TP-DNN initialization")
        
        self.projection_insize = projection_insize
        self.projection_outsize = projection_outsize

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
        self.numpy_rng = numpy_rng

        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        if expand_by_minibatch:
            self.x_proj = T.ivector('x_proj') 
        else:
            self.x_proj = T.matrix('x_proj') 
        self.y = T.matrix('y') 

        if expand_by_minibatch:
            z = theano.tensor.zeros((self.x_proj.shape[0], self.projection_insize))
            indexes = self.x_proj 
            one_hot = theano.tensor.set_subtensor(z[theano.tensor.arange(self.x_proj.shape[0]), indexes], 1)
            
            projection_input = one_hot
        else:
            projection_input = self.x_proj



        ## Make projection layer        
        self.projection_layer = TokenProjectionLayer(rng=numpy_rng,
                                        input=projection_input,
                                        projection_insize = self.projection_insize,
                                        projection_outsize = self.projection_outsize,
                                        initial_projection_distrib=initial_projection_distrib)
 
        self.params.extend(self.projection_layer.params)
        self.delta_params.extend(self.projection_layer.delta_params)

        first_layer_input = T.concatenate([self.x, self.projection_layer.output], axis=1)



        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins + self.projection_outsize
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = first_layer_input
            else:
                layer_input = self.sigmoid_layers[-1].output

            if i == 0 and first_layer_split:
                sigmoid_layer = SplitHiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in1=n_ins, n_in2=self.projection_outsize,
                                            n_out=hidden_layers_sizes[i],
                                            activation=T.tanh)  ##T.nnet.sigmoid)  #             
            else:
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

        ## params for 2 hidden layers, projection, first split layer, will look like this:
        ## [W_proj; W_1a, W_1b, b_1; W_2 b_2; W_o, b_o]

        ### MSE
        self.finetune_cost = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))
        
        self.errors = T.mean(T.sum( (self.final_layer.output-self.y)*(self.final_layer.output-self.y), axis=1 ))
<<<<<<< .mine

        #### Could apply regularisation to ALL params (including biases, unlike in dnn.py,
        ####    and projection weights). Note that no regularisation was used in tpdnn.py
        ####    before June 2015 -- not as in IS2015 paper (misreported)
        ### L1-norm
=======

        #### Apply regularisation to ALL params (including biases, unlike in dnn.py,
        ####    and projection weights). Note that no regularisation was used in tpdnn.py
        ####    before June 2015 -- not in IS2015 paper (misreported)
        ### L1-norm
        if self.l1_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i]
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        ### L2-norm
        if self.l2_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i]
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

#         ### L1-norm
>>>>>>> .r181
#         if self.l1_reg is not None:
#             for i in xrange(self.n_layers):
#                 W = self.params[i]
#                 self.finetune_cost += self.l1_reg * (abs(W).sum())
# 
#         ### L2-norm
#         if self.l2_reg is not None:
#             for i in xrange(self.n_layers):
#                 W = self.params[i]
#                 self.finetune_cost += self.l2_reg * T.sqr(W).sum()


    def get_projection_weights(self):
        weights = self.params[0].get_value()
        return weights
    
    def get_weights(self, i):
        weights = self.params[i].get_value()
        return weights
    
    
    
    def zero_projection_weights(self):
        old_weights = self.params[0].get_value()
        new_weights = numpy.asarray(numpy.zeros(numpy.shape(old_weights)), dtype=theano.config.floatX)
        self.params[0].set_value(new_weights, borrow=True)
            
    def initialise_projection_weights(self):
        old_weights = self.params[0].get_value()
        new_weights = numpy.asarray(self.numpy_rng.normal(0.0, 0.1, size=numpy.shape(old_weights)), \
                                                                dtype=theano.config.floatX)
        self.params[0].set_value(new_weights, borrow=True)
        
        

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_x_proj, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_x_proj, valid_set_y) = valid_shared_xy

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate') ## osw temp
        momentum = T.fscalar('momentum')           ## osw temp
        ##proj_learning_rate = T.dscalar('proj_learning_rate') ## osw temp
        
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

        def make_updates_plain(param_list, delta_param_list, gparam_list, lr_list, params_to_update):
            updates = OrderedDict()
            for dparam, gparam, lrate in zip(delta_param_list, gparam_list, lr_list):
                updates[dparam] = momentum * dparam - gparam * lrate
            for dparam, param in zip(delta_param_list, param_list):
                updates[param] = param + updates[dparam]            
            return updates
            
    
        ## Define updates over various subsets of model parameters. These will be used
        ## in various compiled training/inference functions.
        
        ## As a guide to the structure of params, params for 2 hidden layers, projection, 
        ## first split layer, will look like this:
        # i:   0       1    2     3    4   5    6    7
        ##   [W_proj; W_1a, W_1b, b_1; W_2 b_2; W_o, b_o]

        '''
        updates -- all params
        subword_updates: exclude parameters at 0 and 2  -- proj. weights and proj. half of split layer
        word_updates: exclude all but the word half of the split layer, and bias of that layer, and projection
        projection_updates: exclude all but parameters at 0 -- projection layer
        '''
        
        all_params = range(len(self.params))
        subword_params = [i for i in all_params if i not in [0,2]]
        word_params = [0,2,3]
        projection_params = [0]
        

        if self.use_rprop:
            print '========USING RPROP ========='
            updates = compile_RPROP_train_function(self, gparams)
            subword_updates = compile_RPROP_train_function(self, gparams, params_to_update=subword_params)
            word_updates = compile_RPROP_train_function(self, gparams, params_to_update=word_params)
            projection_updates = compile_RPROP_train_function(self, gparams, params_to_update=projection_params)
            on_unused_input_value = 'warn'
             
        else:    
            print '========NOT USING RPROP ========='            
            updates = make_updates_plain(self.params, self.delta_params, gparams, lr_list, all_params)
            subword_updates = make_updates_plain(self.params, self.delta_params, gparams, lr_list, subword_params)
            word_updates = make_updates_plain(self.params, self.delta_params, gparams, lr_list, word_params)
            projection_updates = make_updates_plain(self.params, self.delta_params, gparams, lr_list, projection_params)
            on_unused_input_value = 'raise'  ## Theano's default 
            
        ##### OLDER VERSION:--
        '''
        ## All updates:
        updates = OrderedDict()
        layer_index = 0
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam * lr_list[layer_index]
            layer_index += 1

        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        ## These updates exclude parameters at 0 and 2  -- proj. weights and proj. half of split layer
        subword_updates = OrderedDict()
        for (i, (dparam, gparam)) in enumerate(zip(self.delta_params, gparams)):
            if i not in [0,2]:  ## proj weights and proj half of split layer
                subword_updates[dparam] = momentum * dparam - gparam * lr_list[i]

        for (i, (dparam, param)) in enumerate(zip(self.delta_params, self.params)):
            if i not in [0,2]:  ## proj weights and proj half of split layer
                subword_updates[param] = param + subword_updates[dparam]

        ## These updates exclude parameters at 1 -- subword half of split layer
        ### NO!!! -- just the word half of the split layer, and bias of that layer
        word_updates = OrderedDict()
        for (i, (dparam, gparam)) in enumerate(zip(self.delta_params, gparams)):
            if i in [0,2,3]:  
                word_updates[dparam] = momentum * dparam - gparam * lr_list[i]

        for (i, (dparam, param)) in enumerate(zip(self.delta_params, self.params)):
            if i in [0,2,3]: 
                word_updates[param] = param + word_updates[dparam]


        ## These updates exclude all but parameters at 0 -- projection layer
        projection_updates = OrderedDict()
        for (i, (dparam, gparam)) in enumerate(zip(self.delta_params, gparams)):
            if i == 0: 
                projection_updates[dparam] = momentum * dparam - gparam * lr_list[i]

        for (i, (dparam, param)) in enumerate(zip(self.delta_params, self.params)):
            if i == 0: 
                projection_updates[param] = param + projection_updates[dparam]
        '''


        ## Update all params -- maybe never used:
        print 'compile train_all_fn'
        train_all_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
	      on_unused_input=on_unused_input_value,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.x_proj: train_set_x_proj[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
                     
        ## Update all but word-projection part of split first hidden layer and projection weights  
        print 'compile train_subword_fn'                        
        train_subword_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=subword_updates,
              on_unused_input=on_unused_input_value,              
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.x_proj: train_set_x_proj[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        print 'compile train_word_fn' 
        train_word_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=word_updates,
              on_unused_input=on_unused_input_value,              
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.x_proj: train_set_x_proj[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})                                          

        print 'compile infer_projections_fn -- NB: to operate by default on validation set' 
        infer_projections_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=projection_updates,
              on_unused_input=on_unused_input_value,              
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.x_proj: valid_set_x_proj[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})                                                                               
                                  
                           
                                          
        valid_fn = theano.function([], 
              outputs=self.errors,
              givens={self.x: valid_set_x,
                      self.x_proj: valid_set_x_proj,
                      self.y: valid_set_y})

        valid_score_i = theano.function([index], 
              outputs=self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.x_proj: valid_set_x_proj[index * batch_size:
                                          (index + 1) * batch_size],                                        
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        print 'finished Theano function compilation'  
        return train_all_fn, train_subword_fn,  train_word_fn, infer_projections_fn, valid_fn, valid_score_i

    def parameter_prediction(self, test_set_x, test_set_x_proj):  #, batch_size

        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.final_layer.output,
              givens={self.x: test_set_x[0:n_test_set_x],
                      self.x_proj: test_set_x_proj[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter
        
    def generate_top_hidden_layer(self, test_set_x):
        
        n_test_set_x = test_set_x.get_value(borrow=True).shape[0]

        test_out = theano.function([], self.sigmoid_layers[-2].output,
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

