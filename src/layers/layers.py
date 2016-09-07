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


import numpy, time, cPickle, gzip, sys, os, copy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import logging


class MixtureDensityOutputLayer(object):
    def __init__(self, rng, input, n_in, n_out, n_component, var_floor):
        self.input = input
        
        W_value = rng.normal(0.0, 1.0/numpy.sqrt(n_in), size=(n_in, n_out*n_component))
        self.W_mu = theano.shared(value=numpy.asarray(W_value, dtype=theano.config.floatX), name='W_mu', borrow=True)
        
        self.W_sigma = theano.shared(value=numpy.asarray(W_value.copy(), dtype=theano.config.floatX), name='W_sigma', borrow=True)
        
        W_mix_value = rng.normal(0.0, 1.0/numpy.sqrt(n_in), size=(n_in, n_component))
        self.W_mix = theano.shared(value=numpy.asarray(W_mix_value, dtype=theano.config.floatX), name='W_mix', borrow=True)

        self.mu = T.dot(self.input, self.W_mu) # assume linear output for mean vectors

        #self.sigma = T.nnet.softplus(T.dot(self.input, self.W_sigma)) # + 0.0001
        self.sigma = T.exp(T.dot(self.input, self.W_sigma)) # Zen et al. 2014
        self.sigma = T.maximum(var_floor, self.sigma) # hard variance flooring
        # note: sigma contains variances, so var_floor=0.01 means that
        # the lowest possible standard deviation is 0.1

        self.mix = T.nnet.softmax(T.dot(self.input, self.W_mix))

        self.delta_W_mu    = theano.shared(value = numpy.zeros((n_in, n_out*n_component),
                                           dtype=theano.config.floatX), name='delta_W_mu')
        self.delta_W_sigma = theano.shared(value = numpy.zeros((n_in, n_out*n_component),
                                           dtype=theano.config.floatX), name='delta_W_sigma')
        self.delta_W_mix   = theano.shared(value = numpy.zeros((n_in, n_component),
                                           dtype=theano.config.floatX), name='delta_W_mix')

        
        self.params = [self.W_mu, self.W_sigma, self.W_mix]
        self.delta_params = [self.delta_W_mu, self.delta_W_sigma, self.delta_W_mix]



class LinearLayer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None):
        n_in = int(n_in)  # ensure sizes have integer type
        n_out = int(n_out)# ensure sizes have integer type

        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_value = rng.normal(0.0, 1.0/numpy.sqrt(n_in), size=(n_in, n_out))
            W = theano.shared(value=numpy.asarray(W_value, dtype=theano.config.floatX), name='W', borrow=True)

        if b is None:
            b = theano.shared(value=numpy.zeros((n_out,),
                                        dtype=theano.config.floatX),
                                   name='b', borrow=True)
        
        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.output = T.dot(self.input, self.W) + self.b

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def errors(self, y):
        L = T.sum( (self.output-y)*(self.output-y), axis=1 )
        errors = T.mean(L)
        return (errors)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates

class SigmoidLayer(object):
    def __init__(self, rng, x, n_in, n_out, W = None, b = None, activation = T.tanh, p=0.0, training=0):
        n_in = int(n_in)  # ensure sizes have integer type
        n_out = int(n_out)# ensure sizes have integer type

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
        
        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.output = T.dot(self.x, self.W) + self.b
        self.output = activation(self.output)
        
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def errors(self, y):
        L = T.sum( (self.output-y)*(self.output-y), axis=1 )
        errors = T.mean(L)
        return (errors)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates
        

class GeneralLayer(object):

    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation = 'linear'):
 
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        logger = logging.getLogger('general_layer')        

        # randomly initialise the activation weights based on the input size, as advised by the 'tricks of neural network book'        
        if W is None:
            W_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, self.W) + self.b
        if activation == 'sigmoid':
            self.output = T.nnet.sigmoid(lin_output)

        elif activation == 'tanh':
            self.output = T.tanh(lin_output)

        elif activation == 'linear':
            self.output = lin_output

        elif activation == 'ReLU':  ## rectifier linear unit
            self.output = T.maximum(0.0, lin_output)

        elif activation == 'ReSU':  ## rectifier smooth unit
            self.output = numpy.log(1.0 + numpy.exp(lin_output))

        else:
            logger.critical('the input activation function: %s is not supported right now. Please modify layers.py to support' % (activation))
            raise

        # parameters of the model

        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
    
    def errors(self, y):
        errors = T.mean(T.sum((self.output-y)**2, axis=1))

        return errors

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1,
                 do_pnorm = False, pnorm_order = 1):
        """ Class for hidden layer """
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        
        if W is None:

            W_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W')

        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, self.W) + self.b
        if do_maxout == True:
            self.last_start = n_out - pool_size
            self.tmp_output = lin_output[:,0:self.last_start+1:pool_size]
            for i in range(1, pool_size):
                cur = lin_output[:,i:self.last_start+i+1:pool_size]
                self.tmp_output = T.maximum(cur, self.tmp_output)
            self.output = activation(self.tmp_output)
        elif do_pnorm == True:
            self.last_start = n_out - pool_size
            self.tmp_output = abs(lin_output[:,0:self.last_start+1:pool_size]) ** pnorm_order
            for i in range(1, pool_size):
                cur = abs(lin_output[:,i:self.last_start+i+1:pool_size]) ** pnorm_order
                self.tmp_output = self.tmp_output + cur
            self.tmp_output = self.tmp_output ** (1.0 / pnorm_order)
            self.output = activation(self.tmp_output)
        else:
            self.output = (lin_output if activation is None
                           else activation(lin_output))

#        self.output = self.rectifier_linear(lin_output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
    
    def rectifier_linear(self, x):
        x = T.maximum(0.0, x)
        
        return  x
        
    def rectifier_smooth(self, x):
        x = numpy.log(1.0 + numpy.exp(x))
        
        return  x


class SplitHiddenLayer(object):
    '''
    The nin x nout matrix is vertically split into 2 portions which can be updated 
    independently. 
    
    n_in1 -- by convention, use this part for subword contexts
    n_in2 -- by convention, use this part for word projections
    
    Bias is not split in any way.
    '''
    def __init__(self, rng, input, n_in1, n_in2, n_out, W1=None, W2=None, b=None,
                 activation=T.tanh, do_maxout = False, pool_size = 1,
                 do_pnorm = False, pnorm_order = 1):
        """ Class for hidden layer """
        self.input = input
        #self.n_in = n_in
        self.n_out = n_out
        
        if W1 is None:

            W1_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in1),
                    size=(n_in1, n_out)), dtype=theano.config.floatX)

            W1 = theano.shared(value=W1_values, name='W1', borrow=True)

        if W2 is None:

            W2_values = numpy.asarray(rng.normal(0.0, 1.0/numpy.sqrt(n_in1),
                    size=(n_in2, n_out)), dtype=theano.config.floatX)

            W2 = theano.shared(value=W2_values, name='W2', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W1 = W1
        self.W2 = W2
        self.b = b

        self.delta_W1 = theano.shared(value = numpy.zeros((n_in1,n_out),
                                     dtype=theano.config.floatX), name='delta_W1')

        self.delta_W2 = theano.shared(value = numpy.zeros((n_in2,n_out),
                                     dtype=theano.config.floatX), name='delta_W2')
                                     
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        lin_output = T.dot(input, T.concatenate([self.W1, self.W2])) + self.b
        if do_maxout == True:
            self.last_start = n_out - pool_size
            self.tmp_output = lin_output[:,0:self.last_start+1:pool_size]
            for i in range(1, pool_size):
                cur = lin_output[:,i:self.last_start+i+1:pool_size]
                self.tmp_output = T.maximum(cur, self.tmp_output)
            self.output = activation(self.tmp_output)
        elif do_pnorm == True:
            self.last_start = n_out - pool_size
            self.tmp_output = abs(lin_output[:,0:self.last_start+1:pool_size]) ** pnorm_order
            for i in range(1, pool_size):
                cur = abs(lin_output[:,i:self.last_start+i+1:pool_size]) ** pnorm_order
                self.tmp_output = self.tmp_output + cur
            self.tmp_output = self.tmp_output ** (1.0 / pnorm_order)
            self.output = activation(self.tmp_output)
        else:
            self.output = (lin_output if activation is None
                           else activation(lin_output))

#        self.output = self.rectifier_linear(lin_output)
        
        # parameters of the model
        self.params = [self.W1, self.W2, self.b]
        self.delta_params = [self.delta_W1, self.delta_W2, self.delta_b]
    
    def rectifier_linear(self, x):
        x = T.maximum(0.0, x)
        
        return  x
        
    def rectifier_smooth(self, x):
        x = numpy.log(1.0 + numpy.exp(x))
        
        return  x



                                        


class TokenProjectionLayer(object):
    '''
    A single projection, not shared. MErging or outputs with non-projected inputs is handled elsewhere.
    '''

    def __init__(self, rng, input, projection_insize, projection_outsize, initial_projection_distrib='gaussian'):
        
        self.input = input
        self.params = []
        self.delta_params = []
        #self.n_in = n_in
        self.projection_insize = projection_insize
        self.projection_outsize = projection_outsize 

        
        if initial_projection_distrib == 'gaussian':
            W_values = numpy.asarray(rng.normal(0.0, 0.1,
                                    size=(projection_insize, projection_outsize)),
                                    dtype=theano.config.floatX)
        elif initial_projection_distrib == 'uniform':
            
            # W_values = numpy.asarray(rng.uniform(low=-0.02, high=0.02,
            W_values = numpy.asarray(rng.uniform(low=0.0, high=1.0,
            size=(projection_insize, projection_outsize)),
            dtype=theano.config.floatX)
        
        elif initial_projection_distrib == 'zeros':
            
            W_values = numpy.zeros((projection_insize, projection_outsize),
            dtype=theano.config.floatX)
   
        elif initial_projection_distrib == '4mix':
    
            ## TODO -- generalise to other n_modes and higher deimneionsal CVs
            means = [(-0.5, -0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.5)]
            var = (0.1, 0.1)
            W_prelim = []            
            for mean in means:
                W_prelim.append(
                    numpy.asarray(rng.normal(mean, var,
                    size=(projection_insize / len(means), projection_outsize)),
                    dtype=theano.config.floatX)
                    )  
            W_values = numpy.vstack(W_prelim)  
            rng.shuffle(W_values)
                                                     
        else:
            
            sys.exit( 'initial_projection_distrib must be one of: gaussian, uniform' )

        W = theano.shared(value=W_values, name='W', borrow=True)

        delta_W = theano.shared(value = numpy.zeros((projection_insize, projection_outsize),
                                 dtype=theano.config.floatX), name='delta_W')

        self.params.append(W)
        self.delta_params.append(delta_W)

        self.output = T.dot(self.input, W)
                
        

class dA(object):
    def __init__(self, theano_rng = None, input = None, 
                 n_visible= None, n_hidden= None, W = None, bhid = None,
                 bvis = None, activation=None, firstlayer = 1, variance   = None ):

        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        if not W:
            initial_W = numpy.asarray(theano_rng.normal(0.0, 1.0/numpy.sqrt(n_in),
                    size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value = initial_W, name ='W')
            #initial_W = numpy.asarray( numpy_rng.uniform( 
            #          low  = -4*numpy.sqrt(6./(n_hidden+n_visible)), 
            #          high =  4*numpy.sqrt(6./(n_hidden+n_visible)), 
            #          size = (n_visible, n_hidden)),
            #                           dtype = theano.config.floatX)

        if not bvis:
            bvis = theano.shared(value = numpy.zeros(n_visible, 
                                         dtype = theano.config.floatX))

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                                dtype = theano.config.floatX), name ='b')


        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T 
        self.theano_rng = theano_rng
        self.activation = activation

        if input == None : 
            self.x = T.dmatrix(name = 'input') 
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        # first layer, use Gaussian noise
        self.firstlayer = firstlayer

        if self.firstlayer == 1 :
            if variance == None :
                self.var = T.vector(name = 'input')
            else :
                self.var = variance
        else :
            self.var = None

    def apply_activation(self, lin_output, activation):
        if activation == 'SIGMOID':
            final_output = T.nnet.sigmoid(lin_output)

        elif activation == 'TANH':
            final_output = T.tanh(lin_output)

        elif activation == 'LINEAR':
            final_output = lin_output

        elif activation == 'ReLU':  ## rectifier linear unit
            final_output = T.maximum(0.0, lin_output)

        elif activation == 'ReSU':  ## rectifier smooth unit
            final_output = numpy.log(1.0 + numpy.exp(lin_output))

        else:
            self.logger.critical('the input activation function: %s is not supported right now. Please modify layers.py to support' % (activation))
            raise

        return final_output

    def get_corrupted_input(self, input, corruption_level):
        if self.firstlayer == 0 :
            return  self.theano_rng.binomial(
                             size = input.shape, 
                             n = 1, 
                             p = 1 - corruption_level, 
                             dtype=theano.config.floatX) * input
        else :
            noise = self.theano_rng.normal( size = input.shape, 
                                            dtype = theano.config.floatX)
            denoises = noise * self.var * corruption_level
            return input+denoises
    
    def get_hidden_values(self, input):
        return self.apply_activation((T.dot(input, self.W) + self.b), self.activation)

    def get_reconstructed_input(self, hidden):
        if self.firstlayer == 1 :
            return T.dot(hidden, self.W_prime) + self.b_prime
        else :
            return self.apply_activation((T.dot(hidden, self.W_prime) + self.b_prime), self.activation)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        #if corruption_level == 0:
        #    tilde_x = self.x
        #else:
        #    tilde_x = self.get_corrupted_input(self.x, corruption_level)
        tilde_x = self.x
        
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = T.sum ( (self.x-z) * (self.x-z), axis=1 ) 
        cost = T.mean(L) / 2

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param -  learning_rate*gparam
    
        return (cost, updates)

    def init_params(self, iparams):
        updates = {}
        for param, iparam in zip(self.params, iparams):
            updates[param] = iparam
        return updates

    def get_test_cost(self, corruption_level):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        
        # tilde_x = self.get_corrupted_input(self.x, corruption_level, 0.5)
        y       = self.get_hidden_values( self.x )
        z       = self.get_reconstructed_input(y)
        L = T.sum ( (self.x-z) * (self.x-z), axis=1)
        cost = T.mean(L)

        return cost
