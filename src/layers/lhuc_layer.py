
import numpy, time, pickle, gzip, sys, os, copy

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

class LstmBase_LHUC(object):
    """ 
    Very similar to the LSTM layer in the gating file
    Extra parameters are 'C' for scaling the hidden value
    """

    def __init__(self, rng, x, n_in, n_h, p, training):
        """ Initialise all the components in a LSTM block, including input gate, output gate, forget gate, peephole connections
        
        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        :param p: the probability of dropout
        :param training: a binary value to indicate training or testing (for dropout training)
        """
    
        self.input = x
                
        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape,p=p), x, 0)
            else:
                self.input =  (1-p) * x 

        self.n_in = int(n_in)
        self.n_h = int(n_h)
        
        # random initialisation 
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=theano.config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=theano.config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=theano.config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')
        self.w_ci = theano.shared(value=Wc_value, name='w_ci')

        # random initialisation 
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=theano.config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=theano.config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=theano.config.floatX)

        # Forget gate weights
        self.W_xf = theano.shared(value=Wx_value, name='W_xf')
        self.W_hf = theano.shared(value=Wh_value, name='W_hf')
        self.w_cf = theano.shared(value=Wc_value, name='w_cf')

        # random initialisation 
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=theano.config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=theano.config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=theano.config.floatX)

        # Output gate weights
        self.W_xo = theano.shared(value=Wx_value, name='W_xo')
        self.W_ho = theano.shared(value=Wh_value, name='W_ho')
        self.w_co = theano.shared(value=Wc_value, name='w_co')

        # random initialisation 
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=theano.config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=theano.config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=theano.config.floatX)

        # Cell weights
        self.W_xc = theano.shared(value=Wx_value, name='W_xc')
        self.W_hc = theano.shared(value=Wh_value, name='W_hc')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=theano.config.floatX), name='b_i')
        self.b_f = theano.shared(value=np.zeros((n_h, ), dtype=theano.config.floatX), name='b_f')
        self.b_o = theano.shared(value=np.zeros((n_h, ), dtype=theano.config.floatX), name='b_o')
        self.b_c = theano.shared(value=np.zeros((n_h, ), dtype=theano.config.floatX), name='b_c')
        

        # scaling factor
        c_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h)), dtype=theano.config.floatX)
        self.C = theano.shared(value=c_value, name='c')
        ### make a layer
        
        # initial value of hidden and cell state
        self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = theano.config.floatX), name = 'h0')
        self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = theano.config.floatX), name = 'c0')


        self.Wix = T.dot(self.input, self.W_xi)
        self.Wfx = T.dot(self.input, self.W_xf)
        self.Wcx = T.dot(self.input, self.W_xc)
        self.Wox = T.dot(self.input, self.W_xo)


        
        [self.h, self.c], _ = theano.scan(self.recurrent_fn, sequences = [self.Wix, self.Wfx, self.Wcx, self.Wox],
                                                             outputs_info = [self.h0, self.c0]) 

        self.output = 2. * T.nnet.sigmoid(self.C) * self.h
        
        
    def recurrent_fn(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1 = None):
        """ This implements a genetic recurrent function, called by self.__init__().
        
        :param Wix: pre-computed matrix applying the weight matrix W on  the input units, for input gate
        :param Wfx: Similar to Wix, but for forget gate
        :param Wcx: Similar to Wix, but for cell memory
        :param Wox: Similar to Wox, but for output gate
        :param h_tm1: hidden activation from previous time step
        :param c_tm1: activation from cell memory from previous time step
        :returns: h_t is the hidden activation of current time step, and c_t is the activation for cell memory of current time step
        """
        h_t, c_t = self.lstm_as_activation_function(Wix, Wfx, Wcx, Wox, h_tm1, c_tm1)
            
        return h_t, c_t

    def lstm_as_activation_function(self):
        """ A genetic recurrent activation function for variants of LSTM architectures.
        The function is called by self.recurrent_fn().
        
        """
        pass


class VanillaLstm_LHUC(LstmBase_LHUC):
    """ This class implements the standard LSTM block, inheriting the genetic class :class:`layers.gating.LstmBase`.
    
    """


    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0):
        """ Initialise a vanilla LSTM block
        
        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """
        
            
        LstmBase_LHUC.__init__(self, rng, x, n_in, n_h, p, training)
        
        self.params = [self.W_xi, self.W_hi, self.w_ci,
                       self.W_xf, self.W_hf, self.w_cf,
                       self.W_xo, self.W_ho, self.w_co, 
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c, self.C]
                       
    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the standard LSTM activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`
        
        """
    
        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.w_ci * c_tm1 + self.b_i)  #
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.w_cf * c_tm1 + self.b_f)  # 
    
        c_t = f_t * c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.w_co * c_t + self.b_o)
                            
        h_t = o_t * T.tanh(c_t)

        return h_t, c_t#, i_t, f_t, o_t