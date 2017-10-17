
import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

class VanillaRNNDecoder(object):
    """ This class implements a standard recurrent neural network decoder:
        h_{t} = f(W^{hx}x_{t} + W^{hh}h_{t-1}+ W^{yh}y_{t-1} + b_{h})
        y_{t} = g(h_{t}W^{hy} + b_{y})

    """
    def __init__(self, rng, x, n_in, n_h, n_out, p, training, rnn_batch_training=False):
        """ This is to initialise a standard RNN hidden unit

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input data to current layer
        :param n_in: dimension of input data
        :param n_h: number of hidden units/blocks
        :param n_out: dimension of output data
        :param p: the probability of dropout
        :param training: a binary value to indicate training or testing (for dropout training)
        """
        self.input = x

        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape,p=p), x, 0)
            else:
                self.input =  (1-p) * x #(1-p) *

        self.n_in  = int(n_in)
        self.n_h   = int(n_h)
        self.n_out = int(n_out)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        #Wy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_h)), dtype=config.floatX)
        Ux_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_out)), dtype=config.floatX)
        Uh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_out)), dtype=config.floatX)
        #Uy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_out)), dtype=config.floatX)

        # identity matrix initialisation
        #Wh_value = np.asarray(np.eye(n_h, n_h), dtype=config.floatX)
        Wy_value = np.asarray(np.eye(n_out, n_h), dtype=config.floatX)
        #Uh_value = np.asarray(np.eye(n_in, n_out), dtype=config.floatX)
        Uy_value = np.asarray(np.zeros(n_out, n_out), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')
        self.W_yi = theano.shared(value=Wy_value, name='W_yi')

        # Output gate weights
        self.U_xi = theano.shared(value=Ux_value, name='U_xi')
        self.U_hi = theano.shared(value=Uh_value, name='U_hi')
        self.U_yi = theano.shared(value=Uy_value, name='U_yi')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_i')
        self.b   = theano.shared(value=np.zeros((n_out, ), dtype=config.floatX), name='b')


        # initial value of hidden and cell state and output
        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((1, n_out), dtype = config.floatX), name = 'y0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
            self.y0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((n_out, ), dtype = config.floatX), name = 'y0')


        self.Wix = T.dot(self.input, self.W_xi)
        self.Uix = T.dot(self.input, self.U_xi)

        [self.h, self.c, self.y], _ = theano.scan(self.recurrent_as_activation_function, sequences = [self.Wix, self.Uix],
                                                                      outputs_info = [self.h0, self.c0, self.y0])

        self.output = self.y

        # simple recurrent decoder params
        #self.params = [self.W_xi, self.W_hi, self.W_yi, self.U_hi, self.b_i, self.b]

        # recurrent output params and additional input params
        self.params = [self.W_xi, self.W_hi, self.W_yi, self.U_xi, self.U_hi, self.U_yi, self.b_i, self.b]

        self.L2_cost = (self.W_xi ** 2).sum() + (self.W_hi ** 2).sum() + (self.W_yi ** 2).sum() + (self.U_hi ** 2).sum()


    def recurrent_as_activation_function(self, Wix, Uix, h_tm1, c_tm1, y_tm1):
        """ Implement the recurrent unit as an activation function. This function is called by self.__init__().

        :param Wix: it equals to W^{hx}x_{t}, as it does not relate with recurrent, pre-calculate the value for fast computation
        :type Wix: matrix
        :param h_tm1: contains the hidden activation from previous time step
        :type h_tm1: matrix, each row means a hidden activation vector of a time step
        :param c_tm1: this parameter is not used, just to keep the interface consistent with LSTM
        :returns: h_t is the hidden activation of current time step
        """

        h_t = T.tanh(Wix + T.dot(h_tm1, self.W_hi) + T.dot(y_tm1, self.W_yi) + self.b_i)  #

        # simple recurrent decoder
        #y_t = T.dot(h_t, self.U_hi) + self.b

        # recurrent output and additional input
        y_t = Uix + T.dot(h_t, self.U_hi) + T.dot(y_tm1, self.U_yi) + self.b

        c_t = h_t

        return h_t, c_t, y_t

class ContextRNNDecoder(object):
    """ This class implements a standard recurrent neural network decoder:
        h_{t} = f(W^{hx}x_{t} + W^{hh}h_{t-1}+ W^{yh}y_{t-1} + b_{h})
        y_{t} = g(h_{t}W^{hy} + b_{y})

    """
    def __init__(self, rng, x, n_in, n_h, n_out, p, training, y=None, rnn_batch_training=False):
        """ This is to initialise a standard RNN hidden unit

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input data to current layer
        :param n_in: dimension of input data
        :param n_h: number of hidden units/blocks
        :param n_out: dimension of output data
        :param p: the probability of dropout
        :param training: a binary value to indicate training or testing (for dropout training)
        """
        self.input = x
        if y is not None:
            self.groundtruth = y

        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape,p=p), x, 0)
            else:
                self.input =  (1-p) * x #(1-p) *

        self.n_in  = int(n_in)
        self.n_h   = int(n_h)
        self.n_out = int(n_out)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        #Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        #Wy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_h)), dtype=config.floatX)
        Ux_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_out)), dtype=config.floatX)
        #Uh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_out)), dtype=config.floatX)
        #Uy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_out)), dtype=config.floatX)

        # identity matrix initialisation
        Wh_value = np.asarray(np.eye(n_h, n_h), dtype=config.floatX)
        Wy_value = np.asarray(np.eye(n_out, n_h), dtype=config.floatX)
        Uh_value = np.asarray(np.eye(n_in, n_out), dtype=config.floatX)
        Uy_value = np.asarray(np.zeros(n_out, n_out), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')
        self.W_yi = theano.shared(value=Wy_value, name='W_yi')

        # Output gate weights
        self.U_xi = theano.shared(value=Ux_value, name='U_xi')
        self.U_hi = theano.shared(value=Uh_value, name='U_hi')
        self.U_yi = theano.shared(value=Uy_value, name='U_yi')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_i')
        self.b   = theano.shared(value=np.zeros((n_out, ), dtype=config.floatX), name='b')

        # initial value of hidden and cell state and output
        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((1, n_out), dtype = config.floatX), name = 'y0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
            self.y0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((n_out, ), dtype = config.floatX), name = 'y0')

        self.h0 = self.input[-1, 0:-4] # hard coded to remove coarse coding features

        self.outytm1 = T.roll(self.groundtruth, 1, 0)
        
        self.Wix = T.dot(self.input, self.W_xi)
        self.Uix = T.dot(self.input, self.U_xi)

        [self.h, self.c], _ = theano.scan(self.recurrent_as_activation_function, sequences = [self.Wix, self.Wiy],
                                                                      outputs_info = [self.h0, self.c0])

        self.y = self.Uix + self.Uiy + T.dot(self.h, self.U_hi) + self.b
        self.output = T.nnet.softmax(self.y)
        
        # recurrent output params and additional input params
        self.params = [self.W_xi, self.W_hi, self.W_yi, self.U_xi, self.U_hi, self.U_yi, self.b_i, self.b]

        self.L2_cost = (self.W_xi ** 2).sum() + (self.W_hi ** 2).sum() + (self.W_yi ** 2).sum() + (self.U_hi ** 2).sum()


    def recurrent_as_activation_function(self, Wix, Wiy, h_tm1, c_tm1):
        """ Implement the recurrent unit as an activation function. This function is called by self.__init__().

        :param Wix: it equals to W^{hx}x_{t}, as it does not relate with recurrent, pre-calculate the value for fast computation
        :type Wix: matrix
        :param h_tm1: contains the hidden activation from previous time step
        :type h_tm1: matrix, each row means a hidden activation vector of a time step
        :param c_tm1: this parameter is not used, just to keep the interface consistent with LSTM
        :returns: h_t is the hidden activation of current time step
        """

        h_t = T.tanh(Wix + T.dot(h_tm1, self.W_hi) + Wiy + self.b_i)  #

        c_t = h_t

        return h_t, c_t

class LstmDecoderBase(object):
    """ This class provides as a base for all long short-term memory (LSTM) related classes.
    Several variants of LSTM were investigated in (Wu & King, ICASSP 2016): Zhizheng Wu, Simon King, "Investigating gated recurrent neural networks for speech synthesis", ICASSP 2016

    """

    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, rnn_batch_training=False):
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
        self.n_h  = int(n_h)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)
        Wy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_h)), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')
        self.w_ci = theano.shared(value=Wc_value, name='w_ci')
        self.W_yi = theano.shared(value=Wy_value, name='W_yi')

        # random initialisation
        Uh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_out)), dtype=config.floatX)

        # Output gate weights
        self.U_ho = theano.shared(value=Uh_value, name='U_ho')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Forget gate weights
        self.W_xf = theano.shared(value=Wx_value, name='W_xf')
        self.W_hf = theano.shared(value=Wh_value, name='W_hf')
        self.w_cf = theano.shared(value=Wc_value, name='w_cf')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Output gate weights
        self.W_xo = theano.shared(value=Wx_value, name='W_xo')
        self.W_ho = theano.shared(value=Wh_value, name='W_ho')
        self.w_co = theano.shared(value=Wc_value, name='w_co')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Cell weights
        self.W_xc = theano.shared(value=Wx_value, name='W_xc')
        self.W_hc = theano.shared(value=Wh_value, name='W_hc')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_i')
        self.b_f = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_f')
        self.b_o = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_o')
        self.b_c = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_c')
        self.b   = theano.shared(value=np.zeros((n_out, ), dtype=config.floatX), name='b')

        ### make a layer

        # initial value of hidden and cell state
        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((1, n_out), dtype = config.floatX), name = 'y0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
            self.y0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')
            self.y0 = theano.shared(value=np.zeros((n_out, ), dtype = config.floatX), name = 'y0')


        self.Wix = T.dot(self.input, self.W_xi)
        self.Wfx = T.dot(self.input, self.W_xf)
        self.Wcx = T.dot(self.input, self.W_xc)
        self.Wox = T.dot(self.input, self.W_xo)

        [self.h, self.c, self.y], _ = theano.scan(self.recurrent_fn, sequences = [self.Wix, self.Wfx, self.Wcx, self.Wox],
                                                             outputs_info = [self.h0, self.c0, self.y0])

        self.output = self.y


    def recurrent_fn(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1=None, y_tm1=None):
        """ This implements a genetic recurrent function, called by self.__init__().

        :param Wix: pre-computed matrix applying the weight matrix W on  the input units, for input gate
        :param Wfx: Similar to Wix, but for forget gate
        :param Wcx: Similar to Wix, but for cell memory
        :param Wox: Similar to Wox, but for output gate
        :param h_tm1: hidden activation from previous time step
        :param c_tm1: activation from cell memory from previous time step
        :returns: h_t is the hidden activation of current time step, and c_t is the activation for cell memory of current time step
        """

        h_t, c_t, y_t = self.lstm_as_activation_function(Wix, Wfx, Wcx, Wox, h_tm1, c_tm1, y_tm1)

        return h_t, c_t, y_t

    def lstm_as_activation_function(self):
        """ A genetic recurrent activation function for variants of LSTM architectures.
        The function is called by self.recurrent_fn().

        """
        pass

class VanillaLstmDecoder(LstmDecoderBase):
    """ This class implements the standard LSTM block, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """


    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a vanilla LSTM block

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        self.n_out = int(n_out)

        LstmDecoderBase.__init__(self, rng, x, n_in, n_h, n_out, p, training, rnn_batch_training)

        self.params = [self.W_xi, self.W_hi, self.w_ci, self.W_yi,
                       self.W_xf, self.W_hf, self.w_cf,
                       self.W_xo, self.W_ho, self.w_co,
                       self.W_xc, self.W_hc,
                       self.U_ho,
                       self.b_i, self.b_f, self.b_o, self.b_c, self.b]

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1, y_tm1):
        """ This function treats the LSTM block as an activation function, and implements the standard LSTM activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.w_ci * c_tm1 + self.b_i)  #
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.w_cf * c_tm1 + self.b_f)  #

        c_t = f_t * c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + T.dot(y_tm1, self.W_yi) + self.b_c)

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.w_co * c_t + self.b_o)

        h_t = o_t * T.tanh(c_t)

        y_t = T.dot(h_t, self.U_ho) + self.b

        return h_t, c_t, y_t     #, i_t, f_t, o_t

class SimplifiedLstmDecoder(LstmDecoderBase):
    """ This class implements a simplified LSTM block which only keeps the forget gate, inheriting the genetic class :class:`layers.gating.LstmBase`.
    
    """

    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with only the forget gate
        
        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """
        
        self.n_out = int(n_out)

        LstmDecoderBase.__init__(self, rng, x, n_in, n_h, n_out, p, training, rnn_batch_training)

        self.params = [self.W_yi,
                       self.W_xf, self.W_hf,
                       self.W_xc, self.W_hc,
                       self.U_ho,
                       self.b_f,  self.b_c, self.b]
                       
    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1, y_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (simplified LSTM) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`
        
        """
    
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.b_f)  #self.w_cf * c_tm1 
    
        c_t = f_t * c_tm1 + (1 - f_t) * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + T.dot(y_tm1, self.W_yi) + self.b_c) 

        h_t = T.tanh(c_t)

        y_t = T.dot(h_t, self.U_ho) + self.b

        return h_t, c_t, y_t

class LstmBase(object):
    """ This class provides as a base for all long short-term memory (LSTM) related classes.
    Several variants of LSTM were investigated in (Wu & King, ICASSP 2016): Zhizheng Wu, Simon King, "Investigating gated recurrent neural networks for speech synthesis", ICASSP 2016

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
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

        n_in = int(n_in)  # ensure sizes have integer type
        n_h = int(n_h)# ensure sizes have integer type

        self.input = x

        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape,p=p), x, 0)
            else:
                self.input =  (1-p) * x

        self.n_in = int(n_in)
        self.n_h  = int(n_h)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')
        self.w_ci = theano.shared(value=Wc_value, name='w_ci')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Forget gate weights
        self.W_xf = theano.shared(value=Wx_value, name='W_xf')
        self.W_hf = theano.shared(value=Wh_value, name='W_hf')
        self.w_cf = theano.shared(value=Wc_value, name='w_cf')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Output gate weights
        self.W_xo = theano.shared(value=Wx_value, name='W_xo')
        self.W_ho = theano.shared(value=Wh_value, name='W_ho')
        self.w_co = theano.shared(value=Wc_value, name='w_co')

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)
        Wc_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, )), dtype=config.floatX)

        # Cell weights
        self.W_xc = theano.shared(value=Wx_value, name='W_xc')
        self.W_hc = theano.shared(value=Wh_value, name='W_hc')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_i')
        self.b_f = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_f')
        self.b_o = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_o')
        self.b_c = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_c')

        ### make a layer

        # initial value of hidden and cell state
        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')

        self.h0 = self.input[-1, 0:-4] # hard coded to remove coarse coding features

        self.Wix = T.dot(self.input, self.W_xi)
        self.Wfx = T.dot(self.input, self.W_xf)
        self.Wcx = T.dot(self.input, self.W_xc)
        self.Wox = T.dot(self.input, self.W_xo)

        [self.h, self.c], _ = theano.scan(self.recurrent_fn, sequences = [self.Wix, self.Wfx, self.Wcx, self.Wox],
                                                             outputs_info = [self.h0, self.c0])

        self.output = self.h


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

class ContextLstm(LstmBase):
    """ This class implements the standard LSTM block, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """


    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a vanilla LSTM block

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xi, self.W_hi, self.w_ci,
                       self.W_xf, self.W_hf, self.w_cf,
                       self.W_xo, self.W_ho, self.w_co,
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]

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


