
### refer Zhizheng and Simon's ICASSP'16 paper for more details
### http://www.zhizheng.org/papers/icassp2016_lstm.pdf

import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

class VanillaRNN(object):
    """ This class implements a standard recurrent neural network: h_{t} = f(W^{hx}x_{t} + W^{hh}h_{t-1}+b_{h})

    """
    def __init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training=False):
        """ This is to initialise a standard RNN hidden unit

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input data to current layer
        :param n_in: dimension of input data
        :param n_h: number of hidden units/blocks
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

        self.n_in = int(n_in)
        self.n_h  = int(n_h)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_h)), dtype=config.floatX)
        Wh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_h)), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_hi = theano.shared(value=Wh_value, name='W_hi')

        # bias
        self.b_i = theano.shared(value=np.zeros((n_h, ), dtype=config.floatX), name='b_i')


        # initial value of hidden and cell state
        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')


        self.Wix = T.dot(self.input, self.W_xi)

        [self.h, self.c], _ = theano.scan(self.recurrent_as_activation_function, sequences = [self.Wix],
                                                                      outputs_info = [self.h0, self.c0])

        self.output = self.h

        self.params = [self.W_xi, self.W_hi, self.b_i]

        self.L2_cost = (self.W_xi ** 2).sum() + (self.W_hi ** 2).sum()


    def recurrent_as_activation_function(self, Wix, h_tm1, c_tm1):
        """ Implement the recurrent unit as an activation function. This function is called by self.__init__().

        :param Wix: it equals to W^{hx}x_{t}, as it does not relate with recurrent, pre-calculate the value for fast computation
        :type Wix: matrix
        :param h_tm1: contains the hidden activation from previous time step
        :type h_tm1: matrix, each row means a hidden activation vector of a time step
        :param c_tm1: this parameter is not used, just to keep the interface consistent with LSTM
        :returns: h_t is the hidden activation of current time step
        """

        h_t = T.tanh(Wix + T.dot(h_tm1, self.W_hi) + self.b_i)  #

        c_t = h_t

        return h_t, c_t

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
        Wy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_h)), dtype=config.floatX)
        Ux_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_out)), dtype=config.floatX)
        Uh_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h), size=(n_h, n_out)), dtype=config.floatX)
        Uy_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_out)), dtype=config.floatX)

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

        [self.h, self.c, self.y], _ = theano.scan(self.recurrent_as_activation_function, sequences = [self.Wix],
                                                                      outputs_info = [self.h0, self.c0, self.y0])

        self.output = self.y

        self.params = [self.W_xi, self.W_hi, self.W_yi, self.U_hi, self.b_i, self.b]

        self.L2_cost = (self.W_xi ** 2).sum() + (self.W_hi ** 2).sum() + (self.W_yi ** 2).sum() + (self.U_hi ** 2).sum()


    def recurrent_as_activation_function(self, Wix, h_tm1, c_tm1, y_tm1):
        """ Implement the recurrent unit as an activation function. This function is called by self.__init__().

        :param Wix: it equals to W^{hx}x_{t}, as it does not relate with recurrent, pre-calculate the value for fast computation
        :type Wix: matrix
        :param h_tm1: contains the hidden activation from previous time step
        :type h_tm1: matrix, each row means a hidden activation vector of a time step
        :param c_tm1: this parameter is not used, just to keep the interface consistent with LSTM
        :returns: h_t is the hidden activation of current time step
        """

        h_t = T.tanh(Wix + T.dot(h_tm1, self.W_hi) + T.dot(y_tm1, self.W_yi) + self.b_i)  #

        y_t = T.dot(h_t, self.U_hi) + self.b

        c_t = h_t

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

class VanillaLstm(LstmBase):
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

class LstmNFG(LstmBase):
    """ This class implements a LSTM block without the forget gate, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """
    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with the forget gate

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xi, self.W_hi, self.w_ci,
                       self.W_xo, self.W_ho, self.w_co,
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_o, self.b_c]

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (without the forget gate) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.w_ci * c_tm1 + self.b_i)  #

        c_t = c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)  #f_t *

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.w_co * c_t + self.b_o)

        h_t = o_t * T.tanh(c_t)

        return h_t, c_t

class LstmNIG(LstmBase):
    """ This class implements a LSTM block without the input gate, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with the input gate

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xf, self.W_hf, self.w_cf,
                       self.W_xo, self.W_ho, self.w_co,
                       self.W_xc, self.W_hc,
                       self.b_f, self.b_o, self.b_c]

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (without the input gate) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.w_cf * c_tm1 + self.b_f)  #

        c_t = f_t * c_tm1 + T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)  #i_t *

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.w_co * c_t + self.b_o)

        h_t = o_t * T.tanh(c_t)

        return h_t, c_t


class LstmNOG(LstmBase):
    """ This class implements a LSTM block without the output gate, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with the output gate

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
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_f,
                       self.b_c]

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (without the output gate) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.w_ci * c_tm1 + self.b_i)  #
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.w_cf * c_tm1 + self.b_f)  #

        c_t = f_t * c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)  #i_t *

        h_t = T.tanh(c_t)

        return h_t, c_t


class LstmNoPeepholes(LstmBase):
    """ This class implements a LSTM block without the peephole connections, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with the peephole connections

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xi, self.W_hi, #self.W_ci,
                       self.W_xf, self.W_hf, #self.W_cf,
                       self.W_xo, self.W_ho, #self.W_co,
                       self.W_xc, self.W_hc,
                       self.b_i, self.b_f,
                       self.b_o, self.b_c]

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (without the output gate) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        i_t = T.nnet.sigmoid(Wix + T.dot(h_tm1, self.W_hi) + self.b_i)
        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.b_f)

        c_t = f_t * c_tm1 + i_t * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)

        o_t = T.nnet.sigmoid(Wox + T.dot(h_tm1, self.W_ho) + self.b_o)

        h_t = o_t * T.tanh(c_t)

        return h_t, c_t


class SimplifiedLstm(LstmBase):
    """ This class implements a simplified LSTM block which only keeps the forget gate, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with only the forget gate

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xf, self.W_hf,
                       self.W_xc, self.W_hc,
                       self.b_f,  self.b_c]

        self.L2_cost = (self.W_xf ** 2).sum() + (self.W_hf ** 2).sum() + (self.W_xc ** 2).sum() + (self.W_hc ** 2).sum()

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (simplified LSTM) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """

        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.b_f)  #self.w_cf * c_tm1

        c_t = f_t * c_tm1 + (1 - f_t) * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)

        h_t = T.tanh(c_t)

        return h_t, c_t

class SimplifiedGRU(LstmBase):
    """ This class implements a simplified GRU block which only keeps the forget gate, inheriting the genetic class :class:`layers.gating.LstmBase`.

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a LSTM with the the forget gate

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        """

        LstmBase.__init__(self, rng, x, n_in, n_h, p, training, rnn_batch_training)

        self.params = [self.W_xf, self.W_hf, self.w_cf,
                       self.W_xc, self.W_hc,
                       self.b_f,  self.b_c]

        self.L2_cost = (self.W_xf ** 2).sum() + (self.W_hf ** 2).sum() + (self.W_xc ** 2).sum() + (self.W_hc ** 2).sum()

    def lstm_as_activation_function(self, Wix, Wfx, Wcx, Wox, h_tm1, c_tm1):
        """ This function treats the LSTM block as an activation function, and implements the LSTM (simplified LSTM) activation function.
            The meaning of each input and output parameters can be found in :func:`layers.gating.LstmBase.recurrent_fn`

        """
        ##can_h_t = T.tanh(Whx + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)

        f_t = T.nnet.sigmoid(Wfx + T.dot(h_tm1, self.W_hf) + self.b_f)  #self.w_cf * c_tm1

        can_h_t = T.tanh(Wcx + f_t * T.dot(h_tm1, self.W_hc) + self.b_c)

        h_t = self.w_cf * (1.0 - f_t) * h_tm1 + f_t * can_h_t
        c_t = h_t

#        c_t = f_t * c_tm1 + (1 - f_t) * T.tanh(Wcx + T.dot(h_tm1, self.W_hc) + self.b_c)

#        h_t = T.tanh(c_t)

        return h_t, c_t

class BidirectionSLstm(SimplifiedLstm):

    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, rnn_batch_training=False):

        fwd = SimplifiedLstm(rng, x, n_in, n_h, p, training, rnn_batch_training)
        bwd = SimplifiedLstm(rng, x[::-1], n_in, n_h, p, training, rnn_batch_training)

        self.params = fwd.params + bwd.params

        self.output = T.concatenate([fwd.output, bwd.output[::-1]], axis=-1)

class BidirectionLstm(VanillaLstm):

    def __init__(self, rng, x, n_in, n_h, n_out, p=0.0, training=0, rnn_batch_training=False):

        fwd = VanillaLstm(rng, x, n_in, n_h, p, training, rnn_batch_training)
        bwd = VanillaLstm(rng, x[::-1], n_in, n_h, p, training, rnn_batch_training)

        self.params = fwd.params + bwd.params

        self.output = T.concatenate([fwd.output, bwd.output[::-1]], axis=-1)


class RecurrentOutput(object):
    def __init__(self, rng, x, n_in, n_out, p=0.0, training=0, rnn_batch_training=False):

        self.W_h = theano.shared(value=np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_in, n_out)), dtype=config.floatX), name='W_h')
        self.W_y = theano.shared(value=np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_out), size=(n_out, n_out)), dtype=config.floatX), name='W_y')

        self.b_y = theano.shared(value=np.zeros((n_out, ), dtype=config.floatX), name='b_y')




# Gated Recurrent Unit
class GatedRecurrentUnit(object):
    """ This class implements a gated recurrent unit (GRU), as proposed in Cho et al 2014 (http://arxiv.org/pdf/1406.1078.pdf).

    """

    def __init__(self, rng, x, n_in, n_h, p=0.0, training=0, rnn_batch_training=False):
        """ Initialise a gated recurrent unit

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input to a network
        :param n_in: number of input features
        :type n_in: integer
        :param n_h: number of hidden units
        :type n_h: integer
        :param p: the probability of dropout
        :param training: a binary value to indicate training or testing (for dropout training)
        """

        self.n_in = int(n_in)
        self.n_h  = int(n_h)

        self.rnn_batch_training = rnn_batch_training

        self.input = x

        if p > 0.0:
            if training==1:
                srng = RandomStreams(seed=123456)
                self.input = T.switch(srng.binomial(size=x.shape,p=p), x, 0)
            else:
                self.input =  (1-p) * x

        self.W_xz = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in),
                     size=(n_in, n_h)), dtype=config.floatX), name = 'W_xz')
        self.W_hz = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h),
                     size=(n_h, n_h)), dtype=config.floatX), name = 'W_hz')

        self.W_xr = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in),
                     size=(n_in, n_h)), dtype=config.floatX), name = 'W_xr')
        self.W_hr = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h),
                     size=(n_h, n_h)), dtype=config.floatX), name = 'W_hr')

        self.W_xh = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in),
                     size=(n_in, n_h)), dtype=config.floatX), name = 'W_xh')
        self.W_hh = theano.shared(value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_h),
                     size=(n_h, n_h)), dtype=config.floatX), name = 'W_hh')

        self.b_z = theano.shared(value = np.zeros((n_h, ), dtype = config.floatX), name = 'b_z')

        self.b_r = theano.shared(value = np.zeros((n_h, ), dtype = config.floatX), name = 'b_r')

        self.b_h = theano.shared(value = np.zeros((n_h, ), dtype = config.floatX), name = 'b_h')

        if self.rnn_batch_training:
            self.h0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((1, n_h), dtype = config.floatX), name = 'c0')

            self.h0 = T.repeat(self.h0, x.shape[1], 0)
            self.c0 = T.repeat(self.c0, x.shape[1], 0)
        else:
            self.h0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'h0')
            self.c0 = theano.shared(value=np.zeros((n_h, ), dtype = config.floatX), name = 'c0')


        ## pre-compute these for fast computation
        self.Wzx = T.dot(self.input, self.W_xz)
        self.Wrx = T.dot(self.input, self.W_xr)
        self.Whx = T.dot(self.input, self.W_xh)

        [self.h, self.c], _ = theano.scan(self.gru_as_activation_function,
                                               sequences = [self.Wzx, self.Wrx, self.Whx],
                                               outputs_info = [self.h0, self.c0])  #


        self.output = self.h

        self.params = [self.W_xz, self.W_hz, self.W_xr, self.W_hr, self.W_xh, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        self.L2_cost = (self.W_xz ** 2).sum() + (self.W_hz ** 2).sum() + (self.W_xr ** 2).sum() + (self.W_hr ** 2).sum() + (self.W_xh ** 2).sum() + (self.W_hh ** 2).sum()

    def gru_as_activation_function(self, Wzx, Wrx, Whx, h_tm1, c_tm1 = None):
        """ This function treats the GRU block as an activation function, and implements the GRU activation function.
            This function is called by :func:`layers.gating.GatedRecurrentUnit.__init__`.
            Wzx, Wrx, Whx have been pre-computed before passing to this function.

            To make the same interface as LSTM, we keep a c_tm1 (means the cell state of previous time step, but GRU does not maintain a cell state).
        """

        z_t = T.nnet.sigmoid(Wzx + T.dot(h_tm1, self.W_hz) + self.b_z)
        r_t = T.nnet.sigmoid(Wrx + T.dot(h_tm1, self.W_hr) + self.b_r)
        can_h_t = T.tanh(Whx + r_t * T.dot(h_tm1, self.W_hh) + self.b_h)

        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        c_t = h_t   ## in order to have the same interface as LSTM

        return h_t, c_t
