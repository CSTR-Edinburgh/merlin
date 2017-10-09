import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams


class RecurrentOutputLayer(object):
    """ This class implements a standard recurrent output layer:
        y_{t} = g(h_{t}W^{hy} + y_{t}W^{yy} + b_{y})

    """
    def __init__(self, rng, x, n_in, n_out, p=0.0, training=1, rnn_batch_training=False):
        """ This is to initialise a standard RNN hidden unit

        :param rng: random state, fixed value for randome state for reproducible objective results
        :param x: input data to current layer
        :param n_in: dimension of input data
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

        self.n_in = int(n_in)
        self.n_out = int(n_out)

        self.rnn_batch_training = rnn_batch_training

        # random initialisation
        Wx_value = np.asarray(rng.normal(0.0, 1.0/np.sqrt(n_in), size=(n_in, n_out)), dtype=config.floatX)
        Wy_value = np.asarray(np.zeros((n_out, n_out)), dtype=config.floatX)

        # Input gate weights
        self.W_xi = theano.shared(value=Wx_value, name='W_xi')
        self.W_yi = theano.shared(value=Wy_value, name='W_yi')

        # bias
        self.b_y = theano.shared(value=np.zeros((n_out, ), dtype=config.floatX), name='b_y')

        # initial value of output
        if self.rnn_batch_training:
            self.y0 = theano.shared(value=np.zeros((1, n_out), dtype = config.floatX), name = 'y0')
            self.y0 = T.repeat(self.y0, x.shape[1], 0)
        else:
            self.y0 = theano.shared(value=np.zeros((n_out, ), dtype = config.floatX), name = 'y0')


        self.Wix = T.dot(self.input, self.W_xi)

        self.y, _ = theano.scan(self.recurrent_as_activation_function, sequences = self.Wix,
                                                                      outputs_info = self.y0)

        self.output = self.y

        self.params = [self.W_xi, self.W_yi, self.b_y]

    def recurrent_as_activation_function(self, Wix, y_tm1):
        """ Implement the recurrent unit as an activation function. This function is called by self.__init__().

        :param Wix: it equals to W^{hx}x_{t}, as it does not relate with recurrent, pre-calculate the value for fast computation
        :type Wix: matrix
        :param y_tm1: contains the output from previous time step
        :type y_tm1: matrix, each row means an output vector of a time step
        """

        y_t = Wix + T.dot(y_tm1, self.W_yi) + self.b_y  #

        return y_t
