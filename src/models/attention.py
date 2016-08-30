import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano import function

class AttentionEncoder(object):

    def __init__(self, rng, x, d):

        self.input = x
        self.dur_input = d

        self.n_fr = T.sum(self.dur_input)
       
        [self.n_ph, self.n_in] = x.shape

        # Identity matrix initialisation 
        self.W_xi = T.eye(self.n_fr, dtype=config.floatX)

        # Input gate weights
        #self.W_xi = theano.shared(value=Wx_value, name='W_xi')

        self.Wix = T.dot(self.W_xi, self.input)

        self.encoded_output = self.encode_weighted_states()
        
        self.params = [self.W_xi]

    ### Attention model: weighted C_1-C_n as input to corresponding decoder frames ###
    def encode_weighted_states(self):
        reps = T.repeat(T.arange(self.dur_input.size), self.dur_input)
        attention_vector = self.Wix[reps]

        return attention_vector
