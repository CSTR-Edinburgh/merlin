import numpy as np
import theano
import theano.tensor as T
from theano import function

class VanillaSequenceEncoder(object):

    def __init__(self, rng, x, d):

        self.input = x
        self.dur_input = d
        self.encoded_output = self.encode_final_state()

    ### default seq-to-seq model: tile C as input to all frames ###
    def encode_final_state(self):
        context_vector       = self.input[-1, ]
        tiled_context_vector = T.tile(context_vector, (T.sum(self.dur_input), 1))

        return tiled_context_vector

class DistributedSequenceEncoder(object):

    def __init__(self, rng, x, d):

        self.input = x
        self.dur_input = d
        self.encoded_output = self.encode_all_states()

    ### Distributed seq-to-seq model: tile C_1-C_n as input to corresponding decoder frames ###
    def encode_all_states(self):
        reps = T.repeat(T.arange(self.dur_input.size), self.dur_input)
        dist_context_vector = self.input[reps]

        return dist_context_vector
