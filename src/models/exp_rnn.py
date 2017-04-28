##
##                       _oo0oo_
##                      o8888888o
##                      88" . "88
##                      (| -_- |)
##                      0\  =  /0
##                    ___/`---'\___
##                  .' \\|     |// '.
##                 / \\|||  :  |||// \
##                / _||||| -:- |||||- \
##               |   | \\\  -  /// |   |
##               | \_|  ''\---/''  |_/ |
##               \  .-\__  '-'  ___/-. /
##             ___'. .'  /--.--\  `. .'___
##          ."" '<  `.___\_<|>_/___.' >' "".
##         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
##         \  \ `_.   \_ __\ /__ _/   .-` /  /
##     =====`-.____`.___ \_____/___.-`___.-'=====
##                       `=---='
##
##     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
##          @Buddha: Bless me Bugfree code
##
##     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys

import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.gating import SimplifiedLstm, BidirectionSLstm, VanillaLstm, VanillaLstmDecoder, BidirectionLstm, VanillaRNN, VanillaRNNDecoder, SimplifiedGRU, GatedRecurrentUnit, LstmNoPeepholes, LstmNOG, LstmNIG, LstmNFG
from layers.layers import LinearLayer, SigmoidLayer
from layers.recurrent_decoders import ContextLstm

from models.seq2seq import VanillaSequenceEncoder, DistributedSequenceEncoder

import logging

class DeepRecurrentNetwork(object):
    """
    This class is to assemble various neural network architectures. From basic feedforward neural network to bidirectional gated recurrent neural networks and hybrid architecture. **Hybrid** means a combination of feedforward and recurrent architecture.

    """


    def __init__(self, n_in, hidden_layer_size, n_out, L1_reg, L2_reg, hidden_layer_type, output_type='LINEAR', network_type='DNN', dropout_rate=0.0):
        """ This function initialises a neural network

        :param n_in: Dimensionality of input features
        :type in: Integer
        :param hidden_layer_size: The layer size for each hidden layer
        :type hidden_layer_size: A list of integers
        :param n_out: Dimensionality of output features
        :type n_out: Integrer
        :param hidden_layer_type: the activation types of each hidden layers, e.g., TANH, LSTM, GRU, BLSTM
        :param L1_reg: the L1 regulasation weight
        :param L2_reg: the L2 regulasation weight
        :param output_type: the activation type of the output layer, by default is 'LINEAR', linear regression.
        :param dropout_rate: probability of dropout, a float number between 0 and 1.
        """

        logger = logging.getLogger("DNN initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)

        self.n_layers = len(hidden_layer_size)

        self.dropout_rate = dropout_rate
        self.is_train = T.iscalar('is_train')


        assert len(hidden_layer_size) == len(hidden_layer_type)

        self.x = T.matrix('x')
        self.y = T.matrix('y')

        if network_type == "S2S":
            self.d = T.ivector('d')
            self.f = T.matrix('f')

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.rnn_layers = []
        self.params = []
        self.delta_params = []

        rng = np.random.RandomState(123)

        BLSTM_variants   = ['BLSTM', 'BSLSTM', 'BLSTME']
        Encoder_variants = ['RNNE', 'LSTME', 'BLSTME', 'SLSTME']
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i-1]
                if hidden_layer_type[i-1] in BLSTM_variants:
                    input_size = hidden_layer_size[i-1]*2

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.rnn_layers[i-1].output

            ### sequence-to-sequence mapping ###
            if hidden_layer_type[i-1] in Encoder_variants:
                dur_input        = self.d
                frame_feat_input = self.f

                if network_type == "S2S":
                    seq2seq_model = DistributedSequenceEncoder(rng, layer_input, dur_input)
                    layer_input   = T.concatenate((seq2seq_model.encoded_output, frame_feat_input), axis=1)
                    input_size    = input_size+4

                else:
                    logger.critical("This network type: %s is not supported right now! \n Please use one of the following: DNN, RNN, S2S\n" %(network_type))
                    sys.exit(1)

            if hidden_layer_type[i] == 'SLSTM':
                hidden_layer = SimplifiedLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'SGRU':
                hidden_layer = SimplifiedGRU(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'GRU':
                hidden_layer = GatedRecurrentUnit(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTM_NFG':
                hidden_layer = LstmNFG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTM_NOG':
                hidden_layer = LstmNOG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTM_NIG':
                hidden_layer = LstmNIG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTM_NPH':
                hidden_layer = LstmNoPeepholes(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTM':
                hidden_layer = VanillaLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTME':
                hidden_layer = VanillaLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'CLSTM':
                hidden_layer = ContextLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'LSTMD':
                hidden_layer = VanillaLstmDecoder(rng, layer_input, input_size, hidden_layer_size[i], self.n_out, p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'BSLSTM':
                hidden_layer = BidirectionSLstm(rng, layer_input, input_size, hidden_layer_size[i], hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'BLSTM':
                hidden_layer = BidirectionLstm(rng, layer_input, input_size, hidden_layer_size[i], hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'BLSTME':
                hidden_layer = BidirectionLstm(rng, layer_input, input_size, hidden_layer_size[i], hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'RNN':
                hidden_layer = VanillaRNN(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'RNNE':
                hidden_layer = VanillaRNN(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'RNND':
                hidden_layer = VanillaRNNDecoder(rng, layer_input, input_size, hidden_layer_size[i], self.n_out, p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'TANH':
                hidden_layer = SigmoidLayer(rng, layer_input, input_size, hidden_layer_size[i], activation=T.tanh, p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'SIGMOID':
                hidden_layer = SigmoidLayer(rng, layer_input, input_size, hidden_layer_size[i], activation=T.nnet.sigmoid, p=self.dropout_rate, training=self.is_train)
            else:
                logger.critical("This hidden layer type: %s is not supported right now! \n Please use one of the following: SLSTM, BSLSTM, TANH, SIGMOID\n" %(hidden_layer_type[i]))
                sys.exit(1)

            self.rnn_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        input_size = hidden_layer_size[-1]
        if hidden_layer_type[-1]  == 'BSLSTM' or hidden_layer_type[-1]  == 'BLSTM':
            input_size = hidden_layer_size[-1]*2

        if hidden_layer_type[-1] == "RNND" or hidden_layer_type[-1] == "LSTMD":
            self.final_layer = self.rnn_layers[-1]
        else:
            if output_type == 'LINEAR':
                self.final_layer = LinearLayer(rng, self.rnn_layers[-1].output, input_size, self.n_out)
            elif output_type == 'SIGMOID':
                self.final_layer = SigmoidLayer(rng, self.rnn_layers[-1].output, input_size, self.n_out, activation=T.nnet.sigmoid)
            else:
                logger.critical("This output layer type: %s is not supported right now! \n Please use one of the following: LINEAR, SIGMOID\n" %(output_type))
                sys.exit(1)

            self.params.extend(self.final_layer.params)


        self.updates = {}
        for param in self.params:
            self.updates[param] = theano.shared(value = np.zeros(param.get_value(borrow = True).shape,
                                                dtype = theano.config.floatX), name = 'updates')

        self.finetune_cost = T.mean(T.sum((self.final_layer.output - self.y) ** 2, axis=1))
        self.errors = T.mean(T.sum((self.final_layer.output - self.y) ** 2, axis=1))

#        self.L2_sqr = (self.W_hy ** 2).sum()

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy):
        """ This function is to build finetune functions and to update gradients

        :param train_shared_xy: theano shared variable for input and output training data
        :type train_shared_xy: tuple of shared variable
        :param valid_shared_xy: theano shared variable for input and output development data
        :type valid_shared_xy: tuple of shared variable
        :returns: finetune functions for training and development

        """

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
#        index = T.scalar('index', dtype='int32')
#        batch_size = T.scalar('batch_size', dtype='int32')

        cost = self.finetune_cost #+ self.L2_reg * self.L2_sqr

        gparams = T.grad(cost, self.params)


        # zip just concatenate two lists
        updates = OrderedDict()

        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        train_model = theano.function(inputs = [lr, mom],  #index, batch_size
                                      outputs = self.errors,
                                      updates = updates,
                                      givens = {self.x: train_set_x, #[index*batch_size:(index + 1)*batch_size]
                                                self.y: train_set_y,
                                                self.is_train: np.cast['int32'](1)}, on_unused_input='ignore')


        valid_model = theano.function(inputs = [],
                                      outputs = self.errors,
                                      givens = {self.x: valid_set_x,
                                                self.y: valid_set_y,
                                                self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        return  train_model, valid_model

    def build_finetune_functions_S2S(self, train_shared_xyd, valid_shared_xyd):
        """ This function is to build finetune functions and to update gradients

        :param train_shared_xy: theano shared variable for input and output training data
        :type train_shared_xy: tuple of shared variable
        :param valid_shared_xy: theano shared variable for input and output development data
        :type valid_shared_xy: tuple of shared variable
        :returns: finetune functions for training and development

        """

        (train_set_x, train_set_y, train_set_d) = train_shared_xyd
        (valid_set_x, valid_set_y, valid_set_d) = valid_shared_xyd

        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum

        cost = self.finetune_cost #+ self.L2_reg * self.L2_sqr

        gparams = T.grad(cost, self.params)


        # zip just concatenate two lists
        updates = OrderedDict()

        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        train_model = theano.function(inputs = [lr, mom],
                                      outputs = self.errors,
                                      updates = updates,
                                      givens = {self.x: train_set_x,
                                                self.y: train_set_y,
                                                self.d: train_set_d,
                                                self.is_train: np.cast['int32'](1)}, on_unused_input='ignore')


        valid_model = theano.function(inputs = [],
                                      outputs = self.errors,
                                      givens = {self.x: valid_set_x,
                                                self.y: valid_set_y,
                                                self.d: valid_set_d,
                                                self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        return  train_model, valid_model

    def build_finetune_functions_S2SPF(self, train_shared_xydf, valid_shared_xydf):
        """ This function is to build finetune functions and to update gradients

        :param train_shared_xy: theano shared variable for input and output training data
        :type train_shared_xy: tuple of shared variable
        :param valid_shared_xy: theano shared variable for input and output development data
        :type valid_shared_xy: tuple of shared variable
        :returns: finetune functions for training and development

        """

        (train_set_x, train_set_y, train_set_d, train_set_f) = train_shared_xydf
        (valid_set_x, valid_set_y, valid_set_d, valid_set_f) = valid_shared_xydf

        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum

        cost = self.finetune_cost #+ self.L2_reg * self.L2_sqr

        gparams = T.grad(cost, self.params)


        # zip just concatenate two lists
        updates = OrderedDict()

        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        train_model = theano.function(inputs = [lr, mom],
                                      outputs = self.errors,
                                      updates = updates,
                                      givens = {self.x: train_set_x,
                                                self.y: train_set_y,
                                                self.d: train_set_d,
                                                self.f: train_set_f,
                                                self.is_train: np.cast['int32'](1)}, on_unused_input='ignore')


        valid_model = theano.function(inputs = [],
                                      outputs = self.errors,
                                      givens = {self.x: valid_set_x,
                                                self.y: valid_set_y,
                                                self.d: valid_set_d,
                                                self.f: valid_set_f,
                                                self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        return  train_model, valid_model

    def parameter_prediction(self, test_set_x):  #, batch_size
        """ This function is to predict the output of NN

        :param test_set_x: input features for a testing sentence
        :type test_set_x: python array variable
        :returns: predicted features

        """


        n_test_set_x = test_set_x.shape[0]

        test_out = theano.function([], self.final_layer.output,
              givens={self.x: test_set_x[0:n_test_set_x], self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_S2S(self, test_set_x, test_set_d):
        """ This function is to predict the output of NN

        :param test_set_x: input features for a testing sentence
        :param test_set_d: phone durations for a testing sentence
        :type test_set_x: python array variable
        :type test_set_d: python array variable
        :returns: predicted features

        """

        n_test_set_x = test_set_x.shape[0]

        test_out = theano.function([], self.final_layer.output,
                givens={self.x: test_set_x[0:n_test_set_x], self.d: test_set_d[0:n_test_set_x], self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_S2SPF(self, test_set_x, test_set_d, test_set_f):
        """ This function is to predict the output of NN

        :param test_set_x: input features for a testing sentence
        :param test_set_d: phone durations for a testing sentence
        :type test_set_x: python array variable
        :type test_set_d: python array variable
        :returns: predicted features

        """

        n_test_set_x  = test_set_x.shape[0]
        num_of_frames = sum(test_set_d)

        test_out = theano.function([], self.final_layer.output,
                givens={self.x: test_set_x[0:n_test_set_x], self.d: test_set_d[0:n_test_set_x], self.f: test_set_f[0:num_of_frames], self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_CTC(self, test_set_x):  #, batch_size

        n_test_set_x = test_set_x.shape[0]

        test_out = theano.function([], self.rnn_layers[-1].output,
              givens={self.x: test_set_x[0:n_test_set_x]})

        predict_parameter = test_out()

        return predict_parameter

    def parameter_prediction_MDN(self, test_set_x):  #, batch_size

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
