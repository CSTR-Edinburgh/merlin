# Test the classes used in Merlin pipeline
# TODO run some very simple training on random data)

import sys
import os
sys.path.append('../src')

import numpy as np
import cPickle
import logging

def build_model(hidden_layer_type):
    logger.info('  DeepRecurrentNetwork '+str(hidden_layer_type))
    nnmodel = DeepRecurrentNetwork(8, 16*np.ones(len(hidden_layer_type)), 4, L1_reg=0.0, L2_reg=0.00001, hidden_layer_type=hidden_layer_type)
    
    # Always try to save it and reload it
    modelfile = 'log/model.pkl'
    cPickle.dump(nnmodel, open(modelfile, 'wb'))
    nnmodel = cPickle.load(open(modelfile, 'rb'))

    logger.info('    OK')

    return nnmodel

if __name__ == '__main__':

    # Get a logger for these tests
    logging.basicConfig(format='%(asctime)s %(levelname)8s%(name)15s: %(message)s')
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)

    logger.info('Testing Merlin classes')

    # Build various models
    logger.info('Build models without training')
    from models.deep_rnn import DeepRecurrentNetwork
    nnmodel = build_model(['TANH'])
    del nnmodel
    nnmodel = build_model(['TANH', 'TANH'])
    del nnmodel
    nnmodel = build_model(['LSTM', 'LSTM'])
    del nnmodel
    nnmodel = build_model(['SLSTM', 'SLSTM'])
    del nnmodel
