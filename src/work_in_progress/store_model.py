
import cPickle
import gzip
import os, sys, errno
import time
import math

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano
#import theano.tensor as T


from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, HTSDurationLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
#from frontend.mlpg_fast import MLParameterGenerationFast

#from frontend.mlpg_fast_layer import MLParameterGenerationFastLayer


import configuration
from models.deep_rnn import DeepRecurrentNetwork

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate

from io_funcs.binary_io import  BinaryIOCollection

#import matplotlib.pyplot as plt
# our custom logging class that can also plot
#from logplot.logging_plotting import LoggerPlotter, MultipleTimeSeriesPlot, SingleWeightMatrixPlot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import StringIO
 

def store_network(nnets_file_name, outdir):
    print('store network')
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))

        
    names = [p.name for p in dnn_model.params]
    param_vals = [p.get_value(borrow=True) for p in dnn_model.params]
    shapes = [numpy.shape(p) for p in param_vals]
    print cfg.hidden_layer_size    
    layer_types = cfg.hidden_layer_type
    if cfg.output_activation == 'linear':
        layer_types.append('LINEAR')
    else:
        sys.exit('unsupported output activation')
    assert len(param_vals) == len(layer_types) * 2 ##  W and b for each layer
    print names
    
    
    p_ix = 0
    for (l_ix, layer_type) in enumerate(layer_types):
        layer_name = 'LAYER_' + str(l_ix+1).zfill(3) + '_' + layer_type + '_'
        #print layer_name
        for part in ['W','b']:
            assert names[p_ix] == part
            fname = layer_name + part
            print fname
            #numpy.savetxt(os.path.join(outdir, fname + '.txt'), param_vals[p_ix])
            numpy.save(os.path.join(outdir, fname + '.npy'), param_vals[p_ix])
            
            p_ix += 1
            
    ### Input normalisation:-
    if cfg.process_labels_in_work_dir:
        label_data_dir = cfg.work_dir
    else:
        label_data_dir = cfg.data_dir
    label_norm_file = 'label_norm_%s.dat' %(cfg.label_style)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)
    
    lab_norm_data = numpy.fromfile(label_norm_file, 'float32')
    labsize = numpy.shape(lab_norm_data)[0]
    
    min_vect = lab_norm_data[:(labsize/2)]
    max_vect = lab_norm_data[(labsize/2):]
     
    print min_vect
    print max_vect
    
    fname = 'NORM_INPUT_MIN'
    numpy.save(os.path.join(outdir, fname + '.npy'), min_vect)
    fname = 'NORM_INPUT_MAX'
    numpy.save(os.path.join(outdir, fname + '.npy'), max_vect)
        
        
    ## output norm
    assert cfg.output_feature_normalisation == 'MVN'
    norm_info_file = os.path.join(cfg.data_dir, \
        'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')    
    
    out_norm_data = numpy.fromfile(norm_info_file, 'float32')
    outsize = numpy.shape(out_norm_data)[0]
    
    mean_vect = out_norm_data[:(outsize/2)]
    std_vect = out_norm_data[(outsize/2):]
     
    print mean_vect
    print std_vect
    
    fname = 'NORM_OUTPUT_MEAN'
    numpy.save(os.path.join(outdir, fname + '.npy'), mean_vect)
    fname = 'NORM_OUTPUT_STD'
    numpy.save(os.path.join(outdir, fname + '.npy'), std_vect)


    in_streams = cfg.in_dimension_dict.keys()
    indims = [str(cfg.in_dimension_dict[s]) for s in in_streams]
    out_streams = cfg.out_dimension_dict.keys()
    outdims = [str(cfg.out_dimension_dict[s]) for s in out_streams]

    f = open(os.path.join(outdir, 'stream_info.txt'), 'w')
    f.write(' '.join(in_streams) + '\n')
    f.write(' '.join(indims) + '\n')
    f.write(' '.join(out_streams) + '\n')    
    f.write(' '.join(outdims) + '\n')
    f.close()
    
def main_function(cfg, outdir, model_pickle_file=None):    
    
    hidden_layer_size = cfg.hyper_params['hidden_layer_size']
    data_dir = cfg.data_dir
    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
#     norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')
	
    ### normalise input full context label
    if cfg.label_style == 'HTS':
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension + cfg.appended_input_dim
        print('Input label dimension is %d' % lab_dim)
        suffix=str(lab_dim)
    elif cfg.label_style == 'HTS_duration':
        label_normaliser = HTSDurationLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension ## + cfg.appended_input_dim
        print('Input label dimension is %d' % lab_dim)
        suffix=str(lab_dim)                
    # no longer supported - use new "composed" style labels instead
    elif cfg.label_style == 'composed':
        # label_normaliser = XMLLabelNormalisation(xpath_file_name=cfg.xpath_file_name)
        suffix='composed'

    

    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)
    
    ## if made with run_lstm:--
    '''
    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.%f.rnn.model' \
                      %(model_dir, cfg.combined_model_name, cfg.combined_feature_name, int(cfg.multistream_switch), 
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number, cfg.hyper_params['learning_rate'])
    '''
    
    ## if made with run_dnn:--
    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.model' \
                      %(model_dir, cfg.model_type, cfg.combined_feature_name, int(cfg.multistream_switch), 
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number)

    ## override the name computed from config variables if model_pickle_file specified:
    if model_pickle_file != None:
        nnets_file_name = model_pickle_file

    print('store DNN')


    store_network(nnets_file_name, outdir)


if __name__ == '__main__':
    cfg=configuration.cfg
    if len(sys.argv) not in [3, 4]:
        print('usage: run_dnn.sh [config file name]')
        sys.exit(1)

    if len(sys.argv) == 3:
        config_file = sys.argv[1]
        outdir = sys.argv[2]
        
        model_pickle_file = None

    

    elif len(sys.argv) == 4:
        config_file = sys.argv[1]
        model_pickle_file = sys.argv[2]
        outdir = sys.argv[3]    
    
    
    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    
    main_function(cfg, outdir, model_pickle_file=model_pickle_file)
        
