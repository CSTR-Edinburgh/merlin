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


import cPickle
import gzip
import os, sys, errno
import time
import math

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, HTSDurationLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
#from frontend.acoustic_normalisation import CMPNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
#from frontend.feature_normalisation_base import FeatureNormBase
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer

import configuration

from models.dnn import DNN
#from models.ms_dnn import MultiStreamDNN
#from models.ms_dnn_gv import MultiStreamDNNGv
#from models.sdae import StackedDenoiseAutoEncoder

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate


#import matplotlib.pyplot as plt
# our custom logging class that can also plot
#from logplot.logging_plotting import LoggerPlotter, MultipleTimeSeriesPlot, SingleWeightMatrixPlot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import StringIO


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def read_file_list(file_name):

    logger = logging.getLogger("read_file_list")

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    logger.debug('Read file list from %s' % file_name)
    return  file_lists


def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

    

def visualize_dnn(dnn):

    layer_num = len(dnn.params) / 2     ## including input and output

    for i in xrange(layer_num):
        fig_name = 'Activation weights W' + str(i)
        fig_title = 'Activation weights of W' + str(i)
        xlabel = 'Neuron index of hidden layer ' + str(i)
        ylabel = 'Neuron index of hidden layer ' + str(i+1)
        if i == 0:
            xlabel = 'Input feature index'
        if i == layer_num-1:
            ylabel = 'Output feature index'

        logger.create_plot(fig_name, SingleWeightMatrixPlot)
        plotlogger.add_plot_point(fig_name, fig_name, dnn.params[i*2].get_value(borrow=True).T)
        plotlogger.save_plot(fig_name, title=fig_name, xlabel=xlabel, ylabel=ylabel)


def train_DNN(train_xy_file_list, valid_xy_file_list, \
              nnets_file_name, n_ins, n_outs, ms_outs, hyper_params, buffer_size, plot=False):
    
    # get loggers for this function
    # this one writes to both console and file
    logger = logging.getLogger("main.train_DNN")
    logger.debug('Starting train_DNN')

    if plot:
        # this one takes care of plotting duties
        plotlogger = logging.getLogger("plotting")
        # create an (empty) plot of training convergence, ready to receive data points
        logger.create_plot('training convergence',MultipleSeriesPlot)

    try:
        assert numpy.sum(ms_outs) == n_outs
    except AssertionError:
        logger.critical('the summation of multi-stream outputs does not equal to %d' %(n_outs))
        raise

    ####parameters#####
    finetune_lr     = numpy.asarray(hyper_params['learning_rate'],  dtype='float32')
    training_epochs = int(hyper_params['training_epochs'])
    batch_size      = int(hyper_params['batch_size'])
    l1_reg          = float(hyper_params['l1_reg'])
    l2_reg          = float(hyper_params['l2_reg'])
#     private_l2_reg  = float(hyper_params['private_l2_reg'])
    warmup_epoch    = int(hyper_params['warmup_epoch'])
    momentum        = float(hyper_params['momentum'])
    warmup_momentum = float(hyper_params['warmup_momentum'])

    use_rprop = int(hyper_params['use_rprop']) 
    
    use_rprop = int(hyper_params['use_rprop']) 
    
    hidden_layers_sizes = hyper_params['hidden_layer_size']    

#     stream_weights       = hyper_params['stream_weights']
#     private_hidden_sizes = hyper_params['private_hidden_sizes']

    buffer_utt_size = buffer_size
    early_stop_epoch = int(hyper_params['early_stop_epochs'])

    hidden_activation = hyper_params['hidden_activation']
    output_activation = hyper_params['output_activation']
    
#     stream_lr_weights = hyper_params['stream_lr_weights']
#     use_private_hidden = hyper_params['use_private_hidden']

    model_type = hyper_params['model_type']
    
    ## use a switch to turn on pretraining
    ## pretraining may not help too much, if this case, we turn it off to save time
    do_pretraining = hyper_params['do_pretraining']
    pretraining_epochs = int(hyper_params['pretraining_epochs'])
    pretraining_lr = float(hyper_params['pretraining_lr'])


    buffer_size = int(buffer_size / batch_size) * batch_size

    ###################
    (train_x_file_list, train_y_file_list) = train_xy_file_list
    (valid_x_file_list, valid_y_file_list) = valid_xy_file_list

    logger.debug('Creating training   data provider')
    train_data_reader = ListDataProvider(x_file_list = train_x_file_list, y_file_list = train_y_file_list, n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True)

    logger.debug('Creating validation data provider')
    valid_data_reader = ListDataProvider(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list, n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False)

    shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_next_partition()
    train_set_x, train_set_y = shared_train_set_xy
    shared_valid_set_xy, temp_valid_set_x, temp_valid_set_y = valid_data_reader.load_next_partition()
    valid_set_x, valid_set_y = shared_valid_set_xy
    train_data_reader.reset()
    valid_data_reader.reset()

    ##temporally we use the training set as pretrain_set_x.
    ##we need to support any data for pretraining
    pretrain_set_x = train_set_x

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    logger.info('building the model')


    dnn_model = None
    pretrain_fn = None  ## not all the model support pretraining right now
    train_fn = None
    valid_fn = None
    valid_model = None ## valid_fn and valid_model are the same. reserve to computer multi-stream distortion
    if model_type == 'DNN':
        dnn_model = DNN(numpy_rng=numpy_rng, n_ins=n_ins, n_outs = n_outs,
                        l1_reg = l1_reg, l2_reg = l2_reg, 
                         hidden_layers_sizes = hidden_layers_sizes, 
                          hidden_activation = hidden_activation, 
                          output_activation = output_activation,
                          use_rprop = use_rprop, rprop_init_update=finetune_lr)
        train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (train_set_x, train_set_y), (valid_set_x, valid_set_y), batch_size=batch_size)

    else: 
        logger.critical('%s type NN model is not supported!' %(model_type))
        raise

    logger.info('fine-tuning the %s model' %(model_type))

    start_time = time.clock()

    best_dnn_model = dnn_model
    best_validation_loss = sys.float_info.max
    previous_loss = sys.float_info.max

    early_stop = 0
    epoch = 0
    previous_finetune_lr = finetune_lr
    
    while (epoch < training_epochs):
        epoch = epoch + 1
        
        current_momentum = momentum
        current_finetune_lr = finetune_lr
        if epoch <= warmup_epoch:
            current_finetune_lr = finetune_lr
            current_momentum = warmup_momentum
        else:
            current_finetune_lr = previous_finetune_lr * 0.5
        
        previous_finetune_lr = current_finetune_lr
        
        train_error = []
        sub_start_time = time.clock()

        while (not train_data_reader.is_finish()):
            shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_next_partition()
            train_set_x.set_value(numpy.asarray(temp_train_set_x, dtype=theano.config.floatX), borrow=True)
            train_set_y.set_value(numpy.asarray(temp_train_set_y, dtype=theano.config.floatX), borrow=True)
            
            n_train_batches = train_set_x.get_value().shape[0] / batch_size

            logger.debug('this partition: %d frames (divided into %d batches of size %d)' %(train_set_x.get_value(borrow=True).shape[0], n_train_batches, batch_size) )

            for minibatch_index in xrange(n_train_batches):
                this_train_error = train_fn(minibatch_index, current_finetune_lr, current_momentum)
                train_error.append(this_train_error)
                
                if numpy.isnan(this_train_error):
                    logger.warning('training error over minibatch %d of %d was %s' % (minibatch_index+1,n_train_batches,this_train_error) )

        train_data_reader.reset()
        
        logger.debug('calculating validation loss')
        validation_losses = valid_fn()
        this_validation_loss = numpy.mean(validation_losses)

        # this has a possible bias if the minibatches were not all of identical size
        # but it should not be siginficant if minibatches are small
        this_train_valid_loss = numpy.mean(train_error)

        sub_end_time = time.clock()

        loss_difference = this_validation_loss - previous_loss

        logger.info('epoch %i, validation error %f, train error %f  time spent %.2f' %(epoch, this_validation_loss, this_train_valid_loss, (sub_end_time - sub_start_time)))
        if plot:
            plotlogger.add_plot_point('training convergence','validation set',(epoch,this_validation_loss))
            plotlogger.add_plot_point('training convergence','training set',(epoch,this_train_valid_loss))
            plotlogger.save_plot('training convergence',title='Progress of training and validation error',xlabel='epochs',ylabel='error')

        if this_validation_loss < best_validation_loss:
            best_dnn_model = dnn_model
            best_validation_loss = this_validation_loss
            logger.debug('validation loss decreased, so saving model')
            early_stop = 0
        else:
            logger.debug('validation loss did not improve')
            dbn = best_dnn_model
            early_stop += 1

        if early_stop >= early_stop_epoch:
            # too many consecutive epochs without surpassing the best model
            logger.debug('stopping early')
            break

        if math.isnan(this_validation_loss):
            break
        
        previous_loss = this_validation_loss

    end_time = time.clock()
    cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))
            
    logger.info('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

    if plot:
        plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')
    
    return  best_validation_loss
    

def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
    
#    visualize_dnn(dnn_model)

    file_number = len(valid_file_list)

    for i in xrange(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        features = features.reshape((-1, n_ins))
        temp_set_x = features.tolist()
        test_set_x = theano.shared(numpy.asarray(temp_set_x, dtype=theano.config.floatX)) 
        
        predicted_parameter = dnn_model.parameter_prediction(test_set_x=test_set_x)
#        predicted_parameter = test_out()

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()
        
##generate bottleneck layer as festures
def dnn_hidden_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
    
    file_number = len(valid_file_list)

    for i in xrange(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        features = features.reshape((-1, n_ins))
        temp_set_x = features.tolist()
        test_set_x = theano.shared(numpy.asarray(temp_set_x, dtype=theano.config.floatX)) 
        
        predicted_parameter = dnn_model.generate_top_hidden_layer(test_set_x=test_set_x)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()


def main_function(cfg):    
    
    
    # get a logger for this main function
    logger = logging.getLogger("main")
    
    # get another logger to handle plotting duties
    plotlogger = logging.getLogger("plotting")

    # later, we might do this via a handler that is created, attached and configured
    # using the standard config mechanism of the logging module
    # but for now we need to do it manually
    plotlogger.set_plot_path(cfg.plot_dir)
    
    #### parameter setting########
    hidden_layers_sizes = cfg.hyper_params['hidden_layer_size']
    
    
    ####prepare environment
    
    try:
        file_id_list = read_file_list(cfg.file_id_scp)
        logger.debug('Loaded file id list from %s' % cfg.file_id_scp)
    except IOError:
        # this means that open(...) threw an error
        logger.critical('Could not load file id list from %s' % cfg.file_id_scp)
        raise
    
    ###total file number including training, development, and testing
    total_file_number = len(file_id_list)
    
    data_dir = cfg.data_dir

    nn_cmp_dir       = os.path.join(data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    nn_cmp_nosil_dir  = os.path.join(data_dir, 'nn_nosil' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    nn_cmp_norm_dir   = os.path.join(data_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    
    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir   = os.path.join(cfg.work_dir, 'gen')    

    in_file_list_dict = {}

    for feature_name in cfg.in_dir_dict.keys():
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
    nn_cmp_nosil_file_list  = prepare_file_path_list(file_id_list, nn_cmp_nosil_dir, cfg.cmp_ext)
    nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

    ###normalisation information
    norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')
	
    ### normalise input full context label

    # currently supporting two different forms of lingustic features
    # later, we should generalise this 

    if cfg.label_style == 'HTS':
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension + cfg.appended_input_dim
        logger.info('Input label dimension is %d' % lab_dim)
        suffix=str(lab_dim)
    elif cfg.label_style == 'HTS_duration':
        label_normaliser = HTSDurationLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension ## + cfg.appended_input_dim
        logger.info('Input label dimension is %d' % lab_dim)
        suffix=str(lab_dim)
    # no longer supported - use new "composed" style labels instead
    elif cfg.label_style == 'composed':
        # label_normaliser = XMLLabelNormalisation(xpath_file_name=cfg.xpath_file_name)
        suffix='composed'

    if cfg.process_labels_in_work_dir:
        label_data_dir = cfg.work_dir
    else:
        label_data_dir = data_dir

    # the number can be removed
    binary_label_dir      = os.path.join(label_data_dir, 'binary_label_'+suffix)
    nn_label_dir          = os.path.join(label_data_dir, 'nn_no_silence_lab_'+suffix)
    nn_label_norm_dir     = os.path.join(label_data_dir, 'nn_no_silence_lab_norm_'+suffix)
#    nn_label_norm_mvn_dir = os.path.join(data_dir, 'nn_no_silence_lab_norm_'+suffix)

    in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
    binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
    nn_label_file_list       = prepare_file_path_list(file_id_list, nn_label_dir, cfg.lab_ext)
    nn_label_norm_file_list  = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)
    
    # to do - sanity check the label dimension here?
    
    
    
    min_max_normaliser = None
    label_norm_file = 'label_norm_%s.dat' %(cfg.label_style)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)
    
    if cfg.NORMLAB and (cfg.label_style in ['HTS', 'HTS_duration']):
        # simple HTS labels 
    	logger.info('preparing label data (input) using standard HTS style labels')
        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list)

        if cfg.label_style == 'HTS':
            remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = cfg.silence_pattern)
            remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)
        elif cfg.label_style == 'HTS_duration':
            ## don't remove silences for duration
            nn_label_file_list = binary_label_file_list
            
        
        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
        ###use only training data to find min-max information, then apply on the whole dataset
        min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)


    if cfg.NORMLAB and (cfg.label_style == 'composed'):
        # new flexible label preprocessor
        
    	logger.info('preparing label data (input) using "composed" style labels')
        label_composer = LabelComposer()
        label_composer.load_label_configuration(cfg.label_config_file)
    
        logger.info('Loaded label configuration')
        # logger.info('%s' % label_composer.configuration.labels )

        lab_dim=label_composer.compute_label_dimension()
        logger.info('label dimension will be %d' % lab_dim)
        
        if cfg.precompile_xpaths:
            label_composer.precompile_xpaths()
        
        # there are now a set of parallel input label files (e.g, one set of HTS and another set of Ossian trees)
        # create all the lists of these, ready to pass to the label composer

        in_label_align_file_list = {}
        for label_style, label_style_required in label_composer.label_styles.iteritems():
            if label_style_required:
                logger.info('labels of style %s are required - constructing file paths for them' % label_style)
                if label_style == 'xpath':
                    in_label_align_file_list['xpath'] = prepare_file_path_list(file_id_list, cfg.xpath_label_align_dir, cfg.utt_ext, False)
                elif label_style == 'hts':
                    in_label_align_file_list['hts'] = prepare_file_path_list(file_id_list, cfg.hts_label_align_dir, cfg.lab_ext, False)
                else:
                    logger.critical('unsupported label style %s specified in label configuration' % label_style)
                    raise Exception
        
            # now iterate through the files, one at a time, constructing the labels for them 
            num_files=len(file_id_list)
            logger.info('the label styles required are %s' % label_composer.label_styles)
            
            for i in xrange(num_files):
                logger.info('making input label features for %4d of %4d' % (i+1,num_files))

                # iterate through the required label styles and open each corresponding label file

                # a dictionary of file descriptors, pointing at the required files
                required_labels={}
                
                for label_style, label_style_required in label_composer.label_styles.iteritems():
                    
                    # the files will be a parallel set of files for a single utterance
                    # e.g., the XML tree and an HTS label file
                    if label_style_required:
                        required_labels[label_style] = open(in_label_align_file_list[label_style][i] , 'r')
                        logger.debug(' opening label file %s' % in_label_align_file_list[label_style][i])

                logger.debug('label styles with open files: %s' % required_labels)
                label_composer.make_labels(required_labels,out_file_name=binary_label_file_list[i],fill_missing_values=cfg.fill_missing_values,iterate_over_frames=cfg.iterate_over_frames)
                    
                # now close all opened files
                for fd in required_labels.itervalues():
                    fd.close()
        
        
        # silence removal
        if cfg.remove_silence_using_binary_labels:
            silence_feature = 0 ## use first feature in label -- hardcoded for now
            logger.info('Silence removal from label using silence feature: %s'%(label_composer.configuration.labels[silence_feature]))
            logger.info('Silence will be removed from CMP files in same way')
            ## Binary labels have 2 roles: both the thing trimmed and the instructions for trimming: 
            trim_silence(binary_label_file_list, nn_label_file_list, lab_dim, \
                                binary_label_file_list, lab_dim, silence_feature, percent_to_keep=5)
        else:
            logger.info('No silence removal done')
            # start from the labels we have just produced, not trimmed versions
            nn_label_file_list = binary_label_file_list
        
        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
        ###use only training data to find min-max information, then apply on the whole dataset
        min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

    if min_max_normaliser != None:
        ### save label normalisation information for unseen testing labels
        label_min_vector = min_max_normaliser.min_vector
        label_max_vector = min_max_normaliser.max_vector
        label_norm_info = numpy.concatenate((label_min_vector, label_max_vector), axis=0)
    
        label_norm_info = numpy.array(label_norm_info, 'float32')
        fid = open(label_norm_file, 'wb')
        label_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' %(label_min_vector.size, label_norm_file))




    ### make output acoustic data
    if cfg.MAKECMP:
    	logger.info('creating acoustic (output) features')
        delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
        acc_win = cfg.acc_win     #[1.0, -2.0, 1.0]
        
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

        if cfg.label_style == 'HTS':

            if cfg.remove_silence_using_binary_labels:
                ## do this to get lab_dim:
                label_composer = LabelComposer()
                label_composer.load_label_configuration(cfg.label_config_file)
                lab_dim=label_composer.compute_label_dimension()
        
                silence_feature = 0 ## use first feature in label -- hardcoded for now
                logger.info('Silence removal from CMP using binary label file') 

                ## overwrite the untrimmed audio with the trimmed version:
                trim_silence(nn_cmp_file_list, nn_cmp_nosil_file_list, cfg.cmp_dim, 
                                    binary_label_file_list, lab_dim, silence_feature)
                                
            else: ## back off to previous method using HTS labels:
                remover = SilenceRemover(n_cmp = cfg.cmp_dim, silence_pattern = cfg.silence_pattern)
                remover.remove_silence(nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
                                       in_label_align_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
                                       nn_cmp_nosil_file_list[0:cfg.train_file_number+cfg.valid_file_number]) # save to itself
        
        
        elif cfg.label_style == 'HTS_duration':
            ## don't remove silences for duration
            nn_cmp_nosil_file_list = nn_cmp_file_list
            pass
            

    ### save acoustic normalisation information for normalising the features back
    var_dir   = os.path.join(data_dir, 'var')
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    var_file_dict = {}
    for feature_name in cfg.out_dimension_dict.keys():
        var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))
        
    ### normalise output acoustic data
    if cfg.NORMCMP:
    	logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
        cmp_norm_info = None
        if cfg.output_feature_normalisation == 'MVN':
            normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
            ###calculate mean and std vectors on the training data, and apply on the whole dataset
            global_mean_vector = normaliser.compute_mean(nn_cmp_nosil_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
            global_std_vector = normaliser.compute_std(nn_cmp_nosil_file_list[0:cfg.train_file_number], global_mean_vector, 0, cfg.cmp_dim)

            normaliser.feature_normalisation(nn_cmp_nosil_file_list[0:cfg.train_file_number+cfg.valid_file_number], 
                                             nn_cmp_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number])
            cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

        elif cfg.output_feature_normalisation == 'MINMAX':        
            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim)
            global_mean_vector = min_max_normaliser.compute_mean(nn_cmp_nosil_file_list[0:cfg.train_file_number])
            global_std_vector = min_max_normaliser.compute_std(nn_cmp_nosil_file_list[0:cfg.train_file_number], global_mean_vector)

            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
            min_max_normaliser.find_min_max_values(nn_cmp_nosil_file_list[0:cfg.train_file_number])
            min_max_normaliser.normalise_data(nn_cmp_nosil_file_list, nn_cmp_norm_file_list)

            cmp_min_vector = min_max_normaliser.min_vector
            cmp_max_vector = min_max_normaliser.max_vector
            cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

        else:
            logger.critical('Normalisation type %s is not supported!\n' %(cfg.output_feature_normalisation))
            raise
 
        cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
        fid = open(norm_info_file, 'wb')
        cmp_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' %(cfg.output_feature_normalisation, norm_info_file))
        # logger.debug(' value was\n%s' % cmp_norm_info)
        
        feature_index = 0
        for feature_name in cfg.out_dimension_dict.keys():
            feature_std_vector = numpy.array(global_std_vector[:,feature_index:feature_index+cfg.out_dimension_dict[feature_name]], 'float32')

            fid = open(var_file_dict[feature_name], 'w')
            feature_std_vector.tofile(fid)
            fid.close()

            logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))
            # logger.debug(' value was\n%s' % feature_std_vector)

            feature_index += cfg.out_dimension_dict[feature_name]

    train_x_file_list = nn_label_norm_file_list[0:cfg.train_file_number]
    train_y_file_list = nn_cmp_norm_file_list[0:cfg.train_file_number]
    valid_x_file_list = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]    
    valid_y_file_list = nn_cmp_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
    test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    
    test_y_file_list  = nn_cmp_norm_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]


    # we need to know the label dimension before training the DNN
    # computing that requires us to look at the labels
    #
    # currently, there are two ways to do this
    if cfg.label_style == 'HTS':
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension + cfg.appended_input_dim

    elif cfg.label_style == 'composed':
        label_composer = LabelComposer()
        label_composer.load_label_configuration(cfg.label_config_file)
        lab_dim=label_composer.compute_label_dimension()

    logger.info('label dimension is %d' % lab_dim)

    combined_model_arch = str(len(hidden_layers_sizes))
    for hid_size in hidden_layers_sizes:
        combined_model_arch += '_' + str(hid_size)
    
    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.model' \
                      %(model_dir, cfg.model_type, cfg.combined_feature_name, int(cfg.multistream_switch), 
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number)

    ### DNN model training
    if cfg.TRAINDNN:

    	logger.info('training DNN')

        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # not an error - just means directory already exists
                pass
            else:
                logger.critical('Failed to create model directory %s' % model_dir)
                logger.critical(' OS error was: %s' % e.strerror)
                raise

        try:
            # print   'start DNN'
            train_DNN(train_xy_file_list = (train_x_file_list, train_y_file_list), \
                      valid_xy_file_list = (valid_x_file_list, valid_y_file_list), \
                      nnets_file_name = nnets_file_name, \
                      n_ins = lab_dim, n_outs = cfg.cmp_dim, ms_outs = cfg.multistream_outs, \
                      hyper_params = cfg.hyper_params, buffer_size = cfg.buffer_size, plot = cfg.plot)
        except KeyboardInterrupt:
            logger.critical('train_DNN interrupted via keyboard')
            # Could 'raise' the exception further, but that causes a deep traceback to be printed
            # which we don't care about for a keyboard interrupt. So, just bail out immediately
            sys.exit(1)
        except:
            logger.critical('train_DNN threw an exception')
            raise
            
    ### generate parameters from DNN
    temp_dir_name = '%s_%s_%d_%d_%d_%d_%d_%d' \
                    %(cfg.model_type, cfg.combined_feature_name, int(cfg.do_post_filtering), \
                      cfg.train_file_number, lab_dim, cfg.cmp_dim, \
                      len(hidden_layers_sizes), hidden_layers_sizes[0])
    gen_dir = os.path.join(gen_dir, temp_dir_name)

    gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    

    if cfg.DNNGEN:
    	logger.info('generating from DNN')

        try:
            os.makedirs(gen_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # not an error - just means directory already exists
                pass
            else:
                logger.critical('Failed to create generation directory %s' % gen_dir)
                logger.critical(' OS error was: %s' % e.strerror)
                raise

        gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)
    
#        dnn_generation(valid_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)
        dnn_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)

    	logger.debug('denormalising generated output using method %s' % cfg.output_feature_normalisation)

        fid = open(norm_info_file, 'rb')
        cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_min_vector = cmp_min_max[0, ] 
        cmp_max_vector = cmp_min_max[1, ]

        if cfg.output_feature_normalisation == 'MVN':
            denormaliser = MeanVarianceNorm(feature_dimension = cfg.cmp_dim)
            denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)
        
        elif cfg.output_feature_normalisation == 'MINMAX':
            denormaliser = MinMaxNormalisation(cfg.cmp_dim, min_value = 0.01, max_value = 0.99, min_vector = cmp_min_vector, max_vector = cmp_max_vector)
            denormaliser.denormalise_data(gen_file_list, gen_file_list)
        else:
            logger.critical('denormalising method %s is not supported!\n' %(cfg.output_feature_normalisation))
            raise

        ##perform MLPG to smooth parameter trajectory
        ## lf0 is included, the output features much have vuv. 
        generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
        generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict, var_file_dict)    

    ### generate wav
    if cfg.GENWAV:
    	logger.info('reconstructing waveform(s)')
    	generate_wav(gen_dir, gen_file_id_list, cfg)     # generated speech
#    	generate_wav(nn_cmp_dir, gen_file_id_list)  # reference copy synthesis speech
    	
    ### evaluation: calculate distortion        
    if cfg.CALMCD:
    	logger.info('calculating MCD')

        ref_data_dir = os.path.join(data_dir, 'ref_data')

        ref_mgc_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.mgc_ext)
        ref_bap_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.bap_ext)
        ref_lf0_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.lf0_ext)

        in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        calculator = IndividualDistortionComp()

        spectral_distortion = 0.0
        bap_mse             = 0.0
        f0_mse              = 0.0
        vuv_error           = 0.0
        
        valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
        test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        if cfg.remove_silence_using_binary_labels:
            ## get lab_dim:
            label_composer = LabelComposer()
            label_composer.load_label_configuration(cfg.label_config_file)
            lab_dim=label_composer.compute_label_dimension()

            ## use first feature in label -- hardcoded for now
            silence_feature = 0
            
            ## Use these to trim silence:
            untrimmed_test_labels = binary_label_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]    


        if cfg.in_dimension_dict.has_key('mgc'):
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_mgc_list, cfg.mgc_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.mgc_dim, silence_pattern =  cfg.silence_pattern)
                remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD

            
        if cfg.in_dimension_dict.has_key('bap'):
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_bap_list, cfg.bap_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.bap_dim, silence_pattern =  cfg.silence_pattern)
                remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
            valid_bap_mse        = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            test_bap_mse         = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            valid_bap_mse = valid_bap_mse / 10.0   
            test_bap_mse  = test_bap_mse / 10.0    
                
        if cfg.in_dimension_dict.has_key('lf0'):
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_lf0_list, cfg.lf0_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.lf0_dim, silence_pattern = ['*-#+*'])
                remover.remove_silence(in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lf0_list)
            valid_f0_mse, valid_f0_corr, valid_vuv_error   = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
            test_f0_mse , test_f0_corr, test_vuv_error    = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)

        logger.info('Develop: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(valid_spectral_distortion, valid_bap_mse, valid_f0_mse, valid_f0_corr, valid_vuv_error*100.))
        logger.info('Test   : DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(test_spectral_distortion , test_bap_mse , test_f0_mse , test_f0_corr, test_vuv_error*100.))

if __name__ == '__main__':
    
    
    
    # these things should be done even before trying to parse the command line

    # create a configuration instance
    # and get a short name for this instance
    cfg=configuration.cfg

    # set up logging to use our custom class
    logging.setLoggerClass(LoggerPlotter)

    # get a logger for this main function
    logger = logging.getLogger("main")


    if len(sys.argv) != 2:
        logger.critical('usage: run_dnn.sh [config file name]')
        sys.exit(1)

    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    
    if cfg.profile:
        logger.info('profiling is activated')
        import cProfile, pstats
        cProfile.run('main_function(cfg)', 'mainstats')

        # create a stream for the profiler to write to
        profiling_output = StringIO.StringIO()
        p = pstats.Stats('mainstats', stream=profiling_output)

        # print stats to that stream
        # here we just report the top 10 functions, sorted by total amount of time spent in each
        p.strip_dirs().sort_stats('tottime').print_stats(10)

        # print the result to the log
        logger.info('---Profiling result follows---\n%s' %  profiling_output.getvalue() )
        profiling_output.close()
        logger.info('---End of profiling result---')
        
    else:
        main_function(cfg)
        
    sys.exit(0)
