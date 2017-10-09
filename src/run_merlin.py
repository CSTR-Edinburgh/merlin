################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
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

import pickle
import gzip
import os, sys, errno
import time
import math

import subprocess
import socket # only for socket.getfqdn()

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano
import theano

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
from frontend.label_modifier import HTSLabelModification
from frontend.merge_features import MergeFeat

import configuration
from models.deep_rnn import DeepRecurrentNetwork

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate

from io_funcs.binary_io import  BinaryIOCollection

# our custom logging class that can also plot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import io
from utils.file_paths import FilePaths
from utils.utils import read_file_list, prepare_file_path_list


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists

def visualize_dnn(dnn):

    plotlogger = logging.getLogger("plotting")

        # reference activation weights in layers
    W = list(); layer_name = list()
    for i in range(len(dnn.params)):
        aa = dnn.params[i].get_value(borrow=True).T
        print(aa.shape, aa.size)
        if aa.size > aa.shape[0]:
            W.append(aa)
            layer_name.append(dnn.params[i].name)

    ## plot activation weights including input and output
    layer_num = len(W)
    for i_layer in range(layer_num):
        fig_name = 'Activation weights W' + str(i_layer) + '_' + layer_name[i_layer]
        fig_title = 'Activation weights of W' + str(i_layer)
        xlabel = 'Neuron index of hidden layer ' + str(i_layer)
        ylabel = 'Neuron index of hidden layer ' + str(i_layer+1)
        if i_layer == 0:
            xlabel = 'Input feature index'
        if i_layer == layer_num-1:
            ylabel = 'Output feature index'
        logger.create_plot(fig_name, SingleWeightMatrixPlot)
        plotlogger.add_plot_point(fig_name, fig_name, W[i_layer])
        plotlogger.save_plot(fig_name, title=fig_name, xlabel=xlabel, ylabel=ylabel)


def load_covariance(var_file_dict, out_dimension_dict):
    var = {}
    io_funcs = BinaryIOCollection()
    for feature_name in list(var_file_dict.keys()):
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

        var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

        var[feature_name] = var_values

    return  var


def train_DNN(train_xy_file_list, valid_xy_file_list, \
              nnets_file_name, n_ins, n_outs, ms_outs, hyper_params, buffer_size, plot=False, var_dict=None,
              cmp_mean_vector = None, cmp_std_vector = None, init_dnn_model_file = None):

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
    finetune_lr     = float(hyper_params['learning_rate'])
    training_epochs = int(hyper_params['training_epochs'])
    batch_size      = int(hyper_params['batch_size'])
    l1_reg          = float(hyper_params['l1_reg'])
    l2_reg          = float(hyper_params['l2_reg'])
    warmup_epoch    = int(hyper_params['warmup_epoch'])
    momentum        = float(hyper_params['momentum'])
    warmup_momentum = float(hyper_params['warmup_momentum'])

    hidden_layer_size = hyper_params['hidden_layer_size']

    buffer_utt_size = buffer_size
    early_stop_epoch = int(hyper_params['early_stop_epochs'])

    hidden_activation = hyper_params['hidden_activation']
    output_activation = hyper_params['output_activation']

    model_type = hyper_params['model_type']
    hidden_layer_type  = hyper_params['hidden_layer_type']

    ## use a switch to turn on pretraining
    ## pretraining may not help too much, if this case, we turn it off to save time
    do_pretraining = hyper_params['do_pretraining']
    pretraining_epochs = int(hyper_params['pretraining_epochs'])
    pretraining_lr = float(hyper_params['pretraining_lr'])

    sequential_training = hyper_params['sequential_training']
    dropout_rate = hyper_params['dropout_rate']

    buffer_size = int(buffer_size / batch_size) * batch_size

    ###################
    (train_x_file_list, train_y_file_list) = train_xy_file_list
    (valid_x_file_list, valid_y_file_list) = valid_xy_file_list

    logger.debug('Creating training   data provider')
    train_data_reader = ListDataProvider(x_file_list = train_x_file_list, y_file_list = train_y_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, 
                            sequential = sequential_training, shuffle = True)

    logger.debug('Creating validation data provider')
    valid_data_reader = ListDataProvider(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, 
                            sequential = sequential_training, shuffle = False)

    if cfg.rnn_batch_training:
        train_data_reader.set_rnn_params(training_algo=cfg.training_algo, batch_size=cfg.batch_size, seq_length=cfg.seq_length, merge_size=cfg.merge_size, bucket_range=cfg.bucket_range)
        valid_data_reader.reshape_input_output()
    
    shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()
    train_set_x, train_set_y = shared_train_set_xy
    shared_valid_set_xy, temp_valid_set_x, temp_valid_set_y = valid_data_reader.load_one_partition()
    valid_set_x, valid_set_y = shared_valid_set_xy
    train_data_reader.reset()
    valid_data_reader.reset()


    ##temporally we use the training set as pretrain_set_x.
    ##we need to support any data for pretraining

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    logger.info('building the model')


    dnn_model = None
    pretrain_fn = None  ## not all the model support pretraining right now
    train_fn = None
    valid_fn = None
    valid_model = None ## valid_fn and valid_model are the same. reserve to computer multi-stream distortion
    if model_type == 'DNN':
        dnn_model = DeepRecurrentNetwork(n_in= n_ins, hidden_layer_size = hidden_layer_size, n_out = n_outs,
                                         L1_reg = l1_reg, L2_reg = l2_reg, hidden_layer_type = hidden_layer_type, output_type = cfg.output_layer_type,
                                         dropout_rate = dropout_rate, optimizer = cfg.optimizer, rnn_batch_training = cfg.rnn_batch_training)

    else:
        logger.critical('%s type NN model is not supported!' %(model_type))
        raise

    ## Model adaptation -- fine tuning the existing model
    ## We can't just unpickle the old model and use that because fine-tune functions
    ## depend on opt_l2e option used in construction of initial model. One way around this
    ## would be to unpickle, manually set unpickled_dnn_model.opt_l2e=True and then call
    ## unpickled_dnn_model.build_finetne_function() again. This is another way, construct
    ## new model from scratch with opt_l2e=True, then copy existing weights over:
    use_lhuc = cfg.use_lhuc
    if init_dnn_model_file != "_":
        logger.info('load parameters from existing model: %s' %(init_dnn_model_file))
        if not os.path.isfile(init_dnn_model_file):
            sys.exit('Model file %s does not exist'%(init_dnn_model_file))
        existing_dnn_model = pickle.load(open(init_dnn_model_file, 'rb'))
        if not use_lhuc and not len(existing_dnn_model.params) == len(dnn_model.params):
            sys.exit('Old and new models have different numbers of weight matrices')
        elif use_lhuc and len(dnn_model.params) < len(existing_dnn_model.params):
            sys.exit('In LHUC adaptation new model must have more parameters than old model.')
        # assign the existing dnn model parameters to the new dnn model
        k = 0
        for i in range(len(dnn_model.params)):
            ## Added for LHUC ##
            # In LHUC, we keep all the old parameters intact and learn only a small set of new
            # parameters
            if dnn_model.params[i].name == 'c':
                continue
            else:
                old_val = existing_dnn_model.params[k].get_value()
                new_val = dnn_model.params[i].get_value()
                if numpy.shape(old_val) == numpy.shape(new_val):
                    dnn_model.params[i].set_value(old_val)
                else:
                    sys.exit('old and new weight matrices have different shapes')
                k = k + 1        
    train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (train_set_x, train_set_y), (valid_set_x, valid_set_y), use_lhuc)  #, batch_size=batch_size
    logger.info('fine-tuning the %s model' %(model_type))

    start_time = time.time()

    best_dnn_model = dnn_model
    best_validation_loss = sys.float_info.max
    previous_loss = sys.float_info.max

    lr_decay  = cfg.lr_decay
    if lr_decay>0:
        early_stop_epoch *= lr_decay

    early_stop = 0
    val_loss_counter = 0

    previous_finetune_lr = finetune_lr

    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1
        
        if lr_decay==0:
            # fixed learning rate 
            reduce_lr = False
        elif lr_decay<0:
            # exponential decay
            reduce_lr = False if epoch <= warmup_epoch else True
        elif val_loss_counter > 0:
            # linear decay
            reduce_lr = False
            if val_loss_counter%lr_decay==0:
                reduce_lr = True
                val_loss_counter = 0
        else:
            # no decay
            reduce_lr = False

        if reduce_lr:
            current_finetune_lr = previous_finetune_lr * 0.5
            current_momentum    = momentum
        else:
            current_finetune_lr = previous_finetune_lr
            current_momentum    = warmup_momentum
        
        previous_finetune_lr = current_finetune_lr

        train_error = []
        sub_start_time = time.time()

        logger.debug("training params -- learning rate: %f, early_stop: %d/%d" % (current_finetune_lr, early_stop, early_stop_epoch))
        while (not train_data_reader.is_finish()):

            _, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()

            # if sequential training, the batch size will be the number of frames in an utterance
            # batch_size for sequential training is considered only when rnn_batch_training is set to True
            if sequential_training == True:
                batch_size = temp_train_set_x.shape[0]

            n_train_batches = temp_train_set_x.shape[0] // batch_size
            for index in range(n_train_batches):
                ## send a batch to the shared variable, rather than pass the batch size and batch index to the finetune function
                train_set_x.set_value(numpy.asarray(temp_train_set_x[index*batch_size:(index + 1)*batch_size], dtype=theano.config.floatX), borrow=True)
                train_set_y.set_value(numpy.asarray(temp_train_set_y[index*batch_size:(index + 1)*batch_size], dtype=theano.config.floatX), borrow=True)

                this_train_error = train_fn(current_finetune_lr, current_momentum)

                train_error.append(this_train_error)

        train_data_reader.reset()

        logger.debug('calculating validation loss')
        validation_losses = []
        while (not valid_data_reader.is_finish()):
            shared_valid_set_xy, temp_valid_set_x, temp_valid_set_y = valid_data_reader.load_one_partition()
            valid_set_x.set_value(numpy.asarray(temp_valid_set_x, dtype=theano.config.floatX), borrow=True)
            valid_set_y.set_value(numpy.asarray(temp_valid_set_y, dtype=theano.config.floatX), borrow=True)

            this_valid_loss = valid_fn()

            validation_losses.append(this_valid_loss)
        valid_data_reader.reset()

        this_validation_loss = numpy.mean(validation_losses)

        this_train_valid_loss = numpy.mean(numpy.asarray(train_error))

        sub_end_time = time.time()

        loss_difference = this_validation_loss - previous_loss

        logger.info('epoch %i, validation error %f, train error %f  time spent %.2f' %(epoch, this_validation_loss, this_train_valid_loss, (sub_end_time - sub_start_time)))
        if plot:
            plotlogger.add_plot_point('training convergence','validation set',(epoch,this_validation_loss))
            plotlogger.add_plot_point('training convergence','training set',(epoch,this_train_valid_loss))
            plotlogger.save_plot('training convergence',title='Progress of training and validation error',xlabel='epochs',ylabel='error')

        if this_validation_loss < best_validation_loss:
            pickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))

            best_dnn_model = dnn_model
            best_validation_loss = this_validation_loss

        if this_validation_loss >= previous_loss:
            logger.debug('validation loss increased')
            val_loss_counter+=1
            early_stop+=1

        if epoch > 15 and early_stop > early_stop_epoch:
            logger.debug('stopping early')
            break

        if math.isnan(this_validation_loss):
            break

        previous_loss = this_validation_loss

    end_time = time.time()

    logger.info('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

    if plot:
        plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')

    return  best_validation_loss


def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list, reshape_io=False):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    for i in range(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size // n_ins))]
        test_set_x = features.reshape((-1, n_ins))
        n_rows = test_set_x.shape[0]
        
        if reshape_io:
            test_set_x = numpy.reshape(test_set_x, (1, test_set_x.shape[0], n_ins))
            test_set_x = numpy.array(test_set_x, 'float32')

        predicted_parameter = dnn_model.parameter_prediction(test_set_x)
        predicted_parameter = predicted_parameter.reshape(-1, n_outs)
        predicted_parameter = predicted_parameter[0:n_rows]
        
        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

##generate bottleneck layer as features
def dnn_hidden_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list, bottleneck_index):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    for i in range(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size // n_ins))]
        features = features.reshape((-1, n_ins))
        temp_set_x = features.tolist()
        test_set_x = theano.shared(numpy.asarray(temp_set_x, dtype=theano.config.floatX))

        predicted_parameter = dnn_model.generate_hidden_layer(test_set_x, bottleneck_index)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()


def main_function(cfg):
    file_paths = FilePaths(cfg)

    # get a logger for this main function
    logger = logging.getLogger("main")

    # get another logger to handle plotting duties
    plotlogger = logging.getLogger("plotting")

    # later, we might do this via a handler that is created, attached and configured
    # using the standard config mechanism of the logging module
    # but for now we need to do it manually
    plotlogger.set_plot_path(cfg.plot_dir)

    # create plot dir if set to True
    if not os.path.exists(cfg.plot_dir) and cfg.plot:
        os.makedirs(cfg.plot_dir)

    #### parameter setting########
    hidden_layer_size = cfg.hyper_params['hidden_layer_size']

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
    assert cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number == total_file_number, 'check train, valid, test file number'

    data_dir = cfg.data_dir

    inter_data_dir = cfg.inter_data_dir
    nn_cmp_dir       = file_paths.nn_cmp_dir
    nn_cmp_norm_dir   = file_paths.nn_cmp_norm_dir
    model_dir = file_paths.model_dir
    gen_dir   = file_paths.gen_dir

    in_file_list_dict = {}

    for feature_name in list(cfg.in_dir_dict.keys()):
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = file_paths.get_nn_cmp_file_list()
    nn_cmp_norm_file_list    = file_paths.get_nn_cmp_norm_file_list()

    ###normalisation information
    norm_info_file = file_paths.norm_info_file

    ### normalise input full context label
    # currently supporting two different forms of lingustic features
    # later, we should generalise this

    assert cfg.label_style == 'HTS', 'Only HTS-style labels are now supported as input to Merlin'

    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    if cfg.VoiceConversion:
        lab_dim = cfg.cmp_dim
    logger.info('Input label dimension is %d' % lab_dim)
    suffix=str(lab_dim)


    if cfg.process_labels_in_work_dir:
        inter_data_dir = cfg.work_dir

    # the number can be removed
    file_paths.set_label_dir(label_normaliser.dimension, suffix, lab_dim)
    file_paths.set_label_file_list()

    binary_label_dir      = file_paths.binary_label_dir
    nn_label_dir          = file_paths.nn_label_dir
    nn_label_norm_dir     = file_paths.nn_label_norm_dir

    in_label_align_file_list = file_paths.in_label_align_file_list
    binary_label_file_list   = file_paths.binary_label_file_list
    nn_label_file_list       = file_paths.nn_label_file_list
    nn_label_norm_file_list  = file_paths.nn_label_norm_file_list

    min_max_normaliser = None

    label_norm_file = file_paths.label_norm_file

    test_id_list = file_paths.test_id_list

    if cfg.NORMLAB:
        # simple HTS labels
        logger.info('preparing label data (input) using standard HTS style labels')
        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_type=cfg.label_type)

        if cfg.additional_features:
            out_feat_file_list = file_paths.out_feat_file_list
            in_dim = label_normaliser.dimension

            for new_feature, new_feature_dim in cfg.additional_features.items():
                new_feat_dir  = os.path.join(data_dir, new_feature)
                new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.'+new_feature)

                merger = MergeFeat(lab_dim = in_dim, feat_dim = new_feature_dim)
                merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
                in_dim += new_feature_dim

                binary_label_file_list = out_feat_file_list

        remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
        remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)

        ###use only training data to find min-max information, then apply on the whole dataset
        if cfg.GenTestList:
            min_max_normaliser.load_min_max_values(label_norm_file)
        else:
            min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])

        ### enforce silence such that the normalization runs without removing silence: only for final synthesis
        if cfg.GenTestList and cfg.enforce_silence:
            min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
        else:
            min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)



    if min_max_normaliser != None and not cfg.GenTestList:
        ### save label normalisation information for unseen testing labels
        label_min_vector = min_max_normaliser.min_vector
        label_max_vector = min_max_normaliser.max_vector
        label_norm_info = numpy.concatenate((label_min_vector, label_max_vector), axis=0)

        label_norm_info = numpy.array(label_norm_info, 'float32')
        fid = open(label_norm_file, 'wb')
        label_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' %(label_min_vector.size, label_norm_file))

    ### make output duration data
    if cfg.MAKEDUR:
        logger.info('creating duration (output) features')
        label_normaliser.prepare_dur_data(in_label_align_file_list, file_paths.dur_file_list, cfg.label_type, cfg.dur_feature_type)

    ### make output acoustic data
    if cfg.MAKECMP:
        logger.info('creating acoustic (output) features')
        delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
        acc_win = cfg.acc_win     #[1.0, -2.0, 1.0]

        if cfg.GenTestList:
            for feature_name in list(cfg.in_dir_dict.keys()):
                in_file_list_dict[feature_name] = prepare_file_path_list(test_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)
            nn_cmp_file_list      = prepare_file_path_list(test_id_list, nn_cmp_dir, cfg.cmp_ext)
            nn_cmp_norm_file_list = prepare_file_path_list(test_id_list, nn_cmp_norm_dir, cfg.cmp_ext)
        
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)

        if 'dur' in list(cfg.in_dir_dict.keys()) and cfg.AcousticModel:
            lf0_file_list = file_paths.get_lf0_file_list()
            acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)

        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

        if cfg.remove_silence_using_binary_labels:
            ## do this to get lab_dim:
            label_composer = LabelComposer()
            label_composer.load_label_configuration(cfg.label_config_file)
            lab_dim=label_composer.compute_label_dimension()

            silence_feature = 0 ## use first feature in label -- hardcoded for now
            logger.info('Silence removal from CMP using binary label file')

            ## overwrite the untrimmed audio with the trimmed version:
            trim_silence(nn_cmp_file_list, nn_cmp_file_list, cfg.cmp_dim,
                                binary_label_file_list, lab_dim, silence_feature)

        elif cfg.remove_silence_using_hts_labels: 
            ## back off to previous method using HTS labels:
            remover = SilenceRemover(n_cmp = cfg.cmp_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
            remover.remove_silence(nn_cmp_file_list, in_label_align_file_list, nn_cmp_file_list) # save to itself

    ### save acoustic normalisation information for normalising the features back
    var_dir  = file_paths.var_dir
    var_file_dict = file_paths.get_var_dic()

    ### normalise output acoustic data
    if cfg.NORMCMP:
        logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
        cmp_norm_info = None
        if cfg.output_feature_normalisation == 'MVN':
            normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
            if cfg.GenTestList:
                # load mean std values
                global_mean_vector, global_std_vector = normaliser.load_mean_std_values(norm_info_file)
            else:
                ###calculate mean and std vectors on the training data, and apply on the whole dataset
                global_mean_vector = normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
                global_std_vector = normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector, 0, cfg.cmp_dim)
                # for hmpd vocoder we don't need to normalize the 
                # pdd values
                if cfg.vocoder_type == 'hmpd':
                    stream_start_index = {}
                    dimension_index = 0
                    recorded_vuv = False
                    vuv_dimension = None
                    for feature_name in cfg.out_dimension_dict.keys():
                        if feature_name != 'vuv':
                            stream_start_index[feature_name] = dimension_index
                        else:
                            vuv_dimension = dimension_index
                            recorded_vuv = True
                        
                        dimension_index += cfg.out_dimension_dict[feature_name]
                    logger.info('hmpd pdd values are not normalized since they are in 0 to 1')
                    global_mean_vector[:,stream_start_index['pdd']: stream_start_index['pdd'] + cfg.out_dimension_dict['pdd']] = 0
                    global_std_vector[:,stream_start_index['pdd']: stream_start_index['pdd'] + cfg.out_dimension_dict['pdd']] = 1
            normaliser.feature_normalisation(nn_cmp_file_list, nn_cmp_norm_file_list)
            cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

        elif cfg.output_feature_normalisation == 'MINMAX':
            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
            if cfg.GenTestList:
                min_max_normaliser.load_min_max_values(norm_info_file)
            else:
                min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
            min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

            cmp_min_vector = min_max_normaliser.min_vector
            cmp_max_vector = min_max_normaliser.max_vector
            cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

        else:
            logger.critical('Normalisation type %s is not supported!\n' %(cfg.output_feature_normalisation))
            raise

        if not cfg.GenTestList:
            cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
            fid = open(norm_info_file, 'wb')
            cmp_norm_info.tofile(fid)
            fid.close()
            logger.info('saved %s vectors to %s' %(cfg.output_feature_normalisation, norm_info_file))

            feature_index = 0
            for feature_name in list(cfg.out_dimension_dict.keys()):
                feature_std_vector = numpy.array(global_std_vector[:,feature_index:feature_index+cfg.out_dimension_dict[feature_name]], 'float32')

                fid = open(var_file_dict[feature_name], 'w')
                feature_var_vector = feature_std_vector**2
                feature_var_vector.tofile(fid)
                fid.close()

                logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))

                feature_index += cfg.out_dimension_dict[feature_name]

    train_x_file_list, train_y_file_list = file_paths.get_train_list_x_y()
    valid_x_file_list, valid_y_file_list = file_paths.get_valid_list_x_y()
    test_x_file_list, test_y_file_list = file_paths.get_test_list_x_y()

    # we need to know the label dimension before training the DNN
    # computing that requires us to look at the labels
    #
    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    if cfg.VoiceConversion:
        lab_dim = cfg.cmp_dim

    logger.info('label dimension is %d' % lab_dim)

    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)

    nnets_file_name = file_paths.get_nnets_file_name()
    temp_dir_name = file_paths.get_temp_nn_dir_name()

    gen_dir = os.path.join(gen_dir, temp_dir_name)

    if cfg.switch_to_keras or cfg.switch_to_tensorflow:
        ### set configuration variables ###
        cfg.inp_dim = lab_dim
        cfg.out_dim = cfg.cmp_dim

        cfg.inp_feat_dir  = nn_label_norm_dir
        cfg.out_feat_dir  = nn_cmp_norm_dir
        cfg.pred_feat_dir = gen_dir

        if cfg.GenTestList and cfg.test_synth_dir!="None":
            cfg.inp_feat_dir  = cfg.test_synth_dir
            cfg.pred_feat_dir = cfg.test_synth_dir
        
    if cfg.switch_to_keras:
        ### call kerasclass and use an instance ###
        from run_keras_with_merlin_io import KerasClass
        keras_instance = KerasClass(cfg)
    
    elif cfg.switch_to_tensorflow:
        ### call Tensorflowclass and use an instance ###
        from run_tensorflow_with_merlin_io import TensorflowClass
        tf_instance = TensorflowClass(cfg)

    ### DNN model training
    if cfg.TRAINDNN:

        var_dict = load_covariance(var_file_dict, cfg.out_dimension_dict)

        logger.info('training DNN')

        fid = open(norm_info_file, 'rb')
        cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_mean_vector = cmp_min_max[0, ]
        cmp_std_vector  = cmp_min_max[1, ]


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
            if cfg.switch_to_keras:
                keras_instance.train_keras_model()
            elif cfg.switch_to_tensorflow:
                tf_instance.train_tensorflow_model()
            else:
                train_DNN(train_xy_file_list = (train_x_file_list, train_y_file_list), \
                      valid_xy_file_list = (valid_x_file_list, valid_y_file_list), \
                      nnets_file_name = nnets_file_name, \
                      n_ins = lab_dim, n_outs = cfg.cmp_dim, ms_outs = cfg.multistream_outs, \
                      hyper_params = cfg.hyper_params, buffer_size = cfg.buffer_size, plot = cfg.plot, var_dict = var_dict,
                      cmp_mean_vector = cmp_mean_vector, cmp_std_vector = cmp_std_vector,init_dnn_model_file=cfg.start_from_trained_model)
        except KeyboardInterrupt:
            logger.critical('train_DNN interrupted via keyboard')
            # Could 'raise' the exception further, but that causes a deep traceback to be printed
            # which we don't care about for a keyboard interrupt. So, just bail out immediately
            sys.exit(1)
        except:
            logger.critical('train_DNN threw an exception')
            raise



    if cfg.GENBNFEA:
        # Please only tune on this step when you want to generate bottleneck features from DNN
        gen_dir = file_paths.bottleneck_features

        bottleneck_size = min(hidden_layer_size)
        bottleneck_index = 0
        for i in range(len(hidden_layer_size)):
            if hidden_layer_size[i] == bottleneck_size:
                bottleneck_index = i

        logger.info('generating bottleneck features from DNN')

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

        gen_file_id_list = file_id_list[0:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        test_x_file_list = nn_label_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)

        dnn_hidden_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list, bottleneck_index)

    ### generate parameters from DNN
    gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

    if cfg.GenTestList:
        gen_file_id_list = test_id_list
        test_x_file_list = nn_label_norm_file_list
        if cfg.test_synth_dir!="None":
            gen_dir = cfg.test_synth_dir

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


        if cfg.switch_to_keras:
            keras_instance.test_keras_model()
        elif cfg.switch_to_tensorflow:
            tf_instance.test_tensorflow_model()
        else:
            reshape_io = True if cfg.rnn_batch_training else False
            dnn_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list, reshape_io)

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

        if cfg.AcousticModel:
            ##perform MLPG to smooth parameter trajectory
            ## lf0 is included, the output features much have vuv.
            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features, enforce_silence = cfg.enforce_silence)
            generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict, var_file_dict, do_MLPG=cfg.do_MLPG, cfg=cfg)

        if cfg.DurationModel:
            ### Perform duration normalization(min. state dur set to 1) ###
            gen_dur_list   = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.dur_ext)
            gen_label_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.lab_ext)
            in_gen_label_align_file_list = prepare_file_path_list(gen_file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)

            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
            generator.duration_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)

            label_modifier = HTSLabelModification(silence_pattern = cfg.silence_pattern, label_type = cfg.label_type)
            label_modifier.modify_duration_labels(in_gen_label_align_file_list, gen_dur_list, gen_label_list)


    ### generate wav
    if cfg.GENWAV:
        logger.info('reconstructing waveform(s)')
        generate_wav(gen_dir, gen_file_id_list, cfg)     # generated speech
#       generate_wav(nn_cmp_dir, gen_file_id_list, cfg)  # reference copy synthesis speech

    ### setting back to original conditions before calculating objective scores ###
    if cfg.GenTestList:
        in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
        binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
        gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

    ### evaluation: RMSE and CORR for duration
    if cfg.CALMCD and cfg.DurationModel:
        logger.info('calculating MCD')

        ref_data_dir = os.path.join(inter_data_dir, 'ref_data')

        ref_dur_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.dur_ext)

        in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        calculator = IndividualDistortionComp()

        valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
        test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        if cfg.remove_silence_using_binary_labels:
            untrimmed_reference_data = in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
            trim_silence(untrimmed_reference_data, ref_dur_list, cfg.dur_dim, \
                                untrimmed_test_labels, lab_dim, silence_feature)
        else:
            remover = SilenceRemover(n_cmp = cfg.dur_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features)
            remover.remove_silence(in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_dur_list)

        valid_dur_rmse, valid_dur_corr = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)
        test_dur_rmse, test_dur_corr = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)

        logger.info('Develop: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
                    %(valid_dur_rmse, valid_dur_corr))
        logger.info('Test: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
                    %(test_dur_rmse, test_dur_corr))

    ### evaluation: calculate distortion
    if cfg.CALMCD and cfg.AcousticModel:
        logger.info('calculating MCD')

        ref_data_dir = os.path.join(inter_data_dir, 'ref_data')
        ref_lf0_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.lf0_ext)
        # for straight or world vocoders
        ref_mgc_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.mgc_ext)
        ref_bap_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.bap_ext)
        # for GlottDNN vocoder
        ref_lsf_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.lsf_ext)
        ref_slsf_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.slsf_ext)
        ref_gain_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.gain_ext)
        ref_hnr_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.hnr_ext)
        # for pulsemodel vocoder
        ref_pdd_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.pdd_ext)

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


        if 'mgc' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_mgc_list, cfg.mgc_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.mgc_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
            else:
                ref_data_dir = os.path.join(data_dir, 'mgc')
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD


        if 'bap' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_bap_list, cfg.bap_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.bap_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
            else:
                ref_data_dir = os.path.join(data_dir, 'bap')
            valid_bap_mse = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            test_bap_mse  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            valid_bap_mse = valid_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
            test_bap_mse  = test_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC

        if 'lf0' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_lf0_list, cfg.lf0_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.lf0_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lf0_list)
            else:
                ref_data_dir = os.path.join(data_dir, 'lf0')
            valid_f0_mse, valid_f0_corr, valid_vuv_error   = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
            test_f0_mse , test_f0_corr, test_vuv_error    = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
        
        if 'lsf' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['lsf'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_lsf_list, cfg.lsf_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.lsf_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['lsf'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lsf_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.lsf_ext, cfg.lsf_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.lsf_ext, cfg.lsf_dim)
        
        if 'slsf' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['slsf'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_slsf_list, cfg.slsf_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.slsf_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['slsf'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_slsf_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.slsf_ext, cfg.slsf_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.slsf_ext, cfg.slsf_dim)
        
        if 'hnr' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['hnr'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_hnr_list, cfg.hnr_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.hnr_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['hnr'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_hnr_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.hnr_ext, cfg.hnr_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.hnr_ext, cfg.hnr_dim)
        
        if 'gain' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['gain'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_gain_list, cfg.gain_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.gain_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['gain'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_gain_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.gain_ext, cfg.gain_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.gain_ext, cfg.gain_dim)
        
        if 'pdd' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['pdd'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_pdd_list, cfg.pdd_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.pdd_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['pdd'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_pdd_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.pdd_ext, cfg.pdd_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.pdd_ext, cfg.pdd_dim)
        

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
        logger.critical('usage: run_merlin.sh [config file name]')
        sys.exit(1)

    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)


    logger.info('Installation information:')
    logger.info('  Merlin directory: '+os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
    logger.info('  PATH:')
    env_PATHs = os.getenv('PATH')
    if env_PATHs:
        env_PATHs = env_PATHs.split(':')
        for p in env_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  LD_LIBRARY_PATH:')
    env_LD_LIBRARY_PATHs = os.getenv('LD_LIBRARY_PATH')
    if env_LD_LIBRARY_PATHs:
        env_LD_LIBRARY_PATHs = env_LD_LIBRARY_PATHs.split(':')
        for p in env_LD_LIBRARY_PATHs:
            if len(p)>0: logger.info('      '+p)
    logger.info('  Python version: '+sys.version.replace('\n',''))
    logger.info('    PYTHONPATH:')
    env_PYTHONPATHs = os.getenv('PYTHONPATH')
    if env_PYTHONPATHs:
        env_PYTHONPATHs = env_PYTHONPATHs.split(':')
        for p in env_PYTHONPATHs:
            if len(p)>0:
                logger.info('      '+p)
    logger.info('  Numpy version: '+numpy.version.version)
    logger.info('  Theano version: '+theano.version.version)
    logger.info('    THEANO_FLAGS: '+os.getenv('THEANO_FLAGS'))
    logger.info('    device: '+theano.config.device)

    # Check for the presence of git
    ret = os.system('git status > /dev/null')
    if ret==0:
        logger.info('  Git is available in the working directory:')
        git_describe = subprocess.Popen(['git', 'describe', '--tags', '--always'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    Merlin version: {}'.format(git_describe))
        git_branch = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE).communicate()[0][:-1]
        logger.info('    branch: {}'.format(git_branch))
        git_diff = subprocess.Popen(['git', 'diff', '--name-status'], stdout=subprocess.PIPE).communicate()[0]
        if sys.version_info.major >= 3:
            git_diff = git_diff.decode('utf-8')
        git_diff = git_diff.replace('\t',' ').split('\n')
        logger.info('    diff to Merlin version:')
        for filediff in git_diff:
            if len(filediff)>0: logger.info('      '+filediff)
        logger.info('      (all diffs logged in '+os.path.basename(cfg.log_file)+'.gitdiff'+')')
        os.system('git diff > '+cfg.log_file+'.gitdiff')

    logger.info('Execution information:')
    logger.info('  HOSTNAME: '+socket.getfqdn())
    logger.info('  USER: '+os.getenv('USER'))
    logger.info('  PID: '+str(os.getpid()))
    PBS_JOBID = os.getenv('PBS_JOBID')
    if PBS_JOBID:
        logger.info('  PBS_JOBID: '+PBS_JOBID)


    if cfg.profile:
        logger.info('profiling is activated')
        import cProfile, pstats
        cProfile.run('main_function(cfg)', 'mainstats')

        # create a stream for the profiler to write to
        profiling_output = io.StringIO()
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
