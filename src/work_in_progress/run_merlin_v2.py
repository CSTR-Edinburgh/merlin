
import pickle
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

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
from frontend.label_modifier import HTSLabelModification
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
import io


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

    layer_num = len(dnn.params)     ## including input and output
    plotlogger = logging.getLogger("plotting")

    for i in range(layer_num):
        fig_name = 'Activation weights W' + str(i) + '_' + dnn.params[i].name
        fig_title = 'Activation weights of W' + str(i)
        xlabel = 'Neuron index of hidden layer ' + str(i)
        ylabel = 'Neuron index of hidden layer ' + str(i+1)
        if i == 0:
            xlabel = 'Input feature index'
        if i == layer_num-1:
            ylabel = 'Output feature index'

        aa = dnn.params[i].get_value(borrow=True).T
        print(aa.shape, aa.size)
        if aa.size > aa.shape[0]:
            logger.create_plot(fig_name, SingleWeightMatrixPlot)
            plotlogger.add_plot_point(fig_name, fig_name, dnn.params[i].get_value(borrow=True).T)
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
              cmp_mean_vector = None, cmp_std_vector = None, init_dnn_model_file = None, seq_dur_file_list = None):

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

#    sequential_training = True

    buffer_size = int(buffer_size / batch_size) * batch_size

    ###################
    (train_x_file_list, train_y_file_list) = train_xy_file_list
    (valid_x_file_list, valid_y_file_list) = valid_xy_file_list

    if cfg.network_type != 'S2S':
        seq_dur_file_list = None

    if not seq_dur_file_list:
        train_dur_file_list = None
        valid_dur_file_list = None
    else:
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, subphone_feats="coarse_coding")
        train_dur_file_list = seq_dur_file_list[0:cfg.train_file_number]
        valid_dur_file_list = seq_dur_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]

    logger.debug('Creating training   data provider')
    train_data_reader = ListDataProvider(x_file_list = train_x_file_list, y_file_list = train_y_file_list, dur_file_list = train_dur_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, sequential = sequential_training, network_type=cfg.network_type, shuffle = True)

    logger.debug('Creating validation data provider')
    valid_data_reader = ListDataProvider(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list, dur_file_list = valid_dur_file_list,
                            n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, sequential = sequential_training, network_type=cfg.network_type, shuffle = False)


    if cfg.network_type == 'S2S':
        shared_train_set_xyd, temp_train_set_x, temp_train_set_y, temp_train_set_d = train_data_reader.load_one_partition()
        shared_valid_set_xyd, temp_valid_set_x, temp_valid_set_y, temp_valid_set_d = valid_data_reader.load_one_partition()
        train_set_x, train_set_y, train_set_d = shared_train_set_xyd
        valid_set_x, valid_set_y, valid_set_d = shared_valid_set_xyd

        temp_train_set_f = label_normaliser.extract_durational_features(dur_data=temp_train_set_d)
        temp_valid_set_f = label_normaliser.extract_durational_features(dur_data=temp_valid_set_d)
        train_set_f = theano.shared(numpy.asarray(temp_train_set_f, dtype=theano.config.floatX), name='f', borrow=True)
        valid_set_f = theano.shared(numpy.asarray(temp_valid_set_f, dtype=theano.config.floatX), name='f', borrow=True)
    else:
        shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()
        shared_valid_set_xy, temp_valid_set_x, temp_valid_set_y = valid_data_reader.load_one_partition()
        train_set_x, train_set_y = shared_train_set_xy
        valid_set_x, valid_set_y = shared_valid_set_xy
    train_data_reader.reset()
    valid_data_reader.reset()


    ##temporally we use the training set as pretrain_set_x.
    ##we need to support any data for pretraining
#    pretrain_set_x = train_set_x

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
                                         L1_reg = l1_reg, L2_reg = l2_reg, hidden_layer_type = hidden_layer_type, output_type=cfg.output_layer_type, network_type=cfg.network_type, dropout_rate = dropout_rate)
        if cfg.network_type == 'S2S':
            train_fn, valid_fn = dnn_model.build_finetune_functions_S2SPF(
                        (train_set_x, train_set_y, train_set_d, train_set_f), (valid_set_x, valid_set_y, valid_set_d, valid_set_f))
        else:
            train_fn, valid_fn = dnn_model.build_finetune_functions(
                    (train_set_x, train_set_y), (valid_set_x, valid_set_y))  #, batch_size=batch_size

    else:
        logger.critical('%s type NN model is not supported!' %(model_type))
        raise

    logger.info('fine-tuning the %s model' %(model_type))

    start_time = time.time()

    best_dnn_model = dnn_model
    best_validation_loss = sys.float_info.max
    previous_loss = sys.float_info.max

    early_stop = 0
    epoch = 0

#    finetune_lr = 0.000125
    previous_finetune_lr = finetune_lr

    print(finetune_lr)

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
        sub_start_time = time.time()

        while (not train_data_reader.is_finish()):

            if cfg.network_type == 'S2S':
                shared_train_set_xyd, temp_train_set_x, temp_train_set_y, temp_train_set_d = train_data_reader.load_one_partition()
                temp_train_set_f = label_normaliser.extract_durational_features(dur_data=temp_train_set_d)
                train_set_d.set_value(numpy.asarray(temp_train_set_d, dtype='int32'), borrow=True)
                train_set_f.set_value(numpy.asarray(temp_train_set_f, dtype=theano.config.floatX), borrow=True)
            else:
                shared_train_set_xy, temp_train_set_x, temp_train_set_y = train_data_reader.load_one_partition()

            # if sequential training, the batch size will be the number of frames in an utterance

            if sequential_training == True:
                #batch_size = temp_train_set_x.shape[0]

                train_set_x.set_value(numpy.asarray(temp_train_set_x, dtype=theano.config.floatX), borrow=True)
                train_set_y.set_value(numpy.asarray(temp_train_set_y, dtype=theano.config.floatX), borrow=True)

                this_train_error = train_fn(current_finetune_lr, current_momentum)
                train_error.append(this_train_error)
                #print train_set_x.eval().shape, train_set_y.eval().shape, this_train_error

            else:
                n_train_batches = temp_train_set_x.shape[0] / batch_size
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

            if cfg.network_type == 'S2S':
                shared_valid_set_xyd, temp_valid_set_x, temp_valid_set_y, temp_valid_set_d = valid_data_reader.load_one_partition()
                temp_valid_set_f = label_normaliser.extract_durational_features(dur_data=temp_valid_set_d)
                valid_set_d.set_value(numpy.asarray(temp_valid_set_d, dtype='int32'), borrow=True)
                valid_set_f.set_value(numpy.asarray(temp_valid_set_f, dtype=theano.config.floatX), borrow=True)
            else:
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
            if epoch > 5:
                pickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))

            best_dnn_model = dnn_model
            best_validation_loss = this_validation_loss
#            logger.debug('validation loss decreased, so saving model')

        if this_validation_loss >= previous_loss:
            logger.debug('validation loss increased')

#            dbn = best_dnn_model
            early_stop += 1

        if epoch > 15 and early_stop > early_stop_epoch:
            logger.debug('stopping early')
            break

        if math.isnan(this_validation_loss):
            break

        previous_loss = this_validation_loss

    end_time = time.time()
#    cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))

    logger.info('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

    if plot:
        plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')

    return  best_validation_loss


def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
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
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))

        predicted_parameter = dnn_model.parameter_prediction(test_set_x)

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

def dnn_generation_S2S(valid_file_list, valid_dur_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, subphone_feats="coarse_coding")
    for i in range(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))

        fid_lab = open(valid_dur_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        test_set_d = features.astype(numpy.int32)

        dur_features = label_normaliser.extract_durational_features(dur_data=test_set_d)
        test_set_f = dur_features.astype(numpy.float32)

        predicted_parameter = dnn_model.parameter_prediction_S2SPF(test_set_x, test_set_d, test_set_f)

        #print b_indices

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

def dnn_generation_lstm(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    plotlogger = logging.getLogger("plotting")

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))

    visualize_dnn(dnn_model)

    file_number = len(valid_file_list)

    for i in range(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        test_set_x = features.reshape((-1, n_ins))

        predicted_parameter = dnn_model.parameter_prediction_lstm(test_set_x)

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

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))

    file_number = len(valid_file_list)

    for i in range(file_number):
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

    data_dir = cfg.data_dir

    nn_cmp_dir       = os.path.join(data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    nn_cmp_norm_dir   = os.path.join(data_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))

    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir   = os.path.join(cfg.work_dir, 'gen')

    in_file_list_dict = {}

    for feature_name in list(cfg.in_dir_dict.keys()):
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
    nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

    ###normalisation information
    norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')

    ### normalise input full context label
    # currently supporting two different forms of lingustic features
    # later, we should generalise this

    if cfg.label_style == 'HTS':
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
        lab_dim = label_normaliser.dimension + cfg.appended_input_dim
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

    in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
    binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
    nn_label_file_list       = prepare_file_path_list(file_id_list, nn_label_dir, cfg.lab_ext)
    nn_label_norm_file_list  = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)
    dur_file_list            = prepare_file_path_list(file_id_list, cfg.in_dur_dir, cfg.dur_ext)
    seq_dur_file_list = prepare_file_path_list(file_id_list, cfg.in_seq_dur_dir, cfg.dur_ext)
    lf0_file_list            = prepare_file_path_list(file_id_list, cfg.in_lf0_dir, cfg.lf0_ext)

    # to do - sanity check the label dimension here?



    min_max_normaliser = None
    label_norm_file = 'label_norm_%s_%d.dat' %(cfg.label_style, lab_dim)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)

    if cfg.GenTestList:
        try:
            test_id_list = read_file_list(cfg.test_id_scp)
            logger.debug('Loaded file id list from %s' % cfg.test_id_scp)
        except IOError:
            # this means that open(...) threw an error
            logger.critical('Could not load file id list from %s' % cfg.test_id_scp)
            raise

        in_label_align_file_list = prepare_file_path_list(test_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
        binary_label_file_list   = prepare_file_path_list(test_id_list, binary_label_dir, cfg.lab_ext)
        nn_label_file_list       = prepare_file_path_list(test_id_list, nn_label_dir, cfg.lab_ext)
        nn_label_norm_file_list  = prepare_file_path_list(test_id_list, nn_label_norm_dir, cfg.lab_ext)

    if cfg.NORMLAB and (cfg.label_style == 'HTS'):
        # simple HTS labels
        logger.info('preparing label data (input) using standard HTS style labels')
        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_type=cfg.label_type)

        remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
        remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
        ###use only training data to find min-max information, then apply on the whole dataset
        if cfg.GenTestList:
            min_max_normaliser.load_min_max_values(label_norm_file)
        else:
            min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

        ### make duration data for S2S network ###
        if cfg.network_type == "S2S":
            logger.info('creating duration (input) features for S2S network')
            label_normaliser.prepare_dur_data(in_label_align_file_list, seq_dur_file_list, feature_type="numerical", unit_size="phoneme")

            if cfg.remove_silence_from_dur:
                remover = SilenceRemover(n_cmp = cfg.seq_dur_dim, silence_pattern = cfg.silence_pattern, remove_frame_features = cfg.add_frame_features)
                remover.remove_silence(seq_dur_file_list, in_label_align_file_list, seq_dur_file_list)

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
        for label_style, label_style_required in label_composer.label_styles.items():
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

            for i in range(num_files):
                logger.info('making input label features for %4d of %4d' % (i+1,num_files))

                # iterate through the required label styles and open each corresponding label file

                # a dictionary of file descriptors, pointing at the required files
                required_labels={}

                for label_style, label_style_required in label_composer.label_styles.items():

                    # the files will be a parallel set of files for a single utterance
                    # e.g., the XML tree and an HTS label file
                    if label_style_required:
                        required_labels[label_style] = open(in_label_align_file_list[label_style][i] , 'r')
                        logger.debug(' opening label file %s' % in_label_align_file_list[label_style][i])

                logger.debug('label styles with open files: %s' % required_labels)
                label_composer.make_labels(required_labels,out_file_name=binary_label_file_list[i],fill_missing_values=cfg.fill_missing_values,iterate_over_frames=cfg.iterate_over_frames)

                # now close all opened files
                for fd in required_labels.values():
                    fd.close()


        # silence removal
        if cfg.remove_silence_using_binary_labels:
            silence_feature = 0 ## use first feature in label -- hardcoded for now
            logger.info('Silence removal from label using silence feature: %s'%(label_composer.configuration.labels[silence_feature]))
            logger.info('Silence will be removed from CMP files in same way')
            ## Binary labels have 2 roles: both the thing trimmed and the instructions for trimming:
            trim_silence(binary_label_file_list, nn_label_file_list, lab_dim, \
                                binary_label_file_list, lab_dim, silence_feature)
        else:
            logger.info('No silence removal done')
            # start from the labels we have just produced, not trimmed versions
            nn_label_file_list = binary_label_file_list

        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
        ###use only training data to find min-max information, then apply on the whole dataset
        min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
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
        feature_type = cfg.dur_feature_type
        label_normaliser.prepare_dur_data(in_label_align_file_list, dur_file_list, feature_type)


    ### make output acoustic data
    if cfg.MAKECMP:
        logger.info('creating acoustic (output) features')
        delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
        acc_win = cfg.acc_win     #[1.0, -2.0, 1.0]

        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
        if 'dur' in list(cfg.in_dir_dict.keys()) and cfg.AcousticModel:
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

        else: ## back off to previous method using HTS labels:
            remover = SilenceRemover(n_cmp = cfg.cmp_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
            remover.remove_silence(nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number],
                                   in_label_align_file_list[0:cfg.train_file_number+cfg.valid_file_number],
                                   nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number]) # save to itself

    ### save acoustic normalisation information for normalising the features back
    var_dir   = os.path.join(data_dir, 'var')
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    var_file_dict = {}
    for feature_name in list(cfg.out_dimension_dict.keys()):
        var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))

    ### normalise output acoustic data
    if cfg.NORMCMP:
        logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
        cmp_norm_info = None
        if cfg.output_feature_normalisation == 'MVN':
            normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
            ###calculate mean and std vectors on the training data, and apply on the whole dataset
            global_mean_vector = normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
            global_std_vector = normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector, 0, cfg.cmp_dim)

            normaliser.feature_normalisation(nn_cmp_file_list[0:cfg.train_file_number+cfg.valid_file_number],
                                             nn_cmp_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number])
            cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

        elif cfg.output_feature_normalisation == 'MINMAX':
            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim)
            global_mean_vector = min_max_normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number])
            global_std_vector = min_max_normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector)

            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
            min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
            min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

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

        feature_index = 0
        for feature_name in list(cfg.out_dimension_dict.keys()):
            feature_std_vector = numpy.array(global_std_vector[:,feature_index:feature_index+cfg.out_dimension_dict[feature_name]], 'float32')

            fid = open(var_file_dict[feature_name], 'w')
            feature_var_vector = feature_std_vector**2
            feature_var_vector.tofile(fid)
            fid.close()

            logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))

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
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
        lab_dim = label_normaliser.dimension + cfg.appended_input_dim

    elif cfg.label_style == 'composed':
        label_composer = LabelComposer()
        label_composer.load_label_configuration(cfg.label_config_file)
        lab_dim=label_composer.compute_label_dimension()

    logger.info('label dimension is %d' % lab_dim)

    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)

    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.%f.rnn.model' \
                      %(model_dir, cfg.combined_model_name, cfg.combined_feature_name, int(cfg.multistream_switch),
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number, cfg.hyper_params['learning_rate'])


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
            train_DNN(train_xy_file_list = (train_x_file_list, train_y_file_list), \
                      valid_xy_file_list = (valid_x_file_list, valid_y_file_list), \
                      nnets_file_name = nnets_file_name, \
                      n_ins = lab_dim, n_outs = cfg.cmp_dim, ms_outs = cfg.multistream_outs, \
                      hyper_params = cfg.hyper_params, buffer_size = cfg.buffer_size, plot = cfg.plot, var_dict = var_dict,
                      cmp_mean_vector = cmp_mean_vector, cmp_std_vector = cmp_std_vector, seq_dur_file_list=seq_dur_file_list)
        except KeyboardInterrupt:
            logger.critical('train_DNN interrupted via keyboard')
            # Could 'raise' the exception further, but that causes a deep traceback to be printed
            # which we don't care about for a keyboard interrupt. So, just bail out immediately
            sys.exit(1)
        except:
            logger.critical('train_DNN threw an exception')
            raise



    if cfg.GENBNFEA:
        '''
        Please only tune on this step when you want to generate bottleneck features from DNN
        '''
        temp_dir_name = '%s_%s_%d_%d_%d_%d_%s_hidden' \
                        %(cfg.model_type, cfg.combined_feature_name, \
                          cfg.train_file_number, lab_dim, cfg.cmp_dim, \
                          len(hidden_layers_sizes), combined_model_arch)
        gen_dir = os.path.join(gen_dir, temp_dir_name)

        bottleneck_size = min(hidden_layers_sizes)
        bottleneck_index = 0
        for i in range(len(hidden_layers_sizes)):
            if hidden_layers_sizes(i) == bottleneck_size:
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
        test_x_file_list  = nn_label_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        test_d_file_list  = seq_dur_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)

        dnn_hidden_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list, bottleneck_index)

    ### generate parameters from DNN
    temp_dir_name = '%s_%s_%d_%d_%d_%d_%d_%d_%d' \
                    %(cfg.combined_model_name, cfg.combined_feature_name, int(cfg.do_post_filtering), \
                      cfg.train_file_number, lab_dim, cfg.cmp_dim, \
                      len(hidden_layer_size), hidden_layer_size[0], hidden_layer_size[-1])
    gen_dir = os.path.join(gen_dir, temp_dir_name)

    gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    test_x_file_list = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    test_d_file_list = seq_dur_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

    if cfg.GenTestList:
        gen_file_id_list = test_id_list
        test_x_file_list = nn_label_norm_file_list
        test_d_file_list = seq_dur_file_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

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
        if cfg.network_type == "S2S":
            dnn_generation_S2S(test_x_file_list, test_d_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)
        else:
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

        if cfg.AcousticModel:
            ##perform MLPG to smooth parameter trajectory
            ## lf0 is included, the output features much have vuv.
            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
            generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict, var_file_dict, do_MLPG=cfg.do_MLPG)

        if cfg.DurationModel:
            ### Perform duration normalization(min. state dur set to 1) ###
            gen_dur_list   = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.dur_ext)
            gen_label_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.lab_ext)
            in_gen_label_align_file_list = prepare_file_path_list(gen_file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)

            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
            generator.duration_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)

            label_modifier = HTSLabelModification(silence_pattern = cfg.silence_pattern)
            label_modifier.modify_duration_labels(in_gen_label_align_file_list, gen_dur_list, gen_label_list)


    ### generate wav
    if cfg.GENWAV:
        logger.info('reconstructing waveform(s)')
        print(len(gen_file_id_list))
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

        ref_data_dir = os.path.join(data_dir, 'ref_data')

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


        if 'mgc' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_mgc_list, cfg.mgc_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.mgc_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD


        if 'bap' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_bap_list, cfg.bap_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.bap_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
            valid_bap_mse        = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            test_bap_mse         = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            valid_bap_mse = valid_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
            test_bap_mse  = test_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC

        if 'lf0' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_lf0_list, cfg.lf0_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            else:
                remover = SilenceRemover(n_cmp = cfg.lf0_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
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

#    if gnp._boardId is not None:
#        import gpu_lock
#        gpu_lock.free_lock(gnp._boardId)

    sys.exit(0)
