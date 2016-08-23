
import cPickle
import gzip
import os, sys, errno
import time
import math
import glob
import struct

import copy

from lxml import etree

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano

from utils.providers import ListDataProviderWithProjectionIndex, expand_projection_inputs, get_unexpanded_projection_inputs # ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
#from frontend.acoustic_normalisation import CMPNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
#from frontend.feature_normalisation_base import FeatureNormBase
from frontend.mean_variance_norm import MeanVarianceNorm

from io_funcs.binary_io import  BinaryIOCollection

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer

import configuration

from models.dnn import DNN
from models.tpdnn import TokenProjectionDNN
from models.ms_dnn import MultiStreamDNN
from models.ms_dnn_gv import MultiStreamDNNGv
from models.sdae import StackedDenoiseAutoEncoder

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




## This should always be True -- tidy up later
expand_by_minibatch = True

if expand_by_minibatch:
    proj_type = 'int32'
else:
    proj_type = theano.config.floatX




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





def infer_projections(train_xy_file_list, valid_xy_file_list, \
              nnets_file_name, n_ins, n_outs, ms_outs, hyper_params, buffer_size, plot=False):

    '''
    Unlike the same function in run_tpdnn.py this *DOESN'T* save model at the
    end -- just returns array of the learned projection weights
    '''

    ####parameters#####
    finetune_lr     = float(hyper_params['learning_rate'])
    training_epochs = int(hyper_params['training_epochs'])
    batch_size      = int(hyper_params['batch_size'])
    l1_reg          = float(hyper_params['l1_reg'])
    l2_reg          = float(hyper_params['l2_reg'])
    private_l2_reg  = float(hyper_params['private_l2_reg'])
    warmup_epoch    = int(hyper_params['warmup_epoch'])
    momentum        = float(hyper_params['momentum'])
    warmup_momentum = float(hyper_params['warmup_momentum'])
    
    hidden_layers_sizes = hyper_params['hidden_layers_sizes']    

    stream_weights       = hyper_params['stream_weights']
    private_hidden_sizes = hyper_params['private_hidden_sizes']

    buffer_utt_size = buffer_size
    early_stop_epoch = int(hyper_params['early_stop_epochs'])

    hidden_activation = hyper_params['hidden_activation']
    output_activation = hyper_params['output_activation']
    
    stream_lr_weights = hyper_params['stream_lr_weights']
    use_private_hidden = hyper_params['use_private_hidden']

    model_type = hyper_params['model_type']
    
    index_to_project = hyper_params['index_to_project']
    projection_insize = hyper_params['projection_insize']
    projection_outsize = hyper_params['projection_outsize']    

    ######### data providers ##########
    (train_x_file_list, train_y_file_list) = train_xy_file_list
    (valid_x_file_list, valid_y_file_list) = valid_xy_file_list

    logger.debug('Creating training   data provider')
    train_data_reader = ListDataProviderWithProjectionIndex(x_file_list = train_x_file_list, y_file_list = train_y_file_list, n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = True, index_to_project=index_to_project, projection_insize=projection_insize, indexes_only=expand_by_minibatch)

    logger.debug('Creating validation data provider')
    valid_data_reader = ListDataProviderWithProjectionIndex(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list, n_ins = n_ins, n_outs = n_outs, buffer_size = buffer_size, shuffle = False, index_to_project=index_to_project, projection_insize=projection_insize, indexes_only=expand_by_minibatch)

    shared_train_set_xy, temp_train_set_x, temp_train_set_x_proj, temp_train_set_y = train_data_reader.load_next_partition_with_projection()
    train_set_x, train_set_x_proj, train_set_y = shared_train_set_xy
    shared_valid_set_xy, temp_valid_set_x, temp_valid_set_x_proj, temp_valid_set_y = valid_data_reader.load_next_partition_with_projection()
    valid_set_x, valid_set_x_proj, valid_set_y = shared_valid_set_xy
    train_data_reader.reset()
    valid_data_reader.reset()
    ####################################


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    logger.info('building the model')


    ############## load existing dnn #####
    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
    train_all_fn, train_subword_fn, train_word_fn, infer_projections_fn, valid_fn, valid_score_i = \
                    dnn_model.build_finetune_functions(
                    (train_set_x, train_set_x_proj, train_set_y), 
                    (valid_set_x, valid_set_x_proj, valid_set_y), batch_size=batch_size)
    ####################################

    logger.info('fine-tuning the %s model' %(model_type))

    start_time = time.clock()

    best_dnn_model = dnn_model
    best_validation_loss = sys.float_info.max
    previous_loss = sys.float_info.max

    early_stop = 0
    epoch = 0
    previous_finetune_lr = finetune_lr

    logger.info('fine-tuning the %s model' %(model_type))


    
    dnn_model.initialise_projection_weights()
    
    inference_epochs = 20  ## <-------- hard coded !!!!!!!!!!
    
    current_finetune_lr = previous_finetune_lr = finetune_lr
    warmup_epoch_3 = 10 # 10  ## <-------- hard coded !!!!!!!!!!
    
    #warmup_epoch_3 = epoch + warmup_epoch_3
    #inference_epochs += epoch
    while (epoch < inference_epochs):
    
        epoch = epoch + 1
        
        current_momentum = momentum
        
        if epoch > warmup_epoch_3:
            previous_finetune_lr = current_finetune_lr
            current_finetune_lr = previous_finetune_lr * 0.5
            

        
        dev_error = []
        sub_start_time = time.clock()

        ## osw -- inferring word reps on validation set in a forward pass in a single batch 
        ##        exausts memory when using 20k projected vocab -- also use minibatches
        logger.debug('infer word representations for validation set')
        valid_error = []
        n_valid_batches = valid_set_x.get_value().shape[0] / batch_size
        for minibatch_index in xrange(n_valid_batches):
            v_loss = infer_projections_fn(minibatch_index, current_finetune_lr, current_momentum)
            valid_error.append(v_loss)

        this_validation_loss = numpy.mean(valid_error)


        #valid_error = infer_projections_fn(current_finetune_lr, current_momentum)
        #this_validation_loss = numpy.mean(valid_error)

#        if plot:
#            ## add dummy validation loss so that plot works:
#            plotlogger.add_plot_point('training convergence','validation set',(epoch,this_validation_loss))
#            plotlogger.add_plot_point('training convergence','training set',(epoch,this_train_valid_loss))
#            
            
        sub_end_time = time.clock()

        
        logger.info('INFERENCE epoch %i, validation error %f, time spent %.2f' %(epoch, this_validation_loss, (sub_end_time - sub_start_time)))


#        if cfg.hyper_params['model_type'] == 'TPDNN':
#            if not os.path.isdir(cfg.projection_weights_output_dir):
#                os.mkdir(cfg.projection_weights_output_dir)
#            weights = dnn_model.get_projection_weights()
#            fname = os.path.join(cfg.projection_weights_output_dir, 'proj_INFERENCE_epoch_%s'%(epoch))
#            numpy.savetxt(fname, weights)
#           
        
        best_dnn_model = dnn_model  ## always update
                        
    end_time = time.clock()
    ##cPickle.dump(best_dnn_model, open(nnets_file_name, 'wb'))
    final_weights = dnn_model.get_projection_weights()


    logger.info('overall  training time: %.2fm validation error %f' % ((end_time - start_time) / 60., best_validation_loss))

#    if plot:
#        plotlogger.save_plot('training convergence',title='Final training and validation error',xlabel='epochs',ylabel='error')
#        
    
    ### ========================================================



    
#    if cfg.hyper_params['model_type'] == 'TPDNN':
#        os.system('python %s %s'%('/afs/inf.ed.ac.uk/user/o/owatts/scripts_NEW/plot_weights_multiple_phases.py', cfg.projection_weights_output_dir))

    return  final_weights
    

def dnn_generation_PROJECTION(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list, cfg=None, synth_mode='constant', projection_end=0, projection_weights_to_use=None, save_weights_to_file=None):
    '''
    Use the (training/dev/test) projections learned in training, but shuffled, for test tokens. 

    -- projection_end is *real* value for last projection index (or some lower value)
    -- this is so the samples / means are of real values learned on training data
    '''

    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation_PROJECTION')

    plotlogger = logging.getLogger("plotting")

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
    
    ## 'remove' word representations by randomising them. As model is unpickled and
    ## not re-saved, this does not throw trained parameters away.
    
    if synth_mode == 'sampled_training':
        ## use randomly chosen training projection -- shuffle in-place = same as sampling wihtout replacement
        P = dnn_model.get_projection_weights()
        numpy.random.shuffle(P[:,:projection_end])  ## shuffle in place along 1st dim (reorder rows)
        dnn_model.params[0].set_value(P, borrow=True)
    elif synth_mode == 'uniform':
        ##  generate utt embeddings uniformly at random within the min-max of the training set (i.e. from a (hyper)-rectangle) 
        P = dnn_model.get_projection_weights()

        column_min = numpy.min(P[:,:projection_end], axis=0)  ## vector like a row of P with min of its columns
        column_max = numpy.max(P[:,:projection_end], axis=0)

        random_proj = numpy.random.uniform(low=column_min, high=column_max, size=numpy.shape(P))
        random_proj = random_proj.astype(numpy.float32)

        dnn_model.params[0].set_value(random_proj, borrow=True)

    elif synth_mode == 'constant':
        ## use mean projection
        P = dnn_model.get_projection_weights()
        mean_row = P[:,:projection_end].mean(axis=0)
        print 'mean row used for projection:'
        print mean_row
        P = numpy.ones(numpy.shape(P), dtype=numpy.float32) * mean_row   ## stack mean rows
        dnn_model.params[0].set_value(P, borrow=True)
    elif synth_mode == 'inferred':
        ## DEBUG
        assert projection_weights_to_use != None
        old_weights = dnn_model.get_projection_weights()
        ## DEBUG:=========
        #projection_weights_to_use = old_weights # numpy.array(numpy.random.uniform(low=-0.3, high=0.3, size=numpy.shape(old_weights)),  dtype=numpy.float32)
        ## =============
        assert numpy.shape(old_weights) == numpy.shape(projection_weights_to_use),  [numpy.shape(old_weights), numpy.shape(projection_weights_to_use)]
        dnn_model.params[0].set_value(projection_weights_to_use, borrow=True)

    elif synth_mode == 'single_sentence_demo':
        ##  generate utt embeddings from a uniform 10 x 10 grid within the min-max of the training set (i.e. from a rectangle) 
        P = dnn_model.get_projection_weights()

        column_min = numpy.min(P[:,:projection_end], axis=0)  ## vector like a row of P with min of its columns
        column_max = numpy.max(P[:,:projection_end], axis=0)
        assert len(column_min) == 2, 'Only 2D projections supported in mode single_sentence_demo'

        ranges = column_max - column_min
        nstep = 10
        steps = ranges / (nstep-1)
        
        grid_params = [numpy.array([1.0, 1.0])]  ## pading to handle 0 index (reserved for defaults)
        for x in range(nstep):
            for y in range(nstep):               
                grid_params.append( column_min + (numpy.array([x, y]) * steps) )
        stacked_params = numpy.vstack(grid_params)
        print stacked_params
        print numpy.shape(stacked_params)
        print 
        print 


        proj = numpy.ones(numpy.shape(P))
        proj[:101, :] = stacked_params        

        proj = proj.astype(numpy.float32)

        dnn_model.params[0].set_value(proj, borrow=True)
    
    elif  synth_mode == 'uniform_sampled_within_std_1':
        ## points uniformly sampled from between the 1.8 - 2.0 stds of a diagonal covariance gaussian fitted to the data
        P = dnn_model.get_projection_weights()

        column_min = numpy.min(P[:,:projection_end], axis=0)  ## vector like a row of P with min of its columns
        column_max = numpy.max(P[:,:projection_end], axis=0)

        std_val = numpy.std(P[:,:projection_end], axis=0)

        dots = numpy.random.uniform(low=column_min, high=column_max, size=(100000, 2))
        dots = within_circle(dots, radius=std_val*2.0)
        dots = outside_circle(dots, radius=std_val*1.8)

        m,n = numpy.shape(P)
        dots = dots[:m, :]

        dots = dots.astype(numpy.float32)
        dnn_model.params[0].set_value(dots, borrow=True)


    elif  synth_mode == 'uniform_sampled_within_std_2':
        ## points uniformly sampled from between the 1.8 - 2.0 stds of a diagonal covariance gaussian fitted to the data
        P = dnn_model.get_projection_weights()

        column_min = numpy.min(P[:,:projection_end], axis=0)  ## vector like a row of P with min of its columns
        column_max = numpy.max(P[:,:projection_end], axis=0)

        std_val = numpy.std(P[:,:projection_end], axis=0)

        dots = numpy.random.uniform(low=column_min, high=column_max, size=(100000, 2))
        dots = within_circle(dots, radius=std_val*3.0)
        dots = outside_circle(dots, radius=std_val*2.8)

        m,n = numpy.shape(P)
        dots = dots[:m, :]

        dots = dots.astype(numpy.float32)
        dnn_model.params[0].set_value(dots, borrow=True)


    elif  synth_mode == 'uniform_sampled_within_std_3':
        ## points uniformly sampled from between the 1.8 - 2.0 stds of a diagonal covariance gaussian fitted to the data
        P = dnn_model.get_projection_weights()

        column_min = numpy.min(P[:,:projection_end], axis=0)  ## vector like a row of P with min of its columns
        column_max = numpy.max(P[:,:projection_end], axis=0)

        std_val = numpy.std(P[:,:projection_end], axis=0)

        dots = numpy.random.uniform(low=column_min, high=column_max, size=(100000, 2))
        dots = within_circle(dots, radius=std_val*4.0)
        dots = outside_circle(dots, radius=std_val*3.8)

        m,n = numpy.shape(P)
        dots = dots[:m, :]

        dots = dots.astype(numpy.float32)
        dnn_model.params[0].set_value(dots, borrow=True)

    else:
        sys.exit('unknow mode: %s'%(synth_mode))

    ##  save used weights for future reference:
    if save_weights_to_file:
        weights = dnn_model.get_projection_weights()
        numpy.savetxt(save_weights_to_file, weights)

    file_number = len(valid_file_list)

    for i in xrange(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        features = features.reshape((-1, n_ins))
        
        #features, features_proj = expand_projection_inputs(features, cfg.index_to_project, \
        #                                                         cfg.projection_insize)
        features, features_proj = get_unexpanded_projection_inputs(features, cfg.index_to_project, \
                                                                 cfg.projection_insize)
        #temp_set_x = features.tolist()  ## osw - why list conversion necessary?
        test_set_x = theano.shared(numpy.asarray(features, dtype=theano.config.floatX)) 
        test_set_x_proj = theano.shared(numpy.asarray(features_proj, dtype='int32')) 
        
        predicted_parameter = dnn_model.parameter_prediction(test_set_x=test_set_x, test_set_x_proj=test_set_x_proj)
#        predicted_parameter = test_out()

        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()



## define a couple of functions for circular rejection sampling:
def within_circle(dots, radius=1.0):    
    standardised_dots = (dots - numpy.mean(dots)) / radius
    ## if x^2 + y^2 <= 1, point is within unit circle
    within_circle = (standardised_dots[:,0]*standardised_dots[:,0]) + (standardised_dots[:,1]*standardised_dots[:,1]) <= 1.0    
    return dots[within_circle]
##
def outside_circle(dots, radius=1.0):    
    standardised_dots = (dots - numpy.mean(dots)) / radius
    ## if x^2 + y^2 <= 1, point is within unit circle
    within_circle = (standardised_dots[:,0]*standardised_dots[:,0]) + (standardised_dots[:,1]*standardised_dots[:,1]) > 1.0    
    return dots[within_circle]

        
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


def add_projection_indices(uttlist, token_xpath, attrib_name, outdir):
    ## Taken from: ~/proj/dnn_tts/script/add_token_index.py
    '''
    For utts in uttlist, add attribute called <attrib_name> to all nodes 
    matching <token_xpath> with a corpus-unique integer value > 0. Add default 
    0-valued attrib at root node.
    '''
    i = 1
    for uttfile in uttlist:
        utt = etree.parse(uttfile)
        ## clear target attribute name from all nodes to be safe:
        for node in utt.xpath('//*'): ## all nodes
            if attrib_name in node.attrib:
                del node.attrib[attrib_name]
        root_node = utt.getroot()
        root_node.attrib[attrib_name] = '0'   ## 0 is the defualt 'n/a' value -- *some* ancestor of all nodes will have the relevant attibute to fall back on
        for node in utt.xpath(token_xpath):
            node.attrib[attrib_name] = str(i)            
            i += 1            
        junk,fname = os.path.split(uttfile)
        outfile = os.path.join(outdir, fname)
        utt.write(outfile, encoding='utf-8', pretty_print=True) 

def add_projection_indices_with_replicates(uttlist, token_xpath, attrib_name, outdir, nreplicates):
    ## Taken from: ~/proj/dnn_tts/script/add_token_index.py
    '''
    For utts in uttlist, add attribute called <attrib_name> to all nodes 
    matching <token_xpath> with a corpus-unique integer value > 0. Add default 
    0-valued attrib at root node.
    '''
    assert len(uttlist) == 1
    uttfile = uttlist[0]

    i = 1
    
    master_utt = etree.parse(uttfile)

    new_utt_names = []

    while i < nreplicates + 1:

        utt = copy.copy(master_utt)

        ## clear target attribute name from all nodes to be safe:
        for node in utt.xpath('//*'): ## all nodes
            if attrib_name in node.attrib:
                del node.attrib[attrib_name]
        root_node = utt.getroot()
        root_node.attrib[attrib_name] = '0'   ## 0 is the defualt 'n/a' value -- *some* ancestor of all nodes will have the relevant attibute to fall back on
        assert len(utt.xpath(token_xpath)) == 1
        for node in utt.xpath(token_xpath):
            node.attrib[attrib_name] = str(i)                    
        junk,fname = os.path.split(uttfile)
        new_utt_name = fname.replace('.utt', '_rep_%s.utt'%(i))
        new_utt_names.append(new_utt_name)
        outfile = os.path.join(outdir, new_utt_name)
        utt.write(outfile, encoding='utf-8', pretty_print=True) 
        i += 1    

    return new_utt_names


def retrieve_normalisation_values(norm_file):
    ## TODO -- move reading and writing into MinMaxNormalisation class

    if not os.path.isfile(norm_file):
        sys.exit('Normalisation file %s does not exist '%(norm_file))

    # reload stored minmax values: 
    fid = open(norm_file, 'rb')

    ## This doesn't work -- precision is lost -- reads in as float64
    #label_norm_info = numpy.fromfile(fid)  ## label_norm_info = numpy.array(label_norm_info, 'float32')

    ## use struct to enforce float32:
    nbytes = os.stat(norm_file)[6]  # length in bytes
    data = fid.read(nbytes)               # = read until bytes run out 
    fid.close()
    m = nbytes / 4  ## number of 32 bit floats
    format = str(m)+"f"
    label_norm_info = struct.unpack(format, data)
    label_norm_info = numpy.array(label_norm_info)

    ## values can be min + max or mean + std, hence non-descript variable names:
    first_vector = label_norm_info[:m/2]
    second_vector = label_norm_info[m/2:]     

    return (first_vector, second_vector)


def main_function(cfg, in_dir, out_dir, token_xpath, index_attrib_name, synth_mode, cmp_dir, projection_end):
    ## TODO: token_xpath & index_attrib_name   should be in config
    
    # get a logger for this main function
    logger = logging.getLogger("main")
    
    # get another logger to handle plotting duties
    plotlogger = logging.getLogger("plotting")

    # later, we might do this via a handler that is created, attached and configured
    # but for now we need to do it manually
    plotlogger.set_plot_path(cfg.plot_dir)
    
    #### parameter setting########
    hidden_layers_sizes = cfg.hyper_params['hidden_layers_sizes']
    
    ####prepare environment    
    synth_utts_input = glob.glob(in_dir + '/*.utt')
    ###synth_utts_input = synth_utts_input[:10]   ### temp!!!!!

    if synth_mode == 'single_sentence_demo':
        synth_utts_input = synth_utts_input[:1]
        print 
        print 'mode: single_sentence_demo'
        print synth_utts_input
        print

    indexed_utt_dir = os.path.join(out_dir, 'utt') ## place to put test utts with tokens labelled with projection indices
    direcs = [out_dir, indexed_utt_dir]
    for direc in direcs:
        if not os.path.isdir(direc):
            os.mkdir(direc)
    

    ## was below -- see comment
    if synth_mode == 'single_sentence_demo':
        synth_utts_input = add_projection_indices_with_replicates(synth_utts_input, token_xpath, index_attrib_name, indexed_utt_dir, 100)
    else:
        add_projection_indices(synth_utts_input, token_xpath, index_attrib_name, indexed_utt_dir)




    file_id_list = []
    for fname in synth_utts_input:
        junk,name = os.path.split(fname)
        file_id_list.append(name.replace('.utt',''))


    data_dir = cfg.data_dir

    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir   = os.path.join(out_dir, 'gen')    

    ###normalisation information
    norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')
    
    ### normalise input full context label
    if cfg.label_style == 'HTS':
        sys.exit('only ossian utts supported')        
    elif cfg.label_style == 'composed':
        suffix='composed'

    # the number can be removed
    binary_label_dir      = os.path.join(out_dir, 'lab_bin')
    nn_label_norm_dir     = os.path.join(out_dir, 'lab_bin_norm')

    binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
    nn_label_norm_file_list  = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)

    ## need this to find normalisation info:
    if cfg.process_labels_in_work_dir:
        label_data_dir = cfg.work_dir
    else:
        label_data_dir = data_dir
    
    min_max_normaliser = None
    label_norm_file = 'label_norm_%s.dat' %(cfg.label_style)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)
    
    if cfg.label_style == 'HTS':
        sys.exit('script not tested with HTS labels')


    ## always do this in synth:
    ## if cfg.NORMLAB and (cfg.label_style == 'composed'):  
    logger.info('add projection indices to tokens in test utts')

    ## add_projection_indices was here

    logger.info('preparing label data (input) using "composed" style labels')
    label_composer = LabelComposer()
    label_composer.load_label_configuration(cfg.label_config_file)

    logger.info('Loaded label configuration')

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
                in_label_align_file_list['xpath'] = prepare_file_path_list(file_id_list, indexed_utt_dir, cfg.utt_ext, False)
            elif label_style == 'hts':
                logger.critical('script not tested with HTS labels')        
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
    
    # no silence removal for synthesis ...
    
    ## minmax norm:
    min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99, exclude_columns=[cfg.index_to_project])

    (min_vector, max_vector) = retrieve_normalisation_values(label_norm_file)
    min_max_normaliser.min_vector = min_vector
    min_max_normaliser.max_vector = max_vector

    ###  apply precompuated and stored min-max to the whole dataset
    min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)


### DEBUG
    if synth_mode == 'inferred':

        ## set up paths -- write CMP data to infer from in outdir:
        nn_cmp_dir = os.path.join(out_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
        nn_cmp_norm_dir = os.path.join(out_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))

        in_file_list_dict = {}
        for feature_name in cfg.in_dir_dict.keys():
            in_direc = os.path.join(cmp_dir, feature_name)
            assert os.path.isdir(in_direc), in_direc
            in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, in_direc, cfg.file_extension_dict[feature_name], False)        
        
        nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
        nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)



        ### make output acoustic data
        #    if cfg.MAKECMP:
        logger.info('creating acoustic (output) features')
        delta_win = [-0.5, 0.0, 0.5]
        acc_win = [1.0, -2.0, 1.0]
        
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

        ## skip silence removal for inference -- need to match labels, which are
        ## not silence removed either


        
    ### retrieve acoustic normalisation information for normalising the features back
    var_dir   = os.path.join(data_dir, 'var')
    var_file_dict = {}
    for feature_name in cfg.out_dimension_dict.keys():
        var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))
        
        
    ### normalise output acoustic data
#    if cfg.NORMCMP:


#### DEBUG
    if synth_mode == 'inferred':


        logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
        cmp_norm_info = None
        if cfg.output_feature_normalisation == 'MVN':
            normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)

            (mean_vector,std_vector) = retrieve_normalisation_values(norm_info_file)
            normaliser.mean_vector = mean_vector
            normaliser.std_vector = std_vector

            ###  apply precompuated and stored mean and std to the whole dataset
            normaliser.feature_normalisation(nn_cmp_file_list, nn_cmp_norm_file_list)

        elif cfg.output_feature_normalisation == 'MINMAX':        
            sys.exit('not implemented')
            #            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim)
            #            global_mean_vector = min_max_normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number])
            #            global_std_vector = min_max_normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector)

            #            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
            #            min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
            #            min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

            #            cmp_min_vector = min_max_normaliser.min_vector
            #            cmp_max_vector = min_max_normaliser.max_vector
            #            cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

        else:
            logger.critical('Normalisation type %s is not supported!\n' %(cfg.output_feature_normalisation))
            raise
 

    combined_model_arch = str(len(hidden_layers_sizes))
    for hid_size in hidden_layers_sizes:
        combined_model_arch += '_' + str(hid_size)
    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.model' \
                      %(model_dir, cfg.model_type, cfg.combined_feature_name, int(cfg.multistream_switch), 
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number)

    ### DNN model training
#    if cfg.TRAINDNN: always do this in synth






#### DEBUG
    inferred_weights = None ## default, for non-inferring synth methods
    if synth_mode == 'inferred':

        ## infer control values from TESTING data

        ## identical lists (our test data) for 'train' and 'valid' -- this is just to
        ##   keep the infer_projections_fn theano function happy -- operates on
        ##    validation set. 'Train' set shouldn't be used here.
        train_x_file_list = copy.copy(nn_label_norm_file_list)
        train_y_file_list = copy.copy(nn_cmp_norm_file_list)
        valid_x_file_list = copy.copy(nn_label_norm_file_list)
        valid_y_file_list = copy.copy(nn_cmp_norm_file_list)

        print 'FILELIST for inferr:'
        print train_x_file_list 
        print 

        try:
            inferred_weights = infer_projections(train_xy_file_list = (train_x_file_list, train_y_file_list), \
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






    ## if cfg.DNNGEN:
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



    gen_file_list = prepare_file_path_list(file_id_list, gen_dir, cfg.cmp_ext)

    #print nn_label_norm_file_list  ## <-- this WAS mangled in inferred due to copying of file list to trainlist_x etc. which is then shuffled. Now use copy.copy
    #print gen_file_list

    weights_outfile = os.path.join(out_dir, 'projection_weights_for_synth.txt')  
    dnn_generation_PROJECTION(nn_label_norm_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list, cfg=cfg, synth_mode=synth_mode, projection_end=projection_end, projection_weights_to_use=inferred_weights, save_weights_to_file=weights_outfile )
    
    logger.debug('denormalising generated output using method %s' % cfg.output_feature_normalisation)
    ## DNNGEN

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

            ## osw: skip MLPG:
#            split_cmp(gen_file_list, ['mgc', 'lf0', 'bap'], cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)    

    ## Variance scaling:
    scaled_dir = gen_dir + '_scaled'
    simple_scale_variance(gen_dir, scaled_dir, var_file_dict, cfg.out_dimension_dict, file_id_list, gv_weight=0.5)  ## gv_weight hardcoded

    ### generate wav ---- glottHMM only!!!
    #if cfg.GENWAV:
    logger.info('reconstructing waveform(s)')
    generate_wav_glottHMM(scaled_dir, file_id_list)   # generated speech
    

def simple_scale_variance(indir, outdir, var_file_dict, out_dimension_dict, file_id_list, gv_weight=1.0):
    ## simple variance scaling (silen et al. 2012, paragraph 3.1)
    ## TODO: Lots of things like stream names hardcoded here; 3 for delta + delta-delta; ...
    all_streams = ['cmp','HNR','F0','LSF','Gain','LSFsource']
    streams_to_scale = ['LSF']

    static_variances = {}
 
    static_dimension_dict = {}
    for (feature_name,size) in out_dimension_dict.items():
        static_dimension_dict[feature_name] = size/3

    io_funcs = BinaryIOCollection()
    for feature_name in var_file_dict.keys():
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)
        static_var_values = var_values[:static_dimension_dict[feature_name], :]
        static_variances[feature_name] = static_var_values

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    assert gv_weight <= 1.0 and gv_weight >= 0.0
    local_weight = 1.0 - gv_weight

    for uttname in file_id_list:
        for stream in all_streams:
            infile = os.path.join(indir, uttname + '.' + stream)
            outfile = os.path.join(outdir, uttname + '.' + stream)
            if not os.path.isfile(infile):
                sys.exit(infile + ' does not exist')
            if stream in streams_to_scale:
                speech, dimension = io_funcs.load_binary_file_frame(infile, static_dimension_dict[stream])
                utt_mean = numpy.mean(speech, axis=0) 
                utt_std =  numpy.std(speech, axis=0) 

                global_std = numpy.transpose((static_variances[stream]))
                weighted_global_std = (gv_weight * global_std) + (local_weight * utt_std)
                std_ratio = weighted_global_std / utt_std 

                nframes, ndim = numpy.shape(speech)
                utt_mean_matrix = numpy.tile(utt_mean, (nframes,1))
                std_ratio_matrix = numpy.tile(std_ratio, (nframes,1))

                scaled_speech = ((speech - utt_mean_matrix) * std_ratio_matrix) + utt_mean_matrix
                io_funcs.array_to_binary_file(scaled_speech, outfile)


            else:
                os.system('cp %s %s'%(infile, outfile))


def log_to_hertz(infile, outfile):
    f = open(infile, 'r')
    log_values = [float(val) for val in f.readlines()]
    f.close()

    def m2h(l):
        h = math.exp(l)
        return h

    hertz = [m2h(l) for l in log_values]
    f = open(outfile, 'w')
    for val in hertz:
        if val > 0:
            f.write(str(val) + '\n')
        else:
            f.write('0.0\n')
    f.close()

def generate_wav_glottHMM(gen_dir, gen_file_id_list):

    x2x='~/repos/simple4all/CSTRVoiceClone/trunk/bin/x2x'
    synthesis='~/sim2/oliver/nst_repos/OSSIAN/ossian-v.1.3/tools/GlottHMM/Synthesis'
    general_glott_conf = '~/sim2/oliver/nst_repos/OSSIAN/ossian-v.1.3/voices/en/ky_02_toy/english_gold_basic_glott_KY/processors/speech_feature_extractor/main_config.cfg'
    user_glott_conf = '~/sim2/oliver/nst_repos/OSSIAN/ossian-v.1.3/voices/en/ky_02_toy/english_gold_basic_glott_KY/processors/speech_feature_extractor/user_config.cfg'

    exports = 'export LIBCONFIG_INSTALL_DIR=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/nst_repos/OSSIAN/ossian-v.1.3/tools/GlottHMM//libconfig-1.4.9 ; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBCONFIG_INSTALL_DIR/lib/.libs ; export LIBRARY_PATH=$LIBRARY_PATH:$LIBCONFIG_INSTALL_DIR/lib/.libs ; export CPATH=$CPATH:$LIBCONFIG_INSTALL_DIR/lib ;'


    streams = ['cmp','HNR','F0','LSF','Gain','LSFsource']
    for uttname in gen_file_id_list:
        all_present = True
        for stream in streams:
            if not os.path.isfile(os.path.join(gen_dir, uttname + '.' + stream)):
                all_present = False
        if all_present:
            for stream in streams:
                extra = ''
                if stream == 'F0':
                    extra = '.NEGVALS'
                fname = os.path.join(gen_dir, uttname + '.' + stream)
                fname_txt = os.path.join(gen_dir, uttname + '.txt.' + stream + extra)
                comm = '%s +fa %s > %s'%(x2x, fname, fname_txt) 
                os.system(comm)
            log_to_hertz(os.path.join(gen_dir, uttname + '.txt.F0.NEGVALS'), \
                                        os.path.join(gen_dir, uttname + '.txt.F0'))

            stem_name = os.path.join(gen_dir, uttname + '.txt')
            comm = '%s %s %s %s %s'%(exports, synthesis, stem_name, general_glott_conf, user_glott_conf)
            print comm
            os.system(comm)


            
        else:
            print 'missing stream(s) for utterance ' + uttname
        


if __name__ == '__main__':
    
    
    
    # these things should be done even before trying to parse the command line

    # create a configuration instance
    # and get a short name for this instance
    cfg=configuration.cfg



    # set up logging to use our custom class
    logging.setLoggerClass(LoggerPlotter)

    # get a logger for this main function
    logger = logging.getLogger("main")

    if len(sys.argv) not in [8,9]:
        print sys.argv
        sys.exit('usage: run_dnn.sh config_file_name utt_dir')

    config_file = sys.argv[1]
    in_dir = sys.argv[2]
    out_dir = sys.argv[3]
    token_xpath = sys.argv[4]
    index_attrib_name = sys.argv[5]
    synth_mode = sys.argv[6]
    projection_end = int(sys.argv[7]) 

    assert synth_mode in ['constant', 'sampled_training', 'inferred', 'uniform', 'single_sentence_demo', 'uniform_sampled_within_std_1', 'uniform_sampled_within_std_2', 'uniform_sampled_within_std_3']

    cmp_dir = None
    if synth_mode == 'inferred':
        cmp_dir = sys.argv[8]
   

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    
#    if cfg.profile:
#        logger.info('profiling is activated')
#        import cProfile, pstats
#        cProfile.run('main_function(cfg)', 'mainstats')

#        # create a stream for the profiler to write to
#        profiling_output = StringIO.StringIO()
#        p = pstats.Stats('mainstats', stream=profiling_output)

#        # print stats to that stream
#        # here we just report the top 10 functions, sorted by total amount of time spent in each
#        p.strip_dirs().sort_stats('tottime').print_stats(10)

#        # print the result to the log
#        logger.info('---Profiling result follows---\n%s' %  profiling_output.getvalue() )
#        profiling_output.close()
#        logger.info('---End of profiling result---')
#        
#    else:
    main_function(cfg, in_dir, out_dir, token_xpath, index_attrib_name, synth_mode, cmp_dir, projection_end)



    sys.exit(0)
