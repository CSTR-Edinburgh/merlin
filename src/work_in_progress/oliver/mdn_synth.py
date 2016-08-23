
import cPickle
import gzip
import os, sys, errno
import time
import math
import glob
import struct

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano 
import theano

from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
#from frontend.acoustic_normalisation import CMPNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
#from frontend.feature_normalisation_base import FeatureNormBase
from frontend.mean_variance_norm import MeanVarianceNorm
##from frontend.mlpg_fast import MLParameterGenerationFast
from frontend.mlpg import MLParameterGeneration as MLParameterGenerationFast  ## osw temp

from io_funcs.binary_io import  BinaryIOCollection

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer

import configuration

from models.dnn import DNN
from models.ms_dnn import MultiStreamDNN
from models.ms_dnn_gv import MultiStreamDNNGv
from models.sdae import StackedDenoiseAutoEncoder
from models.mdn import MixtureDensityNetwork

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


### plain DNN case
# def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list):
#     logger = logging.getLogger("dnn_generation")
#     logger.debug('Starting dnn_generation')
# 
#     plotlogger = logging.getLogger("plotting")
# 
#     dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
#     
# #    visualize_dnn(dbn)
# 
#     file_number = len(valid_file_list)
# 
#     for i in xrange(file_number):
#         logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
#         fid_lab = open(valid_file_list[i], 'rb')
#         features = numpy.fromfile(fid_lab, dtype=numpy.float32)
#         fid_lab.close()
#         features = features[:(n_ins * (features.size / n_ins))]
#         features = features.reshape((-1, n_ins))
#         temp_set_x = features.tolist()
#         test_set_x = theano.shared(numpy.asarray(temp_set_x, dtype=theano.config.floatX)) 
#         
#         predicted_parameter = dnn_model.parameter_prediction(test_set_x=test_set_x)
# #        predicted_parameter = test_out()
# 
#         ### write to cmp file
#         predicted_parameter = numpy.array(predicted_parameter, 'float32')
#         temp_parameter = predicted_parameter
#         fid = open(out_file_list[i], 'wb')
#         predicted_parameter.tofile(fid)
#         logger.debug('saved to %s' % out_file_list[i])
#         fid.close()
#         

### multiple Gaussian components
def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list, target_mean_vector, target_std_vector, out_dimension_dict, file_extension_dict, vocoder='straight'):
    logger = logging.getLogger("dnn_generation")
    logger.debug('Starting dnn_generation')

    inf_float = -1.0e+10

    plotlogger = logging.getLogger("plotting")

    cfg.gen_wav_features
    
    if vocoder == 'straight':
        gen_wav_features = ['mgc', 'lf0', 'bap']
    elif vocoder == 'glotthmm':
        gen_wav_features = ['F0', 'Gain', 'HNR', 'LSF','LSFsource']  ## TODO: take this from config
    else:
        sys.exit('unsupported vocoder %s !'%(vocoder))
        
    stream_start_index = {}
    dimension_index = 0
    for feature_name in out_dimension_dict.keys():
        stream_start_index[feature_name] = dimension_index
        dimension_index += out_dimension_dict[feature_name]

    dnn_model = cPickle.load(open(nnets_file_name, 'rb'))
    
    file_number = len(valid_file_list)
    io_funcs = BinaryIOCollection()

    mlpg = MLParameterGenerationFast()

    for i in xrange(file_number):
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size / n_ins))]
        features = features.reshape((-1, n_ins))
        
        frame_number = features.shape[0]
        
        test_set_x = theano.shared(numpy.asarray(features, dtype=theano.config.floatX)) 

        mean_matrix = numpy.tile(target_mean_vector, (features.shape[0], 1))
        std_matrix = numpy.tile(target_std_vector, (features.shape[0], 1))

        predicted_mix   = dnn_model.parameter_prediction_mix(test_set_x = test_set_x)
        max_index = numpy.argmax(predicted_mix, axis=1)

        temp_predicted_mu = dnn_model.parameter_prediction(test_set_x=test_set_x)
        temp_predicted_sigma = dnn_model.parameter_prediction_sigma(test_set_x = test_set_x)
        predicted_mu = numpy.zeros((temp_predicted_mu.shape[0], n_outs))
        predicted_sigma = numpy.zeros((temp_predicted_sigma.shape[0], n_outs))
        for kk in xrange(temp_predicted_mu.shape[0]):
            predicted_mu[kk, :] = temp_predicted_mu[kk, max_index[kk]*n_outs:(max_index[kk]+1)*n_outs]
            predicted_sigma[kk, :] = temp_predicted_sigma[kk, max_index[kk]*n_outs:(max_index[kk]+1)*n_outs]
#        print   predicted_mu.shape
#        predicted_mu = predicted_mu[aa*n_outs:(aa+1)*n_outs]
        predicted_mu = predicted_mu * std_matrix + mean_matrix
        predicted_sigma = ((predicted_sigma ** 0.5) * std_matrix ) ** 2
        
        dir_name = os.path.dirname(out_file_list[i])
        file_id = os.path.splitext(os.path.basename(out_file_list[i]))[0]
        
        mlpg = MLParameterGenerationFast()
        for feature_name in gen_wav_features:
            current_features = predicted_mu[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]
            current_sigma    = predicted_sigma[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]
            
            gen_features = mlpg.generation(current_features, current_sigma, out_dimension_dict[feature_name]/3)

            if feature_name in ['lf0', 'F0']:
                if stream_start_index.has_key('vuv'):
                    vuv_feature = predicted_mu[:, stream_start_index['vuv']:stream_start_index['vuv']+1]
                    for i in xrange(frame_number):
                        if vuv_feature[i, 0] < 0.5:
                            gen_features[i, 0] = inf_float
#                print   gen_features
            new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])

            io_funcs.array_to_binary_file(gen_features, new_file_name)

        
        
        
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


def main_function(cfg, in_dir, out_dir):    
    
    
    # get a logger for this main function
    logger = logging.getLogger("main")
    
    # get another logger to handle plotting duties
    plotlogger = logging.getLogger("plotting")

    # later, we might do this via a handler that is created, attached and configured
    # using the standard config mechanism of the logging module
    # but for now we need to do it manually
    plotlogger.set_plot_path(cfg.plot_dir)
    
    #### parameter setting########
    hidden_layers_sizes = cfg.hyper_params['hidden_layers_sizes']
    

    synth_utts = glob.glob(in_dir + '/*.utt')
    
    file_id_list = []
    for fname in synth_utts:
        junk,name = os.path.split(fname)
        file_id_list.append(name.replace('.utt',''))

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    ###total file number including training, development, and testing
    #total_file_number = len(file_id_list)
    
    data_dir = cfg.data_dir

    #nn_cmp_dir       = os.path.join(data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    #nn_cmp_norm_dir   = os.path.join(data_dir, 'nn_norm'  + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    
    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir   = os.path.join(out_dir, 'gen')    

    #in_file_list_dict = {}

    #for feature_name in cfg.in_dir_dict.keys():
    #    in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    #nn_cmp_file_list         = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
    #nn_cmp_norm_file_list    = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

    ###normalisation information
    norm_info_file = os.path.join(data_dir, 'norm_info' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim) + '_' + cfg.output_feature_normalisation + '.dat')
    
    ### normalise input full context label

    # currently supporting two different forms of lingustic features
    # later, we should generalise this 

    if cfg.label_style == 'HTS':
        label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name)
        lab_dim = label_normaliser.dimension
        logger.info('Input label dimension is %d' % lab_dim)
        suffix=str(lab_dim)
    # no longer supported - use new "composed" style labels instead
    elif cfg.label_style == 'composed':
        # label_normaliser = XMLLabelNormalisation(xpath_file_name=cfg.xpath_file_name)
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
        # simple HTS labels 
        #        logger.info('preparing label data (input) using standard HTS style labels')
        #        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list)

        #        remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = ['*-#+*'])
        #        remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

        #        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
        #        ###use only training data to find min-max information, then apply on the whole dataset
        #        min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
        #        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)



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
                in_label_align_file_list['xpath'] = prepare_file_path_list(file_id_list, in_dir, cfg.utt_ext, False)
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
    min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)

    # reload stored minmax values: (TODO -- move reading and writing into MinMaxNormalisation class)
    fid = open(label_norm_file, 'rb')
    
    ## This doesn't work -- precision is lost -- reads in as float64
    #label_norm_info = numpy.fromfile(fid)  ## label_norm_info = numpy.array(label_norm_info, 'float32')

    ## use struct to enforce float32:
    nbytes = os.stat(label_norm_file)[6]  # length in bytes
    data = fid.read(nbytes)               # = read until bytes run out 
    fid.close()
    m = nbytes / 4  ## number 32 bit floats
    format = str(m)+"f"
    label_norm_info = struct.unpack(format, data)
    label_norm_info = numpy.array(label_norm_info)

    min_max_normaliser.min_vector = label_norm_info[:m/2]
    min_max_normaliser.max_vector = label_norm_info[m/2:]         

    ###  apply precompuated min-max to the whole dataset
    min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)



    ### make output acoustic data
#    if cfg.MAKECMP:
   
    ### retrieve acoustic normalisation information for normalising the features back
    var_dir   = os.path.join(data_dir, 'var')
    var_file_dict = {}
    for feature_name in cfg.out_dimension_dict.keys():
        var_file_dict[feature_name] = os.path.join(var_dir, feature_name + '_' + str(cfg.out_dimension_dict[feature_name]))
        
        
    ### normalise output acoustic data
#    if cfg.NORMCMP:

    combined_model_arch = str(len(hidden_layers_sizes))
    for hid_size in hidden_layers_sizes:
        combined_model_arch += '_' + str(hid_size)
    nnets_file_name = '%s/%s_%s_%d_%s_%d.%d.train.%d.mdn.model' \
                      %(model_dir, cfg.model_type, cfg.combined_feature_name, int(cfg.multistream_switch), 
                        combined_model_arch, lab_dim, cfg.cmp_dim, cfg.train_file_number)
 
    ### DNN model training
#    if cfg.TRAINDNN:

    ##if cfg.DNNGEN:
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



    assert cfg.output_feature_normalisation == 'MVN'

    #gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)

    fid = open(norm_info_file, 'rb')
    cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_min_max = cmp_min_max.reshape((2, -1))
    target_mean_vector = cmp_min_max[0, ] 
    target_std_vector = cmp_min_max[1, ]

#        dnn_generation(valid_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)
#        dnn_generation(test_x_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list)
    dnn_generation(nn_label_norm_file_list, nnets_file_name, lab_dim, cfg.cmp_dim, gen_file_list, target_mean_vector, target_std_vector, cfg.out_dimension_dict, cfg.file_extension_dict, vocoder='glotthmm')

 

    ## Variance scaling:
    test_var_scaling=False
    scaled_dir = gen_dir + '_scaled'
    if test_var_scaling:
        file_id_list = simple_scale_variance_CONTINUUM(gen_dir, scaled_dir, var_file_dict, cfg.out_dimension_dict, file_id_list)
    else:
        simple_scale_variance(gen_dir, scaled_dir, var_file_dict, cfg.out_dimension_dict, file_id_list, gv_weight=0.5)  ## gv_weight hard coded here!

    ### generate wav ---- glottHMM only!!!
    #if cfg.GENWAV:
    logger.info('reconstructing waveform(s)')
    generate_wav_glottHMM(scaled_dir, file_id_list)   # generated speech
    

def simple_scale_variance(indir, outdir, var_file_dict, out_dimension_dict, file_id_list, gv_weight=1.0):
    ## simple variance scaling (silen et al. 2012, paragraph 3.1)
    ## TODO: Lots of things like stream names hardcoded here; 3 for delta + delta-delta; ...
    all_streams = ['HNR','F0','LSF','Gain','LSFsource']
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



def simple_scale_variance_CONTINUUM(indir, outdir, var_file_dict, out_dimension_dict, file_id_list):
    ## Try range of interpolation weights for combining global & local variance 
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

    file_id_list_out = []
    for uttname in file_id_list:
        for gv_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            local_weight = 1.0 - gv_weight
            for stream in all_streams:
                infile = os.path.join(indir, uttname + '.' + stream)
                extended_uttname = uttname + '_gv' + str(gv_weight)
                print extended_uttname
                outfile = os.path.join(outdir, extended_uttname + '.' + stream)
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
            file_id_list_out.append(extended_uttname)
    return file_id_list_out


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


    streams = ['HNR','F0','LSF','Gain','LSFsource']
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


    if len(sys.argv) != 4:
        logger.critical('usage: run_dnn.sh config_file_name utt_dir')
        sys.exit(1)

    config_file = sys.argv[1]
    in_dir = sys.argv[2]
    out_dir = sys.argv[3]

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
    main_function(cfg, in_dir, out_dir)
        
    sys.exit(0)
