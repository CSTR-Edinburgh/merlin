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

import ConfigParser
import logging
import os
import sys

class configuration(object):

    def __init__(self):
        pass;

    def configure(self, configFile=None):

        # get a logger
        logger = logging.getLogger("configuration")
        # this (and only this) logger needs to be configured immediately, otherwise it won't work
        # we can't use the full user-supplied configuration mechanism in this particular case,
        # because we haven't loaded it yet!
        #
        # so, just use simple console-only logging
        logger.setLevel(logging.DEBUG) # this level is hardwired here - should change it to INFO
        # add a handler & its formatter - will write only to console
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s%(name)15s: %(message)s')
        ch.setFormatter(formatter)

        # first, set up some default configuration values
        self.initial_configuration()

        # next, load in any user-supplied configuration values
        # that might over-ride the default values
        self.user_configuration(configFile)

        # finally, set up all remaining configuration values
        # that depend upon either default or user-supplied values
        self.complete_configuration()
        logger.debug('configuration completed')

    def initial_configuration(self):

        # to be called before loading any user specific values

        # things to put here are
        # 1. variables that the user cannot change
        # 2. variables that need to be set before loading the user's config file

        UTTID_REGEX = '(.*)\..*'

    def user_configuration(self,configFile=None):

        # get a logger
        logger = logging.getLogger("configuration")

        # load and parse the provided configFile, if provided
        if not configFile:
            logger.warn('no user configuration file provided; using only built-in default settings')
            return

        # load the config file
        try:
            configparser = ConfigParser.ConfigParser()
            configparser.readfp(open(configFile))
            logger.debug('successfully read and parsed user configuration file %s' % configFile)
        except:
            logger.fatal('error reading user configuration file %s' % configFile)
            raise

        #work_dir must be provided before initialising other directories
        self.work_dir = None

        if self.work_dir == None:
            try:
                self.work_dir = configparser.get('Paths', 'work')

            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                if self.work_dir == None:
                    logger.critical('Paths:work has no value!')
                    raise Exception

        # default place for some data
        self.data_dir    = os.path.join(self.work_dir, 'data')
        self.tensorflow_dir   = os.path.join(self.work_dir, 'tensorflow')

        self.gen_dir     = os.path.join(self.tensorflow_dir, 'gen')
        self.model_dir   = os.path.join(self.tensorflow_dir, 'models')
        self.stats_dir   = os.path.join(self.tensorflow_dir, 'stats')

        self.inter_data_dir = os.path.join(self.work_dir, 'inter_module')
        self.def_inp_dir    = os.path.join(self.inter_data_dir, 'nn_no_silence_lab_norm_425')
        self.def_out_dir    = os.path.join(self.inter_data_dir, 'nn_norm_mgc_lf0_vuv_bap_187')

        impossible_int=int(-99999)
        impossible_float=float(-99999.0)

        user_options = [

            # Paths
            ('work_dir', self.work_dir, 'Paths','work'),
            ('data_dir', self.data_dir, 'Paths','data'),

            ('inp_feat_dir', self.def_inp_dir, 'Paths', 'inp_feat'),
            ('out_feat_dir', self.def_out_dir, 'Paths', 'out_feat'),

            ('model_dir', self.model_dir, 'Paths', 'models'),
            ('stats_dir', self.stats_dir, 'Paths', 'stats'),
            ('gen_dir'  ,   self.gen_dir, 'Paths', 'gen'),

            ('file_id_scp', os.path.join(self.data_dir, 'file_id_list.scp'), 'Paths', 'file_id_list'),
            ('test_id_scp', os.path.join(self.data_dir, 'test_id_list.scp'), 'Paths', 'test_id_list'),

            # Input-Output
            ('inp_dim', 425, 'Input-Output', 'inp_dim'),
            ('out_dim', 187, 'Input-Output', 'out_dim'),

            ('inp_file_ext', '.lab', 'Input-Output', 'inp_file_ext'),
            ('out_file_ext', '.cmp', 'Input-Output', 'out_file_ext'),

            ('inp_norm', 'MINMAX', 'Input-Output', 'inp_norm'),
            ('out_norm', 'MINMAX', 'Input-Output', 'out_norm'),

            # Architecture
            ('hidden_layer_type', ['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH'], 'Architecture', 'hidden_layer_type'),
            ('hidden_layer_size', [ 1024 ,  1024 ,  1024 ,  1024 ,  1024 ,   1024], 'Architecture', 'hidden_layer_size'),

            ('batch_size'   , 256, 'Architecture', 'batch_size'),
            ('num_of_epochs',   1, 'Architecture', 'training_epochs'),
            ('dropout_rate' , 0.0, 'Architecture', 'dropout_rate'),

            ('output_layer_type', 'linear', 'Architecture', 'output_layer_type'),
            ('optimizer'        ,   'adam', 'Architecture', 'optimizer'),
            ('loss_function'    ,    'mse', 'Architecture', 'loss_function'),

            # RNN
            ('sequential_training', False, 'Architecture', 'sequential_training'),
            ('stateful'           , False, 'Architecture', 'stateful'),
            ('use_high_batch_size', False, 'Architecture', 'use_high_batch_size'),

            ('training_algo',   1, 'Architecture', 'training_algo'),
            ('merge_size'   ,   1, 'Architecture', 'merge_size'),
            ('seq_length'   , 200, 'Architecture', 'seq_length'),
            ('bucket_range' , 100, 'Architecture', 'bucket_range'),
            #encoder_decoder
            ('encoder_decoder'      , False                                           ,  'Architecture','encoder_decoder'),
            ('attention'            , False                                           ,  'Architecture', 'attention'),
            ("cbhg"                 , False                                           ,   "Architecture", "cbhg"),
            # Data
            ('shuffle_data', False, 'Data', 'shuffle_data'),

            ('train_file_number', impossible_int, 'Data','train_file_number'),
            ('valid_file_number', impossible_int, 'Data','valid_file_number'),
            ('test_file_number' , impossible_int, 'Data','test_file_number'),

            # Processes
            ('NORMDATA'  , False, 'Processes', 'NORMDATA'),
            ('TRAINMODEL', False, 'Processes', 'TRAINMODEL'),
            ('TESTMODEL' , False, 'Processes', 'TESTMODEL')

        ]

        # this uses exec(...) which is potentially dangerous since arbitrary code could be executed
        for (variable,default,section,option) in user_options:
            # default value
            value=None

            try:
                # first, look for a user-set value for this variable in the config file
                value = configparser.get(section,option)
                user_or_default='user'

            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                # use default value, if there is one
                if (default == None) or \
                   (default == '')   or \
                   ((type(default) == int) and (default == impossible_int)) or \
                   ((type(default) == float) and (default == impossible_float))  :
                    logger.critical('%20s has no value!' % (section+":"+option) )
                    raise Exception
                else:
                    value = default
                    user_or_default='default'


            if type(default) == str:
                exec('self.%s = "%s"'      % (variable,value))
            elif type(default) == int:
                exec('self.%s = int(%s)'   % (variable,value))
            elif type(default) == float:
                exec('self.%s = float(%s)' % (variable,value))
            elif type(default) == bool:
                exec('self.%s = bool(%s)'  % (variable,value))
            elif type(default) == list:
                exec('self.%s = list(%s)'  % (variable,value))
            elif type(default) == dict:
                exec('self.%s = dict(%s)'  % (variable,value))
            else:
                logger.critical('Variable %s has default value of unsupported type %s',variable,type(default))
                raise Exception('Internal error in configuration settings: unsupported default type')

            logger.info('%20s has %7s value %s' % (section+":"+option,user_or_default,value) )


    def complete_configuration(self):
        # to be called after reading any user-specific settings
        # because the values set here depend on those user-specific settings

        # get a logger
        logger = logging.getLogger("configuration")

        ## create directories if not exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

        # input-output normalization stat files
        self.inp_stats_file = os.path.join(self.stats_dir, "input_%d_%s_%d.norm" %(int(self.train_file_number), self.inp_norm, self.inp_dim))
        self.out_stats_file = os.path.join(self.stats_dir, "output_%d_%s_%d.norm" %(int(self.train_file_number), self.out_norm, self.out_dim))

        # define model file name
        if self.sequential_training:
            self.combined_model_arch = 'RNN'+str(self.training_algo)
        else:
            self.combined_model_arch = 'DNN'

        self.combined_model_arch += '_'+str(len(self.hidden_layer_size))
        self.combined_model_arch += '_'+'_'.join(map(str, self.hidden_layer_size))
        self.combined_model_arch += '_'+'_'.join(map(str, self.hidden_layer_type))

        self.nnets_file_name = '%s_%d_train_%d_%d_%d_%d_%d_model' \
                          %(self.combined_model_arch, int(self.shuffle_data),
                             self.inp_dim, self.out_dim, self.train_file_number, self.batch_size, self.num_of_epochs)

        logger.info('model file: %s' % (self.nnets_file_name))

        # model files
        self.json_model_file = os.path.join(self.model_dir, self.nnets_file_name+'.json')
        self.h5_model_file   = os.path.join(self.model_dir, self.nnets_file_name+'.h5')

        # predicted features directory
        self.pred_feat_dir = os.path.join(self.gen_dir, self.nnets_file_name)
        if not os.path.exists(self.pred_feat_dir):
            os.makedirs(self.pred_feat_dir)

        # string.lower for some architecture values
        self.output_layer_type = self.output_layer_type.lower()
        self.optimizer         = self.optimizer.lower()
        self.loss_function     = self.loss_function.lower()
        for i in range(len(self.hidden_layer_type)):
            self.hidden_layer_type[i] = self.hidden_layer_type[i].lower()

        # set sequential training True if using LSTMs
        if 'lstm' in self.hidden_layer_type:
            self.sequential_training = True

        # set/limit batch size to 25
        if self.sequential_training and self.batch_size>50:
            if not self.use_high_batch_size:
                logger.info('reducing the batch size from %s to 25' % (self.batch_size))
                self.batch_size = 25 ## num. of sentences in this case

        # rnn params
        self.rnn_params = {}
        self.rnn_params['merge_size']   = self.merge_size
        self.rnn_params['seq_length']   = self.seq_length
        self.rnn_params['bucket_range'] = self.bucket_range
        self.rnn_params['stateful']     = self.stateful
