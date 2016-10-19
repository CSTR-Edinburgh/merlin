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


import math
import ConfigParser
import os
import logging
import StringIO
import sys
import textwrap
import datetime

class configuration(object):

    """Configuration settings. Any user-specific values are read from an external file
    and parsed by an instance of the built-in ConfigParser class"""

    def __init__(self):
        # doesn't do anything
        pass


    def configure(self, configFile=None, use_logging=True):

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

        # now that we have loaded the user's configuration, we can load the
        # separate config file for logging (the name of that file will be specified in the config file)
        if use_logging:
            self.logging_configuration()

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

        # look for those items that are user-configurable, and get their values
        # sptk_bindir= ....



        # a list instead of a dict because OrderedDict is not available until 2.7
        # and I don't want to import theano here just for that one class
        # each entry is a tuple of (variable name, default value, section in config file, option name in config file)
        #
        # the type of the default value is important and controls the type that the corresponding
        # variable will have
        #
        # to set a default value of 'undefined' use an empty string
        # or the special value 'impossible', as appropriate
        #
        impossible_int=int(-99999)
        impossible_float=float(-99999.0)

        user_options = [

            ('work_dir', self.work_dir, 'Paths','work'),
            ('data_dir', '', 'Paths','data'),
            ('plot_dir', '', 'Paths','plot'),

            ('plot',      False, 'Utility', 'plot'),
            ('profile',   False, 'Utility', 'profile'),

            ('file_id_scp'       , os.path.join(self.work_dir, 'data/file_id_list.scp')    , 'Paths', 'file_id_list'),
            ('test_id_scp'       , os.path.join(self.work_dir, 'data/test_id_list.scp')    , 'Paths', 'test_id_list'),

            ('GV_dir'       , os.path.join(self.work_dir, 'data/GV' )  , 'Paths', 'GV_dir'),

            ('in_stepw_dir' , os.path.join(self.work_dir, 'data/stepw'), 'Paths', 'in_stepw_dir'),
            ('in_mgc_dir'   , os.path.join(self.work_dir, 'data/mgc')  , 'Paths', 'in_mgc_dir'),
            ('in_lf0_dir'   , os.path.join(self.work_dir, 'data/lf0')  , 'Paths', 'in_lf0_dir'),
            ('in_bap_dir'   , os.path.join(self.work_dir, 'data/bap')  , 'Paths', 'in_bap_dir'),
            ('in_sp_dir'    , os.path.join(self.work_dir, 'data/sp' )  , 'Paths', 'in_sp_dir'),
            ('in_seglf0_dir', os.path.join(self.work_dir, 'data/lf03') , 'Paths', 'in_seglf0_dir'),

	    ## for glottHMM
            ('in_F0_dir'   , os.path.join(self.work_dir, 'data/F0')  , 'Paths', 'in_F0_dir'),
            ('in_Gain_dir'   , os.path.join(self.work_dir, 'data/Gain')  , 'Paths', 'in_Gain_dir'),
            ('in_HNR_dir'   , os.path.join(self.work_dir, 'data/HNR')  , 'Paths', 'in_HNR_dir'),
            ('in_LSF_dir'   , os.path.join(self.work_dir, 'data/LSF')  , 'Paths', 'in_LSF_dir'),
            ('in_LSFsource_dir'   , os.path.join(self.work_dir, 'data/LSFsource')  , 'Paths', 'in_LSFsource_dir'),

            ## for joint duration
            ('in_seq_dur_dir' , os.path.join(self.work_dir, 'data/S2S_dur')  , 'Paths', 'in_seq_dur_dir'),
            ('in_dur_dir'     , os.path.join(self.work_dir, 'data/dur')      , 'Paths', 'in_dur_dir'),


            ('nn_norm_temp_dir', os.path.join(self.work_dir, 'data/step_hidden9'), 'Paths', 'nn_norm_temp_dir'),

            ('process_labels_in_work_dir', False, 'Labels', 'process_labels_in_work_dir'),



            ('label_style'        , 'HTS'                                                 ,    'Labels', 'label_style'),
            ('label_type'         , 'state_align'                                         ,    'Labels', 'label_type'),
            ('in_label_align_dir' , os.path.join(self.work_dir, 'data/label_state_align') ,    'Labels', 'label_align'),
            ('question_file_name' , os.path.join(self.work_dir, 'data/questions.hed')     ,    'Labels', 'question_file_name'),
            ('silence_pattern'    , ['*-#+*']                                             ,    'Labels', 'silence_pattern'),
            ('subphone_feats'     , 'full'                                                ,    'Labels', 'subphone_feats'),
            ('additional_features', {}                                                    ,    'Labels', 'additional_features'),

            ('xpath_file_name',      os.path.join(self.work_dir, 'data/xml_labels/xpaths.txt'), 'Labels', 'xpath_file_name'),

            ('label_config_file',    'configuration/examplelabelconfigfile.py',                 'Labels', 'label_config'),
            ('add_frame_features',      True,                                                   'Labels', 'add_frame_features'),
            ('fill_missing_values',  False,                                                     'Labels', 'fill_missing_values'),
            ('xpath_label_align_dir', os.path.join(self.work_dir, 'data/label_state_align'),    'Labels', 'xpath_label_align'),

            ('enforce_silence', False, 'Labels', 'enforce_silence'),
            ('remove_silence_using_binary_labels', False, 'Labels', 'remove_silence_using_binary_labels'),
            ('precompile_xpaths', True, 'Labels', 'precompile_xpaths'),
            ('iterate_over_frames', True, 'Labels', 'iterate_over_frames'),

            ('appended_input_dim'   ,  0                   ,  'Labels'       ,  'appended_input_dim'),

            ('buffer_size', 200000, 'Data', 'buffer_size'),

            ('train_file_number', impossible_int, 'Data','train_file_number'),
            ('valid_file_number', impossible_int, 'Data','valid_file_number'),
            ('test_file_number' , impossible_int, 'Data','test_file_number'),

            ('log_path', os.path.join(self.work_dir, 'log'), 'Paths', 'log_path'),
            ('log_file', '', 'Paths','log_file'),
            ('log_config_file', 'configuration/exampleloggingconfigfile.conf', 'Paths', 'log_config_file'),

            ('sptk_bindir', 'tools/bin/SPTK-3.9', 'Paths','sptk'),
            ('straight_bindir', 'tools/bin/straight', 'Paths','straight'),
            ('world_bindir', 'tools/bin/WORLD', 'Paths','world'),

            ('network_type'           , 'RNN'                                           , 'Architecture', 'network_type'),
            ('model_type'           , 'DNN'                                             , 'Architecture', 'model_type'),
            ('hidden_layer_type'    , ['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']  , 'Architecture', 'hidden_layer_type'),
            ('output_layer_type'    , 'LINEAR'                                          , 'Architecture', 'output_layer_type'),
            ('sequential_training'  , False                                           , 'Architecture', 'sequential_training'),
            ('dropout_rate'         , 0.0                                               , 'Architecture', 'dropout_rate'),

	    ## some config variables for token projection DNN
            ('scheme'               , 'stagewise'                   , 'Architecture', 'scheme'),
            ('index_to_project'    , 0       , 'Architecture', 'index_to_project'),
            ('projection_insize'    , 10000        , 'Architecture', 'projection_insize'),
            ('projection_outsize'    , 10        , 'Architecture', 'projection_outsize'),
            ('initial_projection_distrib'    , 'gaussian'    , 'Architecture', 'initial_projection_distrib'),
            ('projection_weights_output_dir'    , 'some_path', 'Architecture', 'projection_weights_output_dir'),
            ('layers_with_projection_input'    , [0], 'Architecture', 'layers_with_projection_input'),
            ('projection_learning_rate_scaling'    , 1.0, 'Architecture', 'projection_learning_rate_scaling'),


            ('learning_rate'        , 0.0002                          , 'Architecture', 'learning_rate'),
            ('l2_reg'               , 0.00001                      , 'Architecture', 'L2_regularization'),
            ('l1_reg'               , 0.0                           , 'Architecture', 'L1_regularization'),
            ('batch_size'           , 16                            , 'Architecture', 'batch_size'),
            ('training_epochs'      , 25                            , 'Architecture', 'training_epochs'),
            ('hidden_activation'    , 'tanh'                        , 'Architecture', 'hidden_activation'),
            ('output_activation'    , 'linear'                      , 'Architecture', 'output_activation'),
            ('hidden_layer_size'  , [1024, 1024, 1024, 1024, 1024, 1024], 'Architecture', 'hidden_layer_size'),
            ('private_hidden_sizes' , [1024]                         , 'Architecture', 'private_hidden_sizes'),
            ('stream_weights'       , [1.0]                         , 'Architecture', 'stream_weights'),
            ('private_l2_reg'       , 0.00001                       , 'Architecture', 'private_l2_reg'),
            ('warmup_epoch'         , 5                             , 'Architecture', 'warmup_epoch'),

            ('warmup_momentum' ,    0.3                           , 'Architecture', 'warmup_momentum'),
            ('momentum' ,           0.9                           , 'Architecture', 'momentum'),
            ('warmup_epoch' ,       5                             , 'Architecture', 'warmup_epoch'),
            ('mdn_component',       1                             , 'Architecture', 'mdn_component'),
            ('var_floor',           0.01                          , 'Architecture', 'var_floor'),
            ('beta_opt',            False                         , 'Architecture', 'beta_opt'),
            ('eff_sample_size',     0.8                           , 'Architecture', 'eff_sample_size'),
            ('mean_log_det',        -100.0                        , 'Architecture', 'mean_log_det'),
            ('start_from_trained_model',  '_'                     , 'Architecture', 'start_from_trained_model'),
            ('use_rprop',           0                             , 'Architecture', 'use_rprop'),


            ('mgc_dim' ,60     ,'Outputs','mgc'),
            ('dmgc_dim',60 * 3 ,'Outputs','dmgc'),
            ('vuv_dim' ,1      ,'Outputs','vuv'),
            ('lf0_dim' ,1      ,'Outputs','lf0'),
            ('dlf0_dim',1 * 3  ,'Outputs','dlf0'),
            ('bap_dim' ,25     ,'Outputs','bap'),
            ('dbap_dim',25 * 3 ,'Outputs','dbap'),
            ('cmp_dim'          ,(60 * 3) + 1 + (1 * 3) + (25 * 3) ,'Outputs','cmp'),
            ('stepw_dim'        , 55, 'Outputs', 'stepw_dim'),
            ('temp_sp_dim'      , 1025, 'Outputs', 'temp_sp_dim'),
            ('seglf0_dim'       , 7                 , 'Outputs', 'seglf0_dim'),
            ('delta_win'        , [-0.5, 0.0, 0.5]  , 'Outputs', 'delta_win'),
            ('acc_win'          , [1.0, -2.0, 1.0]  , 'Outputs', 'acc_win'),
            ('do_MLPG'          , True              , 'Outputs', 'do_MLPG'),


	    ## for GlottHMM
            ('F0_dim' ,1     ,'Outputs','F0'),
            ('dF0_dim',1 * 3 ,'Outputs','dF0'),
            ('Gain_dim' ,1     ,'Outputs','Gain'),
            ('dGain_dim',1 * 3 ,'Outputs','dGain'),
            ('HNR_dim' ,5     ,'Outputs','HNR'),
            ('dHNR_dim',5 * 3 ,'Outputs','dHNR'),
            ('LSF_dim' ,30     ,'Outputs','LSF'),
            ('dLSF_dim',30 * 3 ,'Outputs','dLSF'),
            ('LSFsource_dim' ,10     ,'Outputs','LSFsource'),
            ('dLSFsource_dim',10 * 3 ,'Outputs','dLSFsource'),

        ## for joint dur:-
            ('seq_dur_dim' ,1     ,'Outputs','seq_dur'),
            ('remove_silence_from_dur'  , True  , 'Outputs', 'remove_silence_from_dur'),
            ('dur_dim' ,5     ,'Outputs','dur'),
            ('dur_feature_type' , 'numerical' , 'Outputs', 'dur_feature_type'),


            ('output_feature_normalisation', 'MVN', 'Outputs', 'output_feature_normalisation'),

            ('multistream_switch'  , False , 'Streams', 'multistream_switch'),
#            ('use_private_hidden'  , False, 'Streams', 'use_private_hidden'),

            ('output_features' , ['mgc','lf0', 'vuv', 'bap'], 'Streams', 'output_features'),
            ('gen_wav_features', ['mgc', 'bap', 'lf0']      , 'Streams', 'gen_wav_features'),

#            ('stream_mgc_hidden_size'   ,  192 , 'Streams', 'stream_mgc_hidden_size'),
#            ('stream_lf0_hidden_size'   ,  32  , 'Streams', 'stream_lf0_hidden_size'),
#            ('stream_vuv_hidden_size'   ,  32  , 'Streams', 'stream_vuv_hidden_size'),
#            ('stream_bap_hidden_size'   ,  128 , 'Streams', 'stream_bap_hidden_size'),
#            ('stream_stepw_hidden_size' ,  64  , 'Streams', 'stream_stepw_hidden_size'),
#            ('stream_seglf0_hidden_size',  64  , 'Streams', 'stream_seglf0_hidden_size'),
#            ('stream_cmp_hidden_size'   ,  256 , 'Streams', 'stream_cmp_hidden_size'),  #when multi-stream is disabled, use this to indicate the final hidden layer size
                                                                                        #if this is also not provided, use the top common hidden layer size

            ## Glott HMM -- dummy values -- haven't used private streams:--
#            ('stream_F0_hidden_size'   ,  192 , 'Streams', 'stream_F0_hidden_size'),
#            ('stream_Gain_hidden_size'   ,  192 , 'Streams', 'stream_Gain_hidden_size'),
#            ('stream_HNR_hidden_size'   ,  192 , 'Streams', 'stream_HNR_hidden_size'),
#            ('stream_LSF_hidden_size'   ,  192 , 'Streams', 'stream_LSF_hidden_size'),
#            ('stream_LSFsource_hidden_size'   ,  192 , 'Streams', 'stream_LSFsource_hidden_size'),

            ## joint dur -- dummy values -- haven't used private streams:--
#            ('stream_dur_hidden_size'   ,  192 , 'Streams', 'stream_dur_hidden_size'),

#            ('stream_sp_hidden_size'    , 1024, 'Streams', 'stream_sp_hidden_size'),

#            ('stream_weight_mgc'   , 1.0, 'Streams', 'stream_weight_mgc'),
#            ('stream_weight_lf0'   , 3.0, 'Streams', 'stream_weight_lf0'),
#            ('stream_weight_vuv'   , 1.0, 'Streams', 'stream_weight_vuv'),
#            ('stream_weight_bap'   , 1.0, 'Streams', 'stream_weight_bap'),
#            ('stream_weight_stepw' , 0.0, 'Streams', 'stream_weight_stepw'),
#            ('stream_weight_seglf0', 1.0, 'Streams', 'stream_weight_seglf0'),
#            ('stream_weight_sp'    , 1.0, 'Streams', 'stream_weight_sp'),


            ## Glott HMM - unused?
#            ('stream_weight_F0'   , 1.0, 'Streams', 'stream_weight_F0'),
#            ('stream_weight_Gain'   , 1.0, 'Streams', 'stream_weight_Gain'),
#            ('stream_weight_HNR'   , 1.0, 'Streams', 'stream_weight_HNR'),
#            ('stream_weight_LSF'   , 1.0, 'Streams', 'stream_weight_LSF'),
#            ('stream_weight_LSFsource'   , 1.0, 'Streams', 'stream_weight_LSFsource'),

            ## dur - unused?
#            ('stream_weight_dur'   , 1.0, 'Streams', 'stream_weight_dur'),
#            ('stream_lf0_lr'       , 0.5, 'Streams', 'stream_lf0_lr'),
#            ('stream_vuv_lr'       , 0.5, 'Streams', 'stream_vuv_lr'),

            ('vocoder_type'     ,'STRAIGHT'            ,'Waveform'  , 'vocoder_type'),
            ('sr'               ,48000                 ,'Waveform'  , 'samplerate'),
            ('fl'               ,4096                  ,'Waveform'  , 'framelength'),
            ('shift'            ,1000 * 240 / 48000    ,'Waveform'  , 'frameshift'),
            ('sp_dim'           ,(4096 / 2) + 1        ,'Waveform'  , 'sp_dim'),
            # fw_alpha: 'Bark' or 'ERB' allowing deduction of alpha, or explicity float value (e.g. 0.77)
            ('fw_alpha'         ,0.77                  ,'Waveform'  , 'fw_alpha'),
            ('pf_coef'          ,1.4                   ,'Waveform'  , 'postfilter_coef'),
            ('co_coef'          ,2047                  ,'Waveform'  , 'minimum_phase_order'),
            ('use_cep_ap'       ,True                  ,'Waveform'  , 'use_cep_ap'),
            ('do_post_filtering',True                  ,'Waveform'  , 'do_post_filtering'),
            ('apply_GV'         ,False                 ,'Waveform'  , 'apply_GV'),
            ('test_synth_dir'   ,'test_synthesis/wav'  ,'Waveform'  , 'test_synth_dir'),

            ('DurationModel'        , False, 'Processes', 'DurationModel'),
            ('AcousticModel'        , False, 'Processes', 'AcousticModel'),
            ('GenTestList'          , False, 'Processes', 'GenTestList'),

            ('NORMLAB'         , False, 'Processes', 'NORMLAB'),
            ('MAKEDUR'         , False, 'Processes', 'MAKEDUR'),
            ('MAKECMP'         , False, 'Processes', 'MAKECMP'),
            ('NORMCMP'         , False, 'Processes', 'NORMCMP'),
            ('TRAINDNN'        , False, 'Processes', 'TRAINDNN'),
            ('DNNGEN'          , False, 'Processes', 'DNNGEN'),
            ('GENWAV'          , False, 'Processes', 'GENWAV'),
            ('CALMCD'          , False, 'Processes', 'CALMCD'),
            ('NORMSTEP'        , False, 'Processes', 'NORMSTEP'),
            ('GENBNFEA'        , False, 'Processes', 'GENBNFEA'),

            ('mgc_ext'   , '.mgc'     , 'Extensions', 'mgc_ext'),
            ('bap_ext'   , '.bap'     , 'Extensions', 'bap_ext'),
            ('lf0_ext'   , '.lf0'     , 'Extensions', 'lf0_ext'),
            ('cmp_ext'   , '.cmp'     , 'Extensions', 'cmp_ext'),
            ('lab_ext'   , '.lab'     , 'Extensions', 'lab_ext'),
            ('utt_ext'   , '.utt'     , 'Extensions', 'utt_ext'),
            ('stepw_ext' , '.stepw'   , 'Extensions', 'stepw_ext'),
            ('sp_ext'    , '.sp'      , 'Extensions', 'sp_ext'),


            ## GlottHMM
            ('F0_ext'   , '.F0'     , 'Extensions', 'F0_ext'),
            ('Gain_ext'   , '.Gain'     , 'Extensions', 'Gain_ext'),
            ('HNR_ext'   , '.HNR'     , 'Extensions', 'HNR_ext'),
            ('LSF_ext'   , '.LSF'     , 'Extensions', 'LSF_ext'),
            ('LSFsource_ext'   , '.LSFsource'     , 'Extensions', 'LSFsource_ext'),

            ## joint dur
            ('dur_ext'   , '.dur'     , 'Extensions', 'dur_ext'),

        ]


        # this uses exec(...) which is potentially dangerous since arbitrary code could be executed
        for (variable,default,section,option) in user_options:
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


            if   type(default) == str:
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


        self.combined_feature_name = ''
        for feature_name in self.output_features:
            self.combined_feature_name += '_'
            self.combined_feature_name += feature_name

        self.combined_model_name = self.model_type
        for hidden_type in self.hidden_layer_type:
            self.combined_model_name += '_' + hidden_type

        self.combined_model_name += '_' + self.output_layer_type


    def complete_configuration(self):
        # to be called after reading any user-specific settings
        # because the values set here depend on those user-specific settings

        # get a logger
        logger = logging.getLogger("configuration")

        # tools
        self.SPTK = {
            'X2X'    : os.path.join(self.sptk_bindir,'x2x'),
            'MERGE'  : os.path.join(self.sptk_bindir,'merge'),
            'BCP'    : os.path.join(self.sptk_bindir,'bcp'),
            'MLPG'   : os.path.join(self.sptk_bindir,'mlpg'),
            'MGC2SP' : os.path.join(self.sptk_bindir,'mgc2sp'),
            'VSUM'   : os.path.join(self.sptk_bindir,'vsum'),
            'VSTAT'  : os.path.join(self.sptk_bindir,'vstat'),
            'SOPR'   : os.path.join(self.sptk_bindir,'sopr'),
            'VOPR'   : os.path.join(self.sptk_bindir,'vopr'),
            'FREQT'  : os.path.join(self.sptk_bindir,'freqt'),
            'C2ACR'  : os.path.join(self.sptk_bindir,'c2acr'),
            'MC2B'   : os.path.join(self.sptk_bindir,'mc2b'),
            'B2MC'  : os.path.join(self.sptk_bindir,'b2mc')
            }
#        self.NND = {
#            'FEATN'  : os.path.join(self.nndata_bindir,'FeatureNormalization'),
#            'LF0IP'  : os.path.join(self.nndata_bindir,'F0Interpolation'),
#            'F0VUV'  : os.path.join(self.nndata_bindir,'F0VUVComposition')
#            }
        self.STRAIGHT = {
            'SYNTHESIS_FFT' : os.path.join(self.straight_bindir, 'synthesis_fft'),
            'BNDAP2AP'      : os.path.join(self.straight_bindir, 'bndap2ap'),
            }

        self.WORLD = {
            'SYNTHESIS'     : os.path.join(self.world_bindir, 'synth'),
            'ANALYSIS'      : os.path.join(self.world_bindir, 'analysis'),
            }

        # STILL TO DO - test that all the above tools exist and are executable




        ###dimensions for the output features
        ### key name must follow the self.in_dimension_dict.
        ### If do not want to include dynamic feature, just use the same dimension as that self.in_dimension_dict
        ### if lf0 is one of the acoustic featues, the out_dimension_dict must have an additional 'vuv' key
        ### a bit confusing

        ###need to control the order of the key?
        self.in_dir_dict  = {}          ##dimensions for each raw acoustic (output of NN) feature
        self.out_dimension_dict = {}
        self.in_dimension_dict = {}

        self.private_hidden_sizes = []
        self.stream_weights = []

        logger.debug('setting up output features')
        self.cmp_dim = 0
        for feature_name in self.output_features:
            logger.debug(' %s' % feature_name)

            in_dimension = 0
            out_dimension = 0
            in_directory = ''
#            current_stream_hidden_size = 0
#            current_stream_weight = 0.0
#            stream_lr_ratio = 0.0
            if feature_name == 'mgc':
                in_dimension  = self.mgc_dim
                out_dimension = self.dmgc_dim
                in_directory  = self.in_mgc_dir

#                current_stream_hidden_size = self.stream_mgc_hidden_size
#                current_stream_weight      = self.stream_weight_mgc
            elif feature_name == 'bap':
                in_dimension = self.bap_dim
                out_dimension = self.dbap_dim
                in_directory  = self.in_bap_dir

#                current_stream_hidden_size = self.stream_bap_hidden_size
#                current_stream_weight      = self.stream_weight_bap
            elif feature_name == 'lf0':
                in_dimension = self.lf0_dim
                out_dimension = self.dlf0_dim
                in_directory  = self.in_lf0_dir

#                current_stream_hidden_size = self.stream_lf0_hidden_size
#                current_stream_weight      = self.stream_weight_lf0
            elif feature_name == 'vuv':
                out_dimension = 1

#                current_stream_hidden_size = self.stream_vuv_hidden_size
#                current_stream_weight      = self.stream_weight_vuv
            elif feature_name == 'stepw':
                in_dimension = self.stepw_dim
                out_dimension = self.stepw_dim
                in_directory  = self.in_stepw_dir

#                current_stream_hidden_size = self.stream_stepw_hidden_size
#                current_stream_weight      = self.stream_weight_stepw
            elif feature_name == 'sp':
                in_dimension = self.sp_dim
                out_dimension = self.sp_dim
                in_directory  = self.in_sp_dir

#                current_stream_hidden_size = self.stream_sp_hidden_size
#                current_stream_weight      = self.stream_weight_sp

            elif feature_name == 'seglf0':
                in_dimension = self.seglf0_dim
                out_dimension = self.seglf0_dim
                in_directory = self.in_seglf0_dir

#                current_stream_hidden_size = self.stream_seglf0_hidden_size
#                current_stream_weight      = self.stream_weight_seglf0


            ## for GlottHMM (start)
            elif feature_name == 'F0':
                in_dimension = self.F0_dim
                out_dimension = self.dF0_dim
                in_directory  = self.in_F0_dir

#                current_stream_hidden_size = self.stream_F0_hidden_size
#                current_stream_weight      = self.stream_weight_F0

            elif feature_name == 'Gain':
                in_dimension = self.Gain_dim
                out_dimension = self.dGain_dim
                in_directory  = self.in_Gain_dir

#                current_stream_hidden_size = self.stream_Gain_hidden_size
#                current_stream_weight      = self.stream_weight_Gain

            elif feature_name == 'HNR':
                in_dimension = self.HNR_dim
                out_dimension = self.dHNR_dim
                in_directory  = self.in_HNR_dir

#                current_stream_hidden_size = self.stream_HNR_hidden_size
#                current_stream_weight      = self.stream_weight_HNR

            elif feature_name == 'LSF':
                in_dimension = self.LSF_dim
                out_dimension = self.dLSF_dim
                in_directory  = self.in_LSF_dir

#                current_stream_hidden_size = self.stream_LSF_hidden_size
#                current_stream_weight      = self.stream_weight_LSF

            elif feature_name == 'LSFsource':
                in_dimension = self.LSFsource_dim
                out_dimension = self.dLSFsource_dim
                in_directory  = self.in_LSFsource_dir

#                current_stream_hidden_size = self.stream_LSFsource_hidden_size
#                current_stream_weight      = self.stream_weight_LSFsource
            ## for GlottHMM (end)

            ## for joint dur (start)
            elif feature_name == 'dur':
                in_dimension = self.dur_dim
                out_dimension = self.dur_dim
                in_directory  = self.in_dur_dir

#                current_stream_hidden_size = self.stream_dur_hidden_size
#                current_stream_weight      = self.stream_weight_dur
            ## for joint dur (end)


            else:
                logger.critical('%s feature is not supported right now. Please change the configuration.py to support it' %(feature_name))
                raise

            logger.info('  in_dimension: %d' % in_dimension)
            logger.info('  out_dimension : %d' % out_dimension)
            logger.info('  in_directory : %s' %  in_directory)
#            logger.info('  current_stream_hidden_size: %d' % current_stream_hidden_size)
#            logger.info('  current_stream_weight: %d' % current_stream_weight)

            if in_dimension > 0:
                self.in_dimension_dict[feature_name] = in_dimension
                if in_directory == '':
                    logger.critical('please provide the path for %s feature' %(feature_name))
                    raise
                if out_dimension < in_dimension:
                    logger.critical('the dimensionality setting for %s feature is not correct!' %(feature_name))
                    raise

                self.in_dir_dict[feature_name] = in_directory


            if out_dimension > 0:
                self.out_dimension_dict[feature_name] = out_dimension

#                if (current_stream_hidden_size <= 0 or current_stream_weight <= 0.0) and self.multistream_switch:
#                    logger.critical('the hidden layer size or stream weight is not corrected setted for %s feature' %(feature_name))
#                    raise

#                if self.multistream_switch:
#                    self.private_hidden_sizes.append(current_stream_hidden_size)
#                    self.stream_weights.append(current_stream_weight)

                self.cmp_dim += out_dimension



#        if not self.multistream_switch:
#            self.private_hidden_sizes = []
#            if self.stream_cmp_hidden_size > 0:
#                self.private_hidden_sizes.append(self.stream_cmp_hidden_size)
#            else:
#                self.private_hidden_sizes.append(self.hidden_layer_size[-1])  ## use the same number of hidden layers if multi-stream is not supported
#            self.stream_weights = []
#            self.stream_weights.append(1.0)

        self.stream_lr_weights = []

        self.multistream_outs = []
        if self.multistream_switch:
            for feature_name in self.out_dimension_dict.keys():
                self.multistream_outs.append(self.out_dimension_dict[feature_name])

#                stream_lr_ratio = 0.5
#                if feature_name == 'lf0':
#                    stream_lr_ratio = self.stream_lf0_lr
#                if feature_name == 'vuv':
#                    stream_lr_ratio = self.stream_vuv_lr
#                self.stream_lr_weights.append(stream_lr_ratio)
        else:
            ### the new cmp is not the one for HTS, it includes all the features, such as that for main tasks and that for additional tasks
            self.multistream_outs.append(self.cmp_dim)
#            self.stream_lr_weights.append(0.5)

        logger.info('multistream dimensions: %s' %(self.multistream_outs))

        # to check whether all the input and output features' file extensions are here
        self.file_extension_dict = {}
        self.file_extension_dict['mgc'] = self.mgc_ext
        self.file_extension_dict['lf0'] = self.lf0_ext
        self.file_extension_dict['bap'] = self.bap_ext
        self.file_extension_dict['stepw'] = self.stepw_ext
        self.file_extension_dict['cmp'] = self.cmp_ext
        self.file_extension_dict['seglf0'] = self.lf0_ext


        ## gHMM:
        self.file_extension_dict['F0'] = self.F0_ext
        self.file_extension_dict['Gain'] = self.Gain_ext
        self.file_extension_dict['HNR'] = self.HNR_ext
        self.file_extension_dict['LSF'] = self.LSF_ext
        self.file_extension_dict['LSFsource'] = self.LSFsource_ext

        ## joint dur
        self.file_extension_dict['dur'] = self.dur_ext

        ## hyper parameters for DNN. need to be setted by the user, as they depend on the architecture
        self.hyper_params = { 'learning_rate'      : '0.0002',        ###
                              'l2_reg'             : '0.00001',
                              'l1_reg'             : '0.0',
                              'batch_size'         : '16',
                              'training_epochs'    : '25',
                              'early_stop_epochs'  : '5',
                              'hidden_activation'  : 'tanh',
                              'output_activation'  : 'linear',
                              'do_pretraining'     : False,
                              'pretraining_epochs' : '10',
                              'pretraining_lr'     : '0.0001'}

        self.hyper_params['warmup_momentum']      = self.warmup_momentum
        self.hyper_params['momentum']             = self.momentum
        self.hyper_params['warmup_epoch']         = self.warmup_epoch


        self.hyper_params['learning_rate']         = self.learning_rate
        self.hyper_params['l2_reg']                = self.l2_reg
        self.hyper_params['l1_reg']                = self.l1_reg
        self.hyper_params['batch_size']            = self.batch_size
        self.hyper_params['training_epochs']       = self.training_epochs
        self.hyper_params['hidden_activation']     = self.hidden_activation
        self.hyper_params['output_activation']     = self.output_activation
        self.hyper_params['hidden_layer_size']   = self.hidden_layer_size
        self.hyper_params['warmup_epoch']          = self.warmup_epoch
        self.hyper_params['use_rprop']             = self.use_rprop

#        self.hyper_params['private_hidden_sizes']  = self.private_hidden_sizes
#        self.hyper_params['stream_weights']        = self.stream_weights
#        self.hyper_params['private_l2_reg']        = self.private_l2_reg
#        self.hyper_params['stream_lr_weights']     = self.stream_lr_weights
#        self.hyper_params['use_private_hidden']    = self.use_private_hidden

        self.hyper_params['model_type']            = self.model_type
        self.hyper_params['hidden_layer_type']     = self.hidden_layer_type

        self.hyper_params['index_to_project']     = self.index_to_project
        self.hyper_params['projection_insize']    = self.projection_insize
        self.hyper_params['projection_outsize']   = self.projection_outsize
        self.hyper_params['initial_projection_distrib']   = self.initial_projection_distrib
        self.hyper_params['layers_with_projection_input']   = self.layers_with_projection_input
        self.hyper_params['projection_learning_rate_scaling']   = self.projection_learning_rate_scaling

        self.hyper_params['sequential_training'] = self.sequential_training
        self.hyper_params['dropout_rate'] = self.dropout_rate

        for hidden_type in self.hidden_layer_type:
            if 'LSTM' in hidden_type or 'RNN' in hidden_type or 'GRU' in hidden_type:
                self.hyper_params['sequential_training'] = self.sequential_training


        #To be recorded in the logging file for reference
        for param_name in self.hyper_params.keys():
            logger.info('%s : %s' %(param_name, str(self.hyper_params[param_name])))

                # input files


        # set up the label processing
        # currently must be one of two styles
        if self.label_style == 'HTS':
            # xpath_file_name is now obsolete - to remove
            self.xpath_file_name=None
        elif self.label_style == 'HTS_duration':
            self.xpath_file_name=None



        elif self.label_style == 'composed':
            self.question_file_name=None

        else:
            logger.critical('unsupported label style requested: %s' % self.label_style)
            raise Exception


    def logging_configuration(self):

        # get a logger
        logger = logging.getLogger("configuration")

        # logging configuration, see here for format description
        # https://docs.python.org/2/library/logging.config.html#logging-config-fileformat


        # what we really want to do is this dicitonary-based configuration, but it's only available from Python 2.7 onwards
        #    logging.config.dictConfig(cfg.logging_configuration)
        # so we will settle for this file-based configuration procedure instead

        try:
            # open the logging configuration file
            fp = open(self.log_config_file,'r')
            logger.debug("loading logging configuration from %s" % self.log_config_file)
            # load the logging configuration file into a string
            config_string = fp.read()
            fp.close()

        except ValueError:
            # this means that cfg.log_config_file does not exist and that no default was provided
            # NOTE: currently this will never run
            logging.warn('no logging configuration file provided - using default (console only, DEBUG level)')

            # set up a default level and default handlers
            # first, get the root logger - all other loggers will inherit its configuration
            rootogger = logging.getLogger("")
            # default logging level is DEBUG (a highly-verbose level)
            rootlogger.setLevel(logging.DEBUG)
            # add a handler to write to console
            ch = logging.StreamHandler()
            rootlogger.addHandler(ch)
            # and a formatter
            formatter = logging.Formatter('%(asctime)s %(levelname)8s%(name)15s: %(message)s')
            ch.setFormatter(formatter)

        except IOError:
            # this means that open(...) threw an error
            logger.critical('could not load logging configuration file %s' % self.log_config_file)
            raise

        else:

            # inject the config lines for the file handler, now that we know the name of the file it will write to

            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path, 0755)
            log_file_name = '%s_%s_%d_%d_%d_%d_%f_%s.log' %(self.combined_model_name, self.combined_feature_name, self.train_file_number,
                                                                      self.cmp_dim, len(self.hidden_layer_size),
                                                                      self.hidden_layer_size[-1], self.learning_rate,
                                                                      datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y"))

            self.log_file = os.path.join(self.log_path, log_file_name)

            to_inject="""
                [handler_file]
                class=FileHandler
                formatter=file
                args=('"""+self.log_file+"""', 'w')
            """

            # config file format doesn't allow leading white space on lines, so remove it with dedent
            config_string = config_string + textwrap.dedent(to_inject)


            try:
                # pass that string as a filehandle
                fh = StringIO.StringIO(config_string)
                logging.config.fileConfig(fh)
                fh.close()
                logger.info("logging is now fully configured")

            except IOError:
                logger.critical('could not configure logging: perhaps log file path is wrong?')
                sys.exit(1)
