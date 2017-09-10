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

import os
import sys
import time
import tensorflow as tf
from tensorflow_lib import configuration
from tensorflow_lib import data_utils
from tensorflow_lib.train import TrainTensorflowModels,Train_Encoder_Decoder_Models

class TensorflowClass(object):

    def __init__(self, cfg):

        ###################################################
        ########## User configurable variables ############
        ###################################################

        inp_feat_dir  = cfg.inp_feat_dir
        out_feat_dir  = cfg.out_feat_dir
        pred_feat_dir = cfg.pred_feat_dir

        inp_file_ext = cfg.inp_file_ext
        out_file_ext = cfg.out_file_ext

        ### Input-Output ###

        self.inp_dim = cfg.inp_dim
        self.out_dim = cfg.out_dim

        self.inp_norm = cfg.inp_norm
        self.out_norm = cfg.out_norm

        self.inp_stats_file = cfg.inp_stats_file
        self.out_stats_file = cfg.out_stats_file

        self.inp_scaler = None
        self.out_scaler = None

        #### define model params ####

        self.hidden_layer_type = cfg.hidden_layer_type
        self.hidden_layer_size = cfg.hidden_layer_size

        self.sequential_training = cfg.sequential_training
        self.encoder_decoder     = cfg.encoder_decoder
        
        self.attention    = cfg.attention
        self.cbhg         = cfg.cbhg
        self.batch_size   = cfg.batch_size
        self.shuffle_data = cfg.shuffle_data

        self.output_layer_type = cfg.output_layer_type
        self.loss_function     = cfg.loss_function
        self.optimizer         = cfg.optimizer

        self.rnn_params    = cfg.rnn_params
        self.dropout_rate  = cfg.dropout_rate
        self.num_of_epochs = cfg.num_of_epochs

        ### Define the work directory###
        self.model_dir = cfg.model_dir

        ### define train, valid, test ###

        train_file_number = cfg.train_file_number
        valid_file_number = cfg.valid_file_number
        test_file_number  = cfg.test_file_number

        file_id_scp  = cfg.file_id_scp
        test_id_scp  = cfg.test_id_scp

        #### main processess ####

        self.NORMDATA   = cfg.NORMDATA
        self.TRAINMODEL = cfg.TRAINMODEL
        self.TESTMODEL  = cfg.TESTMODEL

        #### Generate only test list ####
        self.GenTestList = cfg.GenTestList
        
        ###################################################
        ####### End of user-defined conf variables ########
        ###################################################
        
        #### Create train, valid and test file lists ####
        file_id_list = data_utils.read_file_list(file_id_scp)

        train_id_list = file_id_list[0: train_file_number]
        valid_id_list = file_id_list[train_file_number: train_file_number + valid_file_number]
        test_id_list  = file_id_list[train_file_number + valid_file_number: train_file_number + valid_file_number + test_file_number]
        
        valid_test_id_list = file_id_list[train_file_number: train_file_number + valid_file_number + test_file_number]

        self.inp_train_file_list = data_utils.prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
        self.out_train_file_list = data_utils.prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)

        self.inp_valid_file_list = data_utils.prepare_file_path_list(valid_id_list, inp_feat_dir, inp_file_ext)
        self.out_valid_file_list = data_utils.prepare_file_path_list(valid_id_list, out_feat_dir, out_file_ext)

        self.inp_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, inp_feat_dir, inp_file_ext)
        self.out_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, out_feat_dir, out_file_ext)

        self.gen_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, pred_feat_dir, out_file_ext)

        if self.GenTestList:
            test_id_list = data_utils.read_file_list(test_id_scp)
            self.inp_test_file_list = data_utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
            self.gen_test_file_list = data_utils.prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)

        if not self.encoder_decoder:
          self.tensorflow_models = TrainTensorflowModels(self.inp_dim, self.hidden_layer_size, self.out_dim, self.hidden_layer_type, self.model_dir, 
                                                output_type=self.output_layer_type, dropout_rate=self.dropout_rate,
                                                loss_function=self.loss_function, optimizer=self.optimizer)
        else:
            self.encoder_decoder_models = Train_Encoder_Decoder_Models(self.inp_dim,self.hidden_layer_size,self.out_dim,self.hidden_layer_type,output_type=self.output_layer_type,\
                                                                     dropout_rate=self.dropout_rate,loss_function=self.loss_function,optimizer=self.optimizer,\
                                                                     attention=self.attention,cbhg=self.cbhg)
    def normlize_data(self):
        ### normalize train data ###
        if os.path.isfile(self.inp_stats_file) and os.path.isfile(self.out_stats_file):
            self.inp_scaler = data_utils.load_norm_stats(self.inp_stats_file, self.inp_dim, method=self.inp_norm)
            self.out_scaler = data_utils.load_norm_stats(self.out_stats_file, self.out_dim, method=self.out_norm)
        else:
            print('preparing train_x, train_y from input and output feature files...')
            train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list, self.out_train_file_list,\
                    self.inp_dim, self.out_dim, sequential_training=True if self.sequential_training or self.encoder_decoder else False)

            print('computing norm stats for train_x...')
            inp_scaler = data_utils.compute_norm_stats(train_x, self.inp_stats_file, method=self.inp_norm)

            print('computing norm stats for train_y...')
            out_scaler = data_utils.compute_norm_stats(train_y, self.out_stats_file, method=self.out_norm)

    def train_tensorflow_model(self):
        print('preparing train_x, train_y from input and output feature files...')
                 #### load the data ####

        train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list, self.out_train_file_list,
                     self.inp_dim, self.out_dim, sequential_training=True if self.sequential_training or self.encoder_decoder else False)
                #### normalize the data ####
        data_utils.norm_data(train_x, self.inp_scaler, sequential_training=True if self.sequential_training or self.encoder_decoder else False)
        data_utils.norm_data(train_y, self.out_scaler, sequential_training=True if self.sequential_training or self.encoder_decoder else False)

        #### define the model ####
        if self.sequential_training:
           utt_length=train_flen["utt2framenum"].values()
           self.tensorflow_models.get_max_step(max(utt_length))
           self.tensorflow_models.define_sequence_model()

        elif self.encoder_decoder:
             utt_length=train_flen["utt2framenum"].values()
             super(Train_Encoder_Decoder_Models,self.encoder_decoder_models).__setattr__("max_step",max(utt_length))
             self.encoder_decoder_models.define_encoder_decoder()
        else:
            self.tensorflow_models.define_feedforward_model()

        #### train the model ####
        print('training...')
        if self.sequential_training:
            ### Train feedforward model ###
            self.tensorflow_models.train_sequence_model(train_x, train_y, batch_size=self.batch_size, num_of_epochs=self.num_of_epochs, shuffle_data=self.shuffle_data,utt_length=utt_length)

        elif self.encoder_decoder:
            self.encoder_decoder_models.train_encoder_decoder_model(train_x,train_y,batch_size=self.batch_size,num_of_epochs=self.num_of_epochs,shuffle_data=True,utt_length=utt_length)
        else:
            self.tensorflow_models.train_feedforward_model(train_x, train_y, batch_size=self.batch_size, num_of_epochs=self.num_of_epochs, shuffle_data=self.shuffle_data)

    def test_tensorflow_model(self):
         #### load the data ####
         print('preparing test_x from input feature files...')
         test_x, test_flen = data_utils.read_test_data_from_file_list(self.inp_test_file_list, self.inp_dim)

          #### normalize the data ####
         data_utils.norm_data(test_x, self.inp_scaler)
          #### compute predictions ####
         if self.encoder_decoder:
             self.encoder_decoder_models.predict(test_x,self.out_scaler,self.gen_test_file_list)
         else:
             self.tensorflow_models.predict(test_x, self.out_scaler, self.gen_test_file_list, self.sequential_training)

    def main_function(self):
         ### Implement each module ###
         if self.NORMDATA:
            self.normlize_data()

         if self.TRAINMODEL:
            self.train_tensorflow_model()

         if self.TESTMODEL:
            self.test_tensorflow_model()

if __name__=="__main__":

    if len(sys.argv) != 2:
        print('usage: python run_tensorflow_with_merlin_io.py [config file name]')
        sys.exit(1)

    # create a configuration instance
    # and get a short name for this instance
    cfg = configuration.configuration()

    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)

    print("--- Job started ---")
    start_time = time.time()

    # main function
    tensorflow_instance = TensorflowClass(cfg)
  # except:
   #         print "inp stats file is %s"%cfg.inp_stats_file
    #        sys.exit(0)
    tensorflow_instance.main_function()

    (m, s) = divmod(int(time.time() - start_time), 60)
    print("--- Job completion time: %d min. %d sec ---" % (m, s))

    sys.exit(0)
