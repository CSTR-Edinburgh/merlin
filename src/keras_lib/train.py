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

import os, sys
import random
import numpy as np

from io_funcs.binary_io import BinaryIOCollection

from keras_lib.model import kerasModels
from keras_lib import data_utils

class TrainKerasModels(kerasModels):

    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam', rnn_params=None):

        kerasModels.__init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

        #### TODO: Find a good way to pass below params ####
        self.merge_size   = rnn_params['merge_size']
        self.seq_length   = rnn_params['seq_length']
        self.bucket_range = rnn_params['bucket_range']

        self.stateful = rnn_params['stateful']

        pass;

    def train_feedforward_model(self, train_x, train_y, valid_x, valid_y, batch_size=256, num_of_epochs=10, shuffle_data=True):
        self.model.fit(train_x, train_y, batch_size=batch_size, epochs=num_of_epochs, shuffle=shuffle_data)

    def train_sequence_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size=1, num_of_epochs=10, shuffle_data=True, training_algo=1):
        if batch_size == 1:
            self.train_recurrent_model_batchsize_one(train_x, train_y, valid_x, valid_y, num_of_epochs, shuffle_data)
        else:
            self.train_recurrent_model(train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data, training_algo)

    def train_recurrent_model_batchsize_one(self, train_x, train_y, valid_x, valid_y, num_of_epochs, shuffle_data):
        ### if batch size is equal to 1 ###
        train_idx_list = list(train_x.keys())
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_idx_list)

        train_file_number = len(train_idx_list)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            for file_name in train_idx_list:
                temp_train_x = train_x[file_name]
                temp_train_y = train_y[file_name]
                temp_train_x = np.reshape(temp_train_x, (1, temp_train_x.shape[0], self.n_in))
                temp_train_y = np.reshape(temp_train_y, (1, temp_train_y.shape[0], self.n_out))
                self.model.train_on_batch(temp_train_x, temp_train_y)
                #self.model.fit(temp_train_x, temp_train_y, epochs=1, shuffle=False, verbose=0)
                file_num += 1
                data_utils.drawProgressBar(file_num, train_file_number)

            sys.stdout.write("\n")

    def train_recurrent_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data, training_algo):
        ### if batch size more than 1 ###
        if training_algo == 1:
            self.train_padding_model(train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data)
        elif training_algo == 2:
            self.train_bucket_model(train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data)
        elif training_algo == 3:
            self.train_split_model(train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data)
        else:
            print("Choose training algorithm for batch training with RNNs:")
            print("1. Padding model -- pad utterances with zeros to maximum sequence length")
            print("2. Bucket model  -- form buckets with minimum and maximum sequence length")
            print("3. Split model   -- split utterances to a fixed sequence length")
            sys.exit(1)

    
    def train_padding_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        ### Method 1 ###
        train_id_list = list(train_flen['utt2framenum'].keys())
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_id_list)

        train_file_number = len(train_id_list)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            while file_num < train_file_number:
                train_idx_list = train_id_list[file_num: file_num + batch_size]
                seq_len_arr    = [train_flen['utt2framenum'][filename] for filename in train_idx_list]
                max_seq_length = max(seq_len_arr)
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=max_seq_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=max_seq_length)
                self.model.train_on_batch(temp_train_x, temp_train_y)
                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))
    
    def train_bucket_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        ### Method 2 ###
        train_fnum_list  = np.array(list(train_flen['framenum2utt'].keys()))
        train_range_list = list(range(min(train_fnum_list), max(train_fnum_list)+1, self.bucket_range))
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_range_list)

        train_file_number = len(train_x)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            for frame_num in train_range_list:
                min_seq_length = frame_num
                max_seq_length = frame_num+self.bucket_range
                sub_train_list = train_fnum_list[(train_fnum_list>=min_seq_length) & (train_fnum_list<max_seq_length)]
                if len(sub_train_list)==0:
                    continue;
                train_idx_list = sum([train_flen['framenum2utt'][framenum] for framenum in sub_train_list], [])
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=max_seq_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=max_seq_length)
                self.model.fit(temp_train_x, temp_train_y, batch_size=batch_size, shuffle=False, epochs=1, verbose=0)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))

    def train_split_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        ### Method 3 ###
        train_id_list = list(train_flen['utt2framenum'].keys())
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_id_list)

        train_file_number = len(train_id_list)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            while file_num < train_file_number:
                train_idx_list = train_id_list[file_num: file_num + batch_size]
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, seq_length=self.seq_length, merge_size=self.merge_size)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, seq_length=self.seq_length, merge_size=self.merge_size)
    
                self.model.train_on_batch(temp_train_x, temp_train_y)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))

    def train_split_model_keras_version(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        """This function is not used as of now 
        """
        ### Method 3 ###
        temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, seq_length=self.seq_length, merge_size=self.merge_size, shuffle_data=shuffle_data)
        print(("Input shape: "+str(temp_train_x.shape)))
        
        temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, seq_length=self.seq_length, merge_size=self.merge_size, shuffle_data=shuffle_data)
        print(("Output shape: "+str(temp_train_y.shape)))
        
        if self.stateful:
            temp_train_x, temp_train_y = data_utils.get_stateful_data(temp_train_x, temp_train_y, batch_size)
    
        self.model.fit(temp_train_x, temp_train_y, batch_size=batch_size, epochs=num_of_epochs)
    
    def train_bucket_model_without_padding(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        """This function is not used as of now
        """
        ### Method 4 ###
        train_count_list = list(train_flen['framenum2utt'].keys())
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_count_list)

        train_file_number = len(train_x)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            for sequence_length in train_count_list:
                train_idx_list = train_flen['framenum2utt'][sequence_length]
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=sequence_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=sequence_length)
                self.model.fit(temp_train_x, temp_train_y, batch_size=batch_size, epochs=1, verbose=0)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            sys.stdout.write("\n")

    def get_validation_error(self, valid_x, valid_y, sequential_training=True, stateful=False):
        valid_id_list = list(valid_x.keys())
        valid_id_list.sort()

        valid_error = 0.0
        valid_file_number = len(valid_id_list)
        for utt_index in range(valid_file_number):
            temp_valid_x = valid_x[valid_id_list[utt_index]]
            temp_valid_y = valid_y[valid_id_list[utt_index]]
            num_of_rows = temp_valid_x.shape[0]

            if stateful:
                temp_valid_x = data_utils.get_stateful_input(temp_valid_x, self.seq_length, self.batch_size)
            elif sequential_training:
                temp_valid_x = np.reshape(temp_valid_x, (1, num_of_rows, self.n_in))

            predictions = self.model.predict(temp_valid_x)
            if sequential_training:
                predictions = np.reshape(predictions, (num_of_rows, self.n_out))

            valid_error += np.mean(np.sum((predictions - temp_valid_y) ** 2, axis=1))

        valid_error = valid_error/valid_file_number

        return valid_error

    def predict(self, test_x, out_scaler, gen_test_file_list, sequential_training=False, stateful=False):
        #### compute predictions ####
        io_funcs = BinaryIOCollection()

        test_id_list = list(test_x.keys())
        test_id_list.sort()

        test_file_number = len(test_id_list)
        print("generating features on held-out test data...")
        for utt_index in range(test_file_number):
            gen_test_file_name = gen_test_file_list[utt_index]
            temp_test_x        = test_x[test_id_list[utt_index]]
            num_of_rows        = temp_test_x.shape[0]

            if stateful:
                temp_test_x = data_utils.get_stateful_input(temp_test_x, self.seq_length, self.batch_size)
            elif sequential_training:
                temp_test_x = np.reshape(temp_test_x, (1, num_of_rows, self.n_in))

            predictions = self.model.predict(temp_test_x)
            if sequential_training:
                predictions = np.reshape(predictions, (num_of_rows, self.n_out))

            data_utils.denorm_data(predictions, out_scaler)

            io_funcs.array_to_binary_file(predictions, gen_test_file_name)
            data_utils.drawProgressBar(utt_index+1, test_file_number)

        sys.stdout.write("\n")
