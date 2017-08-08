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
import time
import random
import numpy as np

from sklearn import preprocessing

from io_funcs.binary_io import BinaryIOCollection

############################
##### Memory variables #####
############################

UTT_BUFFER_SIZE   =   10000
FRAME_BUFFER_SIZE = 3000000


def read_data_from_file_list(inp_file_list, out_file_list, inp_dim, out_dim, sequential_training=True): 
    io_funcs = BinaryIOCollection()

    num_of_utt = len(inp_file_list)

    file_length_dict = {'framenum2utt':{}, 'utt2framenum':{}}

    if sequential_training:
        temp_set_x = {}
        temp_set_y = {}
    else:
        temp_set_x = np.empty((FRAME_BUFFER_SIZE, inp_dim))
        temp_set_y = np.empty((FRAME_BUFFER_SIZE, out_dim))
     
    ### read file by file ###
    current_index = 0
    for i in xrange(num_of_utt):    
        inp_file_name = inp_file_list[i]
        out_file_name = out_file_list[i]
        inp_features, inp_frame_number = io_funcs.load_binary_file_frame(inp_file_name, inp_dim)
        out_features, out_frame_number = io_funcs.load_binary_file_frame(out_file_name, out_dim)

        base_file_name = os.path.basename(inp_file_name).split(".")[0]

        if abs(inp_frame_number-out_frame_number)>5:
            print 'the number of frames in input and output features are different: %d vs %d (%s)' %(inp_frame_number, out_frame_number, base_file_name)
            sys.exit(0)
        else:
            frame_number = min(inp_frame_number, out_frame_number)
  
        if sequential_training:
            temp_set_x[base_file_name] = inp_features[0:frame_number] 
            temp_set_y[base_file_name] = out_features[0:frame_number] 
        else:
            temp_set_x[current_index:current_index+frame_number, ] = inp_features[0:frame_number]
            temp_set_y[current_index:current_index+frame_number, ] = out_features[0:frame_number]
            current_index += frame_number
        
        if frame_number not in file_length_dict['framenum2utt']:
            file_length_dict['framenum2utt'][frame_number] = [base_file_name]
        else:
            file_length_dict['framenum2utt'][frame_number].append(base_file_name)

        file_length_dict['utt2framenum'][base_file_name] = frame_number
    
        drawProgressBar(i+1, num_of_utt)
        
    sys.stdout.write("\n")
        
    if not sequential_training:
        temp_set_x = temp_set_x[0:current_index, ]
        temp_set_y = temp_set_y[0:current_index, ]
    
    return temp_set_x, temp_set_y, file_length_dict

def read_test_data_from_file_list(inp_file_list, inp_dim, sequential_training=True): 
    io_funcs = BinaryIOCollection()

    num_of_utt = len(inp_file_list)

    file_length_dict = {'framenum2utt':{}, 'utt2framenum':{}}

    if sequential_training:
        temp_set_x = {}
    else:
        temp_set_x = np.empty((FRAME_BUFFER_SIZE, inp_dim))
     
    ### read file by file ###
    current_index = 0
    for i in xrange(num_of_utt):    
        inp_file_name = inp_file_list[i]
        inp_features, frame_number = io_funcs.load_binary_file_frame(inp_file_name, inp_dim)

        base_file_name = os.path.basename(inp_file_name).split(".")[0]

        if sequential_training:
            temp_set_x[base_file_name] = inp_features
        else:
            temp_set_x[current_index:current_index+frame_number, ] = inp_features[0:frame_number]
            current_index += frame_number
        
        if frame_number not in file_length_dict['framenum2utt']:
            file_length_dict['framenum2utt'][frame_number] = [base_file_name]
        else:
            file_length_dict['framenum2utt'][frame_number].append(base_file_name)

        file_length_dict['utt2framenum'][base_file_name] = frame_number
    
        drawProgressBar(i+1, num_of_utt)
        
    sys.stdout.write("\n")
        
    if not sequential_training:
        temp_set_x = temp_set_x[0:current_index, ]
    
    return temp_set_x, file_length_dict

def transform_data_to_3d_matrix(data, seq_length=200, max_length=0, merge_size=1, shuffle_data = True, shuffle_type = 1, padding="right"):
    num_of_utt = len(data)
    feat_dim   = data[data.keys()[0]].shape[1]

    if max_length > 0:
        temp_set = np.zeros((num_of_utt, max_length, feat_dim))
        
        ### read file by file ###
        current_index = 0
        for base_file_name, in_features in data.iteritems():
            frame_number = min(in_features.shape[0], max_length)
            if padding=="right":
                temp_set[current_index, 0:frame_number, ] = in_features
            else:
                temp_set[current_index, -frame_number:, ] = in_features
            current_index += 1

    else:
        temp_set = np.zeros((FRAME_BUFFER_SIZE, feat_dim))

        train_idx_list = data.keys()
        train_idx_list.sort()
        
        if shuffle_data:
            if shuffle_type == 1:
                train_idx_list = shuffle_file_list(train_idx_list)
            elif shuffle_type == 2:
                train_idx_list = shuffle_file_list(train_idx_list, shuffle_type=2, merge_size=merge_size)
        
        ### read file by file ###
        current_index = 0
        for file_number in xrange(num_of_utt):
            base_file_name = train_idx_list[file_number]
            in_features    = data[base_file_name]
            frame_number   = in_features.shape[0]
            
            temp_set[current_index:current_index+frame_number, ] = in_features
            current_index += frame_number
    
            if (file_number+1)%merge_size == 0:
                current_index = seq_length * (int(np.ceil(float(current_index)/float(seq_length))))
            
        
        num_of_samples = int(np.ceil(float(current_index)/float(seq_length)))
    
        temp_set = temp_set[0: num_of_samples*seq_length, ]
        temp_set = temp_set.reshape(-1, seq_length, feat_dim)
     
    return temp_set

def read_and_transform_data_from_file_list(in_file_list, dim, seq_length=200, merge_size=1):
    io_funcs = BinaryIOCollection()

    num_of_utt = len(in_file_list)

    temp_set = np.zeros((FRAME_BUFFER_SIZE, dim))

    ### read file by file ###
    current_index = 0
    for i in range(num_of_utt):
        in_file_name = in_file_list[i]
        in_features, frame_number = io_funcs.load_binary_file_frame(in_file_name, dim)
        base_file_name            = os.path.basename(in_file_name).split(".")[0]

        temp_set[current_index:current_index+frame_number, ] = in_features
        current_index += frame_number

        if (i+1)%merge_size == 0:
            current_index = seq_length * (int(np.ceil(float(current_index)/float(seq_length))))
            
        drawProgressBar(i+1, num_of_utt)
        
    sys.stdout.write("\n")

    num_of_samples = int(np.ceil(float(current_index)/float(seq_length)))

    temp_set = temp_set[0: num_of_samples*seq_length, ]
    temp_set = temp_set.reshape(num_of_samples, seq_length)

    return temp_set

def merge_data(train_x, train_y, merge_size):
    temp_train_x = {}
    temp_train_y = {}

    train_id_list     = train_x.keys()
    train_file_number = len(train_id_list)
    train_id_list.sort()

    inp_dim = train_x[train_id_list[0]].shape[1]
    out_dim = train_y[train_id_list[0]].shape[1]
    
    merged_features_x = np.zeros((0, inp_dim))
    merged_features_y = np.zeros((0, out_dim))
    new_file_count = 0
    for file_index in xrange(1, train_file_number+1):
        inp_features      = train_x[train_id_list[file_index-1]]
        out_features      = train_y[train_id_list[file_index-1]]
        merged_features_x = np.vstack((merged_features_x, inp_features))
        merged_features_y = np.vstack((merged_features_y, out_features))
        
        if file_index % merge_size == 0 or file_index==train_file_number:
            base_file_name = "new_utterance_%04d" % (new_file_count)
            temp_train_x[base_file_name] = merged_features_x
            temp_train_y[base_file_name] = merged_features_y
            new_file_count += 1
            merged_features_x = np.zeros((0, inp_dim))
            merged_features_y = np.zeros((0, out_dim))

    return temp_train_x, temp_train_y

def shuffle_file_list(train_idx_list, shuffle_type=1, merge_size=5):
    ### shuffle train id list ###
    random.seed(271638)
    train_file_number = len(train_idx_list)
    
    if shuffle_type==1:  ## shuffle by sentence
        random.shuffle(train_idx_list)
        return train_idx_list
     
    elif shuffle_type==2:  ## shuffle by a group of sentences
        id_numbers = range(0, train_file_number, merge_size)
        random.shuffle(id_numbers)
        new_train_idx_list = []
        for i in xrange(len(id_numbers)):
            new_train_idx_list += train_idx_list[id_numbers[i]:id_numbers[i]+merge_size]
        return new_train_idx_list

def get_stateful_data(train_x, train_y, batch_size):
    num_of_batches = int(train_x.shape[0]/batch_size) 
    train_x   = train_x[0: num_of_batches*batch_size, ]
    train_y   = train_y[0: num_of_batches*batch_size, ]

    stateful_seq = np.zeros(num_of_batches*batch_size, dtype="int32")
    for i in xrange(num_of_batches):
        stateful_seq[i*batch_size:(i+1)*batch_size] = np.array(range(batch_size))*num_of_batches+i

    temp_train_x   = train_x[stateful_seq]
    temp_train_y   = train_y[stateful_seq]

    return temp_train_x, temp_train_y

def get_stateful_input(test_x, seq_length, batch_size=1):
    [n_frames, n_dim] = test_x.shape

    num_of_samples = batch_size*seq_length
    num_of_batches = int(n_frames/num_of_samples) + 1
    new_data_size  = num_of_batches*num_of_samples

    temp_test_x = np.zeros((new_data_size, n_dim))
    temp_test_x[0: n_frames, ] = test_x

    temp_test_x = temp_test_x.reshape(-1, seq_length, n_dim)

    return temp_test_x
    
def compute_norm_stats(data, stats_file, method="MVN"):
    #### normalize training data ####
    io_funcs = BinaryIOCollection()

    if method=="MVN":
        scaler = preprocessing.StandardScaler().fit(data)
        norm_matrix = np.vstack((scaler.mean_, scaler.scale_))
    elif method=="MINMAX":
        scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99)).fit(data)
        norm_matrix = np.vstack((scaler.min_, scaler.scale_))
    
    print norm_matrix.shape
    io_funcs.array_to_binary_file(norm_matrix, stats_file)

    return scaler

def load_norm_stats(stats_file, dim, method="MVN"):
    #### load norm stats ####
    io_funcs = BinaryIOCollection()

    norm_matrix, frame_number = io_funcs.load_binary_file_frame(stats_file, dim)
    assert frame_number==2

    if method=="MVN":
        scaler = preprocessing.StandardScaler()
        scaler.mean_  = norm_matrix[0, :]
        scaler.scale_ = norm_matrix[1, :]
    elif method=="MINMAX":
        scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
        scaler.min_   = norm_matrix[0, :]
        scaler.scale_ = norm_matrix[1, :]

    return scaler

def norm_data(data, scaler, sequential_training=True):
    if scaler is None:
        return;
    
    #### normalize data ####
    if not sequential_training:
        data = scaler.transform(data) 
    else:
        for filename, features in data.iteritems():
            data[filename] = scaler.transform(features)

def denorm_data(data, scaler):
    if scaler is None:
        return;
    
    #### de-normalize data ####
    data = scaler.inverse_transform(data) 
    
def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

def read_file_list(file_name):
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    return  file_lists

def print_status(i, length): 
    pr = int(float(i)/float(length)*100)
    st = int(float(pr)/7)
    sys.stdout.write(("\r%d/%d ")%(i,length)+("[ %d"%pr+"% ] <<< ")+('='*st)+(''*(100-st)))
    sys.stdout.flush()
    
def drawProgressBar(indx, length, barLen = 20):
    percent = float(indx)/length
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[%s] <<< %d/%d (%d%%)" % (progress, indx, length, percent * 100))
    sys.stdout.flush()
