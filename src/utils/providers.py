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

import os, sys
import numpy, theano, random
from io_funcs.binary_io import BinaryIOCollection
import logging
from frontend.label_normalisation import HTSLabelNormalisation

class ListDataProvider(object):
    """ This class provides an interface to load data into CPU/GPU memory utterance by utterance or block by block.

    In speech synthesis, usually we are not able to load all the training data/evaluation data into RAMs, we will do the following three steps:

    - Step 1: a data provide will load part of the data into a buffer

    - Step 2: training a DNN by using the data from the buffer

    - Step 3: Iterate step 1 and 2 until all the data are used for DNN training. Until now, one epoch of DNN training is finished.

    The utterance-by-utterance data loading will be useful when sequential training is used, while block-by-block loading will be used when the order of frames is not important.

    This provide assumes binary format with float32 precision without any header (e.g. HTK header).

    """
    def __init__(self, x_file_list, y_file_list, dur_file_list=None, n_ins=0, n_outs=0, buffer_size=500000, sequential=False, network_type=None, shuffle=False):
        """Initialise a data provider

        :param x_file_list: list of file names for the input files to DNN
        :type x_file_list: python list
        :param y_file_list: list of files for the output files to DNN
        :param n_ins: the dimensionality for input feature
        :param n_outs: the dimensionality for output features
        :param buffer_size: the size of the buffer, indicating the number of frames in the buffer. The value depends on the memory size of RAM/GPU.
        :param shuffle: True/False. To indicate whether the file list will be shuffled. When loading data block by block, the data in the buffer will be shuffle no matter this value is True or False.
        """

        self.logger = logging.getLogger("ListDataProvider")

        self.n_ins = n_ins
        self.n_outs = n_outs

        self.buffer_size = buffer_size

        self.sequential = sequential
        self.network_type = network_type

        self.rnn_batch_training = False
        self.reshape_io = False

        #remove potential empty lines and end of line signs

        try:
            assert len(x_file_list) > 0
        except AssertionError:
            self.logger.critical('first list is empty')
            raise

        try:
            assert len(y_file_list) > 0
        except AssertionError:
            self.logger.critical('second list is empty')
            raise

        try:
            assert len(x_file_list) == len(y_file_list)
        except AssertionError:
            self.logger.critical('two lists are of differing lengths: %d versus %d',len(x_file_list),len(y_file_list))
            raise

        if dur_file_list:
            try:
                assert len(x_file_list) == len(dur_file_list)
            except AssertionError:
                self.logger.critical('two lists are of differing lengths: %d versus %d',len(x_file_list),len(y_file_list))
                raise

        self.x_files_list = x_file_list
        self.y_files_list = y_file_list
        self.dur_files_list = dur_file_list

        self.logger.debug('first  list of items from ...%s to ...%s' % (self.x_files_list[0].rjust(20)[-20:],self.x_files_list[-1].rjust(20)[-20:]) )
        self.logger.debug('second list of items from ...%s to ...%s' % (self.y_files_list[0].rjust(20)[-20:],self.y_files_list[-1].rjust(20)[-20:]) )

        if shuffle:
            random.seed(271638)
            random.shuffle(self.x_files_list)
            random.seed(271638)
            random.shuffle(self.y_files_list)
            if self.dur_files_list:
                random.seed(271638)
                random.shuffle(self.dur_files_list)

        self.file_index = 0
        self.list_size = len(self.x_files_list)

        self.remain_data_x = numpy.empty((0, self.n_ins))
        self.remain_data_y = numpy.empty((0, self.n_outs))
        self.remain_frame_number = 0

        self.end_reading = False

        self.logger.debug('initialised')

    def __iter__(self):
        return self

    def reset(self):
        """When all the files in the file list have been used for DNN training, reset the data provider to start a new epoch.

        """
        self.file_index = 0
        self.end_reading = False

        self.remain_frame_number = 0
        
        self.bucket_index = 0
        self.bucket_file_index = 0
        self.current_bucket_size = 0

        self.logger.debug('reset')

    def make_shared(self, data_set, data_name):
        """To make data shared for theano implementation. If you want to know why we make it shared, please refer the theano documentation: http://deeplearning.net/software/theano/library/compile/shared.html

        :param data_set: normal data in CPU memory
        :param data_name: indicate the name of the data (e.g., 'x', 'y', etc)
        :returns: shared dataset -- data_set
        """
        data_set = theano.shared(numpy.asarray(data_set, dtype=theano.config.floatX), name=data_name, borrow=True)

        return  data_set

    def set_rnn_params(self, training_algo=1, batch_size=25, seq_length=200, merge_size=1, bucket_range=100):
        # get file lengths
        self.get_file_lengths()

        # set training algo
        self.training_algo = training_algo

        # set batch size
        self.batch_size = batch_size

        # set RNN batch training True
        self.rnn_batch_training = True

        # set params for each training algo
        if(self.training_algo == 1):
            self.merge_size = 1
        elif(self.training_algo == 2):
            self.merge_size = 1
            self.bucket_index = 0
            self.bucket_file_index = 0
            self.current_bucket_size = 0
            self.bucket_range = bucket_range
            self.x_frame_list = numpy.array(list(self.file_length_dict['framenum2utt'].keys()))
            self.list_of_buckets = list(range(min(self.x_frame_list), max(self.x_frame_list)+1, self.bucket_range))
        elif(self.training_algo == 3):
            self.seq_length = seq_length
            self.merge_size = merge_size
        else:
            self.logger.critical("Choose training algorithm for batch training with RNNs:")
            self.logger.critical("1. Padding model -- pad utterances with zeros to maximum sequence length")
            self.logger.critical("2. Bucket model  -- form buckets with minimum and maximum sequence length")
            self.logger.critical("3. Split model   -- split utterances to a fixed sequence length")
            sys.exit(1)

    def reshape_input_output(self):
        self.reshape_io = True

    def get_file_lengths(self):
        io_funcs = BinaryIOCollection()

        self.file_length_dict = {'framenum2utt':{}, 'utt2framenum':{}, 'utt2index':{}}

        ### read file by file ###
        while True:
            if  self.file_index >= self.list_size:
                self.end_reading = True
                self.file_index = 0
                break

            in_features, lab_frame_number = io_funcs.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
            out_features, out_frame_number = io_funcs.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)
         
            base_file_name = os.path.basename(self.x_files_list[self.file_index]).split('.')[0]
            if abs(lab_frame_number - out_frame_number) < 5:    ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
                frame_number = min(lab_frame_number, out_frame_number)
            else:
                self.logger.critical("the number of frames in label and acoustic features are different: %d vs %d (%s)" %(lab_frame_number, out_frame_number, base_file_name))
                raise

            if frame_number not in self.file_length_dict['framenum2utt']:
                self.file_length_dict['framenum2utt'][frame_number] = [base_file_name]
            else:
                self.file_length_dict['framenum2utt'][frame_number].append(base_file_name)

            self.file_length_dict['utt2framenum'][base_file_name] = frame_number
            self.file_length_dict['utt2index'][base_file_name] = self.file_index
            self.file_index += 1

        self.reset()

    def set_seq_length_from_current_batch(self):
        temp_list = []
        for indx in range(self.batch_size):
            if  self.file_index+indx >= self.list_size:
                break
            base_file_name = os.path.basename(self.x_files_list[self.file_index+indx]).split('.')[0]
            temp_list.append(self.file_length_dict['utt2framenum'][base_file_name])

        self.seq_length = max(temp_list)

    def get_next_bucket(self):
        min_seq_length = self.list_of_buckets[self.bucket_index]
        max_seq_length = self.list_of_buckets[self.bucket_index] + self.bucket_range
        
        current_bucket = self.x_frame_list[(self.x_frame_list >= min_seq_length) & (self.x_frame_list < max_seq_length)]
        self.current_bucket_list  = sum([self.file_length_dict['framenum2utt'][framenum] for framenum in current_bucket], [])
        
        self.bucket_file_index   = 0  
        self.current_bucket_size = len(self.current_bucket_list)
        
        self.seq_length   = max_seq_length
        self.bucket_index = self.bucket_index + 1

    def set_s2s_division(self, linguistic_feats_file=None, frame_length=4):
        self.MLU_div = {}
        in_f = open(linguistic_feats_file, 'r')
        for newline in in_f.readlines():
            temp_list = newline.strip().split()
            unit  = temp_list[0]
            feat1 = temp_list[1][1:-1].split('-')
            feat2 = temp_list[2][1:-1].split('-')

            self.MLU_div[unit] = [int(feat1[0]), int(feat1[1]), int(feat2[0]), int(feat2[1])]
       
        syl_length = (self.MLU_div['syl'][1] - self.MLU_div['syl'][0])+ (self.MLU_div['syl'][3] - self.MLU_div['syl'][2])
        phone_length = (self.MLU_div['phone'][1] - self.MLU_div['phone'][0]) + (self.MLU_div['phone'][3] - self.MLU_div['phone'][2])
        self.MLU_div['length'] = [0, syl_length, syl_length+phone_length, syl_length+phone_length+frame_length]

        return self.MLU_div

    def load_one_partition(self):
        if self.sequential == True:
            if not self.network_type or self.network_type=="RNN":
                if self.rnn_batch_training:
                    shared_set_xy, temp_set_x, temp_set_y = self.load_next_batch()
                else:
                    shared_set_xy, temp_set_x, temp_set_y = self.load_next_utterance()
            elif self.network_type=="CTC":
                shared_set_xy, temp_set_x, temp_set_y = self.load_next_utterance_CTC()
            elif self.network_type=="S2S":
                shared_set_xyd, temp_set_x, temp_set_y, temp_set_d, temp_set_af = self.load_next_utterance_S2SML()
                return  shared_set_xyd, temp_set_x, temp_set_y, temp_set_d, temp_set_af
            else:
                logger.critical("Unknown network type: %s \n Please use one of the following: DNN, RNN, S2S, CTC\n" %(self.network_type))
                sys.exit(1)
        else:
            shared_set_xy, temp_set_x, temp_set_y = self.load_next_partition()

        return  shared_set_xy, temp_set_x, temp_set_y

    def load_next_batch(self):
        io_funcs = BinaryIOCollection()

        ## set sequence length for batch training 
        if(self.training_algo == 1):
            # set seq length to maximum seq length from current batch
            self.set_seq_length_from_current_batch()
        elif(self.training_algo == 2):
            # set seq length to maximum seq length from current bucket
            while not self.current_bucket_size:
                self.get_next_bucket()
        elif(self.training_algo == 3):
            # seq length is set based on default/user configuration 
            pass;
            
        temp_set_x = numpy.zeros((self.buffer_size, self.n_ins))
        temp_set_y = numpy.zeros((self.buffer_size, self.n_outs))

        ### read file by file ###
        current_index = 0
        while True:
            if current_index >= self.buffer_size:
                print('buffer size reached by file index %d' %(self.file_index))
                break

            if self.training_algo == 2:
                # choose utterance from current bucket list
                base_file_name = self.current_bucket_list[self.bucket_file_index]
                self.utt_index = self.file_length_dict['utt2index'][base_file_name] 
            else: 
                # choose utterance randomly from current file list 
                #self.utt_index = numpy.random.randint(self.list_size)
                ## choose utterance in serial order
                self.utt_index = self.file_index 
                base_file_name = os.path.basename(self.x_files_list[self.utt_index]).split('.')[0]

            in_features, lab_frame_number = io_funcs.load_binary_file_frame(self.x_files_list[self.utt_index], self.n_ins)
            out_features, out_frame_number = io_funcs.load_binary_file_frame(self.y_files_list[self.utt_index], self.n_outs)
         
            frame_number = self.file_length_dict['utt2framenum'][base_file_name]

            temp_set_x[current_index:current_index+frame_number, ] = in_features
            temp_set_y[current_index:current_index+frame_number, ] = out_features
            current_index += frame_number

            if((self.file_index+1)%self.merge_size == 0):
                num_of_samples = int(numpy.ceil(float(current_index)/float(self.seq_length)))
                current_index = self.seq_length * num_of_samples
                
            self.file_index += 1
            
            # break for any of the below conditions
            if self.training_algo == 2:
                self.bucket_file_index += 1
                if(self.bucket_file_index >= self.current_bucket_size):
                    self.current_bucket_size = 0
                    break;
                if(self.bucket_file_index%self.batch_size==0):
                    break;
            else:  
                if(self.file_index%self.batch_size==0) or (self.file_index >= self.list_size):
                    break
        
        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0
        
        num_of_samples = int(numpy.ceil(float(current_index)/float(self.seq_length)))

        temp_set_x = temp_set_x[0: num_of_samples*self.seq_length, ]
        temp_set_y = temp_set_y[0: num_of_samples*self.seq_length, ]
        
        temp_set_x = temp_set_x.reshape(num_of_samples, self.seq_length, self.n_ins)
        temp_set_y = temp_set_y.reshape(num_of_samples, self.seq_length, self.n_outs)

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')

        shared_set_xy = (shared_set_x, shared_set_y)

        return shared_set_xy, temp_set_x, temp_set_y
        
    def load_next_utterance(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).

        """

        temp_set_x = numpy.empty((self.buffer_size, self.n_ins))
        temp_set_y = numpy.empty((self.buffer_size, self.n_outs))

        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        frame_number = lab_frame_number
        if abs(lab_frame_number - out_frame_number) < 5:    ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
            if lab_frame_number > out_frame_number:
                frame_number = out_frame_number
        else:
            base_file_name = os.path.basename(self.x_files_list[self.file_index]).split('.')[0]
            self.logger.critical("the number of frames in label and acoustic features are different: %d vs %d (%s)" %(lab_frame_number, out_frame_number, base_file_name))
            raise

        temp_set_y = out_features[0:frame_number, ]
        temp_set_x = in_features[0:frame_number, ]

        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0
       
        # reshape input-output
        if self.reshape_io:
            temp_set_x = numpy.reshape(temp_set_x, (1, temp_set_x.shape[0], self.n_ins))
            temp_set_y = numpy.reshape(temp_set_y, (1, temp_set_y.shape[0], self.n_outs))
        
            temp_set_x = numpy.array(temp_set_x, 'float32')
            temp_set_y = numpy.array(temp_set_y, 'float32')

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')

        shared_set_xy = (shared_set_x, shared_set_y)

        return shared_set_xy, temp_set_x, temp_set_y

    def load_next_utterance_S2S(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).

        """

        temp_set_x = numpy.empty((self.buffer_size, self.n_ins))
        temp_set_y = numpy.empty((self.buffer_size, self.n_outs))

        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        temp_set_x = in_features[0:lab_frame_number, ]
        temp_set_y = out_features[0:out_frame_number, ]

        if not self.dur_files_list:
            dur_frame_number = out_frame_number
            dur_features = numpy.array([dur_frame_number])
        else:
            dur_features, dur_frame_number = io_fun.load_binary_file_frame(self.dur_files_list[self.file_index], 1)
            assert sum(dur_features) == out_frame_number
           
        dur_features = numpy.reshape(dur_features, (-1, ))
        temp_set_d = dur_features.astype(int)   
        
        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')
        shared_set_d = theano.shared(numpy.asarray(temp_set_d, dtype='int32'), name='d', borrow=True)

        shared_set_xyd = (shared_set_x, shared_set_y, shared_set_d)

        return shared_set_xyd, temp_set_x, temp_set_y, temp_set_d

    def load_next_utterance_S2SML(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).
        
        """
        
        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)
        dur_features, dur_frame_number = io_fun.load_binary_file_frame(self.dur_files_list[self.file_index], 1)
      
        ### MLU features sub-division ###
        temp_set_MLU = in_features[0:lab_frame_number, ]
        temp_set_y   = out_features[0:out_frame_number, ]
      
        temp_set_phone = numpy.concatenate([temp_set_MLU[:, self.MLU_div['phone'][0]: self.MLU_div['phone'][1]], temp_set_MLU[:, self.MLU_div['phone'][2]: self.MLU_div['phone'][3]]], axis = 1)
        temp_set_syl   = numpy.concatenate([temp_set_MLU[:, self.MLU_div['syl'][0]: self.MLU_div['syl'][1]], temp_set_MLU[:, self.MLU_div['syl'][2]: self.MLU_div['syl'][3]]], axis = 1)
        temp_set_word  = numpy.concatenate([temp_set_MLU[:, self.MLU_div['word'][0]: self.MLU_div['word'][1]], temp_set_MLU[:, self.MLU_div['word'][2]: self.MLU_div['word'][3] ]], axis = 1)
        
        ### duration array sub-division ###
        dur_features = numpy.reshape(dur_features, (-1, ))
        temp_set_d   = dur_features.astype(int)   
        dur_word_syl = temp_set_d[0: -lab_frame_number]    
        
        num_ph    = lab_frame_number
        num_syl   = (numpy.where(numpy.cumsum(dur_word_syl[::-1])==lab_frame_number)[0][0] + 1)
        num_words = len(dur_word_syl) - num_syl 
        
        temp_set_dur_phone = temp_set_d[-num_ph:] 
        temp_set_dur_word  = dur_word_syl[0: num_words]
        temp_set_dur_syl   = dur_word_syl[num_words: ]
        
        ### additional feature matrix (syllable+phone+frame=432) ###
        num_frames = sum(temp_set_dur_phone)
        temp_set_af = numpy.empty((num_frames, self.MLU_div['length'][-1]))
        
        temp_set_af[0: num_syl, self.MLU_div['length'][0]: self.MLU_div['length'][1] ] = temp_set_syl[numpy.cumsum(temp_set_dur_syl)-1]
        temp_set_af[0: num_ph, self.MLU_div['length'][1]: self.MLU_div['length'][2]] = temp_set_phone
        
        ### input word feature matrix ###
        temp_set_dur_word_segments = numpy.zeros(num_words, dtype='int32')
        syl_bound = numpy.cumsum(temp_set_dur_word)
        for indx in xrange(num_words):
            temp_set_dur_word_segments[indx] = int(sum(temp_set_dur_syl[0: syl_bound[indx]]))
        temp_set_x = temp_set_word[temp_set_dur_word_segments-1]
        
        ### rest of the code similar to S2S ###
        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        shared_set_x  = self.make_shared(temp_set_x, 'x')
        shared_set_y  = self.make_shared(temp_set_y, 'y')
        shared_set_d  = theano.shared(numpy.asarray(temp_set_d, dtype='int32'), name='d', borrow=True)

        shared_set_xyd = (shared_set_x, shared_set_y, shared_set_d)
        
        return shared_set_xyd, temp_set_x, temp_set_y, temp_set_d, temp_set_af

    def load_next_batch_S2S(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).
        
        """

        temp_set_x = numpy.empty((self.buffer_size, self.n_ins))
        temp_set_y = numpy.empty((self.buffer_size, self.n_outs))
        temp_set_d = numpy.empty((self.buffer_size, 1))

        io_fun = BinaryIOCollection()

        lab_start_frame_number = 0
        lab_end_frame_number   = 0

        out_start_frame_number = 0
        out_end_frame_number   = 0

        new_x_files_list = self.x_files_list[self.file_index].split(',')
        new_y_files_list = self.y_files_list[self.file_index].split(',')
        new_dur_files_list = self.dur_files_list[self.file_index].split(',')

        for new_file_index in xrange(len(new_x_files_list)):
            in_features, lab_frame_number = io_fun.load_binary_file_frame(new_x_files_list[new_file_index], self.n_ins)
            out_features, out_frame_number = io_fun.load_binary_file_frame(new_y_files_list[new_file_index], self.n_outs)
            
            lab_end_frame_number+=lab_frame_number
            out_end_frame_number+=out_frame_number

            temp_set_x[lab_start_frame_number: lab_end_frame_number, ] = in_features[0:lab_frame_number, ]
            temp_set_y[out_start_frame_number: out_end_frame_number, ] = out_features[0:out_frame_number, ]
            if not self.dur_files_list:
                dur_frame_number = out_end_frame_number
                temp_set_d = numpy.array([dur_frame_number])
            else:
                dur_features, dur_frame_number = io_fun.load_binary_file_frame(new_dur_files_list[new_file_index], 1)
                assert sum(dur_features) == out_frame_number
                temp_set_d[lab_start_frame_number: lab_end_frame_number, ] = dur_features[0:lab_frame_number, ]

            lab_start_frame_number = lab_end_frame_number
            out_start_frame_number = out_end_frame_number

        temp_set_x = temp_set_x[0:lab_end_frame_number, ]
        temp_set_y = temp_set_y[0:out_end_frame_number, ]

        temp_set_d = temp_set_d[0:lab_end_frame_number, ]
        temp_set_d = numpy.reshape(temp_set_d, (-1, ))
        temp_set_d = temp_set_d.astype(int)   
        
        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')
        shared_set_d = theano.shared(numpy.asarray(temp_set_d, dtype='int32'), name='d', borrow=True)

        shared_set_xyd = (shared_set_x, shared_set_y, shared_set_d)

        return shared_set_xyd, temp_set_x, temp_set_y, temp_set_d

    def load_next_batch_S2SML(self):
        """Load the data for one utterance. This function will be called when utterance-by-utterance loading is required (e.g., sequential training).
        
        """
       
        inp_length = (self.MLU_div['word'][1] - self.MLU_div['word'][0]) + (self.MLU_div['word'][3] - self.MLU_div['word'][2])
        af_length = self.MLU_div['length'][-1]

        new_temp_set_x  = numpy.empty((self.buffer_size, inp_length))
        new_temp_set_y  = numpy.empty((self.buffer_size, self.n_outs))
        new_temp_set_af = numpy.empty((self.buffer_size, af_length))
        new_temp_set_d  = [numpy.array([], 'int32'),numpy.array([], 'int32'),numpy.array([], 'int32')]

        io_fun = BinaryIOCollection()

        lab_start_frame_number = 0
        lab_end_frame_number   = 0

        out_start_frame_number = 0
        out_end_frame_number   = 0

        new_x_files_list = self.x_files_list[self.file_index].split(',')
        new_y_files_list = self.y_files_list[self.file_index].split(',')
        new_dur_files_list = self.dur_files_list[self.file_index].split(',')

        for new_file_index in xrange(len(new_x_files_list)):
            in_features, lab_frame_number = io_fun.load_binary_file_frame(new_x_files_list[new_file_index], self.n_ins)
            out_features, out_frame_number = io_fun.load_binary_file_frame(new_y_files_list[new_file_index], self.n_outs)
            dur_features, dur_frame_number = io_fun.load_binary_file_frame(new_dur_files_list[new_file_index], 1)
            
            ### MLU features sub-division ###
            temp_set_MLU = in_features[0:lab_frame_number, ]
            temp_set_y   = out_features[0:out_frame_number, ]
        
            temp_set_phone = numpy.concatenate([temp_set_MLU[:, self.MLU_div['phone'][0]: self.MLU_div['phone'][1]], temp_set_MLU[:, self.MLU_div['phone'][2]: self.MLU_div['phone'][3]]], axis = 1)
            temp_set_syl   = numpy.concatenate([temp_set_MLU[:, self.MLU_div['syl'][0]: self.MLU_div['syl'][1]], temp_set_MLU[:, self.MLU_div['syl'][2]: self.MLU_div['syl'][3]]], axis = 1)
            temp_set_word  = numpy.concatenate([temp_set_MLU[:, self.MLU_div['word'][0]: self.MLU_div['word'][1]], temp_set_MLU[:, self.MLU_div['word'][2]: self.MLU_div['word'][3] ]], axis = 1)
        
            ### duration array sub-division ###
            dur_features = numpy.reshape(dur_features, (-1, ))
            temp_set_d   = dur_features.astype(int)   
            dur_word_syl = temp_set_d[0: -lab_frame_number]    
        
            num_ph    = lab_frame_number
            num_syl   = (numpy.where(numpy.cumsum(dur_word_syl[::-1])==lab_frame_number)[0][0] + 1)
            num_words = len(dur_word_syl) - num_syl 
        
            temp_set_dur_phone = temp_set_d[-num_ph:] 
            temp_set_dur_word  = dur_word_syl[0: num_words]
            temp_set_dur_syl   = dur_word_syl[num_words: ]
        
            ### additional feature matrix (syllable+phone+frame=432) ###
            num_frames = sum(temp_set_dur_phone)
            temp_set_af = numpy.empty((num_frames, self.MLU_div['length'][-1]))
        
            temp_set_af[0: num_syl, self.MLU_div['length'][0]: self.MLU_div['length'][1] ] = temp_set_syl[numpy.cumsum(temp_set_dur_syl)-1]
            temp_set_af[0: num_ph, self.MLU_div['length'][1]: self.MLU_div['length'][2]] = temp_set_phone
        
            ### input word feature matrix ###
            temp_set_dur_word_segments = numpy.zeros(num_words, dtype='int32')
            syl_bound = numpy.cumsum(temp_set_dur_word)
            for indx in xrange(num_words):
                temp_set_dur_word_segments[indx] = int(sum(temp_set_dur_syl[0: syl_bound[indx]]))
            temp_set_x = temp_set_word[temp_set_dur_word_segments-1]
        
            ### for batch processing ###
            lab_end_frame_number+=num_words
            out_end_frame_number+=out_frame_number
      
            new_temp_set_x[lab_start_frame_number: lab_end_frame_number, ] = temp_set_x[0:num_words, ]
            new_temp_set_y[out_start_frame_number: out_end_frame_number, ] = temp_set_y[0:out_frame_number, ]
            new_temp_set_af[out_start_frame_number: out_end_frame_number, ] = temp_set_af[0:out_frame_number, ]

            new_temp_set_d[0] = numpy.append(new_temp_set_d[0], temp_set_dur_word)
            new_temp_set_d[1] = numpy.append(new_temp_set_d[1], temp_set_dur_syl)
            new_temp_set_d[2] = numpy.append(new_temp_set_d[2], temp_set_dur_phone)

            lab_start_frame_number = lab_end_frame_number
            out_start_frame_number = out_end_frame_number
        
        new_temp_set_x = new_temp_set_x[0:lab_end_frame_number, ]
        new_temp_set_y = new_temp_set_y[0:out_end_frame_number, ]
        new_temp_set_af = new_temp_set_af[0:out_end_frame_number, ]
        
        new_temp_set_d = numpy.concatenate((new_temp_set_d[0], new_temp_set_d[1], new_temp_set_d[2]))
        
        ### rest of the code similar to S2S ###
        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        shared_set_x  = self.make_shared(new_temp_set_x, 'x')
        shared_set_y  = self.make_shared(new_temp_set_y, 'y')
        shared_set_d  = theano.shared(numpy.asarray(new_temp_set_d, dtype='int32'), name='d', borrow=True)

        shared_set_xyd = (shared_set_x, shared_set_y, shared_set_d)
        
        return shared_set_xyd, new_temp_set_x, new_temp_set_y, new_temp_set_d, new_temp_set_af

    def load_next_utterance_CTC(self):

        temp_set_x = numpy.empty((self.buffer_size, self.n_ins))
        temp_set_y = numpy.empty(self.buffer_size)

        io_fun = BinaryIOCollection()

        in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        frame_number = lab_frame_number
        temp_set_x = in_features[0:frame_number, ]

        temp_set_y = numpy.array([self.n_outs])
        for il in numpy.argmax(out_features, axis=1):
            temp_set_y = numpy.concatenate((temp_set_y, [il, self.n_outs]), axis=0)

        self.file_index += 1

        if  self.file_index >= self.list_size:
            self.end_reading = True
            self.file_index = 0

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = theano.shared(numpy.asarray(temp_set_y, dtype='int32'), name='y', borrow=True)

        shared_set_xy = (shared_set_x, shared_set_y)

        return shared_set_xy, temp_set_x, temp_set_y


    def load_next_partition(self):
        """Load one block data. The number of frames will be the buffer size set during intialisation.

        """

        self.logger.debug('loading next partition')

        temp_set_x = numpy.empty((self.buffer_size, self.n_ins))
        temp_set_y = numpy.empty((self.buffer_size, self.n_outs))
        current_index = 0

        ### first check whether there are remaining data from previous utterance
        if self.remain_frame_number > 0:
            temp_set_x[current_index:self.remain_frame_number, ] = self.remain_data_x
            temp_set_y[current_index:self.remain_frame_number, ] = self.remain_data_y
            current_index += self.remain_frame_number

            self.remain_frame_number = 0

        io_fun = BinaryIOCollection()
        while True:
            if current_index >= self.buffer_size:
                break
            if  self.file_index >= self.list_size:
                self.end_reading = True
                self.file_index = 0
                break

            in_features, lab_frame_number = io_fun.load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
            out_features, out_frame_number = io_fun.load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

            frame_number = lab_frame_number
            if abs(lab_frame_number - out_frame_number) < 5:    ## we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
                if lab_frame_number > out_frame_number:
                    frame_number = out_frame_number
            else:
                base_file_name = os.path.basename(self.x_files_list[self.file_index]).split('.')[0]
                self.logger.critical("the number of frames in label and acoustic features are different: %d vs %d (%s)" %(lab_frame_number, out_frame_number, base_file_name))
                raise

            out_features = out_features[0:frame_number, ]
            in_features = in_features[0:frame_number, ]

            if current_index + frame_number <= self.buffer_size:
                temp_set_x[current_index:current_index+frame_number, ] = in_features
                temp_set_y[current_index:current_index+frame_number, ] = out_features

                current_index = current_index + frame_number
            else:   ## if current utterance cannot be stored in the block, then leave the remaining part for the next block
                used_frame_number = self.buffer_size - current_index
                temp_set_x[current_index:self.buffer_size, ] = in_features[0:used_frame_number, ]
                temp_set_y[current_index:self.buffer_size, ] = out_features[0:used_frame_number, ]
                current_index = self.buffer_size

                self.remain_data_x = in_features[used_frame_number:frame_number, ]
                self.remain_data_y = out_features[used_frame_number:frame_number, ]
                self.remain_frame_number = frame_number - used_frame_number

            self.file_index += 1

        temp_set_x = temp_set_x[0:current_index, ]
        temp_set_y = temp_set_y[0:current_index, ]

        numpy.random.seed(271639)
        numpy.random.shuffle(temp_set_x)
        numpy.random.seed(271639)
        numpy.random.shuffle(temp_set_y)

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')

        shared_set_xy = (shared_set_x, shared_set_y)
#        temp_set_x = self.make_shared(temp_set_x, 'x')
#        temp_set_y = self.make_shared(temp_set_y, 'y')

        return shared_set_xy, temp_set_x, temp_set_y

    def is_finish(self):
        return self.end_reading


class ListDataProviderWithProjectionIndex(ListDataProvider):
    '''
    Added kwarg index_to_project to __init__
    '''

    def __init__(self, x_file_list, y_file_list, n_ins=0, n_outs=0, \
            buffer_size = 500000, shuffle=False, index_to_project=1, projection_insize=10000, indexes_only=False):
        ##ListDataProvider.__init__(x_file_list, \
        ##         y_file_list, n_ins=0, n_outs=0, buffer_size = 500000, shuffle=False)
        super( ListDataProviderWithProjectionIndex, self ).__init__(x_file_list, \
                 y_file_list, n_ins=n_ins, n_outs=n_outs, buffer_size=buffer_size, shuffle=shuffle)
        self.index_to_project = index_to_project
        self.projection_insize = projection_insize
        self.indexes_only = indexes_only

    def load_next_partition_with_projection(self):

        shared_set_xy, temp_set_x, temp_set_y = self.load_next_partition()

        if self.indexes_only:
            temp_set_x, p_indexes = get_unexpanded_projection_inputs(temp_set_x, self.index_to_project, \
                                                            self.projection_insize)
            shared_set_x_proj = theano.shared(p_indexes, name='x_proj', borrow=True)
        else:
            temp_set_x, one_hot = expand_projection_inputs(temp_set_x, self.index_to_project, \
                                                            self.projection_insize)
            shared_set_x_proj = self.make_shared(one_hot, 'x_proj')

        shared_set_x = self.make_shared(temp_set_x, 'x')
        shared_set_y = self.make_shared(temp_set_y, 'y')

        shared_set_xy = (shared_set_x, shared_set_x_proj, shared_set_y)

        if self.indexes_only:
            return shared_set_xy, temp_set_x, p_indexes, temp_set_y
        else:
            return shared_set_xy, temp_set_x, one_hot, temp_set_y

## Put this function at global level so it can be imported for use in dnn_generation
def expand_projection_inputs(temp_set_x, index_to_project, projection_insize):
    ## Turn indexes to words, syllables etc. to one-hot data:
    m,n = numpy.shape(temp_set_x)
    projection_indices = temp_set_x[:, index_to_project]
    #print projection_indices.tolist()
    assert projection_indices.max() < projection_insize,'projection_insize is %s but there is an index %s in the data'%(projection_insize, projection_indices.max())
    one_hot = numpy.zeros((m, projection_insize))

    ## Used advanced indexing to turn the relevant features on:
    projection_indices = projection_indices.astype(int) ## check conversion???!?!?!
    #     print projection_indices.tolist()
    #     print '            ^--- proj indices'
    #     print
    one_hot[list(range(m)),projection_indices] = 1.0
    ## Effectively remove the index from the original data by setting to 0:
    temp_set_x[:, index_to_project] = 0.0
    return temp_set_x, one_hot

def get_unexpanded_projection_inputs(temp_set_x, index_to_project, projection_insize):
    ## Turn indexes to words, syllables etc. to one-hot data:
    m,n = numpy.shape(temp_set_x)
    projection_indices = temp_set_x[:, index_to_project]
    #print projection_indices.tolist()
    assert projection_indices.max() < projection_insize,'projection_insize is %s but there is an index %s in the data'%(projection_insize, projection_indices.max())

    projection_indices = projection_indices.astype('int32') ## check conversion???!?!?!

    temp_set_x[:, index_to_project] = 0.0
    return temp_set_x, projection_indices
