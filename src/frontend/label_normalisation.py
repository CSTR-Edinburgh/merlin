
import os
import numpy, re, sys
from multiprocessing import Pool
from io_funcs.binary_io import BinaryIOCollection
from linguistic_base import LinguisticBase

import lxml
from lxml import etree
from lxml.etree import * 
MODULE_PARSER = etree.XMLParser()

import matplotlib.mlab as mlab
import math

import logging
# from logplot.logging_plotting import LoggerPlotter #, MultipleTimeSeriesPlot, SingleWeightMatrixPlot

class LabelNormalisation(LinguisticBase):

    # this class only knows how to deal with a single style of labels (XML or HTS)
    # (to deal with composite labels, use LabelComposer instead)

    def __init__(self, question_file_name=None,xpath_file_name=None):
        pass
        
    def extract_linguistic_features(self, in_file_name, out_file_name=None, label_type="state_align", dur_file_name=None):
        if label_type=="phone_align":
            A = self.load_labels_with_phone_alignment(in_file_name, dur_file_name)
        elif label_type=="state_align":
            A = self.load_labels_with_state_alignment(in_file_name)
        else:
            logger.critical("we don't support %s labels as of now!!" % (label_type))

        if out_file_name:
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(A, out_file_name)
        else:
            return A

#  -----------------------------



class HTSLabelNormalisation(LabelNormalisation):
    """This class is to convert HTS format labels into continous or binary values, and store as binary format with float32 precision.
    
    The class supports two kinds of questions: QS and CQS.
        **QS**: is the same as that used in HTS
        
        **CQS**: is the new defined question in the system.  Here is an example of the question: CQS C-Syl-Tone {_(\d+)+}. regular expression is used for continous values.
    
    Time alignments are expected in the HTS labels. Here is an example of the HTS labels:
    
    3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]

    3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]

    3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]

    3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]

    3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]

    305000 310000 are the starting and ending time.
    [2], [3], [4], [5], [6] mean the HMM state index. 

    """
    
    # this subclass support HTS labels, which include time alignments
    
    def __init__(self, question_file_name=None, add_frame_features=True, subphone_feats='full', continuous_flag=True):

        logger = logging.getLogger("labels")

        self.question_dict = {}
        self.ori_question_dict = {}
        self.dict_size = 0
        self.continuous_flag = continuous_flag
        try:    
#            self.question_dict, self.ori_question_dict = self.load_question_set(question_file_name)
            self.discrete_dict, self.continuous_dict = self.load_question_set_continous(question_file_name)
        except:
            logger.critical('error whilst loading HTS question set')
            raise
            
        ###self.dict_size = len(self.question_dict)
        
        self.dict_size = len(self.discrete_dict) + len(self.continuous_dict)
        self.add_frame_features = add_frame_features
        self.subphone_feats = subphone_feats

        if self.subphone_feats == 'full':
            self.frame_feature_size = 9   ## zhizheng's original 5 state features + 4 phoneme features
        elif self.subphone_feats == 'minimal_frame':
            self.frame_feature_size = 2   ## the minimal features necessary to go from a state-level to frame-level model 
        elif self.subphone_feats == 'state_only':
            self.frame_feature_size = 1   ## this is equivalent to a state-based system
        elif self.subphone_feats == 'none':
            self.frame_feature_size = 0   ## the phoneme level features only
        elif self.subphone_feats == 'frame_only':
            self.frame_feature_size = 1   ## this is equivalent to a frame-based system without relying on state-features
        elif self.subphone_feats == 'uniform_state':
            self.frame_feature_size = 2   ## this is equivalent to a frame-based system with uniform state-features
        elif self.subphone_feats == 'minimal_phoneme':
            self.frame_feature_size = 3   ## this is equivalent to a frame-based system with minimal features
        elif self.subphone_feats == 'coarse_coding':
            self.frame_feature_size = 4   ## this is equivalent to a frame-based positioning system reported in Heiga Zen's work
            self.cc_features = self.compute_coarse_coding_features(3)
        else:
            sys.exit('Unknown value for subphone_feats: %s'%(subphone_feats))
        
        self.dimension = self.dict_size + self.frame_feature_size   
        
        ### if user wants to define their own input, simply set the question set to empty.
        if self.dict_size == 0:
            self.dimension = 0

        logger.debug('HTS-derived input feature dimension is %d + %d = %d' % (self.dict_size, self.frame_feature_size, self.dimension) )
        
    def prepare_dur_data(self, ori_file_list, output_file_list, label_type="state_align", feature_type=None, unit_size=None, feat_size=None):
        '''
        extracting duration binary features or numerical features.
        '''
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print   "the number of input and output files should be the same!\n";
            sys.exit(1)
               
        ### set default feature type to numerical, if not assigned ###
        if not feature_type:
            feature_type = "numerical"
        
        ### set default unit size to state, if not assigned ###
        if not unit_size:
            unit_size = "state"
        if label_type=="phone_align":
            unit_size = "phoneme"

        ### set default feat size to frame or phoneme, if not assigned ###
        if feature_type=="binary":
            if not feat_size:
                feat_size = "frame"
        elif feature_type=="numerical":
            if not feat_size:
                feat_size = "phoneme"
        else:
            logger.critical("Unknown feature type: %s \n Please use one of the following: binary, numerical\n" %(feature_type))
            sys.exit(1)

        for i in xrange(utt_number):
            self.extract_dur_features(ori_file_list[i], output_file_list[i], label_type, feature_type, unit_size, feat_size)
    
    def extract_dur_features(self, in_file_name, out_file_name=None, label_type="state_align", feature_type=None, unit_size=None, feat_size=None):
        logger = logging.getLogger("dur")
        if label_type=="phone_align":
            A = self.extract_dur_from_phone_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        elif label_type=="state_align":
            A = self.extract_dur_from_state_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        else:
            logger.critical("we don't support %s labels as of now!!" % (label_type))
            sys.exit(1)

        if out_file_name:
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(A, out_file_name)
        else:
            return A
    
    def extract_dur_from_state_alignment_labels(self, file_name, feature_type, unit_size, feat_size): 
        logger = logging.getLogger("dur")

        state_number = 5
        dur_dim = state_number
        
        if feature_type=="binary":
            dur_feature_matrix = numpy.empty((100000, 1))
        elif feature_type=="numerical":
            if unit_size=="state":
                dur_feature_matrix = numpy.empty((100000, dur_dim))
                current_dur_array = numpy.zeros((dur_dim, 1))
            elif unit_size=="phoneme":
                dur_feature_matrix = numpy.empty((100000, 1))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )
		
        current_index = 0
        dur_feature_index = 0
        for line in utt_labels:
            line = line.strip()
            
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            
            full_label = temp_list[2]
            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]
            state_index = int(state_index) - 1

            frame_number = int((end_time - start_time)/50000)
            
            if state_index == 1:
                phone_duration = frame_number
                
                for i in xrange(state_number - 1):
                    line = utt_labels[current_index + i + 1].strip()
                    temp_list = re.split('\s+', line)
                    phone_duration += int((int(temp_list[1]) - int(temp_list[0]))/50000)

            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "state":
                    current_block_array[-1] = 1
                elif unit_size == "phoneme":
                    if state_index == state_number:
                        current_block_array[-1] = 1
                else:
                    logger.critical("Unknown unit size: %s \n Please use one of the following: state, phoneme\n" %(unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "state":
                    current_dur_array[current_index%5] = frame_number 
                    if feat_size == "phoneme" and state_index == state_number:
                        current_block_array =  current_dur_array.transpose() 
                    if feat_size == "frame":
                        current_block_array = numpy.tile(current_dur_array.transpose(), (frame_number, 1))
                elif unit_size == "phoneme":
                    current_block_array = numpy.array([phone_duration])
            
            ### writing into dur_feature_matrix ### 
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index+frame_number,] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif feat_size == "phoneme" and state_index == state_number: 
                dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                dur_feature_index = dur_feature_index + 1

            current_index += 1

        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index,]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape )
        return  dur_feature_matrix

    def extract_dur_from_phone_alignment_labels(self, file_name, feature_type, unit_size, feat_size): 
        logger = logging.getLogger("dur")

        dur_dim = 1 
        
        if feature_type=="binary":
            dur_feature_matrix = numpy.empty((100000, 1))
        elif feature_type=="numerical":
            if unit_size=="phoneme":
                dur_feature_matrix = numpy.empty((100000, 1))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )
		
        current_index = 0
        dur_feature_index = 0
        for line in utt_labels:
            line = line.strip()
            
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            
            full_label = temp_list[2]

            frame_number = int((end_time - start_time)/50000)
            
            phone_duration = frame_number
                
            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "phoneme":
                    current_block_array[-1] = 1
                else:
                    logger.critical("Unknown unit size: %s \n Please use one of the following: phoneme\n" %(unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "phoneme":
                    current_block_array = numpy.array([phone_duration])
            
            ### writing into dur_feature_matrix ### 
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index+frame_number,] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif feat_size == "phoneme": 
                dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                dur_feature_index = dur_feature_index + 1

            current_index += 1

        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index,]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape )
        return  dur_feature_matrix

    def load_labels_with_phone_alignment(self, file_name, dur_file_name):

        # this is not currently used ??? -- it works now :D
        logger = logging.getLogger("labels")
        #logger.critical('unused function ???')
        #raise Exception
        
        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1 ## hard coded for now
            manual_dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        if self.add_frame_features:
            assert self.dimension == self.dict_size+self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size+self.frame_feature_size
        else:
            assert self.dimension == self.dict_size
        
        label_feature_matrix = numpy.empty((100000, self.dimension))

        ph_count=0
        label_feature_index = 0
        fid = open(file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            full_label = temp_list[2]

            # to do - support different frame shift - currently hardwired to 5msec
            # currently under beta testing: support different frame shift 
            if dur_file_name:
                frame_number = manual_dur_data[ph_count]
            else:
                frame_number = int((end_time - start_time)/50000)

            ph_count = ph_count+1
            #label_binary_vector = self.pattern_matching(full_label)
            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label) 
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

            if self.subphone_feats == "coarse_coding":
                cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size+self.frame_feature_size))
                for i in xrange(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector

                    if self.subphone_feats == 'minimal_phoneme':
                        ## features which distinguish frame position in phoneme 
                        current_block_binary_array[i, self.dict_size] = float(i+1)/float(frame_number) # fraction through phone forwards
                        current_block_binary_array[i, self.dict_size+1] = float(frame_number - i)/float(frame_number) # fraction through phone backwards
                        current_block_binary_array[i, self.dict_size+2] = float(frame_number) # phone duration

                    elif self.subphone_feats == 'coarse_coding':
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size+0] = cc_feat_matrix[i, 0]
                        current_block_binary_array[i, self.dict_size+1] = cc_feat_matrix[i, 1]
                        current_block_binary_array[i, self.dict_size+2] = cc_feat_matrix[i, 2]
                        current_block_binary_array[i, self.dict_size+3] = float(frame_number)

                    elif self.subphone_feats == 'none':
                        pass

                    else:
                        sys.exit('unknown subphone_feats type')
            
                label_feature_matrix[label_feature_index:label_feature_index+frame_number,] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number

            elif self.subphone_feats == 'none':
                current_block_binary_array = label_vector 
                label_feature_matrix[label_feature_index:label_feature_index+1,] = current_block_binary_array
                label_feature_index = label_feature_index + 1


        fid.close()

        label_feature_matrix = label_feature_matrix[0:label_feature_index,]

        logger.info('loaded %s, %3d labels' % (file_name, ph_count) )
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix


    def load_labels_with_state_alignment(self, file_name): 
        ## setting add_frame_features to False performs either state/phoneme level normalisation
 
        logger = logging.getLogger("labels")

        if self.add_frame_features:
            assert self.dimension == self.dict_size+self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size+self.frame_feature_size
        else:
            assert self.dimension == self.dict_size
        
        # label_feature_matrix = numpy.empty((100000, self.dict_size+self.frame_feature_size))
        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0

        state_number = 5

        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )
		
        phone_duration = 0
        state_duration_base = 0
        for line in utt_labels:
            line = line.strip()
            
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            full_label = temp_list[2]
            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]
            
#            print state_index
            state_index = int(state_index) - 1
            state_index_backward = 6 - state_index
            full_label = full_label[0:full_label_length]

            frame_number = int((end_time - start_time)/50000)
            
            if state_index == 1:
                current_frame_number = 0
                phone_duration = frame_number
                state_duration_base = 0
                
#                label_binary_vector = self.pattern_matching(full_label)
                label_binary_vector = self.pattern_matching_binary(full_label)

                # if there is no CQS question, the label_continuous_vector will become to empty
                label_continuous_vector = self.pattern_matching_continous_position(full_label) 
                label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

                for i in xrange(state_number - 1):
                    line = utt_labels[current_index + i + 1].strip()
                    temp_list = re.split('\s+', line)
                    phone_duration += int((int(temp_list[1]) - int(temp_list[0]))/50000)

                if self.subphone_feats == "coarse_coding":
                    cc_feat_matrix = self.extract_coarse_coding_features_relative(phone_duration)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size+self.frame_feature_size))
                for i in xrange(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector
		    
                    if self.subphone_feats == 'full':
                        ## Zhizheng's original 9 subphone features:
                        current_block_binary_array[i, self.dict_size] = float(i+1) / float(frame_number)   ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size+1] = float(frame_number - i) / float(frame_number)  ## fraction through state (backwards)
                        current_block_binary_array[i, self.dict_size+2] = float(frame_number)  ## length of state in frames
                        current_block_binary_array[i, self.dict_size+3] = float(state_index)   ## state index (counting forwards)
                        current_block_binary_array[i, self.dict_size+4] = float(state_index_backward) ## state index (counting backwards)

                        current_block_binary_array[i, self.dict_size+5] = float(phone_duration)   ## length of phone in frames
                        current_block_binary_array[i, self.dict_size+6] = float(frame_number) / float(phone_duration)   ## fraction of the phone made up by current state
                        current_block_binary_array[i, self.dict_size+7] = float(phone_duration - i - state_duration_base) / float(phone_duration) ## fraction through phone (forwards)
                        current_block_binary_array[i, self.dict_size+8] = float(state_duration_base + i + 1) / float(phone_duration)  ## fraction through phone (backwards)
                    
                    elif self.subphone_feats == 'state_only':
                        ## features which only distinguish state:
                        current_block_binary_array[i, self.dict_size] = float(state_index)   ## state index (counting forwards)
                    
                    elif self.subphone_feats == 'frame_only':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(phone_duration)   ## fraction through phone (counting forwards)

                    elif self.subphone_feats == 'uniform_state':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(phone_duration)   ## fraction through phone (counting forwards)
                        new_state_index = max(1, round(float(current_frame_number)/float(phone_duration)*5))
                        current_block_binary_array[i, self.dict_size+1] = float(new_state_index)   ## state index (counting forwards)
            
                    elif self.subphone_feats == "coarse_coding":
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size+0] = cc_feat_matrix[current_frame_number, 0]
                        current_block_binary_array[i, self.dict_size+1] = cc_feat_matrix[current_frame_number, 1]
                        current_block_binary_array[i, self.dict_size+2] = cc_feat_matrix[current_frame_number, 2]
                        current_block_binary_array[i, self.dict_size+3] = float(phone_duration)
                        current_frame_number += 1

                    elif self.subphone_feats == 'minimal_frame':
                        ## features which distinguish state and minimally frame position in state:
                        current_block_binary_array[i, self.dict_size] = float(i+1) / float(frame_number)   ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size+1] = float(state_index)   ## state index (counting forwards)
                    elif self.subphone_feats == 'none':
                        pass
                    else:
                        sys.exit('unknown subphone_feats type')

                label_feature_matrix[label_feature_index:label_feature_index+frame_number,] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number
            elif self.subphone_feats == 'state_only' and state_index == state_number:
                current_block_binary_array = numpy.zeros((state_number, self.dict_size+self.frame_feature_size))
                for i in xrange(state_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector
                    current_block_binary_array[i, self.dict_size] = float(i+1)   ## state index (counting forwards)
                label_feature_matrix[label_feature_index:label_feature_index+state_number,] = current_block_binary_array
                label_feature_index = label_feature_index + state_number
            elif self.subphone_feats == 'none' and state_index == state_number:
                current_block_binary_array = label_vector 
                label_feature_matrix[label_feature_index:label_feature_index+1,] = current_block_binary_array
                label_feature_index = label_feature_index + 1

            state_duration_base += frame_number
            
            current_index += 1

        label_feature_matrix = label_feature_matrix[0:label_feature_index,]
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix

    def extract_durational_features(self, dur_file_name=None, dur_data=None):
       
        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1 ## hard coded for now
            dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)
        
        ph_count = len(dur_data)
        total_num_of_frames = int(sum(dur_data))

        duration_feature_array = numpy.zeros((total_num_of_frames, self.frame_feature_size))

        frame_index=0 
        for i in xrange(ph_count):
            frame_number = int(dur_data[i])
            if self.subphone_feats == "coarse_coding":
                cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

                for j in xrange(frame_number):
                    duration_feature_array[frame_index, 0] = cc_feat_matrix[j, 0]
                    duration_feature_array[frame_index, 1] = cc_feat_matrix[j, 1]
                    duration_feature_array[frame_index, 2] = cc_feat_matrix[j, 2]
                    duration_feature_array[frame_index, 3] = float(frame_number)
                    frame_index+=1
        
        return duration_feature_array

    def compute_coarse_coding_features(self, num_states):
        assert num_states == 3

        npoints = 600
        cc_features = numpy.zeros((num_states, npoints))

        x1 = numpy.linspace(-1.5, 1.5, npoints)
        x2 = numpy.linspace(-1.0, 2.0, npoints)
        x3 = numpy.linspace(-0.5, 2.5, npoints)

        mu1 = 0.0
        mu2 = 0.5
        mu3 = 1.0

        sigma = 0.4

        cc_features[0, :] = mlab.normpdf(x1, mu1, sigma)
        cc_features[1, :] = mlab.normpdf(x2, mu2, sigma)
        cc_features[2, :] = mlab.normpdf(x3, mu3, sigma)

        return cc_features

    def extract_coarse_coding_features_relative(self, phone_duration):
        dur = int(phone_duration)
        
        cc_feat_matrix = numpy.zeros((dur, 3))

        for i in xrange(dur):
            rel_indx = int((200/float(dur))*i)
            cc_feat_matrix[i,0] = self.cc_features[0, 300+rel_indx]
            cc_feat_matrix[i,1] = self.cc_features[1, 200+rel_indx]
            cc_feat_matrix[i,2] = self.cc_features[2, 100+rel_indx]

        return cc_feat_matrix

    ### this function is not used now
    def extract_coarse_coding_features_absolute(self, phone_duration):
        dur = int(phone_duration)
        
        cc_feat_matrix = numpy.zeros((dur, 3))
        
        npoints1 = (dur*2)*10+1
        npoints2 = (dur-1)*10+1
        npoints3 = (2*dur-1)*10+1
        
        x1 = numpy.linspace(-dur, dur, npoints1)
        x2 = numpy.linspace(1, dur, npoints2)
        x3 = numpy.linspace(1, 2*dur-1, npoints3)
        
        mu1 = 0
        mu2 = (1+dur)/2
        mu3 = dur
        variance = 1
        sigma = variance*((dur/10)+2)
        sigma1 = sigma
        sigma2 = sigma-1
        sigma3 = sigma
        
        y1 = mlab.normpdf(x1, mu1, sigma1)
        y2 = mlab.normpdf(x2, mu2, sigma2)
        y3 = mlab.normpdf(x3, mu3, sigma3)

        for i in xrange(dur):
            cc_feat_matrix[i,0] = y1[(dur+1+i)*10]
            cc_feat_matrix[i,1] = y2[i*10]
            cc_feat_matrix[i,2] = y3[i*10]

        for i in xrange(3):
            cc_feat_matrix[:,i] = cc_feat_matrix[:,i]/max(cc_feat_matrix[:,i])

        return cc_feat_matrix


    ### this function is not used now
    def pattern_matching(self, label):
        # this function is where most time is spent during label preparation
        #
        # it might be possible to speed it up by using pre-compiled regular expressions?
        # (not trying this now, since we may change to to XML tree format for input instead of HTS labels)
        #
        label_size = len(label)

        lab_binary_vector = numpy.zeros((1, self.dict_size))

        for i in xrange(self.dict_size):
            current_question_list = self.question_dict[str(i)]
            binary_flag = 0
            for iq in xrange(len(current_question_list)):
                current_question = current_question_list[iq]
                current_size = len(current_question)
                if current_question[0] == '*' and current_question[current_size-1] == '*':
                    temp_question = current_question[1:current_size-1]
                    for il in xrange(1, label_size-current_size+2):
                        if temp_question == label[il:il+current_size-2]:
                            binary_flag = 1
                elif current_question[current_size-1] != '*':
                    temp_question = current_question[1:current_size]
                    if temp_question == label[label_size-current_size+1:label_size]:
                        binary_flag = 1
                elif current_question[0] != '*':
                    temp_question = current_question[0:current_size-1]
                    if temp_question == label[0:current_size-1]:
                        binary_flag = 1
                if binary_flag == 1:
                    break
            lab_binary_vector[0, i] = binary_flag
        
        return  lab_binary_vector
        
    def pattern_matching_binary(self, label):
        
        dict_size = len(self.discrete_dict)
        lab_binary_vector = numpy.zeros((1, dict_size))
        
        for i in xrange(dict_size):
            current_question_list = self.discrete_dict[str(i)]
            binary_flag = 0
            for iq in xrange(len(current_question_list)):
                current_compiled = current_question_list[iq]
                
                ms = current_compiled.search(label)
                if ms is not None:
                    binary_flag = 1
                    break
            lab_binary_vector[0, i] = binary_flag
            
        return   lab_binary_vector
        

    def pattern_matching_continous_position(self, label):
        
        dict_size = len(self.continuous_dict)

        lab_continuous_vector = numpy.zeros((1, dict_size))
        
        for i in xrange(dict_size):
            continuous_value = -1.0

            current_compiled = self.continuous_dict[str(i)]
            
            ms = current_compiled.search(label)
            if ms is not None:
#                assert len(ms.group()) == 1
                continuous_value = ms.group(1)
            
            lab_continuous_vector[0, i] = continuous_value

        return  lab_continuous_vector
        
    def load_question_set(self, qs_file_name):
        fid = open(qs_file_name)
        question_index = 0
        question_dict = {}
        ori_question_dict = {}
        for line in fid.readlines():
            line = line.replace('\n', '')
            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                question_list = temp_line.split(',')
                question_dict[str(question_index)] = question_list
                ori_question_dict[str(question_index)] = line
                question_index += 1
        fid.close()

        logger = logging.getLogger("labels")
        logger.debug('loaded question set with %d questions' % len(question_dict))

        return  question_dict, ori_question_dict


    def load_question_set_continous(self, qs_file_name):
        
        logger = logging.getLogger("labels")
        
        fid = open(qs_file_name)
        binary_qs_index = 0
        continuous_qs_index = 0
        binary_dict = {}
        continuous_dict = {}
        
        for line in fid.readlines():
            line = line.replace('\n', '')
            
            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                temp_line = temp_line.strip()
                question_list = temp_line.split(',')
                            
                temp_list = line.split(' ')
                question_key = temp_list[1]
#                print   line
                if temp_list[0] == 'CQS':
                    assert len(question_list) == 1
                    processed_question = self.wildcards2regex(question_list[0], convert_number_pattern=True)
                    continuous_dict[str(continuous_qs_index)] = re.compile(processed_question) #save pre-compiled regular expression 
                    continuous_qs_index = continuous_qs_index + 1
                elif temp_list[0] == 'QS':
                    re_list = []
                    for temp_question in question_list:
                        processed_question = self.wildcards2regex(temp_question)
#                        print   processed_question
                        re_list.append(re.compile(processed_question))
                        
                    binary_dict[str(binary_qs_index)] = re_list
                    binary_qs_index = binary_qs_index + 1
                else:
                    logger.critical('The question set is not defined correctly: %s' %(line))
                    raise Exception
                
#                question_index = question_index + 1       
        return  binary_dict, continuous_dict


    def wildcards2regex(self, question, convert_number_pattern=False):
        """
        Convert HTK-style question into regular expression for searching labels.
        If convert_number_pattern, keep the following sequences unescaped for 
        extracting continuous values):
            (\d+)       -- handles digit without decimal point
            ([\d\.]+)   -- handles digits with and without decimal point
        """
        
        ## handle HTK wildcards (and lack of them) at ends of label:
        if '*' in question:
            if not question.startswith('*'):
                question = '\A' + question
            if not question.endswith('*'):
                question = question + '\Z'
        question = question.strip('*')
        question = re.escape(question)
        ## convert remaining HTK wildcards * and ? to equivalent regex:
        question = question.replace('\\*', '.*')
        question = question.replace('\\?', '.')

        if convert_number_pattern:
            question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
            question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
        return question
                        





class HTSDurationLabelNormalisation(HTSLabelNormalisation):
    """
    Unlike HTSLabelNormalisation, HTSDurationLabelNormalisation does not accept timings.
    One line of labels is converted into 1 datapoint, that is, the label is not 'unpacked'
    into frames. HTK state index [\d] is not handled in any special way.
    """
    def __init__(self, question_file_name=None, subphone_feats='full', continuous_flag=True):
        super(HTSDurationLabelNormalisation, self).__init__(question_file_name=question_file_name, \
                                    subphone_feats=subphone_feats, continuous_flag=continuous_flag)
        ## don't use extra features beyond those in questions for duration labels:
        self.dimension = self.dict_size 


    def load_labels_with_state_alignment(self, file_name, add_frame_features=False): 
        ## add_frame_features not used in HTSLabelNormalisation -- only in XML version
 
        logger = logging.getLogger("labels")

        assert self.dimension == self.dict_size 

        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0

        
        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )

        ## remove empty lines   
        utt_labels = [line for line in utt_labels if line != '']
		
        for (line_number, line) in enumerate(utt_labels):
            temp_list = re.split('\s+', line.strip())
            full_label = temp_list[-1]  ## take last entry -- ignore timings if present
            
            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label) 
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

            label_feature_matrix[line_number, :] = label_vector[:]
            
    
        label_feature_matrix = label_feature_matrix[:line_number+1,:]
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix


#  -----------------------------












class XMLLabelNormalisation(LabelNormalisation):
    
    # this subclass supports XML trees (from Ossian) with time alignments embedded as features of the nodes
        
    def __init__(self, xpath_file_name=None, xpath=None, mapper=None, get_frame_feats=False, target_nodes = "//segment", fill_missing_values=False, use_compiled_xpath=False, iterate_over_frames=False):

        logger = logging.getLogger("labels")

        # specify which nodes in the loaded XML trees will be the targets for the xpaths used to extract features
        self.target_nodes = target_nodes

        self.use_compiled_xpath = use_compiled_xpath
        self.iterate_over_frames = iterate_over_frames
        
        
        if self.use_compiled_xpath:
            ## osw -- compile these once per normaliser -- could do it only once??
            self.start_time_xpath = etree.XPath('./attribute::start')
            self.end_time_xpath   = etree.XPath('./attribute::end')
        else:
            # how to retreive timings
            self.start_time_xpath = './attribute::start'
            self.end_time_xpath   = './attribute::end'

        # to do - make this user-settable via the config file ?
        self.unseen_string='_UNSEEN_'

        self.xpath_dict = [] ## NB 'dict' is now list to ensure feature order
        
        # should rename this variable - it's the feature dimensionality (excluding frame-level features)
        self.dict_size = 0

        # behaviour regarding filling in values for missing frames
        self.fill_missing_values = fill_missing_values
        

        # can read XPATHs from a file OR accept a single XPATH (but not both).
        # In the case of a single XPath, this can be a string or compiled version.
        
        
        # xpath now a list of xpaths [(name, xpath),(name, xpath)...] and
        #      a same length list of mappers -- entries should be None where
        #      feature isn't to be mapped.
        if xpath_file_name:
            assert not xpath
            assert not self.use_compiled_xpath
            self.xpath_dict = self.load_xpath_set(xpath_file_name)
            
            # if we are using a list of XPATHs, then
            # the dictionary size will determine the number of features (excluding frame-level features)
            # each xpath expression will extract a single feature
            self.dict_size = len(self.xpath_dict)
            logger.debug('using XPATH list - feature dimension will be %d' % self.dict_size)
            
            # to be implemented later: using mapping when there is a list of XPATHs
            # for now, do not allow a mapper in this case
            assert not mapper
            
        elif xpath:    ## osw -- now a list
            assert type(xpath) == list,'xpath must be a list of xpath expressions'
            assert not xpath_file_name
            self.xpath_dict = [('dummy_name', xp) for xp in xpath] #   [('dummy_name', xpath)]
            
            self.mapper = [None] * len(self.xpath_dict)
            if mapper:
                self.mapper = mapper
            assert len(self.mapper) == len(self.xpath_dict)
            
            self.dict_size = 0
            for map in self.mapper:
            # if we are using a single XPATH, then the number of features (excluding frame-level features)
            # is either 1, or is determined by the mapper
                if map:
                # pick an arbitrary item in the mapper and measure the length of the feature vector it will provide
                    
                    self.dict_size += len(map.itervalues().next())
                    logger.debug('using mapping - increment feature dimension by %d' % self.dict_size)
                else:
                    self.dict_size += 1
                    logger.debug('no mapping - increment feature dimension by 1')
        
     
                
            
        else:
            logger.critical('must provide one or more XPATHs')
            raise Exception
            


        
        # no access to sub-phonetic alignments in Ossian trees ???
        self.frame_feature_size = 0
        self.dimension = self.dict_size + self.frame_feature_size   

        logger.debug('XPATH feature dimension is %d + %d = %d' % (self.dict_size, self.frame_feature_size, self.dimension) )

        
    def convert_time_to_frames(self,t):
        # time in XML trees is stored as milliseconds, each frame is 5msec (currently hardwired - must change!)
        return numpy.rint(t / 5)

    def convert_frames_to_time(self,f):
        # time in XML trees is stored as milliseconds, each frame is 5msec (currently hardwired - must change!)
        return numpy.rint(f * 5)

    def load_labels_with_state_alignment(self, file_name_or_descriptor, add_frame_features=False):

        logger = logging.getLogger("labels")

        logger.debug('extracting features using this dictionary of XPATHs:')
        logger.debug('%s' % self.xpath_dict)
        
        assert self.dimension == self.dict_size+self.frame_feature_size

        if add_frame_features:
            self.dimension += 1 ## TODO: remove hard coding

        label_feature_matrix = numpy.empty((100000, self.dimension))
        
        # fill the label_feature_matrix with a special value that we can test for later
        label_feature_matrix.fill(numpy.nan)
        
        # label_feature_index = 0
        # state_number = 5

        # each frame will be a vector of features
        lab_binary_vector = numpy.zeros((1, self.dict_size))




        # load XML format labels from file_name
        
        ## Set the UtteranceElement Element as a default element class
        ## (http://lxml.de/element_classes.html):
        # not yet sure if we need to do this ???
        # parser_lookup = etree.ElementDefaultClassLookup(element=UtteranceElement)

        ## Ensure pretty printing
        ## (http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output):
        parser = XMLParser(remove_blank_text=True)
        # parser.set_element_class_lookup(parser_lookup)

        if type(file_name_or_descriptor) == file:
            fid = file_name_or_descriptor
            # rewind the file - it may have been left open from a previous call to this function
            fid.seek(0)
        else:
            try:
                fid = open(file_name_or_descriptor)
            except IOError:
                logger.critical('failed to open file %s' % file_name_or_descriptor)
                raise

        try:    
            tree = parse(fid, parser)
        except lxml.etree.XMLSyntaxError:
            logger.critical('failed to parse file %s' % file_name_or_descriptor)
            raise
            
        if type(file_name_or_descriptor) != file:
            fid.close()

    
        # the target nodes in the XML tree
        # e.g., segments or sub-phonetic states
        # for each of these, we will extract features using the xpath expressions
        targets = tree.getroot().xpath(self.target_nodes)
        
        # make sure there are some targets
        if len(targets)==0:
            logger.critical('pattern %s matches no nodes of utterance %s' % (target_nodes, file_name_or_descriptor) )
            raise Exception

        # iterate over the target nodes (e.g., segments or states)
        label_number = len(targets)
        total_number_of_frames=-1
        frame_num = 0  # osw
        previous_end_time = 0 ## to check target nodes are contiguous segments 
        for node in targets:
            
            logger.debug('extracting features for node %s' % node)

            try:
                if self.use_compiled_xpath:
                    this_segment_start_time = int(self.start_time_xpath(node)[0])
                    this_segment_end_time   = int(self.end_time_xpath(node)[0])            
                else:
                    this_segment_start_time = int(node.xpath(self.start_time_xpath)[0])
                    this_segment_end_time   = int(node.xpath(self.end_time_xpath)[0])
                    logger.debug(' start time: %d    end time: %d' % (this_segment_start_time,this_segment_end_time))
                    assert this_segment_start_time == previous_end_time,'segments not contiguous'
                    previous_end_time = this_segment_end_time
            except:
                logger.critical('problem obtaining start or end time for: %s' % node)
                raise

            segment_data = []
            for ((name, path), map) in zip(self.xpath_dict, self.mapper):
                if self.use_compiled_xpath:
                    pathstring = path.path
                    logger.debug(' evaluating PRECOMPILED xpath %s' % pathstring)
                    try:
                
                        data = path(node)
                    except lxml.etree.XPathEvalError:
                        logger.critical('problem evaluating this precompiled XPATH: %s' % pathstring)
                        raise
                
                else:  
                    pathstring = path   
                    #print 'evaluating PLAIN xpath'         
                    logger.debug(' evaluating xpath %s' % pathstring)
                    try:
                        data = node.xpath(path)
                    except lxml.etree.XPathEvalError:
                        logger.critical('problem evaluating this XPATH: %s' % pathstring)
                        raise


                if data == []:
                    # this means that nothing was matched by the XPATH
                    # so we construct a default value here (following the method used within Ossian)
                    # this will happen (only?) beyond utterance boundaries, where padding is required

## OSW: for now, don't handle padding with padding attributes -- just use _NA_ 
## This means that vectors etc have to be handled with a mapper rather than having 
## their features contained within the XML trees.

#                     fragments = re.split("[/:@\(\)]+", pathstring)  
#                     attribute_name = fragments[-1]          
#                     path_for_padding='ancestor::utt[1]/attribute::%s_PADDING'%(attribute_name)
#                     data = node.xpath(path_for_padding)
                    data = []
                    if data == []:
                        ## No padding attribute was found, use the default _NA_:
#                         logger.warning('failed to find a padding value')
#                         logger.warning(' original XPATH: %s' % pathstring)
#                         logger.warning(' attribute_name: %s' % attribute_name)
#                         logger.warning('  padding XPATH: %s' % path_for_padding)
                        data = ["_NA_"]
                
                if type(data) == list:
                    # currently we do not support lists of features stored in the trees
                    # only single values (which may be returned as a list with a single entry)
                    try:
                        assert len(data) == 1
                    except AssertionError:
                        logger.critical('data extracted using XPATH %s is a list with too many (%d) elements' % (path,len(data)) )
                        raise
                else:
                    # make it into a list with a single entry, because everything after this point
                    # assumes that data is a list of items
                    data=[data]
                    
                # if we are using a mapping, apply it now
                if map != None:  ## 
                    try:
                        data = map[data[0]]
                    except KeyError:
                        data = map[self.unseen_string]
                    except:
                        logger.critical('failed to map %s using mapper %s' % (data,map) )
                        raise
                
                elif (type(data[0]) == lxml.etree._ElementStringResult) or (type(data[0]) == str) \
                                                                or type(data[0] == float): 
                    # any strings coming out of the tree (that have not been mapped already)
                    # must now be co-erced to a numerical value
                    # note that data will always be a list with one entry, by this point.
                    # osw: might already be float -- values returned e.g. by xpath('count(...)') are floats
                    
                    
                    try:
                        data[0] = float(data[0])
                    except ValueError:
                        logger.critical('could not convert %s to a numerical value - problem with tree or mapping?' % data[0])
                        raise
                else:
                    logger.critical('internal error - data of unsupported type %s with value %s' % (type(data[0]) , data[0]) )
                    raise
                    
                    
                # at this point, data is a list of values

                # quick sanity check - just on the first item in the list
                try:
                    assert type(data[0]) in [int, float, bool]
                except AssertionError:
                    logger.critical('data extracted using XPATH %s is of unsupported type %s' % (path,type(data[0])) )
                    raise
                
                logger.debug(' features were %s' % data)

                segment_data.extend(data)

            logger.debug(' stacked features were %s' % segment_data)

	        # we now have a numerical feature vector for the current segment
            # and all we need to do is insert those values into the corresponding frames
            # from the start time to the end time of that segment

            # there are potential problems here converting between times and frame
            # currently, we tick through time in whatever units the XML trees use (msec)
            # and so will write to the same frame many times (wasteful, but avoids skipping any frames)
            # TO DO - be more careful and test for skipped frames 
            
            state_length_ms = this_segment_end_time - this_segment_start_time 
            state_length_frame = self.convert_time_to_frames(state_length_ms)   
          
            if self.iterate_over_frames:         
                for t in xrange(int(state_length_frame)):

                    t_in_frames = frame_num

                    frame_num += 1
                    if add_frame_features:  ## osw
                        since_state_start = t / float(state_length_frame) ## just in one direction -- 2 is redundant
                        #frames_till_state_end = state_length_frame - frames_since_state_start
                        extended_data = segment_data + [state_length_frame, since_state_start]
                    else:
                        extended_data = segment_data
                    assert len(extended_data) == self.dimension,'%s %s'%(len(extended_data),self.dimension)
                    label_feature_matrix[t_in_frames,] = extended_data
                total_number_of_frames = max(total_number_of_frames,t_in_frames)
            else:
                ###             
                for t in xrange(this_segment_start_time,this_segment_end_time):
                    # add 1 to the time in milliseconds before converting (and rounding) to frames
                    # this is to ensure the last frame is written to
                    # there may still be some offset between times and frames - STILL TO VERIFY THIS IS CORRECT
                
                    ## osw: time stamps obtained from HVite, -o flag lets you change how
                    # stamps are made, I used the default (HTKBook):
                    # 'By default start times are set to the start time of the frame and 
                    # end times are set to the end time of the frame.'
                
                    t_in_frames = self.convert_time_to_frames(t+1)
                
                    if add_frame_features:  ## osw -- these features are NOT CORRECT: t_in_frames is 
                                    ## time since utt start, not state start
                        frames_since_state_start = t_in_frames
                        frames_till_state_end = state_length_frame - frames_since_state_start
                        extended_data = segment_data + [state_length_frame, frames_since_state_start, frames_till_state_end]
                    else:
                        extended_data = segment_data
                
                    assert len(extended_data) == self.dimension
                
                    label_feature_matrix[t_in_frames,] = extended_data

                    # print "wrote into frame %d (time=%d)" % (t_in_frames,t)

                total_number_of_frames = max(total_number_of_frames,t_in_frames)
                # print "num frames=",total_number_of_frames
                
        logger.debug('loaded %s, %3d elements matching %s' % (file_name_or_descriptor, label_number, self.target_nodes) )

        total_number_of_frames += 1  ## osw: wrote last frame to row with this index -- need to increment
                                     ## to catch this row in the slice that follows

        # trim the matrix to the correct size
        label_feature_matrix = label_feature_matrix[0:total_number_of_frames,]
        
        
        # optionally, set the values for any frames that were missed thus far
        # (this will be because they were missing in the XML tree for some reason)
        #
        # osw -- checked for contiguous segments above -- can skip this 
        if self.fill_missing_values and numpy.isnan(label_feature_matrix).any():
            # currently naive - just fills with _UNSEEN_ (if there is a mapper), or zero (if there is no mapper)
            if self.mapper:
                data = self.mapper[self.unseen_string]
                # replace entire row if any element in that row is nan
                nan_rows=numpy.isnan(label_feature_matrix).any(axis=1)
                logger.debug('before:\n%s' % label_feature_matrix[nan_rows])
                label_feature_matrix[nan_rows,] = data
                logger.debug('after:\n%s' % label_feature_matrix[nan_rows])
            else:
                data=0
                label_feature_matrix[numpy.isnan(label_feature_matrix)] = data

            logger.debug('XPATH %s left some frames undefined; filled them with %s' % (path,data) )

        
        try:
            assert not numpy.isnan(label_feature_matrix).any()
        except AssertionError:
            logger.critical('XPATH %s left some frames undefined' % path )
            # work out which frames this occurred in, then log that information
            l=label_feature_matrix.shape[0]
            frame_times = [self.convert_frames_to_time(f) for f in range(1,l+1)]
            # append frame times as the first column of the matrix, just for logging purposes
            debug_label_feature_matrix = numpy.hstack( (numpy.array(frame_times).reshape((l,1)) , label_feature_matrix) )
            logger.critical(' here are the problem frames with timings:\n%s' % debug_label_feature_matrix[numpy.isnan(debug_label_feature_matrix).any(axis=1)])
            raise
        
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return label_feature_matrix


    def load_xpath_set(self, xpath_file_name):

        logger = logging.getLogger("labels")

        # logger.debug('Opening xpaths file %s' % xpath_file_name)

        try:
            fid = open(xpath_file_name)
        except IOError:
            logger.critical('failed to open XPATHs file %s' % xpath_file_name)
            raise
            
        # each line contains an xpath expression

        # example line in the xpath file
        # l_segment_vsm_d1 =      preceding::segment[1]/attribute::segment_vsm_d1

        # question_index = 0
        xpath_dict = []
        
        for line in fid.readlines():

            line = line.replace('\n', '')
            
            if line.startswith('#') or (not "=" in line):
                continue
            
            (question_name,xpath)=line.split('=',1)
            question_name=question_name.replace(' ','')
            xpath=xpath.replace(' ','')
            logger.debug('loaded question %s with XPATH %s' % (question_name,xpath))

            # store it in the dictionary
            xpath_dict.append((question_name, xpath))
                
        fid.close()

        logger.debug('loaded XPATH set with %d paths' % len(xpath_dict))

        return  xpath_dict


if __name__ == '__main__':
    
    qs_file_name = '/afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2016/straight_voice/Hybrid_duration_experiments/dnn_tts_release/lstm_rnn/data/questions.hed'
    
    print   qs_file_name
    
    ori_file_list = ['/afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2016/straight_voice/Hybrid_duration_experiments/dnn_tts_release/lstm_rnn/data/label_state_align/AMidsummerNightsDream_000_000.lab']
    output_file_list = ['/afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2016/straight_voice/Hybrid_duration_experiments/dnn_tts_release/lstm_rnn/data/binary_label_601/AMidsummerNightsDream_000_000.lab']
    #output_file_list = ['/afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2016/straight_voice/Hybrid_duration_experiments/dnn_tts_release/lstm_rnn/data/dur/AMidsummerNightsDream_000_000.dur']

    label_operater = HTSLabelNormalisation(qs_file_name)
    label_operater.perform_normalisation(ori_file_list, output_file_list)
    #feature_type="binary"
    #unit_size = "phoneme"
    #feat_size = "phoneme"
    #label_operater.prepare_dur_data(ori_file_list, output_file_list, feature_type, unit_size, feat_size)
    #label_operater.prepare_dur_data(ori_file_list, output_file_list, feature_type)
    print   label_operater.dimension

