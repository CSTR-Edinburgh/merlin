
import os
import numpy, re, sys
from io_funcs.binary_io import BinaryIOCollection

import logging
# from logplot.logging_plotting import LoggerPlotter #, MultipleTimeSeriesPlot, SingleWeightMatrixPlot

class HTSLabelModification(object):
    """This class is to modify HTS format labels with predicted duration.
    
    Time alignments are expected in the HTS labels. Here is an example of the HTS labels:
    
    3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]

    3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]

    3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]

    3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]

    3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]

    305000 310000 are the starting and ending time.
    [2], [3], [4], [5], [6] mean the HMM state index. 

    """
    
    def __init__(self, silence_pattern=['*-#+*'], label_type="state_align"):

        logger = logging.getLogger("labels")

        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.label_type = label_type
        self.state_number = 5

    def check_silence_pattern(self, label): 
        for current_pattern in self.silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0
        

    def modify_duration_labels(self, in_gen_label_align_file_list, gen_dur_list, gen_label_list):
        '''
        modifying duration from label alignments with predicted duration.
        '''
        utt_number = len(gen_dur_list)
        if utt_number != len(in_gen_label_align_file_list):
            print   "the number of input and output files should be the same!\n";
            sys.exit(1)
               
        for i in xrange(utt_number):
            if (self.label_type=="state_align"):
                self.modify_dur_from_state_alignment_labels(in_gen_label_align_file_list[i], gen_dur_list[i], gen_label_list[i])
            elif (self.label_type=="phone_align"):
                self.modify_dur_from_phone_alignment_labels(in_gen_label_align_file_list[i], gen_dur_list[i], gen_label_list[i])
            else:
                logger.critical("we don't support %s labels as of now!!" % (self.label_type))
                sys.exit(1)
    
    def modify_dur_from_state_alignment_labels(self, label_file_name, gen_dur_file_name, gen_lab_file_name): 
        logger = logging.getLogger("dur")

        state_number = self.state_number
        dur_dim = state_number
        
        io_funcs = BinaryIOCollection()
        dur_features, frame_number = io_funcs.load_binary_file_frame(gen_dur_file_name, dur_dim)

        fid = open(label_file_name)
        utt_labels = fid.readlines()
        fid.close()
        
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (label_file_name, label_number) )
		
        out_fid = open(gen_lab_file_name, 'w')

        current_index = 0
        prev_end_time = 0
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

            label_binary_flag = self.check_silence_pattern(full_label)
          
            if label_binary_flag == 1:
                current_state_dur = end_time - start_time
                out_fid.write(str(prev_end_time)+' '+str(prev_end_time+current_state_dur)+' '+full_label+'\n')
                prev_end_time = prev_end_time+current_state_dur
                continue;
            else:
                state_dur = dur_features[current_index, state_index-1]
                state_dur = int(state_dur)*5*10000
                out_fid.write(str(prev_end_time)+' '+str(prev_end_time+state_dur)+' '+full_label+'\n')
                prev_end_time = prev_end_time+state_dur
        
            if state_index == state_number:
                current_index += 1
     
        logger.debug('modifed label with predicted duration of %d frames x %d features' % dur_features.shape )
    
    def modify_dur_from_phone_alignment_labels(self, label_file_name, gen_dur_file_name, gen_lab_file_name): 
        logger = logging.getLogger("dur")

        dur_dim = 1
        
        io_funcs = BinaryIOCollection()
        dur_features, frame_number = io_funcs.load_binary_file_frame(gen_dur_file_name, dur_dim)

        fid = open(label_file_name)
        utt_labels = fid.readlines()
        fid.close()
        
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (label_file_name, label_number) )
		
        out_fid = open(gen_lab_file_name, 'w')

        current_index = 0
        prev_end_time = 0
        for line in utt_labels:
            line = line.strip()
            
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            
            full_label = temp_list[2]

            label_binary_flag = self.check_silence_pattern(full_label)
          
            if label_binary_flag == 1:
                current_phone_dur = end_time - start_time
                out_fid.write(str(prev_end_time)+' '+str(prev_end_time+current_phone_dur)+' '+full_label+'\n')
                prev_end_time = prev_end_time+current_phone_dur
                continue;
            else:
                phone_dur = dur_features[current_index]
                phone_dur = int(phone_dur)*5*10000
                out_fid.write(str(prev_end_time)+' '+str(prev_end_time+phone_dur)+' '+full_label+'\n')
                prev_end_time = prev_end_time+phone_dur
        
            current_index += 1
     
        logger.debug('modifed label with predicted duration of %d frames x %d features' % dur_features.shape )
    
