
import numpy, sys
from io_funcs.binary_io import BinaryIOCollection

import logging

class MergeFeat(object):
    
    def __init__(self, lab_dim = 481, feat_dim = 1):

        self.logger = logging.getLogger("labels")

        self.lab_dim = lab_dim
        self.feat_dim = feat_dim
            

    
    def merge_data(self, binary_label_file_list, new_feat_file_list, out_feat_file_list):
        '''
        merging new features with normalised label features
        '''
        utt_number = len(new_feat_file_list)
        if utt_number != len(binary_label_file_list):
            print   "the number of new feature input files and label files should be the same!\n";
            sys.exit(1)
        
        new_feat_ext   = new_feat_file_list[0].split('/')[-1].split('.')[1]
               
        io_funcs = BinaryIOCollection()
        for i in xrange(utt_number):
            lab_file_name = binary_label_file_list[i]
            new_feat_file_name = new_feat_file_list[i]
            out_feat_file_name = out_feat_file_list[i]
            
            lab_features, lab_frame_number  = io_funcs.load_binary_file_frame(lab_file_name, self.lab_dim)
            new_features, feat_frame_number = io_funcs.load_binary_file_frame(new_feat_file_name, self.feat_dim)
            
             
            if (lab_frame_number - feat_frame_number)>5:
                base_file_name = new_feat_file_list[i].split('/')[-1].split('.')[0]
                self.logger.critical("the number of frames in label and new features are different: %d vs %d (%s)" %(lab_frame_number, feat_frame_number, base_file_name))
                raise

            merged_features = numpy.zeros((lab_frame_number, self.lab_dim+self.feat_dim))
        
            merged_features[0:lab_frame_number, 0:self.lab_dim] = lab_features
            merged_features[0:feat_frame_number, self.lab_dim:self.lab_dim+self.feat_dim] = new_features[0:lab_frame_number, ]
        
            io_funcs.array_to_binary_file(merged_features, out_feat_file_name)
            self.logger.debug('merged new feature %s of %d frames with %d label features' % (new_feat_ext, feat_frame_number,lab_frame_number) )
            
            