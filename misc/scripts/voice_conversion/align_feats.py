
import numpy

from binary_io import BinaryIOCollection

class AlignFeats(object):
    def __init__(self):
        self.io_funcs = BinaryIOCollection()

    def align_src_feats(self, src_feat_file, src_aligned_feat_file, feat_dim, dtw_path_dict):
        '''
        align source feats as per the dtw path (matching target length)
        '''
        src_features, frame_number = self.io_funcs.load_binary_file_frame(src_feat_file, feat_dim)
        
        tgt_length = len(dtw_path_dict)
        src_aligned_features = numpy.zeros((tgt_length, feat_dim))
        
        for i in range(tgt_length):
            src_aligned_features[i, ] = src_features[dtw_path_dict[i]]

        self.io_funcs.array_to_binary_file(src_aligned_features, src_aligned_feat_file)
