#!/usr/bin/env python

import numpy, os
import matplotlib.pyplot as plt

def load_binary_file(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = numpy.fromfile(fid_lab, dtype=numpy.float32)
    fid_lab.close()
    frame_number = features.size / dimension
    features = features[:(dimension * (features.size / dimension))]
    features = features.reshape((-1, dimension))
        
    return  features, frame_number

def read_file_list(dir_name):
    
    #file_paths = []
    #filenames = []
    #for root, directories, files in os.walk(dir_name):
    #    for filename in files:
    #        filepath = os.path.join(root, filename)
    #        file_paths.append(filepath)
    #        filenames.append(filename)
    
    #filenames=filter(os.path.isfile, os.listdir(dir_name))
    filenames=[ f for f in os.listdir(dir_name) if os.path.isfile(dir_name+'/'+f) ]
    #for f in os.listdir(dir_name):
    #    if os.path.isfile(dir_name+'/'+f):
    #        print dir_name+'/'+f+' is a file'
    #    else:
    #        print dir_name+'/'+f+' is not a file'
    #print filenames
    #file_paths=[ dir_name+'/'+f for f in os.listdir(dir_name) if os.path.isfile(f) ]
    file_paths=[ dir_name+'/'+f for f in filenames ]
    
    return  file_paths, filenames


def generate_context_feature(in_data_dir, out_data_dir, context_width, dimension):

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)    

    #print in_data_dir
    file_paths, filenames = read_file_list(in_data_dir)
    
    window_size = context_width * 2 + 1

    file_index = 0
    i = 0
    for file_path, filename in zip(file_paths, filenames):
        features, frame_number = load_binary_file(file_path, dimension)

        print   file_index, features.shape, filename
        expand_features = numpy.zeros((frame_number+window_size-1, dimension))
        for ci in xrange(context_width):
            expand_features[ci, :] = features[0, :]
            expand_features[frame_number+context_width+ci, :] = features[frame_number-1, :]

        expand_features[context_width:frame_number+context_width, :] = features

        context_features = numpy.zeros((frame_number, dimension*window_size))

        for wi in xrange(window_size):
            context_features[0:frame_number, wi*dimension:(wi+1)*dimension] = expand_features[wi:frame_number+wi, :]

        context_filename = out_data_dir + '/' + os.path.splitext(filename)[0] + '.lab'

        context_features = numpy.asarray(context_features, 'float32')
        fid = open(context_filename, 'wb')
        context_features.tofile(fid)
        fid.close()
        
        file_index = file_index + 1

if __name__ == '__main__':
    in_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Merlin/test_version/dnn_tts/experiments/acoustic_model/gen/DNN_TANH_TANH_TANH_TANH_LINEAR__mgc_lf0_vuv_bap_1_200_490_259_4_512_512_hidden/'

    dimension = 32 # 128
    context_width = 10 # 0
    out_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Merlin/test_version/dnn_tts/experiments/acoustic_model/gen/DNN_TANH_TANH_TANH_TANH_LINEAR__mgc_lf0_vuv_bap_1_200_490_259_4_512_512_hidden_stacked/'

    generate_context_feature(in_dir, out_dir, context_width, dimension)




