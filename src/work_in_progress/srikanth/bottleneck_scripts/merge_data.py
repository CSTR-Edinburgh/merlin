
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
    
    file_paths = []
    filenames = []
    for root, directories, files in os.walk(dir_name):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
            filenames.append(filename)

    return  file_paths, filenames


def generate_context_feature(in_data_dir1, in_data_dir2, out_data_dir, dimension1, dimension2):

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)    

    file_paths, filenames = read_file_list(in_data_dir1)
    
    context_features = numpy

    i = 0
    for file_path, filename in zip(file_paths, filenames):
        features1, frame_number1 = load_binary_file(file_path, dimension1)
        features2, frame_number2 = load_binary_file(os.path.join(in_data_dir2, filename), dimension2) 
        if frame_number1 != frame_number2:
            print dimension2
            print filename
            print   "%s %d != %d" %(filename, frame_number1, frame_number2)
            print features1.shape, features2.shape
            os.exit(1)

        context_features = numpy.zeros((frame_number1, dimension1+dimension2))
        
        context_features[0:frame_number1, 0:dimension1] = features1
        context_features[0:frame_number2, dimension1:dimension1+dimension2] = features2
        
        print   filename, features1.shape, features2.shape, context_features.shape
        
        context_filename = out_data_dir + '/' + filename

        context_features = numpy.asarray(context_features, 'float32')
        fid = open(context_filename, 'wb')
        context_features.tofile(fid)
        fid.close()
        

if __name__ == '__main__':
    in_dir1 = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Merlin/test_version/dnn_tts/experiments/acoustic_model/data/nn_no_silence_lab_norm_490'
    in_dir2 = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Merlin/test_version/dnn_tts/experiments/acoustic_model/gen/DNN_TANH_TANH_TANH_TANH_LINEAR__mgc_lf0_vuv_bap_1_200_490_259_4_512_512_hidden_stacked/'

    dimension1 = 490
    dimension2 = 32*21 # 128 * 1

    out_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Merlin/test_version/dnn_tts/experiments/acoustic_model/data/nn_no_silence_lab_norm_1162' 

    generate_context_feature(in_dir1, in_dir2, out_dir, dimension1, dimension2)
