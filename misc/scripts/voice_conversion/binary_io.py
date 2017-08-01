

import numpy

class   BinaryIOCollection(object):

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
        features = features[:(dimension * (features.size / dimension))]
        features = features.reshape((-1, dimension))

        return  features

    def array_to_binary_file(self, data, output_file_name):
        data = numpy.array(data, 'float32')

        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

    def load_binary_file_frame(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
        frame_number = features.size / dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))

        return  features, frame_number

    def load_binary_dtw_file(self, file_name, dimension=2):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype="int32")
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
        frame_number = features.size / dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))
        
        feat_path_dict = {}
        for i in range(frame_number):
            feat_path_dict[features[i][1]] = features[i][0]
        
        return  feat_path_dict
    
    def load_ascii_dtw_file(self, file_name):
        fid_lab = open(file_name, 'r')
        data = fid_lab.readlines()
        fid_lab.close()

        feat_path_dict = {}
        for newline in data[0:-1]:
            temp_list = newline.strip().split()
            feat_path_dict[int(temp_list[0])] = int(temp_list[1])
        
        return  feat_path_dict
