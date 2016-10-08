
from htk_io import HTK_Parm_IO
from htkmfc import HTKFeat_read,HTKFeat_write
import  logging
import  numpy

class   MeanVarianceNorm():
    '''
    plan: 1: support normal MVN and denormalisation for both input and output
          2: support stream-based operation: for example, some streams can use min-max, other streams use MVN, may need one more class
    '''
    def __init__(self, feature_dimension):   
    
        self.mean_vector = None
        self.std_vector  = None     
        self.feature_dimension = feature_dimension

    def feature_normalisation(self, in_file_list, out_file_list):
        logger = logging.getLogger('feature_normalisation')
        
#        self.feature_dimension = feature_dimension
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            logger.critical('The input and output file numbers are not the same! %d vs %d' %(len(in_file_list), len(out_file_list)))
            raise

        if self.mean_vector == None:
            self.mean_vector = self.compute_mean(in_file_list, 0, self.feature_dimension)
        if self.std_vector  == None:
            self.std_vector = self.compute_std(in_file_list, self.mean_vector, 0, self.feature_dimension)
        
        io_funcs = HTKFeat_read()
        file_number = len(in_file_list)
        for i in xrange(file_number):
            features, current_frame_number = io_funcs.getall(in_file_list[i])
#            print   current_frame_number
#            features = io_funcs.data
#            current_frame_number = io_funcs.n_samples

            mean_matrix = numpy.tile(self.mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(self.std_vector, (current_frame_number, 1))
            
            norm_features = (features - mean_matrix) / std_matrix

            htk_writer  = HTKFeat_write(veclen=io_funcs.veclen, sampPeriod=io_funcs.sampPeriod, paramKind=9)            
            htk_writer.writeall(norm_features, out_file_list[i])

#            htk_writter = HTK_Parm_IO(n_samples=io_funcs.n_samples, samp_period=io_funcs.samp_period, samp_size=io_funcs.samp_size, param_kind=io_funcs.param_kind, data=norm_features)    
#            htk_writter.write_htk(out_file_list[i])

        return  self.mean_vector, self.std_vector

    def feature_denormalisation(self, in_file_list, out_file_list, mean_vector, std_vector):
        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            logger.critical('The input and output file numbers are not the same! %d vs %d' %(len(in_file_list), len(out_file_list)))
            raise

        try:
            assert  mean_vector.size == self.feature_dimension and std_vector.size == self.feature_dimension
        except AssertionError:
            logger.critical('the dimensionalities of the mean and standard derivation vectors are not the same as the dimensionality of the feature')
            raise

        for i in xrange(file_number):
            features, current_frame_number = io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))

            norm_features = features * std_matrix + mean_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

    def compute_mean(self, file_list, start_index, end_index):

        logger = logging.getLogger('feature_normalisation')
        
        local_feature_dimension = end_index - start_index
        
        mean_vector = numpy.zeros((1, local_feature_dimension))
        all_frame_number = 0

        io_funcs = HTKFeat_read()
        for file_name in file_list:
            features, current_frame_number = io_funcs.getall(file_name)
#            io_funcs = HTK_Parm_IO()
#            io_funcs.read_htk(file_name)
#            features = io_funcs.data
#            current_frame_number = io_funcs.n_samples

            mean_vector += numpy.reshape(numpy.sum(features[:, start_index:end_index], axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number
            
        mean_vector /= float(all_frame_number)

        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)
        
        self.mean_vector = mean_vector
        
        return  mean_vector
    
    def compute_std(self, file_list, mean_vector, start_index, end_index):
    
        logger = logging.getLogger('feature_normalisation')
        
        local_feature_dimension = end_index - start_index

        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = HTKFeat_read()
        for file_name in file_list:
            features, current_frame_number = io_funcs.getall(file_name)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            
            std_vector += numpy.reshape(numpy.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number
            
        std_vector /= float(all_frame_number)
        
        std_vector = std_vector ** 0.5
        
        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)
        
        self.std_vector = std_vector
        
        return  std_vector
