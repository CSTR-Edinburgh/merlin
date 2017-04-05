import os, sys 
import regex as re
import time
import random
import numpy as np

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM
from keras.layers import Dropout

from io_funcs.binary_io import BinaryIOCollection

BUFFER_SIZE = 4000000

class kerasModels(object):
    
    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam'):
        """ This function initialises a neural network
        
        :param n_in: Dimensionality of input features
        :param hidden_layer_size: The layer size for each hidden layer
        :param n_out: Dimensionality of output features
        :param hidden_layer_type: the activation types of each hidden layers, e.g., TANH, LSTM, GRU, BLSTM
        :param output_type: the activation type of the output layer, by default is 'LINEAR', linear regression.
        :param dropout_rate: probability of dropout, a float number between 0 and 1.
        :type n_in: Integer
        :type hidden_layer_size: A list of integers
        :type n_out: Integrer
        """
        
        self.n_in  = int(n_in)
        self.n_out = int(n_out)
        
        self.n_layers = len(hidden_layer_size)
        
        assert len(hidden_layer_size) == len(hidden_layer_type)
       
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_type = hidden_layer_type
        
        self.output_type   = output_type 
        self.dropout_rate  = dropout_rate
        self.loss_function = loss_function
        self.optimizer     = optimizer

        print "output_type   : "+self.output_type
        print "loss function : "+self.loss_function
        print "optimizer     : "+self.optimizer

    def define_baseline_model(self):
        seed = 12345
        np.random.seed(seed)
        
        # create model
        self.model = Sequential()

        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layer_size[i - 1]

            self.model.add(Dense(
                    units=self.hidden_layer_size[i],
                    activation=self.hidden_layer_type[i],
                    kernel_initializer="normal",
                    input_dim=input_size))
            self.model.add(Dropout(self.dropout_rate))

        # add output layer
        self.final_layer = self.model.add(Dense(
            units=self.n_out,
            activation=self.output_type.lower(),
            kernel_initializer="normal",
            input_dim=self.hidden_layer_size[-1]))

        # Compile the model
        self.compile_model()

    def define_sequence_model(self):
        seed = 12345
        np.random.seed(seed)
        
        # create model
        self.model = Sequential()

        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layer_size[i - 1]
           
            if hidden_layer_type[i]=='lstm': 
                self.model.add(LSTM(
                        units=self.hidden_layer_size[i],
                        input_shape=(None, input_size),
                        return_sequences=True))
            else:
                self.model.add(Dense(
                        units=self.hidden_layer_size[i],
                        activation=self.hidden_layer_type[i],
                        kernel_initializer="normal",
                        input_shape=(None, input_size)))

        # add output layer
        self.final_layer = self.model.add(Dense(
            units=self.n_out,
            input_dim=self.hidden_layer_size[-1],
            kernel_initializer='normal',
            activation=self.output_type.lower()))

        # Compile the model
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        print "model compiled successfully!!"

def read_data_from_file_list(inp_file_list, out_file_list, inp_dim, out_dim, sequential_training=True): 
    io_funcs = BinaryIOCollection()

    utt_len = len(inp_file_list)

    file_length_dict = {}

    if sequential_training:
        temp_set_x = {}
        temp_set_y = {}
    else:
        temp_set_x = np.empty((BUFFER_SIZE, inp_dim))
        temp_set_y = np.empty((BUFFER_SIZE, out_dim))
     
    ### read file by file ###
    current_index = 0
    for i in range(utt_len):    
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
            temp_set_x[base_file_name] = inp_features 
            temp_set_y[base_file_name] = out_features 
        else:
            temp_set_x[current_index:current_index+frame_number, ] = inp_features
            temp_set_y[current_index:current_index+frame_number, ] = out_features
            current_index += frame_number
        
        if frame_number not in file_length_dict:
            file_length_dict[frame_number] = [base_file_name]
        else:
            file_length_dict[frame_number].append(base_file_name)
        
        print_status(i, utt_len)

    sys.stdout.write("\n")

    if not sequential_training:
        temp_set_x = temp_set_x[0:current_index, ]
        temp_set_y = temp_set_y[0:current_index, ]
    
    return temp_set_x, temp_set_y, file_length_dict

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
    pr = int(float(i+1)/float(length)*100)
    st = int(float(pr)/7)
    sys.stdout.write(("\r%d/%d ")%(i+1,length)+("[ %d"%pr+"% ] <<< ")+('='*st)+(''*(100-st)))
    sys.stdout.flush()

            
if __name__ == "__main__":
  
    start_time = time.time()
     
    #### User configurable variables #### 
    
    merlin_dir = "/afs/inf.ed.ac.uk/group/cstr/projects/merlin"
    exp_dir    = os.path.join(merlin_dir, "egs/slt_arctic/s1/experiments/slt_arctic_demo/acoustic_model/")

    inp_dim = 425
    out_dim = 187

    data_dir     = os.path.join(exp_dir, "data")
    inp_feat_dir = os.path.join(data_dir, 'nn_no_silence_lab_norm_'+str(inp_dim))
    out_feat_dir = os.path.join(data_dir, 'nn_norm_mgc_lf0_vuv_bap_'+str(out_dim))
    
    inp_file_ext = '.lab'
    out_file_ext = '.cmp'

    model_dir    = os.path.join(exp_dir, 'keras_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 

    gen_dir      = os.path.join(exp_dir, 'gen_keras')
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir) 

    file_id_scp  = os.path.join(data_dir, 'file_id_list_demo.scp')
    file_id_list = read_file_list(file_id_scp)

    train_file_number = 50
    valid_file_number =  5
    test_file_number  =  5
    
    #### Train, valid and test file lists #### 

    train_id_list = file_id_list[0:train_file_number]
    valid_id_list = file_id_list[train_file_number:train_file_number+valid_file_number]
    test_id_list  = file_id_list[train_file_number+valid_file_number:train_file_number+valid_file_number+test_file_number]
    
    inp_train_file_list = prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
    out_train_file_list = prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)
    
    inp_test_file_list = prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
    out_test_file_list = prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)
   
    ### set to True if training recurrent models ###
    sequential_training = False

    ### set to True if data to be shuffled ###
    shuffle_data        = True
    
    print 'preparing train_x, train_y from input and output feature files...'
    train_x, train_y, train_flen = read_data_from_file_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim, sequential_training=sequential_training)
     
    print 'preparing test_x, test_y from input and output feature files...'
    test_x, test_y, test_flen = read_data_from_file_list(inp_test_file_list, out_test_file_list, inp_dim, out_dim)
     
    #### define Model, train and evaluate ####
    
    if sequential_training:
        hidden_layer_type = ['tanh','tanh','tanh','tanh','lstm','lstm']
        hidden_layer_size = [ 512  , 512  , 512  , 512  , 512  , 512  ]
        batch_size = 1 ## 1 sentence as a batch
    else:
        hidden_layer_type = ['tanh','tanh','tanh','tanh','tanh','tanh']
        hidden_layer_size = [ 1024 , 1024 , 1024 , 1024 , 1024 , 1024 ]
        batch_size = 256 ## 256 frames as a batch

    optimizer     = 'adam'
    output_type   = 'linear'
    loss_function = 'mse'
    
    num_epochs    = 10
    dropout_rate  = 0.0

    TrainModel = True
    TestModel  = True

    diph_classifier = kerasModels(inp_dim, hidden_layer_size, out_dim, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

    if sequential_training:
        combined_model_arch = 'RNN'
    else:
        combined_model_arch = 'DNN'

    combined_model_arch += '_'+str(len(hidden_layer_size))
    combined_model_arch += '_'+'_'.join(map(str, hidden_layer_size))
    combined_model_arch += '_'+'_'.join(map(str, hidden_layer_type))
    
    nnets_file_name = '%s_%d_train_%d_%d_%d_%d_%d_model' \
                      %(combined_model_arch, int(shuffle_data),  
                         inp_dim, out_dim, train_file_number, batch_size, num_epochs)
    
    print 'model file    : '+nnets_file_name    
    
    json_model_file = os.path.join(model_dir, nnets_file_name+'.json')
    h5_model_file   = os.path.join(model_dir, nnets_file_name+'.h5')
    

    pred_feat_dir = os.path.join(gen_dir, nnets_file_name)
    if not os.path.exists(pred_feat_dir):
        os.makedirs(pred_feat_dir)
         
    gen_test_file_list = prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)
    gen_wav_file_list  = prepare_file_path_list(test_id_list, pred_feat_dir, '.wav')
    
    if not TrainModel: 
        #### load the model ####
        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_model_file)
        print("Loaded model from disk")
    
        #### compile the model ####
        diph_classifier.model = loaded_model
        diph_classifier.compile_model()
        model = diph_classifier.model

    else:
        #### define the model ####
        if sequential_training:
            diph_classifier.define_sequence_model()
        else:
            diph_classifier.define_baseline_model()
    
        model = diph_classifier.model
 
        #### train the model ####
    
        if not sequential_training:
            ### Train DNN model ###
            model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, shuffle=shuffle_data)
        else:
            ### Train recurrent model ###
            if batch_size==1:
                ### train each sentence as a batch ###
                train_index_list = range(train_file_number)
                if shuffle_data:
                    random.seed(271638)
                    random.shuffle(train_index_list)        
                for epoch_num in xrange(num_epochs):
                    print 'Epoch: %d/%d ' %(epoch_num+1, num_epochs)
                    utt_count = -1
                    for utt_index in train_index_list:
                        temp_train_x = train_x[train_id_list[utt_index]]
                        temp_train_y = train_y[train_id_list[utt_index]]
                        temp_train_x = np.reshape(temp_train_x, (1, temp_train_x.shape[0], inp_dim))
                        temp_train_y = np.reshape(temp_train_y, (1, temp_train_y.shape[0], out_dim))
                        model.train_on_batch(temp_train_x, temp_train_y)
                        #model.fit(temp_train_x, temp_train_y, epochs=1, shuffle=False, verbose=0)
                        utt_count += 1
                        print_status(utt_count, train_file_number)

                    sys.stdout.write("\n")
            else:
                ### if batch size more than 1 ###
                train_count_list = train_flen.keys()
                if shuffle_data:
                    random.seed(271638)
                    random.shuffle(train_count_list)
                for epoch_num in xrange(num_epochs):
                    print 'Epoch: %d/%d ' %(epoch_num+1, num_epochs)
                    utt_count = -1
                    for frame_number in train_count_list:
                        batch_file_list = train_flen[frame_number]
                        num_of_files    = len(batch_file_list)
                        temp_train_x    = np.zeros((num_of_files, frame_number, inp_dim))
                        temp_train_y    = np.zeros((num_of_files, frame_number, out_dim))
                        for file_index in xrange(num_of_files):
                            temp_train_x[file_index, ] = train_x[batch_file_list[file_index]]
                            temp_train_y[file_index, ] = train_y[batch_file_list[file_index]]
                        model.fit(temp_train_x, temp_train_y, batch_size=batch_size, epochs=1, verbose=0)
                        utt_count += num_of_files
                        print_status(utt_count, train_file_number)

                    sys.stdout.write("\n")

        #### store the model ####

        # serialize model to JSON
        model_json = model.to_json()
        with open(json_model_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(h5_model_file)
        print("Saved model to disk")
   
    if TestModel: 
        #### calculate predictions ####
        
        print "generating acoustic features on held-out test data..."
        if not sequential_training:
            for utt_index in xrange(test_file_number):
                temp_test_x = test_x[test_id_list[utt_index]]
                temp_test_y = test_y[test_id_list[utt_index]]
                
                predictions = model.predict(temp_test_x)
                fid = open(gen_test_file_list[utt_index], 'wb')
                predictions.tofile(fid)
                fid.close()
            
                print_status(utt_index, test_file_number)
            
            sys.stdout.write("\n")
        else:
            for utt_index in xrange(test_file_number):
                temp_test_x = test_x[test_id_list[utt_index]]
                temp_test_y = test_y[test_id_list[utt_index]]
                temp_test_x = np.reshape(temp_test_x, (1, temp_test_x.shape[0], inp_dim))
                temp_test_y = np.reshape(temp_test_y, (1, temp_test_y.shape[0], out_dim))
                
                predictions = model.predict(temp_test_x)
                fid = open(gen_test_file_list[utt_index], 'wb')
                predictions.tofile(fid)
                fid.close()
            
                print_status(utt_index, test_file_number)

            sys.stdout.write("\n")
        
    (m, s) = divmod(int(time.time() - start_time), 60) 
    print("--- Job completion time: %d min. %d sec ---" % (m, s)) 
