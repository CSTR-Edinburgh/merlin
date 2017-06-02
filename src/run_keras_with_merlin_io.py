'''
Created on 8 Mar 2017

@author: Srikanth Ronanki
'''

import os 
import time

from keras_lib.train import TrainKerasModels
from keras_lib import data_utils

def main():
  
    start_time = time.time()
     
    ###################################################
    ########## User configurable variables ############ 
    ###################################################

    merlin_dir = "/group/project/cstr1/srikanth/test/merlin"
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

    inp_norm = "MINMAX"
    out_norm = "MINMAX"

    stats_dir    = os.path.join(exp_dir, 'keras_stats')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir) 

    #### Main switch variables ####
 
    NormData   = False
    TrainModel = True
    TestModel  = True

    demo_mode = True

    if demo_mode:
        train_file_number =  50
        valid_file_number =   5
        test_file_number  =   5
    else:
        train_file_number = 1000
        valid_file_number =   66
        test_file_number  =   66
    
    #### Train, valid and test file lists #### 
    
    file_id_scp  = os.path.join(data_dir, 'file_id_list_demo.scp')
    file_id_list = data_utils.read_file_list(file_id_scp)

    train_id_list = file_id_list[0:train_file_number]
    valid_id_list = file_id_list[train_file_number:train_file_number+valid_file_number]
    test_id_list  = file_id_list[train_file_number+valid_file_number:train_file_number+valid_file_number+test_file_number]
    
    inp_train_file_list = data_utils.prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
    out_train_file_list = data_utils.prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)
    
    inp_test_file_list = data_utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
    out_test_file_list = data_utils.prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)
  
    ### set to True if training recurrent models ###
    sequential_training = False
    stateful = False
    
    ### set to True if data to be shuffled ###
    shuffle_data = True
    
    #### define Model, train and evaluate ####
    if sequential_training:
        if demo_mode:
            hidden_layer_type = ['tanh','lstm']
            hidden_layer_size = [ 512  , 512  ]
        else:
            hidden_layer_type = ['tanh','tanh','tanh','tanh','lstm','lstm']
            hidden_layer_size = [ 1024  ,1024  , 1024  , 1024  , 512  , 512  ]
       
        ### batch size: sentences
        batch_size    = 25
        training_algo = 1
    else:
        hidden_layer_type = ['tanh','tanh','tanh','tanh','tanh','tanh']
        hidden_layer_size = [ 1024 , 1024 , 1024 , 1024 , 1024 , 1024 ]
        
        ### batch size: frames
        batch_size = 256

    optimizer     = 'adam'
    output_type   = 'linear'
    loss_function = 'mse'
    
    num_of_epochs = 25
    dropout_rate  = 0.0

    if sequential_training:
        combined_model_arch = 'RNN'+str(training_algo)
    else:
        combined_model_arch = 'DNN'

    combined_model_arch += '_'+str(len(hidden_layer_size))
    combined_model_arch += '_'+'_'.join(map(str, hidden_layer_size))
    combined_model_arch += '_'+'_'.join(map(str, hidden_layer_type))
    
    nnets_file_name = '%s_%d_train_%d_%d_%d_%d_%d_model' \
                      %(combined_model_arch, int(shuffle_data),  
                         inp_dim, out_dim, train_file_number, batch_size, num_of_epochs)
    
    print 'model file    : '+nnets_file_name    
    
    json_model_file = os.path.join(model_dir, nnets_file_name+'.json')
    h5_model_file   = os.path.join(model_dir, nnets_file_name+'.h5')

    inp_stats_file = os.path.join(stats_dir, "input_%d_%s_%d.norm" %(int(train_file_number), inp_norm, inp_dim))
    out_stats_file = os.path.join(stats_dir, "output_%d_%s_%d.norm" %(int(train_file_number), out_norm, out_dim))
    
    inp_scaler = None
    out_scaler = None
        
    gen_dir       = os.path.join(exp_dir, 'gen')
    pred_feat_dir = os.path.join(gen_dir, nnets_file_name)
    if not os.path.exists(pred_feat_dir):
        os.makedirs(pred_feat_dir)
         
    gen_test_file_list = data_utils.prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)
    gen_wav_file_list  = data_utils.prepare_file_path_list(test_id_list, pred_feat_dir, '.wav')
 
    ###################################################
    ########## End of user-defined variables ##########
    ###################################################

    #### Define keras models class ####
    keras_models = TrainKerasModels(inp_dim, hidden_layer_size, out_dim, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)
    
    if NormData:
        ### normalize train data ###
        if os.path.isfile(inp_stats_file):    
            inp_scaler = data_utils.load_norm_stats(inp_stats_file, inp_dim, method=inp_norm)
        else:
            print 'preparing train_x from input feature files...'
            train_x, train_flen_x = data_utils.read_data_from_file_list(inp_train_file_list, inp_dim, False)
            
            print 'computing norm stats for train_x...'
            inp_scaler = data_utils.compute_norm_stats(train_x, inp_stats_file, method=inp_norm)
            
        if os.path.isfile(out_stats_file):    
            out_scaler = data_utils.load_norm_stats(out_stats_file, out_dim, method=out_norm)
        else:    
            print 'preparing train_y from output feature files...'
            train_y, train_flen_y = data_utils.read_data_from_file_list(out_train_file_list, out_dim, False)
            
            print 'computing norm stats for train_y...'
            out_scaler = data_utils.compute_norm_stats(train_y, out_stats_file, method=out_norm)

        
    if TrainModel:
        #### define the model ####
        if not sequential_training:
            keras_models.define_feedforward_model()
        elif stateful:
            keras_models.define_stateful_model()
        else:
            keras_models.define_sequence_model()
        
        #### load the data ####
        print('preparing train_x, train_y from input and output feature files...')
        train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim, sequential_training=sequential_training)

        #### norm the data ####
        print('normalising the data...')
        data_utils.norm_data(train_x, inp_scaler, sequential_training=sequential_training)
        data_utils.norm_data(train_y, out_scaler, sequential_training=sequential_training)
        
        #### train the model ####
        if not sequential_training:
            ### Train feedforward model ###
            keras_models.train_feedforward_model(train_x, train_y, batch_size=batch_size, num_of_epochs=num_of_epochs, shuffle_data=shuffle_data) 
        else:
            ### Train recurrent model ###
            keras_models.train_sequence_model(train_x, train_y, train_flen, batch_size=batch_size, num_of_epochs=num_of_epochs, 
                                                                                        shuffle_data=shuffle_data, training_algo=training_algo) 

        #### store the model ####
        keras_models.save_model(json_model_file, h5_model_file)
   
    if TestModel: 
        #### load the model ####
        keras_models.load_model(json_model_file, h5_model_file)

        #### load the data ####
        print 'preparing test_x from input feature files...'
        test_x, test_y, test_flen = data_utils.read_data_from_file_list(inp_test_file_list, out_test_file_list, inp_dim, out_dim)
     
        #### norm the data ####
        data_utils.norm_data(test_x, inp_scaler)
        
        #### compute predictions ####
        keras_models.predict(test_x, out_scaler, gen_test_file_list, sequential_training)

    (m, s) = divmod(int(time.time() - start_time), 60) 
    print("--- Job completion time: %d min. %d sec ---" % (m, s)) 


if __name__ == "__main__":
    main()
