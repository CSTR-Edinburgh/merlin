#!/usr/bin/env python
################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

import tensorflow as tf
import numpy as np
import random, os ,sys
from io_funcs.binary_io import BinaryIOCollection
from tensorflow_lib.model1 import TensorflowModels, Encoder_Decoder_Models
from tensorflow_lib import data_utils

class TrainTensorflowModels(TensorflowModels):

    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam', rnn_params=None):

        TensorflowModels.__init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

        #### TODO: Find a good way to pass below params ####
        self.merge_size   = rnn_params['merge_size']
        self.seq_length   = rnn_params['seq_length']
        self.bucket_range = rnn_params['bucket_range']

        self.stateful = rnn_params['stateful']
        pass;

    def train_feedforward_model(self, train_x, train_y, batch_size=256, num_of_epochs=10, shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        print "Input Shape: "+str(train_x.shape)
        print "Output Shape: "+str(train_y.shape)
        with self.graph.as_default() as g:
           output_data=tf.placeholder(dtype=tf.float64,shape=(None,self.n_out),name="output_data")
           input_layer=g.get_collection(name="layer_list")[0][0]
           if self.dropout_rate!=0.0:
              is_training=g.get_collection(name="is_training")[0]
           with tf.name_scope("loss"):
               #final_hidden_layer=g.get_collection(name="hidden_layer"+str(len(self.hidden_layer_size)+1))
               output_layer=g.get_collection(name="output_layer")[0]
               loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")

           with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(loss)

           init=tf.global_variables_initializer()
           self.saver=tf.train.Saver()
           with tf.Session() as sess:
             init.run()
             #input_layer=g.get_collection(name="input_layer")
             if shuffle_data:
                indices=range(train_x.shape[0])
                random.shuffle(indices)
                train_x=train_x[indices,]
                train_y=train_y[indices]
             for epoch in xrange(num_of_epochs):
                 for iteration in range(int(train_x.shape[0]/batch_size)):
            #       # print type(self.input_data)
                    x_batch,y_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size]
                    if self.dropout_rate!=0.0:
                       sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,is_training:True})
                    else:
                       sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch})
                 if self.dropout_rate!=0.0:
                    training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y,is_training:False})
                 else:
                    training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y})
                 print "Epoch ",epoch+1,"Training loss:",training_loss
             self.saver.save(sess,"./temp_checkpoint_file/model.ckpt")
             print "The model parameters are saved"
          # save_path=saver.save(sess,"./temp/tensorflow_deep_feedforward.ckpt")

    def train_sequence_model(self,train_x,train_y,utt_length,batch_size=1,num_of_epochs=10,shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        #Data Preparation

        temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step, shuffle_data=shuffle_data)
        print("Input shape: "+str(temp_train_x.shape))
        temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step, shuffle_data=shuffle_data)
        print("Output shape: "+str(temp_train_y.shape))
        #Shuffle the data
        if shuffle_data==True:
            indices=range(temp_train_x.shape[0])
            random.shuffle(indices)
            temp_train_x=temp_train_x[indices,]
            temp_train_y=temp_train_y[indices]


        with self.graph.as_default() as g:
            #self.define_sequence_model()
            output_layer=g.get_collection(name="output_layer")[0]
            input_layer=g.get_collection(name="layer_list")[0][0]
            utt_length_placeholder=g.get_collection(name="utt_length")[0]
            if self.dropout_rate!=0.0:
                is_training=g.get_collection(name="is_training")[0]
            with tf.name_scope("output_data"):
               output_data=tf.placeholder(tf.float64,shape=(None,None,self.n_out))
            with tf.name_scope("loss"):
                error=output_data-output_layer
                loss=tf.reduce_mean(tf.square(error),name="loss")
            with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(loss)
            init=tf.global_variables_initializer()
            self.saver=tf.train.Saver()
            with tf.Session() as sess:
                 init.run()
                 for epoch in xrange(num_of_epochs):
                     for iteration in range(int(temp_train_x.shape[0]/batch_size)):
                        x_batch,y_batch=temp_train_x[iteration*batch_size:(iteration+1)*batch_size,], temp_train_y[iteration*batch_size:(iteration+1)*batch_size]
                        utt_length_batch=utt_length[iteration*batch_size:(iteration+1)*batch_size]
                        #print len(utt_length_batch)
                        if self.dropout_rate!=0.0:
                           sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,is_training:True})
                        else:
                           sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch})
                     if self.dropout_rate!=0.0:
                         training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,is_training:False})
                     else:
                         training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length})
                     print "Epoch ",epoch+1,"Training loss:",training_loss
                 self.saver.save(sess,"./temp_checkpoint_file/model.ckpt")
                 print "The model parameters are saved"

    def predict(self, test_x, out_scaler, gen_test_file_list, sequential_training=False, stateful=False):
        #### compute predictions ####

        io_funcs = BinaryIOCollection()

        test_id_list = test_x.keys()
        test_id_list.sort()

        test_file_number = len(test_id_list)

        print("generating features on held-out test data...")
        for utt_index in xrange(test_file_number):
            gen_test_file_name = gen_test_file_list[utt_index]
            temp_test_x        = test_x[test_id_list[utt_index]]
            num_of_rows        = temp_test_x.shape[0]
        with self.graph.as_default() as g:
           # init=tf.global_variables_initializer()
            #saver=tf.train.Saver()
            output_layer=g.get_collection(name="output_layer")[0]
            input_layer=g.get_collection(name="layer_list")[0][0]
            with tf.Session() as sess:
                  self.saver.restore(sess,"./temp_checkpoint_file/model.ckpt")
                  print "The model parameters are successfully restored"
                  if not sequential_training:
                     if self.dropout_rate!=0:
                        is_training=g.get_collection(name="is_training")[0]
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training:False})
                     else:
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x})
                  else:
                        utt_length_placeholder=g.get_collection(name="utt_length")[0]
                        utt_length=[len(utt) for utt in test_x.values()]
                        max_step=max(utt_length)
                        temp_test_x = data_utils.transform_data_to_3d_matrix(test_x, max_length=max_step,shuffle_data=False)
                        if self.dropout_rate!=0.0:
                           is_training=g.get_collection(name="is_training")[0]
                           y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length,is_training:False})
                        else:
                           y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length})
        data_utils.denorm_data(y_predict, out_scaler)
        #print y_predict
        io_funcs.array_to_binary_file(y_predict, gen_test_file_name)
        data_utils.drawProgressBar(utt_index+1, test_file_number)
    sys.stdout.write("\n")

class Train_Encoder_Decoder_Models(Encoder_Decoder_Models):

      def __init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam',attention=False):
          Encoder_Decoder_Models.__init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam',attention=False)


      def train_encoder_decoder_model(self,train_x,train_y,utt_length,batch_size=1,num_of_epochs=10,shuffle_data=True):
          temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step,shuffle_data=shuffle_data)
          print("Input shape: "+str(temp_train_x.shape))
          temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step,shuffle_data=shuffle_data)
          print("Output shape: "+str(temp_train_y.shape))
        #Shuffle the data
          #print temp_train_x.shape, temp_train_y.shape
          with self.graph.as_default() as g:
               outputs=g.get_collection(name="decoder_outputs")[0]
               var=g.get_collection(name="trainable_variables")
               targets=g.get_tensor_by_name("target_sequence/targets:0")
               inputs_data=g.get_tensor_by_name("encoder_input/inputs_data:0")
               inputs_sequence_length=g.get_tensor_by_name("encoder_input/inputs_sequence_length:0")
               target_sequence_length=g.get_tensor_by_name("target_sequence/target_sequence_length:0")
               with tf.name_scope("loss"):
                   error=targets-outputs
                   loss=tf.reduce_mean(tf.square(error))
               gradients=self.training_op.compute_gradients(loss)
               capped_gradients=[(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
               self.training_op=self.training_op.apply_gradients(capped_gradients)
               init=tf.global_variables_initializer()
               self.saver=tf.train.Saver()
               with tf.Session() as sess:
                 init.run()
                 for epoch in xrange(num_of_epochs):
                     for iteration in range(int(temp_train_x.shape[0]/batch_size)):
                        x_batch,y_batch=temp_train_x[iteration*batch_size:(iteration+1)*batch_size,], temp_train_y[iteration*batch_size:(iteration+1)*batch_size]
                        utt_length_batch=utt_length[iteration*batch_size:(iteration+1)*batch_size]
                        sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,inputs_sequence_length:utt_length_batch,target_sequence_length:utt_length_batch})
                        #print tf.gradients(loss,var)
                        #g=sess.run([grad[0] for grad in gradients],feed_dict={inputs_data:x_batch,targets:y_batch,inputs_sequence_length:utt_length,target_sequence_length:utt_length})
                     training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,inputs_sequence_length:utt_length,target_sequence_length:utt_length})
                     print "Epoch:",epoch+1, "Training loss:",training_loss
                 self.saver.save(sess,"./temp_checkpoint_file/model.ckpt")
                 print "The model parameters are saved"

      def predict(self,test_x, out_scaler, gen_test_file_list):
          #### compute predictions ####

         io_funcs = BinaryIOCollection()

         test_id_list = test_x.keys()
         test_id_list.sort()

         test_file_number = len(test_id_list)

         print("generating features on held-out test data...")
         for utt_index in xrange(test_file_number):
            gen_test_file_name = gen_test_file_list[utt_index]
            temp_test_x        = test_x[test_id_list[utt_index]]
            num_of_rows        = temp_test_x.shape[0]

         timesteps=[len(utt) for utt in test_x.values()]
         max_step=max(timesteps)
         temp_test_x = data_utils.transform_data_to_3d_matrix(test_x, max_length=max_step,shuffle_data=False)
         new_saver=tf.train.import_meta_graph("./temp_checkpoint_file/model.ckpt.meta")
         with self.graph.as_default() as g:
           # init=tf.global_variables_initializer()
            #saver=tf.train.Saver()
            targets=g.get_tensor_by_name("target_sequence/targets:0")
            inputs_data=g.get_tensor_by_name("encoder_input/inputs_data:0")
            outputs=np.zeros(shape=[len(test_x),max_step,self.n_out],dtype=np.float64)
            inputs_sequence_length=g.get_tensor_by_name("encoder_input/inputs_sequence_length:0")
            target_sequence_length=g.get_tensor_by_name("target_sequence/target_sequence_length:0")
            with tf.Session() as sess:
                print "loading the model parameters..."
                self.saver.restore(sess,"./temp_checkpoint_file/model.ckpt")
                print "Model parameters are successfully restored"
                for t in timesteps:
                   for j in xrange(1,t+1):
                     _outputs=sess.run(targets,feed_dict={inputs_data:temp_test_x,targets:outputs,inputs_sequence_length:timesteps,target_sequence_length:range(1,t+1)})
                     outputs[:,j-1,:]=_outputs[:,j-1,:]
         data_utils.denorm_data(outputs, out_scaler)
        #print y_predict
         io_funcs.array_to_binary_file(outputs, gen_test_file_name)
         data_utils.drawProgressBar(utt_index+1, test_file_number)
      sys.stdout.write("\n")
