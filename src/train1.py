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
           input_layer=g.get_collection(name="input_layer")[0]
           is_training_batch=g.get_collection(name="is_training_batch")[0]
           if self.dropout_rate!=0.0:
              is_training_drop=g.get_collection(name="is_training_drop")[0]
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
             if shuffle_data:
                indices=range(train_x.shape[0])
                random.shuffle(indices)
                train_x=train_x[indices,]
                train_y=train_y[indices]
             for epoch in xrange(num_of_epochs):
                 for iteration in range(int(train_x.shape[0]/batch_size)):
                    x_batch,y_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size]
                    if self.dropout_rate!=0.0:
                       sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,is_training_drop:True,is_training_batch:True})
                    else:
                       sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,is_training_batch:True})
                 if self.dropout_rate!=0.0:
                    training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y,is_training_drop:False,is_training_batch:False})
                 else:
                    training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y,is_training_batch:False})
                 print "Epoch ",epoch+1,"Training loss:",training_loss
             self.saver.save(sess,"./temp_checkpoint_file/model.ckpt")
             print "The model parameters are saved"

    def get_batch(self,train_x,train_y,start,batch_size):

        utt_keys=train_x.keys()
        batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
        batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
        batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
        utt_len_batch=[len(batch_x_dict[k])for k in batch_x_dict.keys()]
        return batch_x_dict, batch_y_dict, utt_len_batch

    def train_sequence_model(self,train_x,train_y,utt_length,batch_size=1,num_of_epochs=10,shuffle_data=False):
        seed=12345
        np.random.seed(seed)
        #Data Preparation
        temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step, shuffle_data=shuffle_data)
        print("Input shape: "+str(temp_train_x.shape))
        temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step, shuffle_data=shuffle_data)
        print("Output shape: "+str(temp_train_y.shape))
        #Shuffle the data

        with self.graph.as_default() as g:
            output_layer=g.get_collection(name="output_layer")[0]
            input_layer=g.get_collection(name="input_layer")[0]
            utt_length_placeholder=g.get_collection(name="utt_length")[0]
            hybrid=0
            if "tanh" in self.hidden_layer_type:
                hybrid=1
                is_training_batch=g.get_collection(name="is_training_batch")[0]
            if self.dropout_rate!=0.0:
                is_training_drop=g.get_collection(name="is_training_drop")[0]
            with tf.name_scope("output_data"):
               output_data=tf.placeholder(tf.float32,shape=(None,None,self.n_out))
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
                     for iteration in range(int(len(train_x.keys())/batch_size)):
                        x_batch_dict,y_batch_dict,utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                        assert [len(v) for v in x_batch_dict.values()]==[len(v) for v in y_batch_dict.values()]
                        assert x_batch_dict.keys()==y_batch_dict.keys()
                        max_length_batch=max(utt_length_batch)
                        x_batch=data_utils.transform_data_to_3d_matrix(x_batch_dict, max_length=max_length_batch, shuffle_data=False)
                        y_batch=data_utils.transform_data_to_3d_matrix(y_batch_dict, max_length=max_length_batch, shuffle_data=False)

                        if self.dropout_rate!=0.0:
                           if hybrid:
                              sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,\
                                       is_training_drop:True,is_training_batch:True})
                           else:
                              sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,\
                                       is_training_drop:True})
                        elif hybrid:
                           sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,is_training_batch:True})
                        else:
                           sess.run(self.training_op,feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch})
                     if self.dropout_rate!=0.0:
                         if hybrid:
                            training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,\
                            is_training_drop:False,is_training_batch:False})
                         else:
                            training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,\
                            is_training_drop:False})
                     elif hybrid:
                         training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,is_training_batch:False})
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
        new_saver=tf.train.import_meta_graph("./temp_checkpoint_file/model.ckpt.meta")
        output_layer=tf.get_collection("output_layer")[0]
        input_layer=tf.get_collection("input_layer")[0]
        with tf.Session() as sess:
                  new_saver.restore(sess,"./temp_checkpoint_file/model.ckpt")
                  print "The model parameters are successfully restored"
                  if not sequential_training:
                     is_training_batch=tf.get_collection("is_training_batch")[0]
                     if self.dropout_rate!=0.0:
                        is_training_drop=tf.get_collection("is_training_drop")[0]
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_drop:False,is_training_batch:False})
                     else:
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_batch:False})
                  else:
                        hybrid=0
                        utt_length_placeholder=tf.get_collection("utt_length")[0]
                        utt_length=[len(utt) for utt in test_x.values()]
                        if "tanh" in self.hidden_layer_type:
                            hybrid=1
                            is_training_batch=tf.get_collection("is_training_batch")[0]
                        max_step=max(utt_length)
                        temp_test_x = data_utils.transform_data_to_3d_matrix(test_x, max_length=max_step,shuffle_data=True)
                        if self.dropout_rate!=0.0:
                           is_training_drop=tf.get_collection("is_training_drop")[0]
                           if hybrid:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length,is_training_drop:False,is_training_batch:False})
                           else:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length,is_training_drop:False})
                        elif hybrid:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length,is_training_batch:False})
                        else:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:utt_length})
        data_utils.denorm_data(y_predict, out_scaler)
        #print y_predict
        io_funcs.array_to_binary_file(y_predict, gen_test_file_name)
        data_utils.drawProgressBar(utt_index+1, test_file_number)
    sys.stdout.write("\n")

class Train_Encoder_Decoder_Models(Encoder_Decoder_Models):

      def __init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam',attention=False,cbhg=False):
          Encoder_Decoder_Models.__init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', \
                                optimizer='adam',attention=attention,cbhg=cbhg)

      def get_batch(self,train_x,train_y,start,batch_size):

             utt_keys=train_x.keys()
             batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
             batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
             batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
             utt_len_batch=[len(batch_x_dict[k])for k in batch_x_dict.keys()]
             return batch_x_dict, batch_y_dict, utt_len_batch


      def train_encoder_decoder_model(self,train_x,train_y,utt_length,batch_size=1,num_of_epochs=10,shuffle_data=False):
          temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step,shuffle_data=False)
          print("Input shape: "+str(temp_train_x.shape))
          temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step,shuffle_data=False)
          print("Output shape: "+str(temp_train_y.shape))

          with self.graph.as_default() as g:
               outputs=g.get_collection(name="decoder_outputs")[0]
               var=g.get_collection(name="trainable_variables")
               targets=g.get_tensor_by_name("target_sequence/targets:0")
               inputs_data=g.get_tensor_by_name("encoder_input/inputs_data:0")
               if not self.cbhg:
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
                        x_batch_dict,y_batch_dict,utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                        assert [len(v) for v in x_batch_dict.values()]==[len(v) for v in y_batch_dict.values()]
                        assert x_batch_dict.keys()==y_batch_dict.keys()
                        max_length_batch=max(utt_length_batch)
                        x_batch=data_utils.transform_data_to_3d_matrix(x_batch_dict, max_length=max_length_batch, shuffle_data=False)
                        y_batch=data_utils.transform_data_to_3d_matrix(y_batch_dict, max_length=max_length_batch, shuffle_data=False)
                        if self.cbhg:
                            sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,target_sequence_length:utt_length_batch})
                        else:
                            sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,inputs_sequence_length:utt_length_batch,target_sequence_length:utt_length_batch})
                     if self.cbhg:
                         training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,target_sequence_length:utt_length})
                     else:
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

         utt_length=[len(utt) for utt in test_x.values()]
         max_step=max(utt_length)
         temp_test_x = data_utils.transform_data_to_3d_matrix(test_x, max_length=max_step,shuffle_data=False)
         new_saver=tf.train.import_meta_graph("./temp_checkpoint_file/model.ckpt.meta")
         #with self.graph.as_default() as g:
         targets=tf.get_collection("targets")[0]
         inputs_data=tf.get_collection("inputs_data")[0]
         decoder_outputs=tf.get_collection("decoder_outputs")[0]
         inputs_sequence_length=tf.get_collection("inputs_sequence_length")[0]
         target_sequence_length=tf.get_collection("target_sequence_length")[0]
         outputs=np.zeros(shape=[len(test_x),max_step,self.n_out],dtype=np.float64)
         with tf.Session() as sess:
                print "loading the model parameters..."
                new_saver.restore(sess,"./temp_checkpoint_file/model.ckpt")
                print "Model parameters are successfully restored"
                print "Generating speech parameters ..."
                for t in range(max_step):
                  _outputs=sess.run(decoder_outputs,feed_dict={inputs_data:temp_test_x,targets:outputs,inputs_sequence_length:utt_length,\
                              target_sequence_length:utt_length})
                  #print _outputs[:,t,:]
                  outputs[:,t,:]=_outputs[:,t,:]

         data_utils.denorm_data(outputs, out_scaler)
        #print y_predict
         io_funcs.array_to_binary_file(outputs, gen_test_file_name)
         data_utils.drawProgressBar(utt_index+1, test_file_number)
      sys.stdout.write("\n")
