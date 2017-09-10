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
from tensorflow_lib.model import TensorflowModels, Encoder_Decoder_Models
from tensorflow_lib import data_utils


class TrainTensorflowModels(TensorflowModels):

    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, model_dir,output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam', rnn_params=None):

        TensorflowModels.__init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

        #### TODO: Find a good way to pass below params ####
        self.ckpt_dir = model_dir

    def train_feedforward_model(self, train_x, train_y, batch_size=256, num_of_epochs=10, shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        print train_x.shape
        with self.graph.as_default() as g:
           output_data=tf.placeholder(dtype=tf.float32,shape=(None,self.n_out),name="output_data")
           input_layer=g.get_collection(name="input_layer")[0]
           is_training_batch=g.get_collection(name="is_training_batch")[0]
           if self.dropout_rate!=0.0:
              is_training_drop=g.get_collection(name="is_training_drop")[0]
           with tf.name_scope("loss"):
               output_layer=g.get_collection(name="output_layer")[0]
               loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")
           with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(loss)
           init=tf.global_variables_initializer()
           self.saver=tf.train.Saver()
           with tf.Session() as sess:
             init.run();summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
             for epoch in xrange(num_of_epochs):
                 L=1;overall_loss=0
                 for iteration in range(int(train_x.shape[0]/batch_size)+1):
                    if (iteration+1)*batch_size>train_x.shape[0]:
                        x_batch,y_batch=train_x[iteration*batch_size:],train_y[iteration*batch_size:]
                        if x_batch!=[]:
                           L+=1
                        else:continue
                    else:
                        x_batch,y_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size]
                        L+=1
                    if self.dropout_rate!=0.0:
                       _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_drop:True,is_training_batch:True})
                       #rs=sess.run(merged,feed_dict={input_layer:x_batch,output_data:y_batch,is_training_drop:True,is_training_batch:True})
                    else:
                       _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_batch:True})
                       #rs=sess.run(merged,feed_dict={input_layer:x_batch,output_data:y_batch,is_training_batch:True})
                    overall_loss+=batch_loss
            #if self.dropout_rate!=0.0:
            #        training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y,is_training_drop:False,is_training_batch:False})
            #     else:
            #        training_loss=loss.eval(feed_dict={input_layer:train_x,output_data:train_y,is_training_batch:False})
                 print "Epoch ",epoch+1, "Finishes","Training loss:",overall_loss/L
             self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
             print "The model parameters are saved"

    def get_batch(self,train_x,train_y,start,batch_size=50):
        utt_keys=train_x.keys()
        if (start+1)*batch_size>len(utt_keys):
            batch_keys=utt_keys[start*batch_size:]
        else:
           batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
        batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
        batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
        utt_len_batch=[len(batch_x_dict[k])for k in batch_x_dict.keys()]
        return batch_x_dict, batch_y_dict, utt_len_batch

    def train_sequence_model(self,train_x,train_y,utt_length,batch_size=256,num_of_epochs=10,shuffle_data=False):
        seed=12345
        np.random.seed(seed)
        #Data Preparation
        temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step, shuffle_data=False)
        print("Input shape: "+str(temp_train_x.shape))
        temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step, shuffle_data=False)
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
            #overall_loss=tf.summary.scalar("training loss",overall_loss)
            with tf.Session() as sess:
                 init.run();summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                 for epoch in xrange(num_of_epochs):
                     L=1;overall_loss=0
                     for iteration in range(int(len(train_x.keys())/batch_size)+1):
                        x_batch,y_batch,utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                        if utt_length_batch==[]:
                            continue
                        else:L+=1
                        max_length_batch=max(utt_length_batch)
                        x_batch=data_utils.transform_data_to_3d_matrix(x_batch, max_length=max_length_batch, shuffle_data=False)
                        y_batch=data_utils.transform_data_to_3d_matrix(y_batch, max_length=max_length_batch, shuffle_data=False)
                        if self.dropout_rate!=0.0:
                           if hybrid:
                              _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,\
                                       is_training_drop:True,is_training_batch:True})
                           else:
                             _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,\
                                       is_training_drop:True})
                        elif hybrid:
                           _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch,is_training_batch:True})
                        else:
                           _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,utt_length_placeholder:utt_length_batch})
                     overall_loss+=batch_loss
                     #summary_writer.add_summary(overall_loss,epoch)
                     #if self.dropout_rate!=0.0:
                     #    if hybrid:
                     #       training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,\
                     #       is_training_drop:False,is_training_batch:False})
                     #    else:
                     #       training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,\
                     #       is_training_drop:False})
                     #elif hybrid:
                     #    training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length,is_training_batch:False})
                     #else:
                     #    training_loss=loss.eval(feed_dict={input_layer:temp_train_x,output_data:temp_train_y,utt_length_placeholder:utt_length})
                     print "Epoch ",epoch+1,"Training loss:",overall_loss/L
                 #model_name="sequence_model"+" hybrid.ckpt" if hybrid==1 else "sequence_model.ckpt"
                 self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                 print "The model parameters are saved"

    def predict(self, test_x, out_scaler, gen_test_file_list, sequential_training=False, stateful=False):
        #### compute predictions ####

        io_funcs = BinaryIOCollection()

        test_id_list = test_x.keys()
        test_id_list.sort()

        test_file_number = len(test_id_list)

        print("generating features on held-out test data...")
        with tf.Session() as sess:
           new_saver=tf.train.import_meta_graph(os.path.join(self.ckpt_dir,"mymodel.ckpt.meta"))
           print "loading the model parameters..."
           output_layer=tf.get_collection("output_layer")[0]
           input_layer=tf.get_collection("input_layer")[0]
           new_saver.restore(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
           print "The model parameters are successfully restored"
           for utt_index in xrange(test_file_number):
               gen_test_file_name = gen_test_file_list[utt_index]
               temp_test_x        = test_x[test_id_list[utt_index]]
               num_of_rows        = temp_test_x.shape[0]
               if not sequential_training:
                   is_training_batch=tf.get_collection("is_training_batch")[0]
                   if self.dropout_rate!=0.0:
                        is_training_drop=tf.get_collection("is_training_drop")[0]
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_drop:False,is_training_batch:False})
                   else:
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_batch:False})
               else:
                        temp_test_x=np.reshape(temp_test_x,[1,num_of_rows,self.n_in])
                        hybrid=0
                        utt_length_placeholder=tf.get_collection("utt_length")[0]
                        if "tanh" in self.hidden_layer_type:
                            hybrid=1
                            is_training_batch=tf.get_collection("is_training_batch")[0]
                        if self.dropout_rate!=0.0:
                           is_training_drop=tf.get_collection("is_training_drop")[0]
                           if hybrid:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_drop:False,is_training_batch:False})
                           else:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_drop:False})
                        elif hybrid:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_batch:False})
                        else:
                              y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows]})
               data_utils.denorm_data(y_predict, out_scaler)
               io_funcs.array_to_binary_file(y_predict, gen_test_file_name)
               data_utils.drawProgressBar(utt_index+1, test_file_number)
    sys.stdout.write("\n")

class Train_Encoder_Decoder_Models(Encoder_Decoder_Models):

      def __init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, model_dir,output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam',attention=False,cbhg=False):
          Encoder_Decoder_Models.__init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear', dropout_rate=0.0, loss_function='mse', \
                                optimizer='adam',attention=attention,cbhg=cbhg)
          self.ckpt_dir=os.path.join(model_dir,"temp_checkpoint_file")

      def get_batch(self,train_x,train_y,start,batch_size):

             utt_keys=train_x.keys()
             if (start+1)*batch_size>len(utt_keys):
                 batch_keys=utt_keys[start*batch_size:]
             else:
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
               overall_loss=0;tf.summary.scalar("training_loss",overall_loss)
               with tf.Session() as sess:
                 init.run();tf.summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                 for epoch in xrange(num_of_epochs):
                     L=1
                     for iteration in range(int(temp_train_x.shape[0]/batch_size)+1):
                        x_batch_dict,y_batch_dict,utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                        if utt_length_batch==[]:
                            continue
                        else:L+=1
                        assert [len(v) for v in x_batch_dict.values()]==[len(v) for v in y_batch_dict.values()]
                        assert x_batch_dict.keys()==y_batch_dict.keys()
                        max_length_batch=max(utt_length_batch)
                        x_batch=data_utils.transform_data_to_3d_matrix(x_batch_dict, max_length=max_length_batch, shuffle_data=False)
                        y_batch=data_utils.transform_data_to_3d_matrix(y_batch_dict, max_length=max_length_batch, shuffle_data=False)
                        if self.cbhg:
                           _,batch_loss=sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,target_sequence_length:utt_length_batch})
                        else:
                           _,batch_loss=sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,inputs_sequence_length:utt_length_batch,target_sequence_length:utt_length_batch})
                        overall_loss+=batch_loss
                     #if self.cbhg:
                     #    training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,target_sequence_length:utt_length})
                     #else:
                     #    training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,inputs_sequence_length:utt_length,target_sequence_length:utt_length})
                     print "Epoch:",epoch+1, "Training loss:",overall_loss/L
                     summary_writer.add_summary(str(overall_loss),epoch)
                 self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                 print "The model parameters are saved"

      def predict(self,test_x, out_scaler, gen_test_file_list):
          #### compute predictions ####

         io_funcs = BinaryIOCollection()

         test_id_list = test_x.keys()
         test_id_list.sort()
         inference_batch_size=len(test_id_list)
         test_file_number = len(test_id_list)
         with tf.Session(graph=self.graph) as sess:
             new_saver=tf.train.import_meta_graph(self.ckpt_dir,"mymodel.ckpt.meta")
             """Notice change targets=tf.get_collection("targets")[0]"""
             inputs_data=self.graph.get_collection("inputs_data")[0]
             """Notice Change decoder_outputs=tf.get_collection("decoder_outputs")[0]"""
             inputs_sequence_length=self.graph.get_collection("inputs_sequence_length")[0]
             target_sequence_length=self.graph.get_collection("target_sequence_length")[0]
             print "loading the model parameters..."
             new_saver.restore(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
             print "Model parameters are successfully restored"
             print("generating features on held-out test data...")
             for utt_index in xrange(test_file_number):
               gen_test_file_name = gen_test_file_list[utt_index]
               temp_test_x        = test_x[test_id_list[utt_index]]
               num_of_rows        = temp_test_x.shape[0]

         #utt_length=[len(utt) for utt in test_x.values()]
         #max_step=max(utt_length)
               temp_test_x = tf.reshape(temp_test_x,[1,num_of_rows,self.n_in])

               outputs=np.zeros(shape=[len(test_x),max_step,self.n_out],dtype=np.float32)
                #dec_cell=self.graph.get_collection("decoder_cell")[0]
               print "Generating speech parameters ..."
               for t in range(num_of_rows):
                 #  outputs=sess.run(inference_output,{inputs_data:temp_test_x,inputs_sequence_length:utt_length,\
                #            target_sequence_length:utt_length})
                   _outputs=sess.run(decoder_outputs,feed_dict={inputs_data:temp_test_x,targets:outputs,inputs_sequence_length:[num_of_rows],\
                             target_sequence_length:[num_of_rows]})
                #   #print _outputs[:,t,:]
                   outputs[:,t,:]=_outputs[:,t,:]

               data_utils.denorm_data(outputs, out_scaler)
               io_funcs.array_to_binary_file(outputs, gen_test_file_name)
               data_utils.drawProgressBar(utt_index+1, test_file_number)
      sys.stdout.write("\n")
