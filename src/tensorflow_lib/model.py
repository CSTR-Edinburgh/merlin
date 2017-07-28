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

import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, BasicLSTMCell,GRUCell, DropoutWrapper


class TensorflowModels(object):

  def __init__ (self,n_in,hidden_layer_size,n_out,hidden_layer_type,output_type="linear",dropout_rate=0,loss_function="mse",optimizer="adam"):

        #self.session=tf.InteractiveSession()
        self.n_in  = int(n_in)
        self.n_out = int(n_out)

        self.n_layers = len(hidden_layer_size)

        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_type = hidden_layer_type

        assert len(self.hidden_layer_size) == len(self.hidden_layer_type)

        self.output_type   = output_type
        self.dropout_rate  = dropout_rate
        self.loss_function = loss_function
        self.optimizer     = optimizer
        #self.activation    ={"tanh":tf.nn.tanh,"sigmoid":tf.nn.sigmoid}
        self.graph=tf.Graph()
        #self.saver=tf.train.Saver()


  def define_feedforward_model(self):
      seed=12345
      np.random.seed(seed)
      layer_list=[]
      with self.graph.as_default() as g:
          #initializer=tf.contrib.layers.variance_scaling_initializer()
          with tf.name_scope("input"):
              input_layer=tf.placeholder(dtype=tf.float64,shape=(None,self.n_in),name="input_layer")
              if self.dropout_rate!=0.0:
                 print "Using dropout to avoid overfitting and the dropout rate is",self.dropout_rate
                 is_training=tf.placeholder(dtype=tf.bool,shape=(),name="is_training")
                 input_layer_drop=dropout(input_layer,self.dropout_rate,is_training=is_training)
                 layer_list.append(input_layer_drop)
                 g.add_to_collection(name="is_training",value=is_training)
              else:
                 layer_list.append(input_layer)
               #g.add_to_collection(name="input_layer",value=input_layer)
          for i in xrange(len(self.hidden_layer_size)):
              with tf.name_scope("hidden_layer_"+str(i+1)):
                if self.dropout_rate!=0.0:
                    last_layer=layer_list[-1]
                    if self.hidden_layer_type[i]=="tanh":
                       new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.tanh)
                    if self.hidden_layer_type[i]=="sigmoid":
                        new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid)
                    new_layer_drop=dropout(new_layer,self.dropout_rate,is_training=is_training)
                    layer_list.append(new_layer_drop)
                else:
                    last_layer=layer_list[-1]
                    if self.hidden_layer_type[i]=="tanh":
                       new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.tanh)
                    #self.layers_list.append(layer)
                    #print self.layers_list[-1]
                    if self.hidden_layer_type[i]=="sigmoid":
                       new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid)
                    layer_list.append(new_layer)
          g.add_to_collection(name="layer_list",value=layer_list)
          with tf.name_scope("output_layer"):
              if self.output_type=="linear":
                 output_layer=fully_connected(layer_list[-1],self.n_out,activation_fn=None)
              if self.output_type=="tanh":
                 output_layer=fully_connected(layer_list[-1],self.n_out,activation_fn=tf.nn.tanh)
              g.add_to_collection(name="output_layer",value=output_layer)
          with tf.name_scope("training_op"):
               if self.optimizer=="adam":
                  self.training_op=tf.train.AdamOptimizer()
       # initializer=tf.contrib.layers.variance_scaling_initializer()
       # with tf.name_scope("feedfoward_model"):
       #     for i in range(len(self.hidden_layer_size)):
       #         hidden_layer=fully_connected(layers_list[i],self.hidden_layer_size[i],weights_initializer=initializer,
        #            activation_fn=self.activation[self.hidden_layer_type[i]],biases_initializer=tf.zeros_initializer(),scope="hidden"+str(i+1))
  def define_sequence_model(self):
      seed=12345
      np.random.seed(12345)
      #self.sentence_length=[]
      layer_list=[]
      with self.graph.as_default() as g:
          #initializer=tf.contrib.layers.variance_scaling_initializer()
          utt_length=tf.placeholder(tf.int64,shape=(None))
          g.add_to_collection(name="utt_length",value=utt_length)
          with tf.name_scope("input"):
               input_layer=tf.placeholder(dtype=tf.float64,shape=(None,None,self.n_in),name="input_layer")
               if self.dropout_rate!=0.0:
                  print "Using dropout to avoid overfitting and the dropout rate is",self.dropout_rate
                  is_training=tf.placeholder(dtype=tf.bool,shape=(),name="is_training")
                  input_layer_drop=dropout(input_layer,self.dropout_rate,is_training=is_training)
                  layer_list.append(input_layer_drop)
                  g.add_to_collection(name="is_training",value=is_training)
               else:
                  layer_list.append(input_layer)
          #for i in xrange(len(self.hidden_layer_size)):
          with tf.name_scope("hidden_layer"):
             basic_cell=[]
             for i in xrange(len(self.hidden_layer_type)):
                 if self.dropout_rate!=0.0:
                     if self.hidden_layer_type[i]=="tanh":
                         basic_cell.append(MyDropoutWrapper(BasicRNNCell(num_units=self.hidden_layer_size[i]),input_keep_prob=self.dropout_rate,is_training=is_training))
                     if self.hidden_layer_type[i]=="lstm":
                         basic_cell.append(MyDropoutWrapper(BasicLSTMCell(num_units=self.hidden_layer_size[i]),input_keep_prob=self.dropout_rate,is_training=is_training))
                     if self.hidden_layer_type[i]=="gru":
                         basic_cell.append(MyDropoutWrapper(GRUCell(num_units=self.hidden_layer_size[i]),input_keep_prob=self.dropout_rate,is_training=is_training))
                 else:
                     if self.hidden_layer_type[i]=="tanh":
                        basic_cell.append(BasicRNNCell(num_units=self.hidden_layer_size[i]))
                     if self.hidden_layer_type[i]=="lstm":
                        basic_cell.append(BasicLSTMCell(num_units=self.hidden_layer_size[i]))
                     if self.hidden_layer_type[i]=="GRU":
                        basic_cell.append(GRUCell(num_untis=self.hidden_layer_size[i]))
             multi_cell=MultiRNNCell(basic_cell)
             rnn_outputs,rnn_states=tf.nn.dynamic_rnn(multi_cell,layer_list[0],dtype=tf.float64,sequence_length=utt_length)
             layer_list.append(rnn_outputs)
             g.add_to_collection(name="layer_list",value=layer_list)
          with tf.name_scope("output_layer"):
               if self.output_type=="linear" :
                  #stacked_rnn_outputs=tf.reshape(layer_list[-1],[-1,self.hidden_layer_size[-1]])
                  #stacked_outputs=fully_connected(stacked_rnn_outputs,self.n_out,activation_fn=None)
                  #output_layer=tf.reshape(stacked_outputs,[-1,self.max_step,self.n_out])
                  output_layer=tf.layers.dense(rnn_outputs,self.n_out)
                  print output_layer.shape
                  layer_list.append(output_layer)
               if self.output_type=="tanh":
                  output_cell=BasicRNNCell(num_units=self.n_out)
                  output_layer=tf.nn.dynamic_rnn(output_cell,layer_list[-1],tf.float64,utt_length)
                  layer_list.append(output_layer)
               g.add_to_collection(name="output_layer",value=output_layer)
          with tf.name_scope("training_op"):
               if self.optimizer=="adam":
                   self.training_op=tf.train.AdamOptimizer()

  def get_max_step(self,max_step):
       ##This method is only used when a sequence model is TrainTensorflowModels
       self.max_step=max_step

class MyDropoutWrapper(DropoutWrapper):

    def __init__(self, cell, is_training,input_keep_prob=1.0, output_keep_prob=1.0,
               state_keep_prob=1.0, variational_recurrent=False,
                input_size=None, dtype=None, seed=None):

        DropoutWrapper.__init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               state_keep_prob=1.0, variational_recurrent=False,
               input_size=None, dtype=None, seed=None)
        self.is_training=is_training

    def __call__(self, inputs, state, scope=None):

           return self._cell(dropout(inputs,self._input_keep_prob,is_training=self.is_training,scope=None),state,scope=None)

class Encoder_Decoder_Models(TensorflowModels):

      def __init__(self,n_in,encoder_layer_size,n_out,encoder_layer_type,output_type="linear",dropout_rate=0,loss_function="mse",optimizer="adam",attention=False):
          TensorflowModels.__init__(self,n_in,encoder_layer_size,n_out,encoder_layer_type,output_type="linear",dropout_rate=0,loss_function="mse",optimizer="adam")
          self.encoder_layer_size=self.hidden_layer_size
          self.encoder_layer_type=self.hidden_layer_type
          self.attention=attention

      def encoder(self,inputs,inputs_sequence_length):
           with tf.variable_scope("encoder"):
                basic_cell=[]
                for i in xrange(len(self.hidden_layer_size)):
                    if self.hidden_layer_type[i]=="tanh":
                        basic_cell.append(tf.contrib.rnn.BasicRNNCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="lstm":
                        basic_cell.append(tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="gru":
                         basic_cell.append(tf.contrib.rnn.GRUCell(num_units=self.encoder_layer_size[i]))
                multicell=MultiRNNCell(basic_cell)
                enc_output, enc_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell,cell_bw=multicell,inputs=inputs,\
                             sequence_length=inputs_sequence_length,dtype=tf.float64)
                return enc_output, enc_state

      def process_decoder_input(self,target_sequence):
          decode_input=tf.concat((tf.zeros_like(target_sequence[:, :1, :]), target_sequence[:, :-1, :]), 1)

          return decode_input

      def decoder(self,decoder_inputs,enc_output,enc_states,target_sequence_length):
          """Memory is a tuple containing the forward and backward final states (output_states_fw,output_states_bw)"""
          with tf.variable_scope("decoder"):
              basic_cell=[]
              for i in xrange(len(self.hidden_layer_size)):
                    if self.hidden_layer_type[i]=="tanh":
                        basic_cell.append(tf.contrib.rnn.BasicRNNCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="lstm":
                        basic_cell.append(tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="gru":
                         basic_cell.append(tf.contrib.rnn.GRUCell(num_units=self.encoder_layer_size[i]))
              multicell=MultiRNNCell(basic_cell)
          if not self.attention:
              dec_output,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell,cell_bw=multicell,inputs=decoder_inputs,initial_state_fw=enc_states[0],\
                                                           initial_state_bw=enc_states[1],sequence_length=target_sequence_length)
          else:
              attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(self.hidden_layer_size[-1],enc_output,normalize=True,probability_fn=tf.nn.softmax)
              cell_with_attention=tf.contrib.seq2seq.AttentionWrapper(multicell,attention_mechanism,self.hidden_layer_size[-1])
              dec_output,_=tf.nn.bidirectional_dyanamic_rnn(cell_fw=cell_with_attention,cell_bw=cell_with_attention,inputs=decoder_inputs,\
                                                            sequence_length=target_sequence_length,initial_state_fw=enc_states[0],initial_state_bw=enc_states[1],dtype=tf.float64)
          return dec_output

      def define_encoder_decoder(self):
          with self.graph.as_default() as g:
              with tf.name_scope("encoder_input"):
                  inputs_data=tf.placeholder(dtype=tf.float64,shape=[None,None,self.n_in],name="inputs_data")
                  inputs_sequence_length=tf.placeholder(tf.int64,shape=[None],name="inputs_sequence_length")
              with tf.name_scope("target_sequence"):
                   targets=tf.placeholder(dtype=tf.float64,shape=[None,None,self.n_out],name="targets")
                   target_sequence_length=tf.placeholder(tf.int64,shape=[None],name="target_sequence_length")
                   #max_target_length=tf.reduce_max(target_sequence_length)
              #assert inputs_sequence_length==target_sequence_length
              with tf.name_scope("encoder_output"):
                   enc_out,enc_states=self.encoder(inputs_data,inputs_sequence_length)
              with tf.name_scope("decoder_inputs"):
                   dec_inputs=self.process_decoder_input(targets)
              with tf.name_scope("decoder_outputs"):
                   dec_output=self.decoder(dec_inputs,enc_out,enc_states,target_sequence_length)
                   dec_output=tf.concat(dec_output,2)
              with tf.name_scope("outputs"):
                  if self.output_type=="linear":
                     outputs=tf.layers.dense(dec_output,self.n_out)
                  g.add_to_collection(name="decoder_outputs",value=outputs)
              with tf.name_scope("training_op"):
                   if self.optimizer=="adam":
                       self.training_op=tf.train.AdamOptimizer()
