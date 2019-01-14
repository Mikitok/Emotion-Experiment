from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time
import numpy as np
import tensorflow as tf
import read_data
import data_helper
from random import shuffle
import random
import pickle
import math
from keras.preprocessing.sequence import pad_sequences
from general_utils import Progbar
from sklearn.model_selection import StratifiedKFold
import tensorflow.contrib.slim as slim
from sst_config import Config

filename='sinaheadline'
vector_path='./run_result/'+filename+'_vectors'
file_path='./data/nopunc_'+filename+'.pkl'
iteration_num = 3
window_size = 1
dataset_name = 'sinaheadline'
model_name = 'lstm'

def lstm_layer(initial_hidden_states, config, keep_prob, mask):
    with tf.variable_scope('forward'):
        fw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
        fw_lstm = tf.contrib.rnn.DropoutWrapper(fw_lstm, output_keep_prob=keep_prob)

    with tf.variable_scope('backward'):
        bw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
        bw_lstm = tf.contrib.rnn.DropoutWrapper(bw_lstm, output_keep_prob=keep_prob)


    #bidirectional rnn
    with tf.variable_scope('bilstm'):
        lstm_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=initial_hidden_states, sequence_length=mask, time_major=False, dtype=tf.float32)
        lstm_output=tf.concat(lstm_output[0], 2)

    return lstm_output

class Classifer(object):

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        #padding zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,:-step,:]
        #concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)
        #return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        #padding zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,step:,:]
        #concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)
        #return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def sum_together(self, l):
        combined_state=None
        for tensor in l:
            if combined_state==None:
                combined_state=tensor
            else:
                combined_state=combined_state+tensor
        return combined_state
    
    def slstm_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states, num_layers):
        with tf.name_scope(name_scope_name):
            #Word parameters 
            #forget gate for left 
            with tf.name_scope("f1_gate"):
                #current
                Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                #left right
                Whf1 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                #initial state
                Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                #dummy node
                Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for right 
            with tf.name_scope("f2_gate"):
                Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf2 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for inital states     
            with tf.name_scope("f3_gate"):
                Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf3 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for dummy states     
            with tf.name_scope("f4_gate"):
                Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf4 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #input gate for current state     
            with tf.name_scope("i_gate"):
                Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxi")
                Whi = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whi")
                Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wii")
                Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdi")
            #input gate for output gate
            with tf.name_scope("o_gate"):
                Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                Who = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
                Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wio")
                Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdo")
            #bias for the gates    
            with tf.name_scope("biases"):
                bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf1")
                bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf2")
                bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf3")
                bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf4")

            #dummy node gated attention parameters
            #input gate for dummy state
            with tf.name_scope("gated_d_gate"):
                gated_Wxd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                gated_Whd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            #output gate
            with tf.name_scope("gated_o_gate"):
                gated_Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Who = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #forget gate for states of word
            with tf.name_scope("gated_f_gate"):
                gated_Wxf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Whf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #biases
            with tf.name_scope("gated_biases"):
                gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")

        #filters for attention        
        mask_softmax_score=tf.cast(tf.sequence_mask(lengths), tf.float32)*1e25-1e25
        mask_softmax_score_expanded=tf.expand_dims(mask_softmax_score, dim=2)               
        #filter invalid steps
        sequence_mask=tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32),axis=2)
        #filter embedding states
        initial_hidden_states=initial_hidden_states*sequence_mask
        initial_cell_states=initial_cell_states*sequence_mask
        #record shape of the batch
        shape=tf.shape(initial_hidden_states)
        
        #initial embedding states
        embedding_hidden_state=tf.reshape(initial_hidden_states, [-1, hidden_size])      
        embedding_cell_state=tf.reshape(initial_cell_states, [-1, hidden_size])

        #randomly initialize the states
        if config.random_initialize:
            initial_hidden_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
            initial_cell_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
            #filter it
            initial_hidden_states=initial_hidden_states*sequence_mask
            initial_cell_states=initial_cell_states*sequence_mask

        #inital dummy node states
        dummynode_hidden_states=tf.reduce_mean(initial_hidden_states, axis=1)
        dummynode_cell_states=tf.reduce_mean(initial_cell_states, axis=1)

        for i in range(num_layers):
            #update dummy node states
            #average states
            combined_word_hidden_state=tf.reduce_mean(initial_hidden_states, axis=1)
            reshaped_hidden_output=tf.reshape(initial_hidden_states, [-1, hidden_size])
            #copy dummy states for computing forget gate
            transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            #input gate
            gated_d_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd
            )
            #output gate
            gated_o_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo
            )
            #forget gate for hidden states
            gated_f_t = tf.nn.sigmoid(
                tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf
            )

            #softmax on each hidden dimension 
            reshaped_gated_f_t=tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size])+ mask_softmax_score_expanded
            gated_softmax_scores=tf.nn.softmax(tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
            #split the softmax scores
            new_reshaped_gated_f_t=gated_softmax_scores[:,:shape[1],:]
            new_gated_d_t=gated_softmax_scores[:,shape[1]:,:]
            #new dummy states
            dummy_c_t=tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) + tf.squeeze(new_gated_d_t, axis=1)*dummynode_cell_states
            dummy_h_t=gated_o_t * tf.nn.tanh(dummy_c_t)

            #update word node states
            #get states before
            initial_hidden_states_before=[tf.reshape(self.get_hidden_states_before(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_hidden_states_before=self.sum_together(initial_hidden_states_before)
            initial_hidden_states_after= [tf.reshape(self.get_hidden_states_after(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_hidden_states_after=self.sum_together(initial_hidden_states_after)
            #get states after
            initial_cell_states_before=[tf.reshape(self.get_hidden_states_before(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_cell_states_before=self.sum_together(initial_cell_states_before)
            initial_cell_states_after=[tf.reshape(self.get_hidden_states_after(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_cell_states_after=self.sum_together(initial_cell_states_after)
            
            #reshape for matmul
            initial_hidden_states=tf.reshape(initial_hidden_states, [-1, hidden_size])
            initial_cell_states=tf.reshape(initial_cell_states, [-1, hidden_size])

            #concat before and after hidden states
            concat_before_after=tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

            #copy dummy node states 
            transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            transformed_dummynode_cell_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1],1]), [-1, hidden_size])

            f1_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) + 
                tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1)+ bf1
            )

            f2_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) + 
                tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2)+ bf2
            )

            f3_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) + 
                tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
            )

            f4_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) + 
                tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
            )
            
            i_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) + 
                tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi)+ bi
            )
            
            o_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) + 
                tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
            )
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1),tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)


            five_gates=tf.concat([f1_t, f2_t, f3_t, f4_t,i_t], axis=1)
            five_gates=tf.nn.softmax(five_gates, dim=1)
            f1_t,f2_t,f3_t, f4_t,i_t= tf.split(five_gates, num_or_size_splits=5, axis=1)
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1),tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1),tf.squeeze(i_t, axis=1)

            c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after)+(f3_t * embedding_cell_state)+ (f4_t * transformed_dummynode_cell_states)+ (i_t * initial_cell_states)
            
            h_t = o_t * tf.nn.tanh(c_t)

            #update states
            initial_hidden_states=tf.reshape(h_t, [shape[0], shape[1], hidden_size])
            initial_cell_states=tf.reshape(c_t, [shape[0], shape[1], hidden_size])
            initial_hidden_states=initial_hidden_states*sequence_mask
            initial_cell_states=initial_cell_states*sequence_mask

            dummynode_hidden_states=dummy_h_t
            dummynode_cell_states=dummy_c_t

        initial_hidden_states = tf.nn.dropout(initial_hidden_states,self.dropout)
        initial_cell_states = tf.nn.dropout(initial_cell_states, self.dropout)

        return initial_hidden_states, initial_cell_states, dummynode_hidden_states



    def __init__(self, config, session):
        #inputs: features, mask, keep_prob, labels
        self.input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels=tf.placeholder(tf.int64, [None,], name="labels")
        self.mask=tf.placeholder(tf.int32, [None,], name="mask")
        self.dropout=self.keep_prob=keep_prob=tf.placeholder(tf.float32, name="keep_prob")
        self.config=config
        shape=tf.shape(self.input_data)
        #if model_name=='lstm':
        #    self.dummy_input = tf.placeholder(tf.float32, [None, None], name="dummy")
        #embedding
        self.embedding=embedding = tf.Variable(tf.random_normal([config.vocab_size, config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="embedding", trainable=config.embedding_trainable)
        #apply embedding
        initial_hidden_states=tf.nn.embedding_lookup(embedding, self.input_data) #初始化每个细胞的输出状态
        initial_cell_states=tf.identity(initial_hidden_states) # 初始化每个细胞的状态。返回一个和输入的 tensor 大小和数值都一样的 tensor

        initial_hidden_states = tf.nn.dropout(initial_hidden_states,keep_prob)  #丢失神经元，keep_prob:每个保留的概率
        initial_cell_states = tf.nn.dropout(initial_cell_states, keep_prob)

        #create layers 
        if model_name=='slstm':
            new_hidden_states,new_cell_state, dummynode_hidden_states=self.slstm_cell("word_slstm", config.hidden_size,self.mask, initial_hidden_states, initial_cell_states, config.layer)
            
            softmax_w = tf.Variable(tf.random_normal([2*config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")
            #representation=dummynode_hidden_states
            representation=tf.reduce_mean(tf.concat([new_hidden_states, tf.expand_dims(dummynode_hidden_states, axis=1)], axis=1), axis=1)
            
            softmax_w2 = tf.Variable(tf.random_normal([config.hidden_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w2")
            softmax_b2 = tf.Variable(tf.random_normal([2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b2")
            representation=tf.nn.tanh(tf.matmul(representation, softmax_w2)+softmax_b2)

        elif model_name=='lstm':
            initial_hidden_states=lstm_layer(initial_hidden_states,config, self.keep_prob, self.mask)
            softmax_w = tf.Variable(tf.random_normal([2*config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")
            representation=tf.reduce_sum(initial_hidden_states,axis=1)
            config.hidden_size_sum=2*config.hidden_size
        elif model_name=='cnn':
            initial_hidden_states=tf.reshape(initial_hidden_states, [-1, 700, config.hidden_size])            
            initial_hidden_states = tf.expand_dims(initial_hidden_states, -1)
            pooled_outputs = []
            for i, filter_size in enumerate([3]):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, config.hidden_size, 1, config.hidden_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b")

                    W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                    b2 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b2")

                    W3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W3")
                    b3 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b3")

                    W4 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W4")
                    b4 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b4")

                    conv = tf.nn.conv2d(
                        initial_hidden_states,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    print(h.get_shape())
                    h=tf.transpose(h, [0,1,3,2])
                    # Apply nonlinearity


                    conv2 = tf.nn.conv2d(
                        h,
                        W2,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv2")
                    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                    print(h2.get_shape())
                    h2=tf.transpose(h2, [0,1,3,2])

                    conv3 = tf.nn.conv2d(
                        h2,
                        W3,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv3")  
                    h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
                    print(h3.get_shape())

                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h3,
                        ksize=[1, 700 - 3*filter_size + 3, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = 1 * config.hidden_size
            self.h_pool = tf.concat(pooled_outputs, axis=3)
            representation = tf.reshape(self.h_pool, [-1, num_filters_total])

            softmax_w = tf.Variable(tf.random_normal([2*config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")

            softmax_w2 = tf.Variable(tf.random_normal([config.hidden_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w2")
            softmax_b2 = tf.Variable(tf.random_normal([2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b2")
            representation=tf.nn.tanh(tf.matmul(representation, softmax_w2)+softmax_b2)
        else:
            print("Invalid model")
            exit(1)
        
        self.logits=logits = tf.matmul(representation, softmax_w) + softmax_b
        self.to_print=tf.nn.softmax(logits)
        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, self.labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        #cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.cost=cost=tf.reduce_mean(loss)+ config.l2_beta*tf.nn.l2_loss(embedding)

        #designate training variables
        tvars=tf.trainable_variables()
        self.lr = tf.Variable(0.0, trainable=False)
        grads=tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)
        self.grads=grads
        optimizer = tf.train.AdamOptimizer(config.learning_rate)        
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    #assign value to learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

def get_minibatches_idx(n, batch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def run_epoch(session, config, model, data, eval_op, keep_prob, is_training):
    n_samples = len(data[0])
    print("Running %d samples:"%(n_samples))  
    minibatches = get_minibatches_idx(n_samples, config.batch_size, shuffle=False)

    correct = 0.
    total = 0
    total_cost=0
    prog = Progbar(target=len(minibatches))
    #dummynode_hidden_states_collector=np.array([[0]*config.hidden_size])

    to_print_total=np.zeros([2,6])
    for i, inds in enumerate(minibatches):
        x = data[0][inds]
        if model_name=='cnn':
            x=pad_sequences(x, maxlen=700, dtype='int32',padding='post', truncating='post', value=0.)
        else:
            x=pad_sequences(x, maxlen=None, dtype='int32',padding='post', truncating='post', value=0.)
        y = data[1][inds]
        mask = data[2][inds]

        count, _, cost, to_print= \
        session.run([model.accuracy, eval_op,model.cost, model.to_print],\
            {model.input_data: x, model.labels: y, model.mask:mask, model.keep_prob:keep_prob})
        if not is_training:
            to_print_total=np.concatenate((to_print_total, to_print), axis=0)
        # to_print_total = np.concatenate((to_print_total, to_print), axis=0)

        correct += count 
        total += len(inds)
        total_cost+=cost
        prog.update(i + 1, [("train loss", cost)])
    # if not is_training:
        # print(to_print_total.shape[0])
        # print(to_print_total.shape[1])
        # for i in range(to_print_total.shape[0]):
        #     print(to_print_total[i].tolist())
        # f = open('./output/slstm-text.txt', 'a+')
        # for i in range(2, to_print_total.shape[0]):
        #     for d in to_print_total[i]:
        #         f.write(str(d) + ' ')
        #     f.write('\n')
        # f.close()
    print("Total loss:")
    print(total_cost)
    accuracy = correct/total
    return accuracy, to_print_total

def train_test_model(config, i, session, model, train_dataset, valid_dataset, test_dataset, best_acc, times):
    #compute lr_decay
    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
    #update learning rate
    model.assign_lr(session, config.learning_rate * lr_decay)

    #training            
    print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(model.lr)))
    start_time = time.time()
    train_acc, to_print_total = run_epoch(session, config, model, train_dataset, model.train_op, config.keep_prob, True)
    print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc, time.time()-start_time))

    #valid 
    # valid_acc, to_print_total = run_epoch(session, config, model, valid_dataset, tf.no_op(),1, True)
    # print("Valid Accuracy = %.4f\n" % valid_acc)

    #testing
    start_time = time.time()
    test_acc, to_print_total = run_epoch(session, config, model, test_dataset, tf.no_op(),1, False)
    print("Test Accuracy = %.4f\n" % test_acc)    
    print("Time = %.3f seconds\n"%(time.time()-start_time))

    if test_acc > best_acc:
        best_acc=test_acc
        print("Best accuracy, save results to file.")
        path='./run_result/'+dataset_name+'_outputs_'+str(times)+'.txt'
        f = open(path, 'w')
        f.write(str(best_acc)+'\n')
        for i in range(2, to_print_total.shape[0]):
            for d in to_print_total[i]:
                f.write(str(d) + ' ')
            f.write('\n')
        f.close()
    #return valid_acc, test_acc
    return best_acc

def start_epoches(config, session,classifier, train_dataset, valid_dataset, test_dataset, times):
    #record max
    #max_val_acc=-1
    #max_test_acc=-1
    acc = 0
    for i in range(config.max_max_epoch):
        acc=train_test_model(config, i, session, classifier, train_dataset, valid_dataset, test_dataset, acc, times)

def word_to_vec(matrix, session,config, *args):
    
    print("word2vec shape: ", matrix.shape)
    for model in args:
        session.run(tf.assign(model.embedding, matrix))

if __name__ == "__main__":
    #configs
    config = Config()

    config.layer=int(iteration_num)
    config.step=int(window_size)
    print("dataset: "+dataset_name)
    print("iteration: "+str(config.layer))
    print("step: "+str(config.step))
    print("model: "+str(model_name))

    labels, texts = data_helper.load_sina(file_path, "./data/sinalabel.pkl")
    one_labels=data_helper.onehot(labels)
    print('load data finished')
    # tfidf, texts = data_helper.tfidf_words(texts, [], 50)

    wordindex, wordvectordict = data_helper.word2vec_embedding(texts, vector_path, 300)
    embedding_mat = [wordvectordict[word] for index, word in enumerate(wordindex.keys())]  # dict，序号key和向量value。得到每个单词对应的向量
    embedding_mat = np.array(embedding_mat, dtype=np.float32)  # array，得到每个单词对应的向量
    textvec = data_helper.doc2vec(texts, wordindex)

    matrix=embedding_mat
    config.vocab_size = matrix.shape[0]

    times=1
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(textvec, one_labels):
        x_train=[textvec[i] for i in train_index]
        y_train = [one_labels[i] for i in train_index]
        x_test = [textvec[i] for i in test_index]
        y_test = [labels[i] for i in test_index]

        path='./run_result/'+dataset_name+'_label_'+str(times)+'.txt'
        f = open(path, 'w')
        for i in y_test:
            for d in i:
                f.write(str(d) + ' ')
            f.write('\n')
        f.close()

        y_test = data_helper.onehot(y_test)
        print('data split finished')

        transformed_text = [x_train] + [[]] + [x_test]
        transformed_label = [y_train] + [[]] + [y_test]
        pickle.dump(((transformed_text, transformed_label)), open('./run_result/'+filename+'_dataset_'+str(times), 'wb'))
        path='./run_result/'+filename+'_dataset_'+str(times)

        train_dataset, valid_dataset, test_dataset = read_data.load_data(path=path, n_words=config.vocab_size)

        config.num_label = len(set(y_train))
        print("number label: " + str(config.num_label))
        train_dataset = read_data.prepare_data(train_dataset[0], train_dataset[1])
        valid_dataset = read_data.prepare_data(valid_dataset[0], valid_dataset[1])
        test_dataset = read_data.prepare_data(test_dataset[0], test_dataset[1])

        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_normal_initializer(0, 0.05) #用正态分布产生张量的初始化器。(mean, stddev, seed=None, dtype=tf.float32)

            classifier= Classifer(config=config, session=session)

            total=0
            #print trainable variables
            for v in tf.trainable_variables():
                print(v.name)
                shape=v.get_shape()
                try:
                    size=shape[0].value*shape[1].value
                except:
                    size=shape[0].value
                total+=size
            print(total)
            #initialize
            init = tf.global_variables_initializer()

            session.run(init)
            #train test model
            word_to_vec(matrix, session,config, classifier)
            start_epoches(config, session,classifier, train_dataset, valid_dataset, test_dataset, times)

        times=times+1

