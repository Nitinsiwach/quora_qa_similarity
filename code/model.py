
import time
import logging
from datetime import datetime
import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from utils.general_utils import get_minibatches
import os
import pickle
import functools
import copy

logging.basicConfig(level=logging.INFO)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn
 
        
class Encoder():
    def __init__(self, encoder_size):
        print("building encoder")
        self.size = encoder_size

    def encode(self, inputs, masks):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
         masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with tf.variable_scope("encoder") as scope_encoder:
            #compute sequence length
            sequence_lengths = tf.reduce_sum(masks, axis = 1) 
            #create a forward cell
            fw_cell = tf.contrib.rnn.LSTMCell(self.size)

            #pass the cells to bilstm and create the bilstm
            bw_cell = tf.contrib.rnn.LSTMCell(self.size)
            output, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, \
                                                                  bw_cell, inputs, \
                                                                  sequence_length = sequence_lengths, \
                                                                  dtype = tf.float32, \
                                                                  parallel_iterations = 256)
            output_lstm = tf.concat([output[0], output[1]], axis = -1)
            final_state_lstm = tf.concat([final_state[0], final_state[1]], axis = -1)
            return output_lstm, final_state_lstm
    
    

class QSystem(object):
    def __init__(self, encoder, pretrained_embeddings, config, train_flag = True):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        print("building question similarity calculator")
        self.encoder = encoder
        
        self.pretrained_embeddings_vars = tf.constant(pretrained_embeddings, dtype = tf.float32)
        
        self.embed_dim = config['embed_dim']
        self.optimizer = config['optimizer']
        self.minibatch_size = config['minibatch_size']
        self.learning_rate = config['learning_rate']
        self.max_grad_norm = config['max_grad_norm']
        self.max_sent_len = config['max_sent_len']
        self.encoder_size = config['encoder_size']
        
        self.q1_masks = tf.placeholder(tf.int32, shape = [None, self.max_sent_len])
        self.q2_masks = tf.placeholder(tf.int32, shape = [None, self.max_sent_len])
        
        self.q1_id = tf.placeholder(tf.int32, shape = [None, self.max_sent_len]) #batch_size x question_length
        self.q2_id = tf.placeholder(tf.int32, shape = [None, self.max_sent_len]) #batch_size x sent_max_length

        self.a = tf.placeholder(tf.int32, shape = [None, ]) #batch_size x 1
        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.embed_lookup()
            self.setup_system()
            self.setup_loss()
            self.make_optimizer()
            self.saver = self.saver_prot()

        # ==== set up training/updating procedure ====
        
        
        
        
    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        with tf.variable_scope("questionRNN", reuse=tf.AUTO_REUSE):
            encoder_q1, final_state_q1 = self.encoder.encode(self.q1, self.q1_masks)
            encoder_q2, final_state_q2 = self.encoder.encode(self.q2, self.q2_masks)
        dot = tf.multiply(encoder_q1,encoder_q2)
        diff = encoder_q1 - encoder_q2
        batch = tf.shape(diff)[0]
        similarity = tf.concat([encoder_q1, encoder_q2, dot, diff], axis = -1)
        similarity = tf.reshape(similarity, [batch, self.max_sent_len*self.encoder_size*2*4])
        with tf.variable_scope("affine", regularizer = tf.contrib.layers.l2_regularizer(0.001)):
            self.logits = tf.contrib.layers.fully_connected(similarity, 2, activation_fn=None) #batch, 2
            
        
        self.prediction = tf.argmax(self.logits, axis = 1) #batch

    def setup_loss(self):
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.a, logits = self.logits))
#             + tf.losses.get_regularization_loss()
            
            
            
    def make_optimizer(self):
        optimizer = get_optimizer(self.optimizer)
        _optimizer_op = optimizer(self.learning_rate)
        gradients, variables = zip(*_optimizer_op.compute_gradients(self.loss))
        clipped_gradients, self.global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        self.optimizer_op = _optimizer_op.apply_gradients(zip(gradients, variables))
        self.optimizer_op = _optimizer_op.minimize(self.loss)
    

    def embed_lookup(self):
        self.q1 = tf.nn.embedding_lookup(self.pretrained_embeddings_vars, self.q1_id) #batch,q_max_len,embed_dim
        self.q2 = tf.nn.embedding_lookup(self.pretrained_embeddings_vars, self.q2_id)

    
    def predict(self, session, prediction_dataset, mode = 'train'):
        input_feed = self.feed_dict(prediction_dataset, mode)
        output = self.prediction
        
        predictions = session.run(output, input_feed)
        return predictions
        

    def evaluate_answer(self, session, dataset, log=False):
        """
        dataset: [q1, q2, a]
        naive accuracy measure
        """
        predictions = np.array(self.predict(session, dataset))
        
        gold = np.array(dataset[2])
        print(predictions.shape,gold.shape)
#         print(predictions[:10] == gold[:10] )
        correct = predictions == gold
        accuracy = np.sum(correct)/gold.shape
        
        
        if log:
            logging.info("accuracy: {}".format(accuracy))
        return accuracy
    
    
    def pad(self, datalist):
        '''pads q1 and q2 to max length of q1 and q2
        Params: datalist: [q1,q2]
                where q1, q2: [[w1, w2, ..], [w6,w7, ..]]'''
        

        padded = []
        masks = []
        m_len = self.max_sent_len
        q1 = datalist[0]
        q2 = datalist[1]
        q1 = [i[:self.max_sent_len] if len(i) > m_len else i for i in q1]
        q2 = [i[:self.max_sent_len] if len(i) > m_len else i for i in q2]
        datalist = [q1, q2]
        padded = [[k + [0]*(m_len-len(k)) for k in j] for j in datalist]
        masks = [[[1 if t != 0 else 0 for t in k] for k in j] for j in padded]
        return padded, masks
    
    def feed_dict(self, dataset_feed, mode = 'train'):
        '''dataset_feed: ([q1,q2,a])'''
        input_feed = {}
        if mode == 'train':
            a = dataset_feed[2]
            input_feed[self.a] = a
        
        padded, padded_masks = self.pad(dataset_feed[:2])
        input_feed[self.q1_id], input_feed[self.q2_id] = padded
        input_feed[self.q1_masks], input_feed[self.q2_masks]= padded_masks
        return input_feed
         
    def run_epoch(self, dataset, sess):
        '''dataset is a list [q1, q2, a]'''
        
        n_minibatches = 0.
        total_loss = 0.
        for dataset_mini in get_minibatches(dataset, self.minibatch_size):
            n_minibatches += 1
            feed_dict = self.feed_dict(dataset_mini)
            output = [self.optimizer_op , self.loss, self.global_norm]
            _, loss, global_norm = sess.run(output, feed_dict)
            if not n_minibatches % 1:
                print("n_minibatch = {}".format(n_minibatches), "loss: {}".format(loss), "global_norm{}".format(global_norm))
            total_loss += loss  
        return total_loss/n_minibatches
    
    def saver_prot(self):
        return tf.train.Saver()

    def train(self, session, dataset,sent_max_len, epochs, train_dir, test):
        """
        Implement main training loop
        :param session: it should be passed in from train.py
               dataset:[[train_q1, train_q2, train_a], [val_q1, val_q2, val_a]]
                        
               train_dir: path to the directory where you save the model checkpoint
        :return: best_score
        """                                                  
        results_path = os.path.join(train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        
       
        dataset_train = dataset[0]
        dataset_val = dataset[1]
        best_score = 0
        accuracy = -1
#         best_score, _ = self.evaluate_answer(session, dataset_val,log=True)
    
        
        
        for epoch in range(epochs):
            logging.info("Epoch %d out of %d", epoch + 1, epochs)
            logging.info("Best score so far: " + str(best_score))
            loss = self.run_epoch(dataset_train, session)
            accuracy = self.evaluate_answer(session, dataset_train, log=True)
            logging.info("loss: " + str(loss) + " accuracy: "+str(accuracy))
            if accuracy > best_score:
                best_score = accuracy
                logging.info("New best score! Saving model in %s", results_path)
                self.saver.save(session, results_path)    
            print("")

        return best_score
    