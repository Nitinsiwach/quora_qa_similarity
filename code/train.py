



import os
import json

import tensorflow as tf
import numpy as np

from model import Encoder, QSystem
from os.path import join as pjoin
import argparse

import logging
logging.basicConfig(level=logging.INFO)



class _FLAGS():
    def __init__(self):
        self.learning_rate = 0.001
        self.max_gradient_norm = 5.0
        self.dropout = 0.15
        self.batch_size = 32
        self.epochs = 60
        self.state_size = 10
        self.embedding_size = 300
        self.data_dir = "data/quora"
        self.train_dir = "train"
        self.load_train_dir = ""
        self.log_dir = "log"
        self.optimizer = "adam"
        self.vocab_path = "data/quora/vocab.dat"
        self.max_sent_len = 40
        self.mode = 'test'
FLAGS = _FLAGS()

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def get_device_name():
    print('device in use:',tf.test.gpu_device_name() )
    return 'CPU' if tf.test.gpu_device_name() == '' else tf.test.gpu_device_name()

def get_normalized_train_dir(train_dir):
    global_train_dir = 'dir_node_wordvar'
    if os.path.exists(global_train_dir):
#         os.unlink(global_train_dir) #forlinux
        os.system('rmdir "%s"' % global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir, True)
    return global_train_dir

def get_pretrained_embeddings(embed_path):
    glove = np.load(embed_path)
    return glove['glove']


def main():
    get_device_name()

    print('Device in use {}'.format(get_device_name()))
    dataset = None

    embed_path = pjoin("data", "quora", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    pretrained_embeddings = get_pretrained_embeddings(embed_path)
    config = {}
    config['embed_dim'] = FLAGS.embedding_size
    config['optimizer'] = FLAGS.optimizer
    config['minibatch_size'] = FLAGS.batch_size
    config['learning_rate'] = FLAGS.learning_rate
    config['max_grad_norm'] = FLAGS.max_gradient_norm
    config['max_sent_len'] = FLAGS.max_sent_len
    config['encoder_size'] = FLAGS.state_size

    encoder = Encoder(FLAGS.state_size)
    qsystem = QSystem(encoder, pretrained_embeddings, config)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with open(vocab_path, encoding = 'utf-8') as file1:
        f = file1.readlines()

    print((vars(FLAGS)))

    with open("data/quora"+"/.ids.train.question1", encoding = 'utf8') as tq1, \
        open("data/quora"+"/.ids.train.question2", encoding = 'utf8') as tq2, \
        open("data/quora"+"/.ids.val.question1", encoding = 'utf8') as vq1, \
        open("data/quora"+"/.ids.val.question2", encoding = 'utf8') as vq2, \
        open("data/quora"+"/labels_train.txt", encoding = 'utf8') as ta, \
        open("data/quora"+"/labels_val.txt", encoding = 'utf8') as va:
                tq1 = tq1.readlines()
                tq2 = tq2.readlines()
                ta = ta.readlines()
                vq1 = vq1.readlines()
                vq2 = vq2.readlines()
                va = va.readlines()
    ta = [int(i) for i in list(ta[0])]
    va = [int(i) for i in list(va[0])]
    tq1 = [[int(i) for i in j.replace("\n", "").split()] for j in tq1]
    tq2 = [[int(i) for i in j.replace("\n", "").split()] for j in tq2]
    vq1 = [[int(i) for i in j.replace("\n", "").split()] for j in vq1]
    vq2 = [[int(i) for i in j.replace("\n", "").split()] for j in vq2]
    dataset = [[tq1[1560:11560],tq2[1560:11560],ta[1560:11560]],[vq1[1560:11560],vq2[1560:11560],va[1560:11560]]]

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qsystem, load_train_dir)
        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        if FLAGS.mode == 'train':
            
            qsystem.train(sess, dataset, FLAGS.max_sent_len, FLAGS.epochs, save_train_dir, test = args.test)
            accuracy = qa.evaluate_answer(sess, dataset_train, log=True)
        if Flags.mode == 'test':
            with open("data/quora"+"/.ids.test.question1", encoding = 'utf8') as tq1, \
                open("data/quora"+"/.ids.test.question2", encoding = 'utf8') as tq2:
                    tq1 = tq1.readlines()
                    tq2 = tq2.readlines()
            tq1 = [[int(i) for i in j.replace("\n", "").split()] for j in tq1]
            tq2 = [[int(i) for i in j.replace("\n", "").split()] for j in tq2]
            prediction_dataset = [tq1, tq2]
            predictions = qa.predict(sess, prediction_dataset, mode = 'test')
            with open("results_test.txt", 'w', encoding = 'utf-8') as f:
                f.write(predictions)

if __name__ == "__main__":
    main()