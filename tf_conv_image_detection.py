# Get the assiged number of cores for this job. This is stored in
# the NSLOTS variable, If NSLOTS is not defined throw an exception.
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')

import copy
import os
import math
import numpy as np
import scipy
import scipy.io
from six.moves import range
import read_data
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


@read_data.restartable
def svhn_dataset_generator(dataset_name, batch_size):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    
    path = './svhn_mat/' # path to the SVHN dataset
    file_name = '%s_32x32.mat' % dataset_name
    file_dict = scipy.io.loadmat(os.path.join(path, file_name))
    X_all = file_dict['X'].transpose((3, 0, 1, 2))
    y_all = file_dict['y']
    data_len = X_all.shape[0]
    batch_size = batch_size if batch_size > 0 else data_len
    
    X_all_padded = np.concatenate([X_all, X_all[:batch_size]], axis=0)
    y_all_padded = np.concatenate([y_all, y_all[:batch_size]], axis=0)
    y_all_padded[y_all_padded == 10] = 0
    
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        X_batch = X_all_padded[idx:idx + batch_size]
        y_batch = np.ravel(y_all_padded[idx:idx + batch_size])
        yield X_batch, y_batch

# The following defines a simple CovNet Model.


def SVHN_net_v0(x_):
    # expanded svhn net
    #followed by one additional convolutional layer, and
    #followed by one additional pooling layer.
    conv1 = tf.layers.conv2d(
            inputs=x_,
            filters=36,  # number of filters
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2], 
                                    strides=5)  # convolution stride
    
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=36, # number of filters
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                    pool_size=[2, 2], 
                                    strides=5)  # convolution stride
    conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=36, # number of filters
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, 
                                    pool_size=[2, 2], 
                                    strides=5)  # convolution stride
    
    
        
    pool_flat = tf.contrib.layers.flatten(pool3, scope='pool2flat')
    dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)
    return logits


def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
            y_ = tf.placeholder(tf.int32, [None])
            y_logits = model_function(x_)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = trainer.minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict

#save weights after the training is complete if save_model is True, and
#load weights on start-up before training if load_model is True.


def train_model(model_dict, dataset_generators, epoch_n, print_every,save_model=False, load_model=False):
    
    
    
    
        tf.reset_default_graph()
        with model_dict['graph'].as_default(), tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            
            if load_model:
                #tf.reset_default_graph()
                saver.restore(sess, "/tmp/model.ckpt")
                print("Model restored.")
                
                #saver = tf.train.import_meta_graph('cnn_expanded')
                #new_saver = tf.train.import_meta_graph(os.path.join(saved_path, 'model'))
                #new_saver.restore(sess, tf.train.latest_checkpoint('./'))
                
                
                
            for epoch_i in range(epoch_n):
                for iter_i, data_batch in enumerate(dataset_generators['train']):
                    train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                    sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                    
                    if iter_i % print_every == 0:
                        collect_arr = []
                        for test_batch in dataset_generators['test']:
                            test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                            to_compute = [model_dict['loss'], model_dict['accuracy']]
                            collect_arr.append(sess.run(to_compute, test_feed_dict))
                        averages = np.mean(collect_arr, axis=0)
                        fmt = (epoch_i, iter_i, ) + tuple(averages)
                        print('iteration {:d} {:d}\t loss: {:.3f}, '
                              'accuracy: {:.3f}'.format(*fmt))
                        
                        
            if save_model:
                saved_path = saver.save(sess,"/tmp/model.ckpt")
                print("Model saved in path: %s" % saved_path)

dataset_generators = {
        'train': svhn_dataset_generator('train', 256),
        'test': svhn_dataset_generator('test', 256)
}
    
model_dict = apply_classification_loss(SVHN_net_v0)
train_model(model_dict, dataset_generators, epoch_n=100, print_every=10,save_model=True)
train_model(model_dict, dataset_generators, epoch_n=10, print_every=1,load_model=True)
