import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def load_data(dir,ratio,dataset):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = dir + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
  DATA = np.concatenate((data_train,data_test_val),axis=0)
  N = DATA.shape[0]

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = DATA[ind[:ratio[0]],1:]
  X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_test = DATA[ind[ratio[1]:],1:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA[ind[:ratio[0]],0]-1
  y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  y_test = DATA[ind[ratio[1]:],0]-1
  return X_train,X_val,X_test,y_train,y_val,y_test


def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch