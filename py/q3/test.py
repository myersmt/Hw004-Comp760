import numpy as np
import matplotlib.cm as cm
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt
def init_params():
    W1 = np.random.rand(300,784) 
    W2 = np.random.rand(200, 300) 
    W3 = np.random.rand(10, 200)
    return W1, W2, W3
def sigmoid (x):
    z = 1/(1 + np.exp(-x))
    return z 
def softmax(x):
    a = np.exp(x)/np.exp(x).sum()
    return a
def forward_prop(In, W1, W2, W3):
    #print("shape of input : ",In.shape)
    Y1 = W1.dot(In)
    #print("shape of Y1 : ",Y1.shape)
    Z1 = sigmoid(Y1)
    Y2 = W2.dot(Z1)
    Z2 = sigmoid(Y2)
    #print("shape of Z2_T : ",Z2.T.shape)
    Y3 = W3.dot(Z2)
    Z3 = softmax(Y3)
    #print("shape of Z3 : ",Z3.shape)
    return Z1, Z2, Z3, Y1, Y2, Y3
def softmax_deriv(y_pred, y_actual):
    return y_pred - y_actual
def sigmoid_deriv(x) :
    return np.diag(x*(1-x))
def one_hot(Y):
    Y_hot = np.atleast_2d(np.zeros(10))
    #print("shape of Y_hot : ",Y_hot.shape)
    Y_hot[0][Y] = 1
    return Y_hot.T
def back_prop (Z1,Z2, Z3, In, Y,W2,W3):
    dW3 = (Z3 - one_hot(Y)).dot(Z2.T)
    Z2_deriv = sigmoid_deriv(Z2.squeeze())
    A = np.matmul (Z2_deriv,W3.T)
    B = np.matmul(A,(Z3-one_hot(Y)))
    dW2 = np.matmul(B,Z1.T)
    Z1_deriv = sigmoid_deriv(Z1.squeeze())
    C = np.matmul (Z1_deriv, W2.T)
    D = np.matmul ( C, B)
    dW1 = np.matmul(D, In.T)
    return dW1, dW2, dW3
def update_params(W1, W2, W3, dW1, dW2, dW3, alpha):
    W1 = W1 - alpha * dW1  
    W2 = W2 - alpha * dW2  
    W3 = W3 - alpha * dW3    
    return W1, W2, W3

def get_predictions(Z3):
    return np.argmax(Z3)

def gradient_descent(X, Y, alpha, W1, W2, W3):
    #W1, W2, W3 = init_params()
    X = np.atleast_2d(X)
    X = X.T
    for i in range(3):
        Z1,Z2,Z3,Y1,Y2,Y3 = forward_prop(X,W1, W2, W3)
        dW1, dW2, dW3 = back_prop(Z1,Z2,Z3,X, Y,W2,W3)
        W1, W2, W3 = update_params(W1, W2, W3, dW1, dW2, dW3, alpha)
    
    return W1, W2, W3, Z3


mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
#train_data_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size=1, shuffle=False)
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True, transform=ToTensor())

iterations = 10
for k in [100, 500, 1000, 5000, 10000, 15000]:

  print("###### " + str(k) + " Images ######")

  #### BEGIN TRAINING ####
  W1_train, W2_train, W3_train = init_params()
  for j in range(iterations):
      for i in range(k):
          train_features, train_labels = mnist_data_train[i]
          #print("labels unflattened : ",train_labels)
          #print("features : ")
          #print(train_features.shape)
          #print(train_features)
          train_features_flatten = torch.flatten(train_features[0].squeeze())
          train_features_arr = train_features_flatten.numpy()
          #train_labels_flatten = torch.flatten(train_labels[0].squeeze())
          #train_labels_arr = train_labels.numpy()
          #print("train labels array : ",train_labels_arr)
          #plt.imshow(train_features[0].reshape(28,28), cmap=cm.binary)
          W1_train,W2_train,W3_train,Z3_train = gradient_descent(train_features_arr,train_labels,0.01, W1_train, W2_train, W3_train)
          #predictions = get_predictions(Z3)
      #loss = - np.matmul(one_hot(train_labels).T, np.log(Z3))
      #print("LOSS in IMG " + str(i) + " in iteration " + str(j) + " : " + str(loss))
      #print("true labels : {}".format(train_labels))
      #print("predictions : {}\n".format(predictions))
  
  #### BEGIN TESTING ####
  accurate_count = 0
  for t in range(len(mnist_data_test)):
    test_features, test_labels = mnist_data_test[t]
    test_features_flatten = torch.flatten(test_features[0].squeeze())
    test_features_arr = test_features_flatten.numpy()
    
    Z1_test,Z2_test,Z3_test,Y1_test,Y2_test,Y3_test = forward_prop(test_features_arr, W1_train, W2_train, W3_train)
    Y_pred = get_predictions(Z3_test)
    if (Y_pred == test_labels):
      accurate_count = accurate_count + 1
  
  print("Accurate count value : " + str(accurate_count))
  accuracy = accurate_count/10000

  print("Accuracy for " + str(k) + " Images : " + str(accuracy))