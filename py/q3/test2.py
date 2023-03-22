# Q3.2

# Vectorized version #

from torchvision import datasets, transforms

import numpy as np
from scipy.special import softmax, expit  # sigmoid on nparrays
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


def get_features(data):
    X = np.zeros(len(data) * 28 * 28).reshape(len(data), 784)
    for i in range(len(data)):
        X[i] = np.array(data[i][0]).reshape(1, 784)[0]
    return (X.T)  # dxn i.e. 784x60000 for full dataset


def get_labels(data):
    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = int(data[i][1])
    return (y)


X_train = get_features(mnist_train)
y_train = get_labels(mnist_train)
X_test = get_features(mnist_test)
y_test = get_labels(mnist_test)


def get_predictions(X, W1, W2, W3):
    return (softmax(np.matmul(W3, expit(np.matmul(W2, expit(np.matmul(W1, X)))))))  # g(kxn vector)


# returns kxn vector of predictions for each digit for each image. Note k = 10.

def make_one_hot_vector(labels):
    M = np.zeros((10 * len(labels))).reshape(10, len(labels))
    for i in range(len(labels)):
        index = int(labels[i] - 1)
        M[index, i] = 1
    return (M)


# returns 10xn matrix of one_hot encoded labels; each column is a different label

def get_losses(labels, y_hats):
    eps = 0.00001
    y_hats = y_hats + eps  # to avoid numerical issues
    loss = labels * np.log(y_hats)  # element-wise multiplication of two 10xn matrices
    out = -np.sum(loss, axis=0)  # sums across rows (vertically), returns 1xn vector
    out = out.reshape(1, len(labels))  # make sure dimension is as expected
    return (out)


# initialize weights
W1 = np.random.uniform(low=-1, high=1, size=(300 * 784)).reshape(300, 784)
W2 = np.random.uniform(low=-1, high=1, size=(200 * 300)).reshape(200, 300)
W3 = np.random.uniform(low=-1, high=1, size=(10 * 200)).reshape(10, 200)

# training
epochs = 50  # we make 15 passes thru the entire dataset
batch_size = 60
y_matrix = make_one_hot_vector(y_train)

# Training loop

train_loss = np.zeros(epochs)
test_loss = np.zeros(epochs)
errors = np.zeros(epochs)

for m in range(epochs):

    if not (m % 5): print(f'--- Start of epoch {m + 1} ---')

    batch_ids = np.random.choice(range(len(y_train)), size=len(y_train), replace=False)
    batches = int(len(y_train) / batch_size)

    alpha = 0.01 / ((m + 1) * batch_size)  # dynamic (shrinking) step size

    if not (m % 5): print("Started learning weight parameters, mini-batch SGD")

    for b in range(int(batches)):
        start = b * batch_size
        stop = start + batch_size
        ids = batch_ids[start:stop]
        X_batch = X_train[:, ids]  # dxn
        y_matrix_batch = y_matrix[:, ids]  # kxn

        a1 = expit(np.matmul(W1, X_batch))  # d1xn matrix
        a2 = expit(np.matmul(W2, a1))  # d2xn matrix
        g = get_predictions(X_batch, W1, W2, W3)  # 10xn matrix of predictions (all n images, one column each)

        dW3_tilde = (g - y_matrix_batch)  # 10xn matrix

        tmp1 = np.matmul(W3.T, dW3_tilde)  # d2xn matrix
        tmp2 = (1 - a2)  # d2xn matrix (1 is understood to be a matrix of correct size of all ones)

        dW2_tilde = tmp1 * a2 * tmp2  # d2xn

        tmp3 = np.matmul(W2.T, dW2_tilde)  # d1xn
        tmp4 = (1 - a1)  # d1xn

        dW1_tilde = tmp3 * a1 * tmp4  # d1xn

        W3 = W3 - alpha * np.matmul(dW3_tilde, a2.T)  # 10xd2
        W2 = W2 - alpha * np.matmul(dW2_tilde, a1.T)  # d2xd1
        W1 = W1 - alpha * np.matmul(dW1_tilde, X_batch.T)  # d1xd

    if not (m % 5): print("Finished learning weight parameters")

    y_hats_train = get_predictions(X_train, W1, W2, W3)
    train_loss[m] = np.sum(get_losses(y_train, y_hats_train) / len(y_train))
    if not (m % 5): print(f'Training loss: {train_loss[m]}')

    y_hats_test = get_predictions(X_test, W1, W2, W3)
    test_loss[m] = np.sum(get_losses(y_test, y_hats_test) / len(y_test))
    if not (m % 5): print(f'Test loss: {test_loss[m]}')

    errors_tmp = 0
    for i in range(len(y_test)):
        pred = np.argmax(y_hats_test[:, i])  # gets position of largest activation in output layer for image i
        if pred != int(y_test[i]): errors_tmp += 1
    if not (m % 5): print(f'Number of errors out of sample: {errors_tmp}')

    errors[m] = errors_tmp / len(y_test)

for i in range(len(errors)):
    print(i + 1, "&", round(errors[i] * 100, 4), "\%", "\\\\")

plt.plot(np.arange(1, epochs+1), train_loss, label='Training Loss')
plt.plot(np.arange(1, epochs+1), test_loss, label='Test Loss')
plt.legend(loc="right")
plt.title("Learning Curve")
plt.ylabel("Empirical Risk")
plt.xlabel("epoch")
plt.show()