import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    e = np.exp(z - np.max(z, axis = 1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def forward(x, W1, W2, W3):
    z1 = np.dot(W1, x)
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1)
    h2 = sigmoid(z2)
    z3 = np.dot(W3, h2)
    y_pred = softmax(z3)
    return y_pred, h1, h2

def backward(X, y, y_hat, h1, h2, W1, W2, W3):
    dz3 = y_hat - y
    dW3 = np.matmul(dz3, h2.T)
    dh2 = np.matmul(W3.T, dz3)
    dz2 = dh2 * (h2 > 0)
    dW2 = np.matmul(dz2, h1.T)
    dh1 = np.matmul(W2.T, dz2)
    dz1 = dh1 * (h1 > 0)
    dW1 = np.matmul(dz1, X.T)
    return dW1, dW2, dW3


def train(X_train, y_train, X_test, y_test, n_hidden, learning_rate, n_epochs):
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]

    # Initialize weights
    W1 = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
    W2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(1 / n_hidden)
    W3 = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)

    # Train the network
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        # Forward pass on training set
        train_pred, h1, h2 = forward(X_train.T, W1, W2, W3)
        train_loss = cross_entropy_loss(y_train.T, train_pred)
        train_losses.append(train_loss)

        # Backward pass
        dW1, dW2, dW3 = backward(X_train.T, y_train.T, train_pred, h1, h2, W1, W2, W3)

        # Update weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3

        # Evaluate on test set
        test_pred, _, _ = forward(X_test.T, W1, W2, W3)
        test_loss = cross_entropy_loss(y_test.T, test_pred)
        test_losses.append(test_loss)

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        # Early stopping if the test loss doesn't improve for 5 epochs
        if epoch > 4:
            if all(test_losses[-1] >= x for x in test_losses[-5:-1]):
                print(f"Early stopping after epoch {epoch+1}")
                break

        # Shuffle training data for the next epoch
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Learning rate schedule
        if epoch == n_epochs // 2:
            learning_rate /= 10

        # Save the model with the lowest test loss
        if test_loss == min(test_losses):
            np.savez("model.npz", W1=W1, W2=W2, W3=W3)

    # Return the trained model
    return {"W1": W1, "W2": W2, "W3": W3}

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the hyperparameters
n_hidden = 100
learning_rate = 0.1
n_epochs = 50

# Train the network
model = train(X_train, y_train, X_val, y_val, n_hidden, learning_rate, n_epochs)

# Evaluate the network on the test set
test_pred, _, _ = forward(X_test.T, model["W1"], model["W2"], model["W3"])
test_loss = cross_entropy_loss(y_test.T, test_pred)
test_acc = np.mean(np.argmax(y_test, axis=1) == np.argmax(test_pred, axis=1))
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")

# Plot the learning curve
import matplotlib.pyplot as plt
plt.plot(range(1, len(model['train_losses']) + 1), model['train_losses'], label='Train Loss')
plt.plot(range(1, len(model['test_losses']) + 1), model['test_losses'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
