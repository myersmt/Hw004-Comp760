"""
Created by: Matt Myers
Question 003
"""
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', download=True, train=True, transform=transform)
testset = datasets.MNIST('mnist_test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define model parameters
d = 784
d1 = 256
d2 = 128
k = 10
W1 = np.random.normal(0, 0.1, (d1, d))
W2 = np.random.normal(0, 0.1, (d2, d1))
W3 = np.random.normal(0, 0.1, (k, d2))

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax_derivative(x):
    p = softmax(x)
    return p * (1 - p)

# Define cross-entropy loss function
def cross_entropy_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat))

# Define learning rate
lr = 0.01

# Train the model
epochs = 20
train_losses = []
test_losses = []
test_error = []
for epoch in range(epochs):
    num_correct_test = 0
    num_total_test = 0
    train_loss = 0.0
    for images, labels in trainloader:
        batch_size = images.shape[0]
        x = images.view(batch_size, -1)
        y = np.eye(k)[labels]  # one-hot encoding
        # Forward pass
        z1 = np.dot(x, W1.T)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2.T)
        a2 = sigmoid(z2)
        z3 = np.dot(a2, W3.T)
        y_hat = softmax(z3)
        train_loss += cross_entropy_loss(y, y_hat)
        # Backward pass
        dz3 = y_hat - y
        dW3 = lr * np.dot(dz3.T, a2)
        da2 = np.dot(dz3, W3)
        dz2 = da2 * a2 * (1 - a2)
        dW2 = lr * np.dot(dz2.T, a1)
        da1 = np.dot(dz2, W2)
        dz1 = da1 * a1 * (1 - a1)
        dW1 = lr * np.dot(dz1.T, x)
        # Update weights
        W3 -= dW3
        W2 -= dW2
        W1 -= dW1
    # Compute train and test losses
    train_loss = train_loss / len(trainloader)
    train_losses.append(train_loss)
    test_loss = 0.0
    for images, labels in testloader:
        batch_size = images.shape[0]
        x = images.view(batch_size, -1)
        y = np.eye(k)[labels]  # one-hot encoding
        # Forward pass
        z1 = np.dot(x, W1.T)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2.T)
        a2 = sigmoid(z2)
        z3 = np.dot(a2, W3.T)
        y_hat = softmax(z3)
        test_loss += cross_entropy_loss(y, y_hat)
        predictions = np.argmax(y_hat, axis=1)
        num_correct_test += np.sum(predictions == labels.numpy())
        num_total_test += batch_size
    test_loss = test_loss / len(testloader)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    err = (num_total_test - num_correct_test) / num_total_test
    test_error.append(err)
    print(f"Test error: {err:.4f}")


plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.show()

num_correct = 0
num_total = 0
for images, labels in testloader:
    batch_size = images.shape[0]
    x = images.view(batch_size, -1)
    y = np.eye(k)[labels] # one-hot encoding
    # Forward pass
    z1 = np.dot(x, W1.T)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2.T)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3.T)
    y_hat = softmax(z3)
# Calculate accuracy
predictions = np.argmax(y_hat, axis=1)
num_correct += np.sum(predictions == labels.numpy())
num_total += batch_size
accuracy = num_correct / num_total
print(f'Test accuracy: {accuracy:.4f}')

for ind, err in enumerate(test_error):
    if ind == 0:
        print(r'\hline'+'\nEpoch & Test Error (\%) \\\\'+r'\begin{center}'+'\n'+r'\begin{tabular}{|c|c|}')
    print(r'\hline', '\n',ind+1, ' & ', str(err*100)+'% \\\\')
print(r'\hline\end{tabular}\end{center}')
