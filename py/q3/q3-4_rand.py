import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        nn.init.uniform_(self.fc1.weight, -1, 1)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(512, 256)
        nn.init.uniform_(self.fc2.weight, -1, 1)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(256, 10)
        nn.init.uniform_(self.fc3.weight, -1, 1)
        nn.init.zeros_(self.fc3.bias)


    def forward(self, x):
        x = x.view(-1, 784)
        x = x.to(self.fc1.weight.dtype)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)

def train(model, optimizer, train_loader):
    model.train()
    train_loss = 0
    correct = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.nll_loss(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    train_accuracy = correct / len(train_loader.dataset)
    return train_loss / len(train_loader.dataset), train_accuracy


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += nn.functional.nll_loss(output, y, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


model = Net()
train_loss_list = []
test_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
test_error = []
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    train_loss, train_accuracy = train(model, optimizer, train_loader)
    test_loss, test_accuracy = test(model, test_loader)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    test_error.append(1-test_accuracy)
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
    print('Epoch:', epoch+1, 'Train Loss:', train_loss, 'Train Accuracy:', train_accuracy, 'Test Loss:', test_loss, 'Test Accuracy:', test_accuracy, 'Test Error:', 1-test_accuracy)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax[0].plot(range(1, 11), train_loss_list, label='Train')
ax[0].plot(range(1, 11), test_loss_list, label='Test')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].set_title('Training and Testing Loss')

ax[1].plot(range(1, 11), train_accuracy_list, label='Train')
ax[1].plot(range(1, 11), test_accuracy_list, label='Test')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].set_title('Training and Testing Accuracy')

plt.show()

for ind, err in enumerate(test_error):
    if ind == 0:
        print(r'\hline'+'\nEpoch & Test Error (\%) \\\\'+r'\begin{center}'+'\n'+r'\begin{tabular}{|c|c|}')
    print(r'\hline', '\n',ind+1, ' & ', str(err*100)+'% \\\\')
print(r'\hline\end{tabular}\end{center}')