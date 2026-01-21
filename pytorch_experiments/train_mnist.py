import numpy as np
import math
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

verbose = False

def vprint(*args):
    if verbose:
        print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vprint('running on', device)

transform = transforms.Compose([
    transforms.ToTensor(),  
])

# --- PARAMS ---
BATCH_SIZE = 64
EPOCHS = 10
HIDDEN_LAYERS = [128, 32]
OPTIM = 'sgd' # 'adam'
ACTIVATION = 'relu' # 'sigmoid'
WEIGHT_INIT = 'normalized_uniform'
# --------------

trainset = datasets.MNIST('data/mnist', download=True, train=True, transform=transform)
testset = datasets.MNIST('data/mnist', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE)
testloader = torch.utils.data.DataLoader(testset, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, HIDDEN_LAYERS[0]),
    torch.nn.ReLU() if ACTIVATION == 'relu' else torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
    torch.nn.ReLU() if ACTIVATION == 'relu' else torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_LAYERS[1], 10),
).to(device)

for layer in model:
    if isinstance(layer, torch.nn.Linear):
        if WEIGHT_INIT == 'normalized_uniform':
            div = math.sqrt(layer.weight.shape[1])
            torch.nn.init.uniform_(layer.weight, -1/div, 1/div)

opt = torch.optim.SGD(model.parameters(), lr=0.01) if OPTIM == 'sgd' \
        else torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        opt.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        running_loss += loss
        opt.step()
    else:
        vprint(f'Epoch: {epoch} Loss: {running_loss / len(trainloader)}')

# get test acc
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = correct / total

# print run summary as dict
print({
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'hidden_layers': HIDDEN_LAYERS,
    'optim': OPTIM,
    'activation': ACTIVATION,
    'weight_init': WEIGHT_INIT,
    'test_acc': test_acc,
})