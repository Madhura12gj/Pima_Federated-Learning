#importing the necesssary packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
import syft as sy
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

#downloading the dataset
url='https://raw.githubusercontent.com/Madhura12gj/Hands-on-Pytorch/master/diabetes.csv'
df=pd.read_csv(url)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#splitting it into testing and training set
x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=1)

x_train=np.array(x_train, dtype=np.float32)
y_train=np.array(y_train, dtype=np.float32)
x_test=np.array(x_test, dtype=np.float32)
y_test=np.array(y_test, dtype=np.float32)

#converting the training set to tensors
inputs=torch.from_numpy(x_train).float()
targets=torch.from_numpy(y_train).long()
inputs_test=torch.from_numpy(x_test).float()
targets_test=torch.from_numpy(y_test).long()


bob_dataset = sy.BaseDataset(inputs[:300], targets[:300]).send(bob)
alice_dataset = sy.BaseDataset(inputs[300:],targets[300:]).send(alice)

federated_train_dataset =sy.FederatedDataset([bob_dataset,alice_dataset])
federated_train_loader = sy.FederatedDataLoader(federated_train_dataset,shuffle=True)

test = torch.utils.data.TensorDataset (inputs_test, targets_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size, shuffle=True)

print(inputs.shape)
print(targets.shape)

#create your model
class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8,16)
        self.linear2 = nn.Linear(16,8)
        self.linear3 = nn.Linear(8,1)
        #self.linear4 = nn.Linear(16,8)
        #self.linear5 = nn.Linear(8,1)
    # Perform the computation
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x))
        #x = F.relu(self.linear4(x))
        x = torch.sigmoid(self.linear3(x))
        return x
model = SimpleNet()

#create optimizer and calculate the loss
opt = torch.optim.SGD(params=model.parameters(),lr=args.lr)
criterion=torch.nn.BCELoss()
#criterion = torch.nn.MSELoss(size_average=False)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(federated_train_loader): # <-- now it is a distributed dataset
        model.send(inputs.location) # <-- NEW: send the model to the right location
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs.float())
        loss = criterion(output, targets.float())
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get the model back
        #print(batch_idx)
        if (batch_idx % args.log_interval) == 0:
          loss_int = loss.get()
          print("Epoch:",epoch, "Batch_id:",batch_idx, "Loss:",loss_int.item())

device = torch.device("cpu")
for epoch in range(1, args.epochs):
    train(args, model, device, federated_train_loader, opt, epoch)
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    mean_accuracy = 0
    final_output = []
    final_test = []
    with torch.no_grad():
        for (inputs_test, targets_test) in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            output = model(inputs_test.float())
            final_output.append(output)
            final_test.append(targets_test)
        final_output = torch.cat(final_output, dim=0)
        final_test = torch.cat(final_test, dim=0)
        accuracy = accuracy_score(final_test.float(), (final_output>0.5))
        print(confusion_matrix(final_test, final_output>0.5))
        print(accuracy)
 
device = torch.device("cpu")
test(args, model, device, test_loader)
