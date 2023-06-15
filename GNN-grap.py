import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

def make_loaders(data_file,test_size=0.2,bsize=5):

    data_list = torch.load(data_file)
    # Split data into training and test sets
    train_data, test_data = train_test_split(data_list, test_size=test_size)#, random_state=42)
    
    # Normalize node features
    ##transform = NormalizeScale()
    #try:

    ##train_data = [transform(data) for data in train_data]
    ##test_data = [transform(data) for data in test_data]
    
    # Create data loaders for training and test sets
    train_loader = DataLoader(train_data, batch_size=bsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=bsize, shuffle=False)
 

    return train_loader, test_loader

import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv, GCNConv, CGConv, GatedGraphConv, ResGatedGraphConv, NNConv

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GraphConv(num_features, 16)
        self.conv2 = GraphConv(16, 32)
        self.conv3 = GraphConv(32, 64)
        self.conv4 = GraphConv(64, 128)
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.lin1(x)
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        #x = x.relu()
        return x.reshape(-1)  

# Define the Graph Neural Network model
class GNN2(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN2, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.lin = Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        #x = x.relu()
        return x.reshape(-1) 
        
class GNNS(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNS, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GraphConv(num_features, 16)
        self.conv2 = GraphConv(16, 32)
        self.conv3 = GraphConv(32, 64)
        self.conv4 = GraphConv(64, 128)
        #self.lin1 = Linear(128, 64)
        self.lin2 = Linear(128, num_classes)

    def forward(self, x, edge_index, batch):
        indx = torch.tensor([0,1,2,3,4,5,6,7,8,9]).long()
        indx = indx.to(device)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        #x = self.lin1(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        #print(x)
       # x=softmax(x) #,indx,dim=1)
       # x = Softmax(x)
        return x.reshape(-1)

def train(model, train_loader, optimizer):
  for epoch in range(100):
    model.train()
    loss_all = 0
    for data in train_loader:
        #data.x.type
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        #yy = torch.tensor([data.y[0],data.y.item()]).to(device)
        #print(out,yy)
        loss = F.mse_loss(out[:-1:5], data.y[:-1:5]) #.view(-1, 1))
        loss.backward()
        optimizer.step()
        loss_all += loss.item()# * data.num_graphs
        
    print('Epoch:',epoch, loss_all / len(train_loader))
  return loss_all / len(train_loader)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = make_loaders(data_file='data-new.pth', test_size=0.25,bsize=128) 
#or data in train_loader:
#  print(data.y)
model = GNN(1,5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ll=train(model,train_loader,optimizer)

model.eval()
test_loss = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out[:-1:5], data.y[:-1:5]) 
        test_loss += loss.item()
print(f"Test Loss: {test_loss / len(test_loader)}")


model.eval()
test_loss = 0
true_energies = []
predicted_energies = []
with torch.no_grad():
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out[:-1:5], data.y[:-1:5]) 
        test_loss += loss.item()
        #print(out)
        #print(out,data.y)
       # pred = out.argmax(dim=0)  # Use the class with highest probability.
     #   correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     #return correct / len(loader.dataset)  # Derive ratio of correct predictions.
        true_energies.extend(data.y.tolist())
        predicted_energies.extend(out.tolist())
        #print(data.y,out)
        #print(f"Test Loss: {correct / len(test_loader)}")
print(f"Test Loss: {test_loss / len(test_loader)}")
#print(true_energies[:-1:5])
plt.scatter(true_energies[:-1:5], predicted_energies[:-1:5], alpha=0.5)
plt.plot(true_energies[:-1:5],true_energies[:-1:5])
plt.xlabel('True Energy')
plt.ylabel('Predicted Energy')
plt.title('True Energy vs Predicted Energy')
plt.show()

  
