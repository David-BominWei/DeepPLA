datapath1 = '/home/wbm001/data/dti_data/Data/BindingDB_All_Mol.csv'
datapath2 = '/home/wbm001/data/dti_data/Data/BindingDB_All_Seq.csv'

seq_modelpath = '/home/wbm001/data/dti_data/Model/seq_model/saved_models/prose_mt_3x1024.sav'
mol_modelpath = '/home/wbm001/data/dti_data/Model/mol_model/model_300dim.pkl'

profilepath = '/home/wbm001/data/Result/version8/modelprofile'
figurepath = '/home/wbm001/data/Result/version8/modelfigure/'
modelsavepath = '/home/wbm001/data/Result/version8/premodels/'
finalmodel = '/home/wbm001/data/Result/version8/modelfinal.pkl'
internaltestpath = '/home/wbm001/data/Result/version8/internaltest.csv'

RANDOMSEED = 42

import pandas as pd
import gc

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(profilepath)

print("Start Reading Data")
data1 = pd.read_csv(datapath1, low_memory=False)
print('data1 done')
data2 = pd.read_csv(datapath2, low_memory=False)
print('data2 done')

import numpy as np
data = pd.concat([data1,data2], axis = 1).iloc[:,3:].astype(np.float32).dropna()

del data1
del data2
gc.collect()

print('data1&data2 removed')

data["Ki (log10)"] = data["Ki (log10)"].map(lambda x : x if x > -8 else None)
data = data.dropna()

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6, 6))
plt.hist(data["Ki (log10)"],bins=100)
plt.savefig("temp-checkhist.png")
writer.add_figure(tag='data distribution', figure=fig)


print('-'*20)
print('Start Building Network')


import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset,SequentialSampler,RandomSampler
from torch import tensor, nn
import torch

train_data, internal_test = train_test_split(data,test_size=0.2, random_state=RANDOMSEED)

train_data, test_data = train_test_split(train_data,test_size=0.1, random_state=RANDOMSEED)

# train
train_seq = tensor(np.array(train_data.iloc[:,301:])).unsqueeze(dim=1).to(torch.float32)
train_mol = tensor(np.array(train_data.iloc[:,1:301])).unsqueeze(dim=1).to(torch.float32)
train_Ki = tensor(np.array(train_data.iloc[:,0]))

trainDataset = TensorDataset(train_mol,train_seq,train_Ki)
trainDataLoader = DataLoader(trainDataset, batch_size=256)

#test
test_seq = tensor(np.array(test_data.iloc[:,301:])).unsqueeze(dim=1).to(torch.float32)
test_mol = tensor(np.array(test_data.iloc[:,1:301])).unsqueeze(dim=1).to(torch.float32)
test_Ki = tensor(np.array(test_data.iloc[:,0]))

testDataset = TensorDataset(test_mol,test_seq,test_Ki)
testDataLoader = DataLoader(testDataset, batch_size=256)

gc.collect()

# internal_test.to_csv(internaltestpath)

from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch

# block ------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1=False, strides=1):
        super().__init__()
        
        self.process = nn.Sequential (
            nn.Conv1d(in_channels, out_channels, 3, stride=strides, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        if use_conv1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv1 = None
        
    def forward(self, x):
        left = self.process(x)
        right = x if self.conv1 is None else self.conv1(x)
        
        return F.relu(left + right)

# cnnNet ------------------------------------------------------------------------
class cnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pre = nn.Sequential (
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(2)
        )
        
        self.layer1 = self._make_layer(32, 16, 2)
        
        
    def _make_layer(self, in_channels, out_channels, block_num, strides=1):

        layers = [Block(in_channels, out_channels, use_conv1=True, strides=strides)] # build the first layer with conv1
        
        for i in range(block_num):
            layers.append(Block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        
        return x

torch.cuda.empty_cache()
gc.collect()

# mainNet ------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mol_cnn = cnnNet()
        self.seq_cnn = cnnNet()
        
        self.pooling = nn.AvgPool1d(5, stride = 3)
        
        self.lstm = nn.LSTM(538, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        self.linear = nn.Sequential (
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(512, 32),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
            
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
            #nn.ReLU(),
#             nn.Dropout(p=0.5),
            
            nn.Linear(32, 1),
        )

    def forward(self, mol, seq):
        mol = self.mol_cnn(mol)
        seq = self.seq_cnn(seq)
        
        # put data into lstm        
        x = torch.cat((mol,seq),2)
        x = self.pooling(x)
        x,_ = self.lstm(x)
        
        # fully connect layer
        x = x.flatten(1)
        x = self.linear(x)
        
        return x.flatten()

# initialize weight
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

print('-'*20)
print("Start Training")

from pytorch_lightning.metrics.functional import r2score
from matplotlib import pyplot as plt
import io

############################################################
def train_loop(model, optimizer, train_dataloader, epoch):
    model.train()
    
    totalloss = 0
    avgloss = 0
    valli = None
    labli = None

    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("lr", current_lr, epoch)


    for step, batch in enumerate(train_dataloader):

        input_mols, input_seqs, labels = batch
        input_mols, input_seqs, labels = input_mols.to('cuda'), input_seqs.to('cuda'), labels.to('cuda')
        model.zero_grad()
        logits = model(input_mols, input_seqs)
        loss = loss_fn(logits, labels.float())
        
        loss.backward()
        optimizer.step()
        
        totalloss += loss.item()
        avgloss += loss.item()
        if valli is None:
            valli = logits
        else:
            valli = torch.cat([valli,logits],0)
        if labli is None:
            labli = labels.float()
        else:
            labli = torch.cat([labli,labels.float()],0)
        
        # for each 20 step print the loss
        if step%20 ==19:
            print("step: " + str(step-19) + "-" + str(step) + "|" + '\t'*2 + "avg: " + str(avgloss/20))
            avgloss = 0

        
            
    # after all print the mse and r^2
    with torch.no_grad():
        val_r2 = np.array(r2score(valli,labli).cpu())
    val_mse = totalloss / len(trainDataLoader)
    
    print("train mse score: " + str(val_mse))
    print("train r^2 score: " + str(val_r2))
    
    return val_mse, val_r2
    
        
############################################################
def test_loop(model, test_dataloader, epoch):

    model.eval()
    val_loss = []
    valli = None
    labli = None

    for batch in test_dataloader:
        input_mols, input_seqs, labels = tuple(t.to("cuda") for t in batch)
        with torch.no_grad():
            logits = model(input_mols, input_seqs)

        # test
        
        loss = loss_fn(logits, labels.float())
        val_loss.append(loss.item())
        
        if valli is None:
            valli = logits
        else:
            valli = torch.cat([valli,logits],0)
            
        if labli is None:
            labli = labels.float()
        else:
            labli = torch.cat([labli,labels.float()],0)

    fig = plt.figure(figsize=(6, 6))
    plt.xlabel("true value")
    plt.ylabel("predict value")
    plt.scatter(labli.cpu(), valli.cpu(), alpha = 0.2, color='Black')
    plt.plot((int(np.array(labli.cpu()).min()),int(np.array(labli.cpu()).max()+1)),(int(np.array(labli.cpu()).min()),int(np.array(labli.cpu()).max()+1)),color="r",linewidth=2)
    plt.savefig(figurepath + str(epoch) + '.svg')
    writer.add_figure(tag='test evaluate', figure=fig, global_step=epoch)
    
    with torch.no_grad():
        val_r2 = np.array(r2score(valli,labli).cpu())
    val_mse = np.array(val_loss).mean()
    
    print("test mse score: " + str(val_mse))
    print("test r^2 score: " + str(val_r2))
    
    return val_mse, val_r2

import torch.optim as optim
import time
import numpy as np

net = Net()
net.apply(initialize_weights)
net = net.to('cuda')

# draw a net graph and save into Tensor Board
net.eval()
writer.add_graph(net, (torch.randn(128,1,300).to('cuda'), torch.randn(128,1,6165).to('cuda')))

torch.cuda.empty_cache()
gc.collect()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=0.00001)


for epoch in range(1000):

    time0 = time.time()

    print("-"*20)
    print("epoch: " + str(epoch))
    trainmse, trainr2 = train_loop(net,optimizer,trainDataLoader,epoch)    
    time1 = time.time()
    writer.add_text("train time", str(time1-time0), epoch)

    testmse, testr2 = test_loop(net, testDataLoader, epoch)
    writer.add_text("test time", str(time.time()-time1), epoch)

    writer.add_scalars('MSE', { "train MSE" : trainmse,
                                            "test MSE" : testmse} , epoch)
    writer.add_scalars('R^2', { "train R^2" : trainr2,
                                            "test R^2" : testr2} , epoch)
    print("use time: " + str(time.time() - time0))

    net.eval()
    torch.save(net,finalmodel)
    
    scheduler.step(testmse)

    if epoch % 50 == 49:
        try:
            net.eval()
            torch.save(net, modelsavepath + 'model' + str(epoch) + '.pkl')
        except:
            print("can not save the model")
            pass

