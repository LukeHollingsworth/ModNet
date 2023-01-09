import numpy as np
import matplotlib 
from matplotlib import rcParams
import matplotlib.pyplot as plt
from cycler import cycler 
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=1
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from tqdm.autonotebook import tqdm
import time

class what_where_network(nn.Module):
    def __init__(self, hyperparameters=None):
        if hyperparameters == None:
            hyperparameters = self.get_default_hyperparameters()
        self.hidden_size = hyperparameters['hidden_size']
        super(what_where_network, self).__init__()

        self.fc1 = nn.Linear(83, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, 9)

        # Initialise hyperparameter
        self.N_train = hyperparameters['N_train']
        self.N_test = hyperparameters['N_test']
        self.epochs = hyperparameters['epochs']
        self.lr = hyperparameters['lr']
        self.batch_size = hyperparameters['batch_size']
        self.train_mode = hyperparameters['train_mode']
        self.fraction = hyperparameters['fraction']

        # Initialises some attributes
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.type_of_network = 'simple_network'

        # Arrays for later analysis
        self.hist = []
        self.RI = [[],[],[],[],[]]
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]

        # Training and testing data 
        self.set_data()
        self.initialise_biases()
        # self.hist.append(self.abs_error())
        # self.hist.append(self.ce_error())
        self.hist.append(self.accuracy())

        self.task1_description = 'Where?'
        self.task2_description = 'What?'

    def get_default_hyperparameters(self):
        hps = {'N_train' : 1000, #size of training dataset 
               'N_test' : 100, #size of test set x
               'lr' : 0.001, #SGD learning rate 
               'epochs' : 10, #training epochs
               'batch_size' : 10,  #batch size (large will probably fail)           
               'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
               'train_mode' : 'random', #training mode 'random' vs 'replay' 
               'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
               'hidden_size' : 100} #hidden layer width
        return hps
    
    def set_data(self):
        self.N_train = 50*81
        self.N_test = 5*81
        self.shapes = { 'T' : [[1,1,1],
                               [0,1,0],
                               [0,1,0]],
                        'K' : [[1,0,1],
                               [1,1,0],
                               [1,0,0]],
                        '+' : [[0,1,0],
                               [1,1,1],
                               [0,1,0]],
                        'L' : [[1,0,0],
                               [1,0,0],
                               [1,1,1]],
                        'Z' : [[1,1,0],
                               [0,1,0],
                               [0,1,1]],
                        'X' : [[1,0,1],
                               [0,1,0],
                               [1,0,1]],
                        'n' : [[0,1,0],
                               [1,0,1],
                               [1,0,1]],
                        'u' : [[1,0,1],
                               [1,1,1],
                               [0,0,0]],
                        '=' : [[0,1,1],
                               [0,0,0],
                               [1,1,1]],
                        'E' : [[0,0,0],
                               [0,0,0],
                               [0,0,0]]}
        self.shape_labels = {'T' : 0, 'K' : 1, '+' : 2, 'L' : 3, 'Z' : 4, 'X' : 5, 'n' : 6, 'u' : 7, '=' : 8}
        self.location_labels = {'TL' : 0, 'TM' : 1, 'TR' : 2, 'CL' : 3,'CM' : 4, 'CR' : 5, 'BL' : 6, 'BM' : 7, 'BR' : 8}
        self.base = [self.shapes['E']] * 9

        self.c1, self.c2 = np.asarray([1,0]), np.asarray([0,1])
        self.X1_train, self.X2_train, self.X1_test, self.X2_test = np.zeros((1,83)), np.zeros((1,83)), np.zeros((1,83)), np.zeros((1,83))
        self.Y1_train, self.Y2_train, self.Y1_test, self.Y2_test = np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)
        
        for i in range(9):
            shape = np.asarray(list(self.shapes.values())[i])
            shape_label = one_hot(i)
            for j in range(9):
                location_label = one_hot(j)
                copy = np.asarray(self.base.copy())
                copy[j] = shape
                x = np.array([])
                for n1 in range(3):
                    row1 = np.asarray([copy[3*n1][0], copy[3*n1+1][0], copy[3*n1+2][0]])
                    row2 = np.asarray([copy[3*n1][1], copy[3*n1+1][1], copy[3*n1+2][1]])
                    row3 = np.asarray([copy[3*n1][2], copy[3*n1+1][2], copy[3*n1+2][2]])
                    block = [row1, row2, row3]
                    x = np.append(x, block).astype(np.int32)
                
                for i in range(int(self.N_train/162)):
                    self.X1_train = np.vstack((self.X1_train, np.concatenate((x, self.c1))))
                    self.X2_train = np.vstack((self.X2_train, np.concatenate((x, self.c2))))

                    self.Y1_train = np.vstack((self.Y1_train, location_label))
                    self.Y2_train = np.vstack((self.Y2_train, shape_label))

                for i in range(int(self.N_test/81)):
                    self.X1_test = np.vstack((self.X1_test, np.concatenate((x, self.c1))))
                    self.X2_test = np.vstack((self.X2_test, np.concatenate((x, self.c2))))

                    self.Y1_test = np.vstack((self.Y1_test, location_label))
                    self.Y2_test = np.vstack((self.Y2_test, shape_label))

        self.X_train = np.vstack((np.asarray(self.X1_train[1:]), np.asarray(self.X2_train[1:])))
        self.Y_train = np.vstack((np.asarray(self.Y1_train[1:]), np.asarray(self.Y2_train[1:])))

        self.X1_test = self.X1_test[1:]
        self.X2_test = self.X2_test[1:]
        self.Y1_test = self.Y1_test[1:]
        self.Y2_test = self.Y2_test[1:]

        shuf_idx_train = np.arange(0, self.N_train)
        shuf_idx_test = np.arange(0, self.N_test)
        np.random.shuffle(shuf_idx_train)
        np.random.shuffle(shuf_idx_test)

        for i in range(self.N_train):
            self.X_train[i] = self.X_train[shuf_idx_train[i]]
            self.Y_train[i] = self.Y_train[shuf_idx_train[i]]
        
        for i in range(self.N_test):
            self.X1_test[i] = self.X1_test[shuf_idx_test[i]]
            self.X2_test[i] = self.X2_test[shuf_idx_test[i]]
            self.Y1_test[i] = self.Y1_test[shuf_idx_test[i]]
            self.Y2_test[i] = self.Y2_test[shuf_idx_test[i]]

        self.X_train = torch.from_numpy(self.X_train).float()
        self.X1_test = torch.from_numpy(np.asarray(self.X1_test)).float()
        self.X2_test = torch.from_numpy(np.asarray(self.X2_test)).float()

        self.Y_train = torch.from_numpy(self.Y_train).float()
        self.Y1_test = torch.from_numpy(np.asarray(self.Y1_test)).float()
        self.Y2_test = torch.from_numpy(np.asarray(self.Y2_test)).float()

    def initialise_biases(self): #probably unneccesary, set biases initially to zero
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.zeros_(self.fc4.bias)
        torch.nn.init.zeros_(self.fc5.bias)

    def forward(self, input, mode='normal'):
        x = nn.functional.relu(self.fc1(input))        
        x1 = nn.functional.relu(self.fc2(x))        
        x2 = nn.functional.relu(self.fc3(x1))
        x3 = nn.functional.relu(self.fc4(x2))
        x4 = nn.functional.softmax(self.fc5(x3), dim=1)
        if mode == 'normal':
            return x4
        elif mode == 'other':
            return x, x1, x2, x3, x4

    def abs_error(self):
        task1_error = (self.forward(self.X1_test) - self.Y1_test).abs().mean(dim=0)
        task2_error = (self.forward(self.X2_test) - self.Y2_test).abs().mean(dim=0)
        return [task1_error.item(), task2_error.item()]

    def ce_error(self):
        loss = nn.CrossEntropyLoss()
        task1_loss = loss(self.forward(self.X1_test), self.Y1_test)
        task2_loss = loss(self.forward(self.X2_test), self.Y2_test)
        return [task1_loss.item(), task2_loss.item()]

    def accuracy(self):
        output1 = self.forward(self.X1_test[:int(0.2*self.N_test)])
        output2 = self.forward(self.X2_test[:int(0.2*self.N_test)])
        correct1 = (F.softmax(output1, dim=1).max(dim=1)[1] == (self.Y1_test[:int(0.2*self.N_test)]).max(dim=1)[1]).sum()
        correct2 = (F.softmax(output2, dim=1).max(dim=1)[1] == (self.Y2_test[:int(0.2*self.N_test)]).max(dim=1)[1]).sum()
        accuracy1 = correct1.item() / len(self.X1_test[:int(0.2*self.N_test)])
        accuracy2 = correct2.item() / len(self.X2_test[:int(0.2*self.N_test)])
        # for i in range(5):
        #     print(np.reshape(self.X2_test[i][:-2], (9,9)), self.X2_test[i][-2:], '\n', self.forward(self.X2_test)[i], '\n', self.Y2_test[i])
        return accuracy1, accuracy2

    def train_model(self):
        if self.train_mode == 'random':
            is_learnt = False
            for epoch in range(self.epochs):
                for i in range(int(0.2*(self.N_train/self.batch_size))): # train on 20%
                    idx = np.random.choice(self.N_train,self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() #test/evaluation model 
                with torch.no_grad():
                    # self.hist.append(self.abs_error())
                    # self.hist.append(self.ce_error())
                    self.hist.append(self.accuracy())
                    if np.mean(self.accuracy()) >= 0.90 and is_learnt != True:
                        self.learning_speed = epoch
                        is_learnt = True
        
        if self.train_mode == 'replay':
            N_task1 = int(self.N_train*self.fraction)
            for epoch in range(self.epochs):
                for i in range(int(0.2*(N_task1/self.batch_size))):
                    idx = np.random.choice(N_task1,self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() # test/evaluation model 
                with torch.no_grad():
                    self.hist.append(self.accuracy())     
            for epoch in range(self.epochs):
                for i in range(int((self.N_train-N_task1)/self.batch_size)): #input.shape == [2]
                    if (i+1)%10 == 0: 
                        idx = np.random.choice(N_task1,self.batch_size,replace=False)
                    else:    
                        idx = np.random.choice(range(N_task1,self.N_train),self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() #test/evaluation model 
                with torch.no_grad():
                    # self.hist.append(self.abs_error())
                    self.hist.append(self.accuracy())

    def do_train_step(self, idx):
        sample=self.X_train[idx] 
        self.train() # we move the model to train regime because some models have different train/test behavior, e.g., dropout.
        self.optimizer.zero_grad()
        output = self.forward(sample)
        # loss = F.mse_loss(output,self.Y_train[idx])
        loss = F.cross_entropy(output, self.Y_train[idx])
        loss.backward()
        self.optimizer.step()

    def get_RI(self):
        self.RI = [[],[],[],[],[]]
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]

        hidden_task1 = self.forward(self.X1_test, mode='other')        
        hidden_task2 = self.forward(self.X2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.Y1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.Y2_test, hidden_task2[-1])
        
        for i in range(len(hidden_task1)):
            hidden_task1[i].retain_grad()
            hidden_task2[i].retain_grad()
            
        error_task1.backward()
        error_task2.backward()    

        for i in range(len(hidden_task1)):
            Itask1 = (((hidden_task1[i] * hidden_task1[i].grad)**2).mean(0)).detach().numpy()
            Itask2 = (((hidden_task2[i] * hidden_task2[i].grad)**2).mean(0)).detach().numpy()
            Itask1[Itask1 < np.mean(Itask1)/10]=0
            Itask2[Itask2 < np.mean(Itask2)/10]=0
            RI_ = (Itask1 - Itask2) / (Itask1 + Itask2)
            self.Itask1[i].extend(list(Itask1)) 
            self.Itask2[i].extend(list(Itask2))
            self.RI[i].extend(list(RI_))

    def get_I(self):
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]

        hidden_task1 = self.forward(self.X1_test, mode='other')        
        hidden_task2 = self.forward(self.X2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.Y1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.Y2_test, hidden_task2[-1])
        
        for i in range(len(hidden_task1)):
            hidden_task1[i].retain_grad()
            hidden_task2[i].retain_grad()
            
        error_task1.backward()
        error_task2.backward()    

        for i in range(len(hidden_task1)):
            Itask1 = (((hidden_task1[i] * hidden_task1[i].grad)**2).mean(0)).detach().numpy()
            Itask2 = (((hidden_task2[i] * hidden_task2[i].grad)**2).mean(0)).detach().numpy()
            Itask1[Itask1 < np.mean(Itask1)/10]=0
            Itask2[Itask2 < np.mean(Itask2)/10]=0
            self.Itask1[i].extend(list(Itask1))
            self.Itask2[i].extend(list(Itask2))

class LamarckianModel(what_where_network):
    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.initialise_weights()
        self.type_of_network = 'Lamarckian'

    def initialise_weights(self):
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.xavier_normal_(self.fc5.weight)

def one_hot(index):
    onehot = np.zeros(9)
    onehot[index] = 1
    return onehot
