import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
import random
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, ConcatDataset


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

class SimpleGaussian(nn.Module):
    def __init__(self, hyperparameters=None):
        if hyperparameters == None:
            hyperparameters = self.get_default_hyperparameters()
        self.hidden_size = hyperparameters['hidden_size']

        super(SimpleGaussian, self).__init__()
        self.fc1 = nn.Linear(3, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

        # Initialise hyperparameters
        self.N_train = hyperparameters['N_train']
        self.N_test = hyperparameters['N_test']
        self.epochs = hyperparameters['epochs']
        self.lr = hyperparameters['lr']
        self.batch_size = hyperparameters['batch_size']
        self.train_mode = hyperparameters['train_mode']
        self.second_task = hyperparameters['second_task']
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
        self.make_data()
        self.initialise_biases()
        self.hist.append(self.abs_error())

        # Task Description
        self.task1_description = "First Gaussian"
        self.task2_description = "Second Gaussian"

    def initialise_biases(self): #probably unneccesary, set biases initially to zero
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

    def get_default_hyperparameters(self):
        hps = {'N_train' : 1000, #size of training dataset 
            'N_test' : 100, #size of test set x
            'lr' : 0.001, #SGD learning rate 
            'epochs' : 10, #training epochs
            'batch_size' : 10,  #batch size (large will probably fail)           
            'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
            'train_mode' : 'random', #training mode 'random' vs 'replay' 
            'second_task' : 'prod', #first task adds x+y, second task 'prod' = xy or 'add1.5' = x+1.5y
            'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
            'hidden_size' : 0} #hidden layer width 
        return hps

    # Forward pass
    def forward(self, input, mode='normal'):
        x = nn.functional.relu(self.fc1(input))        
        x1 = nn.functional.relu(self.fc2(x))        
        x2 = nn.functional.relu(self.fc3(x1))      

        if mode == 'normal':
            return x2
        elif mode == 'other':
            return x, x1, x2

    def do_train_step(self, idx):
        sample=self.x_train[idx] 
        self.train() # we move the model to train regime because some models have different train/test behavior, e.g., dropout.
        self.optimizer.zero_grad()
        output = self.forward(sample)
        loss = F.mse_loss(output,self.y_train[idx])
        loss.backward()
        self.optimizer.step()

    # Make Training Data
    def make_data(self):
        self.N_task1 = int(self.fraction*self.N_train)
        self.N_task2 = self.N_train - self.N_task1

        self.x_task1_train = np.concatenate((np.random.uniform(0,100,(self.N_task1,1)),np.ones((self.N_task1,1)),np.zeros((self.N_task1,1))),axis=1)
        self.x_task2_train = np.concatenate((np.random.uniform(0,100,(self.N_task2,1)),np.zeros((self.N_task2,1)),np.ones((self.N_task2,1))),axis=1)
        self.x_task1_test = np.concatenate((np.random.uniform(0,100,(self.N_test,1)),np.ones((self.N_test,1)),np.zeros((self.N_test,1))),axis=1)
        self.x_task2_test = np.concatenate((np.random.uniform(0,100,(self.N_test,1)),np.zeros((self.N_test,1)),np.ones((self.N_test,1))),axis=1)

        # self.y_task1_train = np.ones(self.N_task1)
        # self.y_task2_train = np.zeros(self.N_task2)
        # self.y_task1_test = np.ones(self.N_test)
        # self.y_task2_test = np.zeros(self.N_test)

        self.y_task1_train = 2*np.ones(self.N_task1)
        self.y_task2_train = np.ones(self.N_task2)
        self.y_task1_test = 2*np.ones(self.N_test)
        self.y_task2_test = np.ones(self.N_test)
        
        self.x_train = np.concatenate((self.x_task1_train, self.x_task2_train))
        self.y_train = np.concatenate((self.y_task1_train, self.y_task2_train))

        self.x_train = torch.from_numpy(self.x_train).float()
        self.x_task1_train = torch.from_numpy(self.x_task1_train).float()
        self.x_task2_train = torch.from_numpy(self.x_task2_train).float()
        self.x_task1_test = torch.from_numpy(self.x_task1_test).float()
        self.x_task2_test = torch.from_numpy(self.x_task2_test).float()
        
        self.y_train = torch.from_numpy(self.y_train).float().unsqueeze(1)
        self.y_task1_train = torch.from_numpy(self.y_task1_train).float().unsqueeze(1)
        self.y_task2_train = torch.from_numpy(self.y_task2_train).float().unsqueeze(1)
        self.y_task1_test = torch.from_numpy(self.y_task1_test).float().unsqueeze(1)
        self.y_task2_test = torch.from_numpy(self.y_task2_test).float().unsqueeze(1)

    def abs_error(self):
        task1_error = (self.forward(self.x_task1_test) - self.y_task1_test).abs().mean(dim=0).item()
        task2_error = (self.forward(self.x_task2_test) - self.y_task2_test).abs().mean(dim=0).item()
        return [task1_error, task2_error]

    def train_model(self):
        for epoch in range(self.epochs):
            for i in range(int(self.N_train/self.batch_size)):
                idx = np.random.choice(self.N_train, self.batch_size, replace=False)
                self.do_train_step(idx)
            self.eval()
            with torch.no_grad():
                self.hist.append(self.abs_error())

    def get_RI(self):
        self.RI = [[],[],[],[],[]]
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]

        hidden_task1 = self.forward(self.x_task1_test, mode='other')        
        hidden_task2 = self.forward(self.x_task2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.y_task1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.y_task2_test, hidden_task2[-1])
        
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
