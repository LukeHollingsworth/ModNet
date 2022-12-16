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

class simple_network(nn.Module):
    def __init__(self, hyperparameters=None):
        if hyperparameters == None:
            hyperparameters = self.get_default_hyperparameters()
        self.hidden_size = hyperparameters['hidden_size']
        self.A1 = hyperparameters['rule1_grad']
        self.A2 = hyperparameters['rule2_grad']
        self.delta_theta = hyperparameters['delta_theta']
        super(simple_network, self).__init__()

        self.fc1 = nn.Linear(4, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

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
        self.hist.append(self.abs_error())

        self.task1_description = '$y = {}x + {}$'.format(self.A1, self.C1_test[0])
        self.task2_description = '$y = {}x + {}$'.format(self.A2, self.C2_test[0])

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
               'hidden_size' : 50, #hidden layer width
               'rule1_grad' : 0.5,
               'rule2_grad' : 5}
        return hps
    
    def set_data(self):
        self.N_task1 = int(self.fraction*self.N_train)
        self.N_task2 = self.N_train - self.N_task1

        # For the line Ax + By + C = 0
        self.B1, self.B2 = -1, -1
        self.C1_train, self.C2_train = np.zeros(self.N_task1), np.zeros(self.N_task2)
        self.C1_test, self.C2_test = np.zeros(self.N_test), np.zeros(self.N_test)

        train_points = np.random.uniform(-1, 1, (int(self.N_train), 2))
        test_points = np.random.uniform(-1, 1, (int(self.N_test), 2))
        train_points1, train_points2 = train_points[:int(self.N_task1)], train_points[-int(self.N_task2):]

        self.x1_train = np.concatenate((train_points1, np.ones((self.N_task1, 1)), np.zeros((self.N_task1,1))), axis=1)
        self.x2_train = np.concatenate((train_points2, np.zeros((self.N_task2, 1)), np.ones((self.N_task2,1))), axis=1)
        self.x1_test = np.concatenate((test_points, np.ones((self.N_test, 1)), np.zeros((self.N_test,1))), axis=1)
        self.x2_test = np.concatenate((test_points, np.zeros((self.N_test, 1)), np.ones((self.N_test,1))), axis=1)

        self.y1_train = np.abs(self.A1*self.x1_train[:,0] + self.B1*self.x1_train[:,1] + self.C1_train) / np.sqrt(self.A1**2 + self.B1**2)
        self.y2_train = np.abs(self.A2*self.x2_train[:,0] + self.B2*self.x2_train[:,1] + self.C2_train) / np.sqrt(self.A2**2 + self.B2**2)
        self.y1_test = np.abs(self.A1*self.x1_test[:,0] + self.B1*self.x1_test[:,1] + self.C1_test) / np.sqrt(self.A1**2 + self.B1**2)
        self.y2_test = np.abs(self.A2*self.x2_test[:,0] + self.B2*self.x2_test[:,1] + self.C2_test) / np.sqrt(self.A2**2 + self.B2**2)

        self.x_train = np.concatenate((self.x1_train, self.x2_train))
        self.y_train = np.concatenate((self.y1_train, self.y2_train))

        self.x_train = torch.from_numpy(self.x_train).float()
        self.x1_train = torch.from_numpy(self.x1_train).float()
        self.x2_train = torch.from_numpy(self.x2_train).float()
        self.x1_test = torch.from_numpy(self.x1_test).float()
        self.x2_test = torch.from_numpy(self.x2_test).float()
        
        self.y_train = torch.from_numpy(self.y_train).float().unsqueeze(1)
        self.y1_train = torch.from_numpy(self.y1_train).float().unsqueeze(1)
        self.y2_train = torch.from_numpy(self.y2_train).float().unsqueeze(1)
        self.y1_test = torch.from_numpy(self.y1_test).float().unsqueeze(1)
        self.y2_test = torch.from_numpy(self.y2_test).float().unsqueeze(1)

    def rules(self):
        if self.A1 > 1:
            r1 = [[-1 / self.A1, -1], [1 / self.A1, 1]]
        else:
            r1 = [[-1, -self.A1], [1, self.A1]]
        if self.A2 > 1:
            r2 = [[-1 / self.A2, -1], [1 / self.A2, 1]]
        else:
            r2 = [[-1, -self.A2], [1, self.A2]]

        return r1, r2

    def initialise_biases(self): #probably unneccesary, set biases initially to zero
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.zeros_(self.fc4.bias)

    def forward(self, input, mode='normal'):
        x = nn.functional.relu(self.fc1(input))        
        x1 = nn.functional.relu(self.fc2(x))        
        x2 = nn.functional.relu(self.fc3(x1))
        x3 = nn.functional.relu(self.fc4(x2))
        if mode == 'normal':
            return x3
        elif mode == 'other':
            return x, x1, x2, x3

    def abs_error(self):
        task1_error = (self.forward(self.x1_test) - self.y1_test).abs().mean(dim=0).item()
        task2_error = (self.forward(self.x2_test) - self.y2_test).abs().mean(dim=0).item()
        return [task1_error, task2_error]

    def train_model(self):
        self.IS_history = np.zeros((4,self.epochs))
        for epoch in range(self.epochs):
            for i in range(int(self.N_train/self.batch_size)): #input.shape == [2]
                idx = np.random.choice(self.N_train,self.batch_size,replace=False)
                self.do_train_step(idx)
            self.eval() #test/evaluation model 
            self.IS_history[:,epoch] = self.get_IS()
            with torch.no_grad():
                self.hist.append(self.abs_error())

        return self.IS_history

    def do_train_step(self, idx):
        sample=self.x_train[idx] 
        self.train() # we move the model to train regime because some models have different train/test behavior, e.g., dropout.
        self.optimizer.zero_grad()
        output = self.forward(sample)
        loss = F.mse_loss(output,self.y_train[idx])
        loss.backward()
        self.optimizer.step()

    def get_RI(self):
        self.RI = [[],[],[],[]]
        self.Itask1 = [[],[],[],[]]
        self.Itask2 = [[],[],[],[]]

        hidden_task1 = self.forward(self.x1_test, mode='other')        
        hidden_task2 = self.forward(self.x2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.y1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.y2_test, hidden_task2[-1])
        
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
        self.Itask1 = [[],[],[],[]]
        self.Itask2 = [[],[],[],[]]

        hidden_task1 = self.forward(self.x1_test, mode='other')        
        hidden_task2 = self.forward(self.x2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.y1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.y2_test, hidden_task2[-1])
        
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

    def get_IS(self):
        self.IS = []
        self.Itask1 = [[],[],[],[]]
        self.Itask2 = [[],[],[],[]]

        hidden_task1 = self.forward(self.x1_test, mode='other')
        hidden_task2 = self.forward(self.x2_test, mode='other')

        error_task1 = F.mse_loss(self.y1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.y2_test, hidden_task2[-1])
        
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
            IS_= np.dot(Itask1, Itask2) / (np.linalg.norm(Itask1) * np.linalg.norm(Itask2))
            self.Itask1[i].extend(list(Itask1)) 
            self.Itask2[i].extend(list(Itask2))
            self.IS.append(IS_)

        return self.IS
