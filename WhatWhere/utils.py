# =============================================================================
# DESCRIPTION: 
#This file contain utility functions for training and analysing the simple_network and MNIST_network model classes in networks.py

#Functions are as follows (described below)
# simple_network() functions:
#  • plot_training()
#  • plot_RI()
#  • plot_lesion_test()
#  • train_and_analyse()
# MNIST_network() functions:
#  • MNIST_subset()
#  • EWC()
# =============================================================================

from copy import deepcopy
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import time 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# SIMPLE_NETWORK ANALYSIS FUNCTIONS
# =============================================================================
# =============================================================================

# The following three functions all take a list of trained simple_network models and prefrom analysis on them 
# =============================================================================
# plot_training plots the testing abs error throughout training for all models (and shows an average) 
# =============================================================================
def plot_training(models, title=None, axis_scale='linear'):  #only works for simple_network lists, not MNIST_networks
    if models[0].type_of_network == 'simple_network':    
        fig, ax = plt.subplots(figsize = (2,1.5))
        task1 = np.zeros(len(np.array(models[0].hist)))
        task2 = np.zeros(len(np.array(models[0].hist)))    
        for i in range(len(models)):
            ax.plot(np.array(models[i].hist)[:,0], linewidth=0.5, alpha=0.2, c='C0')
            task1 += np.array(models[i].hist)[:,0]
            ax.plot(np.array(models[i].hist)[:,1], linewidth=0.5, alpha=0.2, c='C1')
            task2 += np.array(models[i].hist)[:,1]
        task1, task2 = task1/len(models), task2/len(models)
        ax.plot(task1, linewidth=1, c='C0', label = r'Task 1: %s' %models[-1].task1_description )
        ax.plot(task2, linewidth=1, c='C1', label =r'Task 2: %s' %models[-1].task2_description)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epochs')
        if models[-1].train_mode == 'replay':
            ax.axvspan(0, models[-1].epochs,color='C0', alpha=0.1)  #vertical shading
            ax.axvspan(models[-1].epochs, 2*models[-1].epochs,color='C1', alpha=0.1)  #vertical shading
        ax.legend()
        if title != None:
            fig.suptitle("%s" %title)
        if axis_scale == 'log':
            ax.set_xscale('log')
        plt.show()
        return
        
    if models[0].type_of_network == 'MNIST_network': 
        fig, axs = plt.subplots(1,2,figsize=(4,1.5))
        if models[0].train_mode != 'random':
            all_models_losses = {}
            for j in range(models[0].task_count):
                all_models_losses[j] = np.zeros(len(models[0].loss[j]))
            for i in range(len(models)):
                for t, v in models[i].loss.items():
                    axs[0].plot(list(range(t * models[i].epochs, (t + 1) * models[i].epochs + 1)), v, linewidth=0.5, alpha=0.2, color='C%g'%t)
                    all_models_losses[t] += np.array(v)
            for t, v in models[0].loss.items():
                axs[0].plot(list(range(t * models[0].epochs, (t + 1) * models[0].epochs + 1)), all_models_losses[t]/len(models), label="Task %g" %(t+1), linewidth=1, alpha=1, color='C%g'%t)
    
        else:
            all_models_loss = np.zeros(len(models[0].loss[0]))
            for i in range(len(models)):
                for t, v in models[i].loss.items():
                    axs[0].plot(list(range(t * models[i].epochs, (t + 1) * models[i].epochs + 1)), v, color='C6', linewidth=0.5, alpha=0.2)
                    all_models_loss += v
            for t, v in models[0].loss.items():
                axs[0].plot(list(range(t * models[i].epochs, (t + 1) * models[i].epochs + 1)), all_models_loss/len(models), label="All tasks", color='C6', linewidth=1, alpha=1)
            
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross entropy loss")
        axs[0].legend(loc = 1)
        
        all_models_acc = {}
        for j in range(models[0].task_count):
            all_models_acc[j] = np.zeros(len(models[0].acc[j]))
        for i in range(len(models)):
            for t, v in models[i].acc.items():
                if models[i].train_mode == 'random':
                    axs[1].plot(list(range(models[i].total_epochs + 1)), v,linewidth=0.5, alpha=0.2, color='C%g'%t)
                    all_models_acc[t] += v
                else:
                    axs[1].plot(list(range(t * models[i].epochs, models[i].total_epochs + 1)), v,linewidth=0.5, alpha=0.2, color='C%g'%t)
                    all_models_acc[t] += v
        for t, v in models[i].acc.items():
            if models[0].train_mode == 'random':
                axs[1].plot(list(range(models[i].total_epochs + 1)), all_models_acc[t]/len(models), label="Task %g" %(t+1), color='C%g'%t)
            else:
                axs[1].plot(list(range(t * models[i].epochs, models[i].total_epochs + 1)), all_models_acc[t]/len(models), label="Task %g" %(t+1), color='C%g'%t)
        
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Test accuracy")
        axs[1].set_ylim(0,1)
        axs[1].legend(loc=4)
        
        if models[0].train_mode != 'random':
            for i in range(models[0].task_count):
                axs[0].axvspan(i*models[0].epochs, (i+1)*models[0].epochs,color='C%g' %i, alpha=0.1)  #vertical shading
                axs[1].axvspan(i*models[0].epochs, (i+1)*models[0].epochs,color='C%g' %i, alpha=0.1)  #vertical shading
        plt.show()
        return 

# =============================================================================
# plot_RI plots the fractional task variance throughout the layers 
# =============================================================================
def plot_RI(models, show_threshold=False, title=None, learning_speeds=None):  #only works for simple_network lists, not MNIST_networks
    if models[0].type_of_network == 'simple_network':
        RI = [[],[],[],[],[]]
        for model in models:
            for i in range(len(RI)):
                RI[i].extend(list(model.RI[i]))
        fig, axs = plt.subplots(1,4,sharey = True, figsize = (4,0.8))
        for i in range(4):
            n, bins, patches = axs[i].hist(RI[i], weights=np.ones(len(RI[i])) / len(RI[i]),bins=np.linspace(-1,1,11))
            bin_centre = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
            col = (bin_centre - min(bin_centre))/(max(bin_centre) - min(bin_centre))
            cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C1','C0'], N=1000)
            for c, p in zip(col, patches):
                    plt.setp(p, 'facecolor', cm(c))
            plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            axs[i].set_title("Hidden layer %g" %(i+1))
            axs[i].set_xlim([-1,1])
            axs[i].set_xlabel(r'$\mathcal{RI}$')
            if i == 0:
                axs[i].set_ylabel('Proportion')            
            if i == 3 and show_threshold == True: 
                axs[i].axvline(0.9,color='r',linestyle='--',linewidth=0.8)
                axs[i].axvline(-0.9,color='r',linestyle='--',linewidth=0.8)
        for i in range(4):
            axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(RI[i])))/len(RI[i])))), fontdict = {'color':'grey', 'fontsize':4})
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return

    if models[0].type_of_network == 'Lamarckian':
        for i in range(len(models)):
            if learning_speeds[i] == None:
                del learning_speeds[i]
                del models[i]
        N_models = len(models)
        learning_speeds = np.asarray(learning_speeds)
        inds = learning_speeds.argsort()
        sorted_models = []
        sorted_learning_speeds = []
        for i, ind in enumerate(inds):
            sorted_models.append(models[ind])
            sorted_learning_speeds.append(learning_speeds[ind])

        model_groups = [sorted_models[x:x+round(N_models/5)] for x in range(0, N_models, round(N_models/5))]
        ls_groups = [sorted_learning_speeds[x:x+round(N_models/5)] for x in range(0, N_models, round(N_models/5))]
        ls_means = [sum(group)/len(group) for group in ls_groups]

        for l, group in enumerate(model_groups):
            print('Average number of epochs taken to learn: ', ls_means[l])
            RI = [[],[],[],[],[]]
            for model in group:
                for i in range(len(RI)):
                    RI[i].extend(list(model.RI[i]))
            fig, axs = plt.subplots(1,4,sharey = True, figsize = (4,0.8))
            for i in range(4):
                n, bins, patches = axs[i].hist(RI[i], weights=np.ones(len(RI[i])) / len(RI[i]),bins=np.linspace(-1,1,11))
                bin_centre = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
                col = (bin_centre - min(bin_centre))/(max(bin_centre) - min(bin_centre))
                cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C1','C0'], N=1000)
                for c, p in zip(col, patches):
                        plt.setp(p, 'facecolor', cm(c))
                plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                axs[i].set_title("Hidden layer %g" %(i+1))
                axs[i].set_xlim([-1,1])
                axs[i].set_xlabel(r'$\mathcal{RI}$')
                if i == 0:
                    axs[i].set_ylabel('Proportion')            
                if i == 3 and show_threshold == True: 
                    axs[i].axvline(0.9,color='r',linestyle='--',linewidth=0.8)
                    axs[i].axvline(-0.9,color='r',linestyle='--',linewidth=0.8)
            for i in range(4):
                axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(RI[i])))/len(RI[i])))), fontdict = {'color':'grey', 'fontsize':4})
            if title != None:
                fig.suptitle("%s" %title)
            plt.show()
        return
        
    if models[0].type_of_network == 'MNIST_network':
        combined_RI = {}
        for i in range(models[0].task_count):
            for j in range(models[0].task_count):
                combined_RI[i,j] = []
        for i in range(models[0].task_count):
            for j in range(models[0].task_count):
                for k in range(len(models)):
                    combined_RI[i,j].extend(list(models[k].RI[i,j]))
                combined_RI[i,j] = np.array(combined_RI[i,j]) 
                
        fig, axs = plt.subplots(models[0].task_count-1, models[0].task_count-1, figsize=(1*(models[0].task_count-1),1*(models[0].task_count-1)), sharex=True, sharey=True)
        # fig.suptitle("RI histograms across task pairs")
        for i in range(models[0].task_count-1):
            for j in range(i):
                axs[i][j].axis("off")
            for j in range(i,models[0].task_count-1):
                n, bins, patches = axs[i][j].hist(combined_RI[i,j+1],  weights=np.ones(len(combined_RI[i,j+1])) / len(combined_RI[i,j+1]), bins=np.linspace(-1,1,11))
                bin_centres = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
                col = [(bin_centre + 1) / 2 for bin_centre in bin_centres]
                cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C%g' %(j+1),'C%g' %i], N=1000)
                for c, p in zip(col, patches):
                        plt.setp(p, 'facecolor', cm(c))
                plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                colors = {}
                for color in range(models[0].task_count):
                    colors[color] = matplotlib.colors.to_rgba_array('C%g' %color)
                    colors[color][0][-1] = 0.3
                if i == 0:
                    # axs[i][j].set_title("Task %g" %(j+2), backgroundcolor=colors[j+1][0])
                    axs[i][j].set_title("Task %g" %(j+2), color='C%g' %(j+1))
                    if j == 0:
                        axs[i][j].set_xlim([-1,1])
                        axs[i][j].set_xticks([-1,0,1])
                        axs[i][j].set_xticklabels(['-1','0','1'])
                if j == i:
                    # axs[i][j].set_ylabel("Task %g" %(j+1), backgroundcolor=colors[j][0])
                    axs[i][j].set_ylabel("Task %g" %(j+1), color='C%g' %(j))
        for i in range(models[0].task_count-1):
            for j in range(i,models[0].task_count-1):
                    axs[i][j].text(0.5,axs[i][j].get_ylim()[-1]*0.9, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(combined_RI[i,j+1])))/len(combined_RI[i,j+1])))),fontsize=4, color='grey')
        plt.show()
        return 

def plot_IS(models, show_threshold=False, title=None):    
    if models[0].type_of_network == 'simple_network':
        IS = np.zeros((np.shape(models[0].IS)))
        fig, ax = plt.subplots(figsize = (2,1.5))
        for i in range(len(models)):
            ax.plot(np.linspace(0,4,5), models[i].IS, alpha=0.2, lw=0.5)
            IS += models[i].IS
        IS = IS / len(models)
        ax.plot(np.linspace(0,4,5), IS, color='k', lw=1)
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return

def plot_IS_history(models, IS_history, show_threshold=False, title=None):    
    if models[0].type_of_network == 'simple_network':
        fig, axs = plt.subplots(1,4,sharey = True, figsize = (4,0.8))
        for i in range(4):
            IS_avg = np.zeros((10))
            for j in range(len(IS_history[0])):
                axs[i].plot(np.linspace(0, 9, 10), IS_history[i][j], alpha=0.2, lw=0.5)
                IS_avg += IS_history[i][j]
            IS_avg = IS_avg / len(IS_history[0])
            axs[i].plot(np.linspace(0, 9, 10), IS_avg, color='k', lw=1)
            axs[i].set_title("Hidden layer %g" %(i+1))
            axs[i].set_xlabel(r'$\mathcal{IS}$')
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return

# =============================================================================
# plot_lesion_test lesions (sets to zero) penultimate layer neurons with high or low RI then performs a test and plots the performance drop.
# this ONLY works for simple_models not MNIST_models
# =============================================================================
def plot_lesion_test(models, RI_threshold=0.9, title=None): #only works for simple_network lists, not MNIST_networks
    
    derr_h1_large, derr_h1_small, derr_h2_large, derr_h2_small = [], [], [], []
    
    for model in models:
        initial_error = model.abs_error()
        h1 = model.forward(model.x_task1_test, mode='other')[-2]
        h2 = model.forward(model.x_task2_test, mode='other')[-2]
        
        h1_large = h1 * torch.Tensor(np.less(np.array(model.RI[-2]),RI_threshold))
        h1_small = h1 * torch.Tensor(np.greater(np.array(model.RI[-2]),-RI_threshold))
        h2_large = h2 * torch.Tensor(np.less(np.array(model.RI[-2]),RI_threshold))
        h2_small = h2 * torch.Tensor(np.greater(np.array(model.RI[-2]),-RI_threshold))
        
        derr_h1_large.append(((nn.functional.relu(model.fc5(h1_large)) - model.y_task1_test).abs().mean(dim=0).item() - initial_error[0])/initial_error[0])
        derr_h1_small.append(((nn.functional.relu(model.fc5(h1_small)) - model.y_task1_test).abs().mean(dim=0).item() - initial_error[0])/initial_error[0])
        derr_h2_large.append(((nn.functional.relu(model.fc5(h2_large)) - model.y_task2_test).abs().mean(dim=0).item() - initial_error[1])/initial_error[1])
        derr_h2_small.append(((nn.functional.relu(model.fc5(h2_small)) - model.y_task2_test).abs().mean(dim=0).item() - initial_error[1])/initial_error[1])

    fig, axs = plt.subplots(1, 2, figsize = (1.5,0.8), sharey = True)
    axs[0].bar([-0.5,0.5],[100*np.mean(np.array(derr_h1_small)),100*np.mean(np.array(derr_h2_small))],width=1,color=['C0','C1'])
    axs[1].bar([-0.5,0.5],[100*np.mean(np.array(derr_h1_large)),100*np.mean(np.array(derr_h2_large))],width=1, color=['C0','C1'])
    axs[0].set_xticks([-0.5,0.5])
    axs[0].set_xticklabels([r'Task 1:''\n'r'%s' %(models[-1].task1_description),r'Task 2:''\n'r'%s' %(models[-1].task2_description)])
    axs[1].set_xticks([-0.5,0.5])
    axs[1].set_xticklabels([r'Task 1:''\n'r'%s' %(models[-1].task1_description),r'Task 2:''\n'r'%s' %(models[-1].task2_description)])
    axs[0].set_title(r'Lesion $\mathcal{RI}$ < %.2f' %-RI_threshold)
    axs[1].set_title('Lesion $\mathcal{RI}$ > %.2f' %RI_threshold)
    axs[0].set_ylabel(r'$\Delta$ Error (%)')
    if title != None:
        fig.suptitle("%s" %title)
    plt.show()
    return 
    

# =============================================================================
# train_multiple trains N_models simple_networks and returns the trained networkas a list
# a network is only accepted as 'trained' if the abs_error on BOTH tasks is < 0.05 after training 
# if a network fails to train 10 times in a row the training is aborted and an error returned
# =============================================================================
def train_multiple(model_class, hyperparameters = None,  N_models=20, ):
    models = []
    learning_speeds = []  
    
    if model_class == 'simple_network':
        from networks import what_where_network
        for _ in tqdm(range(N_models), desc="Model"):
            model = what_where_network(hyperparameters)
            model.train_model()
            model.get_RI()
            models.append(model)
        return models
    
    if model_class == 'Lamarckian':
        from networks import LamarckianModel
        for _ in tqdm(range(N_models), desc="Model"):
            model = LamarckianModel(hyperparameters)
            model.train_model()
            try:
                model.learning_speed
                learning_speeds.append(model.learning_speed)
            except:
                learning_speeds.append(None)
            model.get_RI()
            models.append(model)
        return models, learning_speeds

def train_IS_history(model_class, hyperparameters = None,  N_models=20, ):
    models = []
    IS_history = [np.zeros((N_models, hyperparameters['epochs'])),np.zeros((N_models, hyperparameters['epochs'])),np.zeros((N_models, hyperparameters['epochs'])),
                  np.zeros((N_models, hyperparameters['epochs'])),np.zeros((N_models, hyperparameters['epochs']))]
    
    if model_class == 'simple_network':
        from networks import what_where_network
        for n in tqdm(range(N_models), desc="Model"):
            fail_count = 0
            for i in range(hyperparameters['epochs']):
                if fail_count >= 10:
                    print("\n This model doesn't train well, aborting")
                    return models
                model = what_where_network(hyperparameters)
                model.train_model()
                if model.accuracy()[0]>0.95 and model.accuracy()[1]>0.95:
                    # model.get_RI()
                    model.get_IS()
                    for j in range(len(IS_history)):
                        IS_history[j][n][i] = model.IS[j]
                    models.append(model)
                else:
                    fail_count += 1
        return models, IS_history


# =============================================================================
# =============================================================================
# MNIST_NETWORK ANALYSIS FUNCTIONS
# =============================================================================
# =============================================================================

# =============================================================================
# MNIST_subset defines a pytorch dataset containing subsets of the MNIST dataset labelled as defined in task_sets

#__init__(task_sets) initialises the dataset             
    #task_sets defines all the tasks. Here's an example of how we'd write 3 seperate tasks which would all train on the same model:
    #task 1: digits 0 and 7 --> 0, digits 1 and 8 --> 1
    #task 2: digits 0 and 8 --> 0, digits 1 and 7 --> 1
    #task 3: digits 0, 1, 2, 3 and 4 --> 0, digits 5,6,7,8 and 9 --> 1
    #would give: 
    #task_sets = [  [[0,7],[1,8]]  ,  [[0,8],[1,7]]  ,  [[0,1,2,3,4],[5,6,7,8,9]] ]
    # note all tasks must have the same number of output classes (here 2 but could be more, or len(task_sets[0]))

# __getitem__() returns a flattened mnist image to which is appended the task context.
# i.e. the first 784 digits are the image, the remainder is the context label (a one-hot vector).
# =============================================================================
class MNIST_subset(Dataset):
    def __init__(self, task_sets = [[0,2,4,6,8],[1,3,5,7,9]], train=True, task_id=0, task_count=1):
        global trainset, testset

        try: 
            trainset
        except NameError:
            # print("Downloading MNIST datasets")
            trainset = datasets.MNIST('./mnistdata', train=True, download=True)
            testset = datasets.MNIST('./mnistdata', train=False, download=True)
        else: 
            pass 
        
        if train == True:
            dataset = trainset
        elif train == False:
            dataset = testset
        for i in range(len(task_sets)):
            for j in range(len(task_sets[i])):
                if i == 0 and j == 0: 
                    self.data = dataset.data[dataset.targets==task_sets[i][j]]
                    self.targets = i*torch.ones(sum(dataset.targets==task_sets[i][j])).byte()
                else:
                    self.data = torch.cat((self.data,dataset.data[dataset.targets==task_sets[i][j]]))
                    self.targets = torch.cat((self.targets,i*torch.ones(sum(dataset.targets==task_sets[i][j])).byte()))
                
        idx = torch.randperm(len(self.targets)) 
        self.data = self.data[idx].float()/255
        self.targets = self.targets[idx].long() 
        
        #add in context:
        context = torch.zeros((len(self.data),task_count))
        context[:,task_id] = torch.ones(len(self.data))
        context = context.float()
        
        self.data = torch.cat((self.data.reshape(len(self.data),-1),context),dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target
    
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]



# =============================================================================
# EWC is used to compute the EWC penaly for elastic weights training. 

#__init__ takes the current model and stores a deepcopy of it's state when initialised.
# This is done so we know how far the weights have drifted from thiswhen calculating the penalty
#it also takes a dataset of previous task examples in order to calculate the fischer information matrix
# =============================================================================
class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = Variable(input)
            input = input.view(-1,input.shape[0])
            output = self.model.forward(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss



