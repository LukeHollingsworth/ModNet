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

def train_multiple(model_class, hyperparameters = None,  N_models=20, ):
    models = []    
    
    if model_class == 'simple_network':
        from networks import simple_network
        for _ in tqdm(range(N_models), desc="Model"):
            fail_count = 0
            current_model_successful = False
            while current_model_successful == False:
                if fail_count >= 10:
                    print("\n This model doesn't train well, aborting")
                    return models
                model = simple_network(hyperparameters)
                model.train_model()
                if model.abs_error()[0]<0.05 and model.abs_error()[1]<0.05:
                    model.get_RI()
                    models.append(model)
                    current_model_successful = True
                else:
                    fail_count += 1
        return models

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
        ax.set_ylabel('Absolute test error')
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

def plot_rulespace(rule1, rule2):
    origin = [0,0]

    plt.figure(figsize=(2,2))
    plt.arrow(origin[0],  origin[1], rule1[0], rule1[1], color='b', length_includes_head=True, head_width = 0.01, label="Rule 1")
    plt.arrow(origin[0], origin[1], rule2[0], rule2[1], color='r', length_includes_head=True, head_width = 0.01, label="Rule 2")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show

def plot_RI(models, show_threshold=False, title=None):  #only works for simple_network lists, not MNIST_networks
    
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