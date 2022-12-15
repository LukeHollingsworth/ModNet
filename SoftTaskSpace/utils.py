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

def train_IS_history(model_class, hyperparameters = None,  N_models=20, ):
    models = []
    IS_history = np.zeros((3,hyperparameters['epochs'],N_models))
    
    if model_class == 'simple_network':
        from networks import simple_network
        for n in tqdm(range(N_models), desc="Model"):
            fail_count = 0
            current_model_successful = False
            while current_model_successful == False:
                if fail_count >= 10:
                    print("\n This model doesn't train well, aborting")
                    return models
                model = simple_network(hyperparameters)
                IS_history[:,:,n] = model.train_model()
                if model.abs_error()[0]<0.05 and model.abs_error()[1]<0.05:
                    # model.get_RI()
                    model.get_IS()
                    models.append(model)
                    current_model_successful = True
                else:
                    fail_count += 1
        return models, IS_history

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

def plot_rulespace(rule1, rule2, data):
    plt.figure(figsize=(2,2))
    plt.scatter(data[:,0], data[:,1], color='gray', s=0.5)
    plt.arrow(rule1[0][0], rule1[0][1], (rule1[1][0]-rule1[0][0]), (rule1[1][1]-rule1[0][1]), color='b', length_includes_head=True, head_width = 0.01, label="Rule 1")
    plt.arrow(rule2[0][0], rule2[0][1], (rule2[1][0]-rule2[0][0]), (rule2[1][1]-rule2[0][1]), color='r', length_includes_head=True, head_width = 0.01, label="Rule 2")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.show

def plot_RI(models, show_threshold=False, title=None):  #only works for simple_network lists, not MNIST_networks
    
    if models[0].type_of_network == 'simple_network':
        RI = [[],[],[],[]]
        for model in models:
            for i in range(len(RI)):
                RI[i].extend(list(model.RI[i]))
        fig, axs = plt.subplots(1,3,sharey = True, figsize = (4,0.8))
        for i in range(3):
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
        for i in range(3):
            axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(RI[i])))/len(RI[i])))), fontdict = {'color':'grey', 'fontsize':4})
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return
        
def plot_full_RI(theta_models, title=None):
    if theta_models[0][0].type_of_network == 'simple_network':
        theta_RI = []
        delta_thetas = np.zeros(shape=(len(theta_models)))
        for theta in range(len(theta_models)):
            RI = [[],[],[],[]]
            for model in theta_models[theta]:
                for i in range(len(RI)):
                    RI[i].extend(list(model.RI[i]))
            delta_thetas[theta] = model.delta_theta
            theta_RI.append(RI[:-1])
        
        fig, axs = plt.subplots(int(len(delta_thetas)/5),5,sharey=True, figsize=(5,4))
        for i in range(int(len(delta_thetas)/5)):
            for j in range(5):
                n, bins, patches = axs[i,j].hist(theta_RI[5*i+j][2], weights=np.ones(len(theta_RI[5*i+j][2])) / len(theta_RI[5*i+j][2]),bins=np.linspace(-1,1,11))
                bin_centre = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
                col = (bin_centre - min(bin_centre))/(max(bin_centre) - min(bin_centre))
                cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C1','C0'], N=1000)
                for c, p in zip(col, patches):
                        plt.setp(p, 'facecolor', cm(c))
                plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                # axs[i,j].set_title("Hidden layer %g" %(i+1))
                axs[i,j].set_xlim([-1,1])
                # axs[i,j].set_xlabel(r'$\mathcal{RI}$')
        for i in range(int(len(delta_thetas)/5)):
            for j in range(5):
                axs[i,j].text(0.51,axs[i,j].get_ylim()[-1]*0.92, round(delta_thetas[5*i+j]), fontdict = {'color':'grey', 'fontsize':4})
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()

def plot_RI_variance(theta_models, variances, title=None):
    if theta_models[0][0].type_of_network == 'simple_network':
        delta_thetas = np.zeros(shape=(len(theta_models)))
        for theta in range(len(theta_models)):
            RI = [[],[],[],[]]
            for model in theta_models[theta]:
                for i in range(len(RI)):
                    RI[i].extend(list(model.RI[i]))
            delta_thetas[theta] = model.delta_theta
        
        layer_1_variances = [item[0] for item in variances]
        layer_2_variances = [item[1] for item in variances]
        layer_3_variances = [item[2] for item in variances]

        plt.plot(delta_thetas, layer_1_variances, color='orange', linestyle='--', marker='o')
        plt.plot(delta_thetas, layer_2_variances, color='springgreen', linestyle='--', marker='o')
        plt.plot(delta_thetas, layer_3_variances, color='darkviolet', linestyle='--', marker='o')
        plt.xlabel(r'$\Delta$ $\theta$ (degrees)')
        plt.ylabel(r'Variance ($\sigma^2$)')
        plt.show()


def plot_I(models, show_threshold=False, title=None):  #only works for simple_network lists, not MNIST_networks
    
    if models[0].type_of_network == 'simple_network':
        I1 = [[],[],[],[]]
        I2 = [[],[],[],[]]
        for model in models:
            for i in range(len(I1)):
                I1[i].extend(list(model.Itask1[i]))
            for i in range(len(I2)):
                I2[i].extend(list(model.Itask2[i]))
        fig, axs = plt.subplots(1,3,sharey = True, figsize = (4,0.8))
        for i in range(3):
            n, bins1, patches1 = axs[i].hist(I1[i], weights=np.ones(len(I1[i])) / len(I1[i]),bins=np.linspace(-1,1,11))
            n, bins2, patches2 = axs[i].hist(I2[i], weights=np.ones(len(I2[i])) / len(I2[i]),bins=np.linspace(-1,1,11))
            bin_centre1 = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins1[1:]),list(bins1[:-1]))]
            bin_centre2 = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins2[1:]),list(bins2[:-1]))]

            col1 = (bin_centre1 - min(bin_centre1))/(max(bin_centre1) - min(bin_centre1))
            col2 = (bin_centre2 - min(bin_centre2))/(max(bin_centre2) - min(bin_centre2))
            cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C1','C0'], N=1000)
            for c, p in zip(col1, patches1):
                    plt.setp(p, 'facecolor', cm(c))
            for c, p in zip(col2, patches2):
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
        for i in range(3):
            axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(I1[i])))/len(I1[i])))), fontdict = {'color':'grey', 'fontsize':4})
            axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(I2[i])))/len(I2[i])))), fontdict = {'color':'grey', 'fontsize':4})
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return
    
def plot_IS(models, show_threshold=False, title=None):    
    if models[0].type_of_network == 'simple_network':
        IS = np.zeros((np.shape(models[0].IS)))
        fig, ax = plt.subplots(figsize = (2,1.5))
        for i in range(len(models)):
            ax.plot(np.linspace(1,4,4), models[i].IS, alpha=0.2, lw=0.5)
            IS += models[i].IS
        IS = IS / len(models)
        ax.plot(np.linspace(1,4,4), IS, color='k', lw=1)
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return

def plot_IS_history(models, IS_history, hyperparameters=None, show_threshold=False, title=None):    
    if models[0].type_of_network == 'simple_network':
        fig, axs = plt.subplots(1,3,sharey = True, figsize = (4,0.8))
        for i in range(3):
            IS_avg = np.zeros((hyperparameters['epochs']))
            for j in range(len(models)):
                axs[i].plot(np.linspace(0, 9, hyperparameters['epochs']), IS_history[i,:,j], alpha=0.2, lw=0.5)
                IS_avg += IS_history[i,:,j]
            IS_avg = IS_avg / len(models)
            axs[i].plot(np.linspace(0, 9, hyperparameters['epochs']), IS_avg, color='k', lw=1)
            axs[i].set_title("Hidden layer %g" %(i+1))
            axs[i].set_xlabel(r'$\mathcal{IS}$')
        if title != None:
            fig.suptitle("%s" %title)
        plt.show()
        return

def theta_variation(model_class, hyperparameters=None, N_models=20):
    models = []
    coord_range = list(np.linspace(0.2, 1, 9))
    coord_range.insert(0, 0.001)
    coords = coord_range
    grads = [1/i for i in coord_range]
    coords.reverse()
    grads = list(np.concatenate((grads, coords)))
    grads.pop(9)
    
    if model_class == 'simple_network':
        from networks import simple_network
        for i in range(len(grads)):
            print('Gradient of r2 = ', grads[i])
            hyperparameters = {'N_train' : 1000, #size of training dataset 
                          'N_test' : 100, #size of test set x
                          'lr' : 0.001, #SGD learning rate 
                          'epochs' : 10, #training epochs
                          'batch_size' : 10,  #batch size (large will probably fail)           
                          'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
                          'train_mode' : 'random', #training mode 'random' vs 'replay' 
                          'second_task' : 'prod', #first task adds x+y, second task 'prod' = xy or 'add1.5' = x+1.5y
                          'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
                          'hidden_size' : 25, #hidden layer width
                          'rule1_grad' : 0,
                          'rule2_grad' : grads[i]}
            data = simple_network(hyperparameters).x1_test[:,:2]

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
                        # print("Gradient of Rule 1 = {}, Gradient of Rule 2 = {}.".format(model.A1, model.A2))
                        # print("Generated point = ({},{})".format(model.x1_test[0][0], model.x1_test[0,1]))
                        # print("X_1 = {}, Y_1 = {}".format(model.forward(model.x1_test)[0].T, model.y1_test[0].T))
                        # print("X_2 = {}, Y_2 = {}".format(model.forward(model.x2_test)[0].T, model.y2_test[0].T))
                    else:
                        fail_count += 1
            rule1, rule2 = model.rules()
            plot_rulespace(rule1, rule2, data)
            plot_RI(models)

def theta_sampling(model_class, hyperparameters=None, N_models=20, N_theta = 20):
    thetas = np.random.uniform(0, 360, N_theta)
    delta_thetas = np.random.uniform(-180, 180, N_theta)
    delta_thetas.sort()
    print(delta_thetas)
    theta_models = []
    RI_variances = []

    if model_class == 'simple_network':
        from networks import simple_network
        for i in tqdm(range(N_theta), desc="Theta"):
            models = []
            theta, delta_theta = thetas[i], delta_thetas[i]
            grad1 = np.tan(np.deg2rad(theta))
            grad2 = np.tan(np.deg2rad(theta + delta_theta))
            hyperparameters = {'N_train' : 1000, #size of training dataset 
                          'N_test' : 100, #size of test set x
                          'lr' : 0.001, #SGD learning rate 
                          'epochs' : 10, #training epochs
                          'batch_size' : 10,  #batch size (large will probably fail)           
                          'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
                          'train_mode' : 'random', #training mode 'random' vs 'replay' 
                          'second_task' : 'prod', #first task adds x+y, second task 'prod' = xy or 'add1.5' = x+1.5y
                          'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
                          'hidden_size' : 25, #hidden layer width
                          'rule1_grad' : grad1,
                          'rule2_grad' : grad2,
                          'delta_theta': delta_theta}
            data = simple_network(hyperparameters).x1_test[:,:2]
            RI_data = [[],[],[],[]]
            RI_variance = []
            IS_data = np.zeros((4, hyperparameters['epochs'], N_models))

            for n in range(N_models):
                fail_count = 0
                current_model_successful = False
                while current_model_successful == False:
                    if fail_count >= 10:
                        print("\n This model doesn't train well, aborting")
                        return models
                    model = simple_network(hyperparameters)
                    model.train_model()
                    IS_data[:,:,n] = model.train_model()
                    if model.abs_error()[0]<0.05 and model.abs_error()[1]<0.05:
                        model.get_RI()
                        models.append(model)
                        current_model_successful = True
                    else:
                        fail_count += 1

                    for i in range(len(RI_data)):
                        RI_data[i].extend(list(model.RI[i]))
            
            for i in range(len(RI_data)):
                cleaned_RI_data = [x for x in RI_data[i] if np.isnan(x) == False]
                RI_variance.append(np.var(cleaned_RI_data))
            
            RI_variances.append(list(RI_variance))
            theta_models.append(models)
            rule1, rule2 = model.rules()
        plot_full_RI(theta_models)
        plot_RI_variance(theta_models, RI_variances)
