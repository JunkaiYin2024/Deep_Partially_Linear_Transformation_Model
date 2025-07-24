import numpy as np
import torch
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from Algorithms import *

def DPLCM_selection(z, x, time, delta):
    n_hidden_set = [1, 2, 3, 4, 5]
    n_neurons_set = [5, 10, 15, 20, 50]
    n_epochs_set = [100, 200, 500]
    learning_rate_set = [1e-3, 2e-3, 5e-3, 1e-2]
    p_dropout_set = [0, 0.1, 0.2, 0.3]
    iter = product(n_hidden_set, n_neurons_set, n_epochs_set, learning_rate_set, p_dropout_set)
   
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_decay = 1e-3
    max_likelihood = -1e7
    best_hyperparams = {}

    z = torch.from_numpy(z).to(torch.float32).to(device)
    x = torch.from_numpy(x).to(torch.float32).to(device)
    time = torch.from_numpy(time).to(torch.float32).to(device)
    delta = torch.from_numpy(delta).to(torch.float32).to(device)

    n = z.shape[0]
    n_train = int(n * 0.8)
    z_train, z_val = z[: n_train], z[n_train: ]
    x_train, x_val = x[: n_train], x[n_train: ]
    time_train, time_val = time[: n_train], time[n_train: ]
    delta_train, delta_val = delta[: n_train], delta[n_train: ]
    
    for hyperparams in iter:
        n_hidden = hyperparams[0]
        n_neurons = hyperparams[1]
        n_epochs = hyperparams[2]
        learning_rate = hyperparams[3]
        p_dropout = hyperparams[4]
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(3407)
        else:
            torch.manual_seed(3407)

        train_data = TensorDataset(z_train, x_train, time_train, delta_train)
        val_data = TensorDataset(z_val, x_val, time_val, delta_val)
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

        model = DPLCM(z_dim = z_train.shape[1], x_dim = x_train.shape[1], n_hidden = n_hidden, n_neurons = n_neurons, p_dropout = p_dropout)
        model = model.to(device)
        loss_fn = PartialLikelihood()
        loss_fn = loss_fn.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)                        

        train_loss = np.zeros(n_epochs)
        val_loss = np.zeros(n_epochs)
        early_stopping_flag = 0

        for epoch in range(n_epochs):
            model.train()
            train_loss_all = 0
            for z, x, time, delta in train_loader:
                beta, gx = model(x)
                loss = loss_fn(beta, gx, z, time, delta)                            
                loss.backward()
                train_loss_all += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            train_loss[epoch] = train_loss_all

            model.eval()
            val_loss_all = 0
            for z, x, time, delta in val_loader:
                beta, gx = model(x)
                loss = loss_fn(beta, gx, z, time, delta)
                val_loss_all += loss.item()

            val_loss[epoch] = val_loss_all

            if epoch == 0:
                early_stopping_flag = 0
            else:
                if val_loss[epoch] <= val_loss[epoch - 1]:
                    early_stopping_flag = 0
                else:
                    early_stopping_flag += 1
                    if early_stopping_flag > patience:
                        break

        beta, gx = model(x_val)
        Likelihood = - loss_fn(beta, gx, z_val, time_val, delta_val).item()

        if Likelihood > max_likelihood:
            max_likelihood = Likelihood
            best_hyperparams = {'n_hidden': n_hidden, 'n_neurons': n_neurons, 'n_epochs': n_epochs, 'learning_rate': learning_rate, 'p_dropout': p_dropout}
    return best_hyperparams