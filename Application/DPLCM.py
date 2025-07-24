import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from Algorithms import *
from Evaluation import *
from ModelSelection_DPLCM import *

def DPLCM_Estimation(z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test):
    n, z_dim = z_train.shape
    n_test, x_dim = x_test.shape
    n_points = 80
    ICI = np.zeros(n_points)

    best_hyperparams = DPLCM_selection(z_train, x_train, time_train, delta_train)
    n_hidden = best_hyperparams['n_hidden']
    n_neurons = best_hyperparams['n_neurons']
    n_epochs = best_hyperparams['n_epochs']
    learning_rate = best_hyperparams['learning_rate']
    p_dropout = best_hyperparams['p_dropout']

    weight_decay = 1e-3
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z_train = torch.from_numpy(z_train).to(torch.float32).to(device)
    x_train = torch.from_numpy(x_train).to(torch.float32).to(device)
    time_train = torch.from_numpy(time_train).to(torch.float32).to(device)
    delta_train = torch.from_numpy(delta_train).to(torch.float32).to(device)

    n_train = int(n * 0.8)
    z_train, z_val = z_train[: n_train], z_train[n_train: ]
    x_train, x_val = x_train[: n_train], x_train[n_train: ]
    time_train, time_val = time_train[: n_train], time_train[n_train: ]
    delta_train, delta_val = delta_train[: n_train], delta_train[n_train: ]

    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    else:
        torch.manual_seed(3407)

    train_data = TensorDataset(z_train, x_train, time_train, delta_train)
    val_data = TensorDataset(z_val, x_val, time_val, delta_val)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

    model = DPLCM(z_dim = z_dim, x_dim = x_dim, n_hidden = n_hidden, n_neurons = n_neurons, p_dropout = p_dropout)
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
    Likelihood = loss_fn(beta, gx, z_val, time_val, delta_val).item()
    
    x_test = torch.from_numpy(x_test).to(torch.float32).to(device)
    beta, gx_test = model(x_test)
    beta = beta.detach().cpu().numpy()
    gx_test = gx_test.detach().cpu().numpy()

    risk_test = np.dot(z_test, beta) + gx_test
    c_index = c_index_func(risk_test, time_test, delta_test)

    for k in range(n_points):
        time_point = (k + 1) / 12
        ICI[k] = ICI_func_Cox(time_point, risk_test, time_test, delta_test)
        
    return Likelihood, c_index, ICI