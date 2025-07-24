import numpy as np
import math
import torch
from SimData import *
from NeuralNetwork import *
from itertools import product
from torch.utils.data import TensorDataset, DataLoader

def B_spline(i, k, node_list, t):
    if k == 0:
        result = 1 if node_list[i] <= t < node_list[i + 1] else 0
    elif k > 0:
        coef_1 = 0 if (node_list[i + k] == node_list[i]) else (t - node_list[i]) / (node_list[i + k] - node_list[i])
        coef_2 = 0 if (node_list[i + k + 1] == node_list[i + 1]) else (node_list[i + k + 1] - t) / (node_list[i + k + 1] - node_list[i + 1])
        result = coef_1 * B_spline(i, k - 1, node_list, t) + coef_2 * B_spline(i + 1, k - 1, node_list, t)
    return result
    
def spline_derivative(i, k, node_list, t):
    if k == 0:
        result = 0 
    elif k > 0:
        coef_1 = 0 if (node_list[i + k] == node_list[i]) else k / (node_list[i + k] - node_list[i])
        coef_2 = 0 if (node_list[i + k + 1] == node_list[i + 1]) else k / (node_list[i + k + 1] - node_list[i + 1])        
        result =  coef_1 *  B_spline(i, k - 1, node_list, t) - coef_2 * B_spline(i + 1, k - 1, node_list, t)
    return result

def model_selection(n, r, c, case):
    n_hidden_set = [1, 2, 3, 4, 5]
    n_neurons_set = [5, 10, 15, 20, 50]
    n_epochs_set = [100, 200, 500]
    learning_rate_set = [1e-3, 2e-3, 5e-3, 1e-2]
    p_dropout_set = [0, 0.1, 0.2, 0.3]
    n_splines_H_set = [i for i in range(math.floor(math.pow(n, 1/3 + 1e-7)), 2 * math.floor(math.pow(n, 1/3 + 1e-7)) + 1)]
    iter = product(n_hidden_set, n_neurons_set, n_epochs_set, learning_rate_set, p_dropout_set, n_splines_H_set)
   
    n_sim = 10
    tau = c
    batch_size = n // 100
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_decay = 1e-3
    max_likelihood = -1e7
    best_hyperparams = {}
    
    for hyperparams in iter:
        n_hidden = hyperparams[0]
        n_neurons = hyperparams[1]
        n_epochs = hyperparams[2]
        learning_rate = hyperparams[3]
        p_dropout = hyperparams[4]
        n_splines_H = hyperparams[5]
        Likelihood = 0

        for i in range(n_sim):
            z, x, gx, time, delta = DataGenerator(n = n, r = r, c = c, sim = i, set = 'train', case = case)

            node_list_H = np.zeros(n_splines_H + 4)
            node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
            node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
            t_splines = np.array([[B_spline(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])
            t_spline_derivatives = np.array([[spline_derivative(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])

            z = torch.from_numpy(z).to(torch.float32).to(device)
            x = torch.from_numpy(x).to(torch.float32).to(device)
            t_splines = torch.from_numpy(t_splines).to(torch.float32).to(device)
            t_spline_derivatives = torch.from_numpy(t_spline_derivatives).to(torch.float32).to(device)
            delta = torch.from_numpy(delta).to(torch.float32).to(device)

            n_train = int(n * 0.8)
            z_train, z_val = z[: n_train], z[n_train: ]
            x_train, x_val = x[: n_train], x[n_train: ]
            t_splines_train, t_splines_val = t_splines[: n_train], t_splines[n_train: ]
            t_spline_derivatives_train, t_spline_derivatives_val = t_spline_derivatives[: n_train], t_spline_derivatives[n_train: ]
            delta_train, delta_val = delta[: n_train], delta[n_train: ]
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(3407 * (i + 1))
            else:
                torch.manual_seed(3407 * (i + 1))

            train_data = TensorDataset(z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train)
            val_data = TensorDataset(z_val, x_val, t_splines_val, t_spline_derivatives_val, delta_val)
            train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
            val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

            model = DNN(z_dim = z.shape[1], x_dim = x.shape[1], n_splines_H = n_splines_H, n_hidden = n_hidden, n_neurons = n_neurons, p_dropout = p_dropout)
            model = model.to(device)
            loss_fn = LogLikelihood(r)
            loss_fn = loss_fn.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)                        

            train_loss = np.zeros(n_epochs)
            val_loss = np.zeros(n_epochs)
            early_stopping_flag = 0

            for epoch in range(n_epochs):
                model.train()
                train_loss_all = 0
                for z, x, t_splines, t_spline_derivatives, delta in train_loader:
                    beta, gamma_tilde, gx = model(x)
                    loss = loss_fn(beta, gamma_tilde, gx, z, t_splines, t_spline_derivatives, delta)                            
                    loss.backward()
                    train_loss_all += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss[epoch] = train_loss_all

                model.eval()
                val_loss_all = 0
                for z, x, t_splines, t_spline_derivatives, delta in val_loader:
                    beta, gamma_tilde, gx = model(x)
                    loss = loss_fn(beta, gamma_tilde, gx, z, t_splines, t_spline_derivatives, delta)
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

            beta, gamma_tilde, gx = model(x_val)
            Likelihood -= loss_fn(beta, gamma_tilde, gx, z_val, t_splines_val, t_spline_derivatives_val, delta_val).item()

        Likelihood /= n_sim
        if Likelihood > max_likelihood:
            max_likelihood = Likelihood
            best_hyperparams = {'n_hidden': n_hidden, 'n_neurons': n_neurons, 'n_epochs': n_epochs, 'learning_rate': learning_rate, 'p_dropout': p_dropout, 'n_splines_H': n_splines_H}
    return best_hyperparams