import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from Algorithms import *
from Evaluation import *
from ModelSelection_DPLTM import *

def Deep_Estimation(r, z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test):
    n, z_dim = z_train.shape
    n_test, x_dim = x_test.shape
    tau = np.maximum(np.max(time_train), np.max(time_test))
    n_points = 80
    ICI = np.zeros(n_points)

    best_hyperparams = DPLTM_selection(r, tau, z_train, x_train, time_train, delta_train)
    n_hidden = best_hyperparams['n_hidden']
    n_neurons = best_hyperparams['n_neurons']
    n_epochs = best_hyperparams['n_epochs']
    learning_rate = best_hyperparams['learning_rate']
    p_dropout = best_hyperparams['p_dropout']
    n_splines_H = best_hyperparams['n_splines_H']

    node_list_H = np.zeros(n_splines_H + 4)
    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
    t_splines_train = np.array([[B_spline(k, 3, node_list_H, time_train[j]) for k in range(n_splines_H)] for j in range(n)])
    t_spline_derivatives_train = np.array([[spline_derivative(k, 3, node_list_H, time_train[j]) for k in range(n_splines_H)] for j in range(n)])

    weight_decay = 1e-3
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z_train = torch.from_numpy(z_train).to(torch.float32).to(device)
    x_train = torch.from_numpy(x_train).to(torch.float32).to(device)
    t_splines_train = torch.from_numpy(t_splines_train).to(torch.float32).to(device)
    t_spline_derivatives_train = torch.from_numpy(t_spline_derivatives_train).to(torch.float32).to(device)
    delta_train = torch.from_numpy(delta_train).to(torch.float32).to(device)

    n_train = int(n * 0.8)
    z_train, z_val = z_train[: n_train], z_train[n_train: ]
    x_train, x_val = x_train[: n_train], x_train[n_train: ]
    t_splines_train, t_splines_val = t_splines_train[: n_train], t_splines_train[n_train: ]
    t_spline_derivatives_train, t_spline_derivatives_val = t_spline_derivatives_train[: n_train], t_spline_derivatives_train[n_train: ]
    delta_train, delta_val = delta_train[: n_train], delta_train[n_train: ]

    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    else:
        torch.manual_seed(3407)
    
    train_data = TensorDataset(z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train)
    val_data = TensorDataset(z_val, x_val, t_splines_val, t_spline_derivatives_val, delta_val)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

    model = DPLTM(z_dim = z_dim, x_dim = x_dim, n_splines_H = n_splines_H, n_hidden = n_hidden, n_neurons = n_neurons, p_dropout = p_dropout)
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

    beta, gamma_tilde, gx = model(x_train)
    beta, gamma_tilde, gx = beta.detach(), gamma_tilde.detach(), gx.detach()
    gamma = torch.zeros_like(gamma_tilde)
    gamma[0] = gamma_tilde[0]
    gamma[1: ] = torch.exp(gamma_tilde[1: ])
    gamma = torch.cumsum(gamma, dim = 0)
    beta_se = Est_SE(r, z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train, beta, gamma, gx)

    beta, gamma_tilde, gx = model(x_val)
    Likelihood = -loss_fn(beta, gamma_tilde, gx, z_val, t_splines_val, t_spline_derivatives_val, delta_val).item()

    x_test = torch.from_numpy(x_test).to(torch.float32).to(device)
    beta, gamma_tilde, gx_test = model(x_test)
    beta = beta.detach().cpu().numpy()
    gamma = gamma.detach().cpu().numpy()
    gx_test = gx_test.detach().cpu().numpy()

    risk_test = np.dot(z_test, beta) + gx_test
    c_index = c_index_func(risk_test, time_test, delta_test)                    

    for k in range(n_points):
        time_point = (k + 1) / 12
        ICI[k] = ICI_func(r, time_point, risk_test, gamma, tau, time_test, delta_test)
    
    return beta, beta_se, Likelihood, c_index, ICI