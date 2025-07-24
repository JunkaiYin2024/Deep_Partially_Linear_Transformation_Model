import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from SimData import *
from NeuralNetwork import *
from Evaluation import *
from Est_SE import *
from ModelSelection import *
    
if __name__ == "__main__":
    n_set = [1000, 2000]
    r_set = [0, 0.5, 1]
    c_rate = [0.4, 0.6]
    c_set = [2.95, 0.85, 2.75, 0.9, 2.55, 1]
    n_sim = 200
    for i1 in range(len(r_set)): 
        for i2 in range(len(c_rate)):
            for i3 in range(len(n_set)):
                c = c_set[2 * i1 + i2]
                tau = c
                r = r_set[i1]
                n = n_set[i3]
                n_test = n // 5

                best_hyperparams = model_selection(n = n, r = r, c = c, case = 'Linear')
                n_hidden = best_hyperparams['n_hidden']
                n_neurons = best_hyperparams['n_neurons']
                n_epochs = best_hyperparams['n_epochs']
                learning_rate = best_hyperparams['learning_rate']
                p_dropout = best_hyperparams['p_dropout']
                n_splines_H = best_hyperparams['n_splines_H']

                beta_est = np.zeros((n_sim, 2))                
                beta_se = np.zeros((n_sim, 2))
                WISE = np.zeros(n_sim)
                RE = np.zeros(n_sim)
                c_index = np.zeros(n_sim)
                ICI_25 = np.zeros(n_sim)
                ICI_50 = np.zeros(n_sim)
                ICI_75 = np.zeros(n_sim)

                weight_decay = 1e-3
                batch_size = n // 100
                patience = 7
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               
                for i in range(n_sim):
                    z, x, gx_true, time, delta = DataGenerator(n = n, r = r, c = c, sim = i, set = 'train', case = 'Linear')

                    node_list_H = np.zeros(n_splines_H + 4)
                    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
                    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
                    t_splines = np.array([[B_spline(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])
                    t_spline_derivatives = np.array([[spline_derivative(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])

                    z = torch.from_numpy(z).to(torch.float32).to(device)
                    x = torch.from_numpy(x).to(torch.float32).to(device)
                    time = torch.from_numpy(time).to(torch.float32).to(device)
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
                                
                    beta, gamma_tilde, gx = model(x_train)
                    beta, gamma_tilde, gx = beta.detach(), gamma_tilde.detach(), gx.detach()

                    gamma = torch.zeros_like(gamma_tilde)
                    gamma[0] = gamma_tilde[0]
                    gamma[1: ] = torch.exp(gamma_tilde[1: ])
                    gamma = torch.cumsum(gamma, dim = 0)

                    beta_se[i] = Est_SE(r, z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train, beta, gamma, gx)
                    beta_est[i] = beta.cpu().numpy()

                    g_bar = gx.cpu().numpy().mean()
                    gamma = gamma.cpu().numpy()
                    WISE[i] = WISE_func(r, gamma, tau, g_bar)

                    z_test, x_test, gx_true_test, time_test, delta_test = DataGenerator(n = n_test, r = r, c = c, sim = i, set = 'test', case = 'Linear')
                    x_test = torch.from_numpy(x_test).to(torch.float32).to(device)
                    beta, gamma_tilde, gx_test = model(x_test)
                    x_test = x_test.cpu().detach().numpy()
                    beta = beta.cpu().detach().numpy()
                    gamma_tilde = gamma_tilde.cpu().detach().numpy()
                    gx_test = gx_test.cpu().detach().numpy()
                    RE[i] = np.sqrt(np.sum((gx_test - np.mean(gx_test) - gx_true_test) ** 2) / np.sum(gx_true_test ** 2))

                    risk = np.dot(z_test, beta) + gx_test
                    c_index[i] = c_index_func(risk, time_test, delta_test)
                    ICI_25[i] = ICI_func(r, 25, risk, gamma, tau, time_test, delta_test)
                    ICI_50[i] = ICI_func(r, 50, risk, gamma, tau, time_test, delta_test)
                    ICI_75[i] = ICI_func(r, 75, risk, gamma, tau, time_test, delta_test)

                est_mean = beta_est.mean(0)
                est_se = beta_est.std(0)
                se_mean = beta_se.mean(0)

                b1 = beta_est + 1.96 * beta_se
                b2 = beta_est - 1.96 * beta_se

                cov_p1 = (((b1[:, 0] >= 1) * (b2[:, 0] <= 1)).sum()) / n_sim
                cov_p2 = (((b1[:, 1] >= -1) * (b2[:, 1] <= -1)).sum()) / n_sim

                print("n: {}, r: {}, censoring_rate: {}, c: {}".format(n, r, c_rate[i2], c))
                print("Bias: beta_1: {:.4f}, beta_2:{:.4f}\nSSE: beta_1:{:.4f}, beta_2:{:.4f}\nESE: beta_1:{:.4f}, beta_2:{:.4f}\nCP: beta_1:{}, beta_2:{}" \
                    .format(est_mean[0] - 1, est_mean[1] + 1, est_se[0], est_se[1], se_mean[0], se_mean[1], cov_p1, cov_p2))
                print("WISE: mean: {:.4f}, std: {:.4f}".format(WISE.mean(), WISE.std()))
                print("RE: mean: {:.4f}, std: {:.4f}".format(RE.mean(), RE.std()))
                print("c_index: mean: {:.4f}, std: {:.4f}" .format(c_index.mean(), c_index.std()))
                print("ICI at t_25: mean: {:.4f}, std: {:.4f}" .format(ICI_25[~np.isnan(ICI_25)].mean(), ICI_25[~np.isnan(ICI_25)].std()))
                print("ICI at t_50: mean: {:.4f}, std: {:.4f}" .format(ICI_50[~np.isnan(ICI_50)].mean(), ICI_50[~np.isnan(ICI_50)].std()))
                print("ICI at t_75: mean: {:.4f}, std: {:.4f}" .format(ICI_75[~np.isnan(ICI_75)].mean(), ICI_75[~np.isnan(ICI_75)].std()))