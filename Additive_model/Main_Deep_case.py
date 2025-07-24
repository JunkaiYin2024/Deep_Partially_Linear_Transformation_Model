import numpy as np
import torch
from SimData import *
from Evaluation import *
from Algorithm import *
from Est_SE import *

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
                n_splines_H = 20
                n_splines_g = 20
                
                beta_est = np.zeros((n_sim, 2))
                beta_se = np.zeros((n_sim, 2))
                WISE = np.zeros(n_sim)
                RE = np.zeros(n_sim)
                c_index = np.zeros(n_sim)
                ICI_25 = np.zeros(n_sim)
                ICI_50 = np.zeros(n_sim)
                ICI_75 = np.zeros(n_sim)

                for i in range(n_sim):
                    z, x, gx_true, time, delta = DataGenerator(n = n, r = r, c = c, sim = i, set = 'train', case = 'Deep')

                    node_list_H = np.zeros(n_splines_H + 4)
                    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
                    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
                    t_splines = np.array([[B_spline(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])
                    t_spline_derivatives = np.array([[spline_derivative(k, 3, node_list_H, time[j]) for k in range(n_splines_H)] for j in range(n)])
        
                    x_dim = x.shape[1]
                    node_list_g = np.linspace(0, 2, n_splines_g - 1)
                    x_splines = np.array([[natural_spline(m, 3, node_list_g, x[k, l]) for l in range(x_dim) for m in range(n_splines_g)] for k in range(n)])

                    parameters = fit_model(r, z, x_splines, t_splines, t_spline_derivatives, delta, n_iter = 100)
                    
                    beta = parameters[: z.shape[1]]
                    beta_est[i] = beta

                    gamma_tilde = parameters[z.shape[1]: z.shape[1] + n_splines_H]
                    gamma = np.zeros_like(gamma_tilde)
                    gamma[0] = gamma_tilde[0]
                    gamma[1: ] = np.exp(gamma_tilde[1: ])
                    gamma = np.cumsum(gamma, axis = 0)

                    spline_coefs = parameters[z.shape[1] + n_splines_H: ]
                    gx = np.dot(x_splines, spline_coefs)
                    g_bar = np.mean(gx)
                    WISE[i] = WISE_func(r, gamma, tau, g_bar)

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    z = torch.from_numpy(z).to(torch.float32).to(device)
                    x = torch.from_numpy(x).to(torch.float32).to(device)
                    t_splines = torch.from_numpy(t_splines).to(torch.float32).to(device)
                    t_spline_derivatives = torch.from_numpy(t_spline_derivatives).to(torch.float32).to(device)
                    delta = torch.from_numpy(delta).to(torch.float32).to(device)
                    beta = torch.from_numpy(beta).to(torch.float32).to(device)
                    gx = torch.from_numpy(gx).to(torch.float32).to(device)
                    gamma = torch.from_numpy(gamma).to(torch.float32).to(device)
                    beta_se[i] = Est_SE(r, z, x, t_splines, t_spline_derivatives, delta, beta, gamma, gx)

                    z_test, x_test, gx_true_test, time_test, delta_test = DataGenerator(n = n_test, r = r, c = c, sim = i, set = 'test', case = 'Deep')
                    x_splines_test = np.array([[natural_spline(m, 3, node_list_g, x_test[k, l]) for l in range(x_dim) for m in range(n_splines_g)] for k in range(n_test)])
                    gx_test = np.dot(x_splines_test, spline_coefs)

                    RE[i] = np.sqrt(np.sum((gx_test - np.mean(gx_test) - gx_true_test) ** 2) / np.sum(gx_true_test ** 2))
                    risk_test = np.dot(z_test, beta) + gx_test
                    c_index[i] = c_index_func(risk_test, time_test, delta_test)
                    ICI_25[i] = ICI_func(r, 25, risk_test, gamma, tau, time_test, delta_test)
                    ICI_50[i] = ICI_func(r, 50, risk_test, gamma, tau, time_test, delta_test)
                    ICI_75[i] = ICI_func(r, 75, risk_test, gamma, tau, time_test, delta_test)

                est_mean = beta_est.mean(axis = 0)
                est_se = beta_est.std(axis = 0)
                se_mean = beta_se.mean(axis = 0)

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