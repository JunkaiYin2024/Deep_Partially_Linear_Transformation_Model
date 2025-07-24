import numpy as np
from Algorithms import *
from Evaluation import *

def Linear_Estimation(r, z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test):
    n, z_dim = z_train.shape
    tau = np.maximum(np.max(time_train), np.max(time_test))
    n_splines_H = 20
    n_points = 80
    ICI = np.zeros(n_points)

    node_list_H = np.zeros(n_splines_H + 4)
    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
    t_splines_train = np.array([[B_spline(k, 3, node_list_H, time_train[j]) for k in range(n_splines_H)] for j in range(n)])
    t_spline_derivatives_train = np.array([[spline_derivative(k, 3, node_list_H, time_train[j]) for k in range(n_splines_H)] for j in range(n)])

    parameters = fit_model(r, z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train, n_iter = 10)
                
    beta = parameters[: z_dim]

    gamma_tilde = parameters[z_dim: z_dim + n_splines_H]
    gamma = np.zeros_like(gamma_tilde)
    gamma[0] = gamma_tilde[0]
    gamma[1: ] = np.exp(gamma_tilde[1: ])
    gamma = np.cumsum(gamma, axis = 0)

    g_coefs = parameters[z_dim + n_splines_H: ]
    gx = np.dot(x_train, g_coefs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_train = torch.from_numpy(z_train).to(torch.float32).to(device)
    x_train = torch.from_numpy(x_train).to(torch.float32).to(device)
    t_splines_train = torch.from_numpy(t_splines_train).to(torch.float32).to(device)
    t_spline_derivatives_train = torch.from_numpy(t_spline_derivatives_train).to(torch.float32).to(device)
    delta_train = torch.from_numpy(delta_train).to(torch.float32).to(device)
    beta = torch.from_numpy(beta).to(torch.float32).to(device)
    gx = torch.from_numpy(gx).to(torch.float32).to(device)
    gamma = torch.from_numpy(gamma).to(torch.float32).to(device)
    beta_se = Est_SE(r, z_train, x_train, t_splines_train, t_spline_derivatives_train, delta_train, beta, gamma, gx)

    beta = beta.detach().cpu().numpy()
    gamma = gamma.detach().cpu().numpy()

    gx_test = np.dot(x_test, g_coefs)
    risk_test = np.dot(z_test, beta) + gx_test
    c_index = c_index_func(risk_test, time_test, delta_test)
    
    for k in range(n_points):
        time_point = (k + 1) / 12
        ICI[k] = ICI_func(r, time_point, risk_test, gamma, tau, time_test, delta_test)
    
    return beta, beta_se, c_index, ICI