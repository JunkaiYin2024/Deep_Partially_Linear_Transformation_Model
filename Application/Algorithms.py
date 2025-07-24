import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def natural_spline(i, k, node_list, t):
    if 0 <= i <= k - 1:
        result = np.power(t, i + 1)
    else:
        if node_list[0] <= t <= node_list[i - k + 1]:
            result = 0
        else:
            result = np.power(t - node_list[i - k + 1], k)
    return result

def B_spline(i, k, node_list, t):
    if k == 0:
        result = 1 if node_list[i] <= t <= node_list[i + 1] else 0
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

class DPLTM(nn.Module):
    def __init__(self, z_dim, x_dim, n_splines_H, n_hidden, n_neurons, p_dropout):
        super(DPLTM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(z_dim))
        self.gamma_tilde = nn.Parameter(-torch.ones(n_splines_H))

        layers = []
        layers.append(nn.Linear(x_dim, n_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p_dropout))
        for i in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(n_neurons, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        gx = torch.squeeze(self.model(x), dim = 1)
        return self.beta, self.gamma_tilde, gx

class LogLikelihood(nn.Module):
    def __init__(self, r):
        super(LogLikelihood, self).__init__()
        self.r = r 
    
    def forward(self, beta, gamma_tilde, gx, z, t_splines, t_spline_derivatives, delta):
        gamma = torch.zeros_like(gamma_tilde)
        gamma[0] = gamma_tilde[0]
        gamma[1: ] = torch.exp(gamma_tilde[1: ])
        gamma = torch.cumsum(gamma, dim = 0)

        H_t = torch.matmul(t_splines, gamma)
        H_t_derivative = torch.matmul(t_spline_derivatives, gamma)
        phi_t = H_t + torch.matmul(z, beta) + gx

        if self.r == 0:
            hazard = H_t_derivative * torch.exp(phi_t)
            cumhazard = torch.exp(phi_t)
        elif self.r > 0:
            hazard = H_t_derivative * torch.exp(phi_t) / (1 + self.r * torch.exp(phi_t))
            cumhazard = torch.log(1 + self.r * torch.exp(phi_t)) / self.r

        Log_Likelihood = delta * torch.log(hazard) - cumhazard
        return - Log_Likelihood.sum()

class DPLCM(nn.Module):
    def __init__(self, z_dim, x_dim, n_hidden, n_neurons, p_dropout):
        super(DPLCM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(z_dim))

        layers = []
        layers.append(nn.Linear(x_dim, n_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p_dropout))
        for i in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(n_neurons, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        gx = torch.squeeze(self.model(x), dim = 1)
        return self.beta, gx

class PartialLikelihood(nn.Module):
    def __init__(self):
        super(PartialLikelihood, self).__init__()
    
    def forward(self, beta, gx, z, time, delta):
        risk = torch.matmul(z, beta) + gx
        sort_time = torch.argsort(time, 0, descending = True)
        delta = torch.gather(delta, 0,  sort_time)
        risk = torch.gather(risk, 0, sort_time)
        exp_risk = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(exp_risk, 0))
        censored_likelihood = (risk - log_risk) * delta
        return - censored_likelihood.sum()

def loglikelihood(beta, gamma_tilde, g_coefs, r, z, x, t_splines, t_spline_derivatives, delta):
    gamma = np.zeros_like(gamma_tilde)
    gamma[0] = gamma_tilde[0]
    gamma[1: ] = np.exp(gamma_tilde[1: ])
    gamma = np.cumsum(gamma)
        
    H_t = np.dot(t_splines, gamma)
    H_t_derivative = np.dot(t_spline_derivatives, gamma)
    phi_t = H_t + np.dot(z, beta) + np.dot(x, g_coefs)

    if r == 0:
        hazard = H_t_derivative * np.exp(phi_t)
        cumhazard = np.exp(phi_t)
    elif r > 0:
        hazard = H_t_derivative * np.exp(phi_t) / (1 + r * np.exp(phi_t))
        cumhazard = np.log(1 + r * np.exp(phi_t)) / r
    hazard = np.maximum(hazard, 1e-10)
    Log_Likelihood = delta * np.log(hazard) - cumhazard

    return - Log_Likelihood.mean()

def fit_model(r, z, x, t_splines, t_spline_derivatives, delta, n_iter):
    z_dim = z.shape[1]
    x_dim = x.shape[1]
    n_splines_H = t_splines.shape[1]
    def loss_fn(parameters):
        beta = parameters[: z_dim]
        gamma_tilde = parameters[z_dim: z_dim + n_splines_H]
        g_coefs = parameters[z_dim + n_splines_H: ]
        return loglikelihood(beta, gamma_tilde, g_coefs, r, z, x, t_splines, t_spline_derivatives, delta)
    for i in range(n_iter):
        if i == 0:
            initial_values = np.zeros(z_dim + n_splines_H + x_dim)
            initial_values[z_dim: z_dim + n_splines_H] = -1
            result = minimize(fun = loss_fn, x0 = initial_values, method = 'SLSQP')
        else:
            result = minimize(fun = loss_fn, x0 = temp_result, method = 'SLSQP')
        temp_result = result['x']
    final_result = temp_result
    return final_result

class DNN_SE(nn.Module):
    def __init__(self, x_dim, n_hidden, n_neurons, p_dropout, n_splines_a):
        super(DNN_SE, self).__init__()
        self.theta = nn.Parameter(torch.zeros(n_splines_a))

        layers = []
        layers.append(nn.Linear(x_dim, n_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p_dropout))
        for i in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(n_neurons, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        b = torch.squeeze(self.model(x), dim = 1)
        return b, self.theta
    
class SE_loss(nn.Module):
    def __init__(self):
        super(SE_loss, self).__init__() 
    
    def forward(self, z, delta, Phi, H_t_derivatives, t_splines, t_spline_derivatives, b, theta):
        a = torch.matmul(t_splines, theta)
        a_derivatives = torch.matmul(t_spline_derivatives, theta)
        information = (z - a - b) * Phi - delta * a_derivatives / H_t_derivatives
        SE_loss = information ** 2      
        return SE_loss.sum() / z.shape[0]
    
def Est_SE(r, z, x, t_splines, t_spline_derivatives, delta, beta, gamma, gx):
    z_dim = z.shape[1]
    x_dim = x.shape[1]
    se = np.zeros(z_dim)
    learning_rate = 2e-3
    weight_decay = 1e-3
    batch_size = 128
    n_epochs = 100
    device = z.device

    H_t = torch.matmul(t_splines, gamma)
    H_t_derivatives = torch.matmul(t_spline_derivatives, gamma)
    phi_t = H_t + torch.matmul(z, beta) + gx

    if r == 0:
        hazard = torch.exp(phi_t)
        hazard_derivative = torch.exp(phi_t)
    elif r > 0:
        hazard = torch.exp(phi_t) / (1 + r * torch.exp(phi_t))
        hazard_derivative = torch.exp(phi_t) / (1 + r * torch.exp(phi_t)) ** 2

    Phi = delta * hazard_derivative / hazard - hazard

    for i in range(z_dim):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(3407 * (i + 1))
        else:
            torch.manual_seed(3407 * (i + 1))

        z0 = z[:, i]
        data = TensorDataset(z0, x, delta, Phi, H_t_derivatives, t_splines, t_spline_derivatives)
        loader = DataLoader(data, batch_size = batch_size, shuffle = True)
        model = DNN_SE(x_dim = x_dim, n_hidden = 2, n_neurons = 10, p_dropout = 0, n_splines_a = gamma.shape[0])
        model.to(device)
        loss_fn = SE_loss()
        loss_fn.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

        for epoch in range(n_epochs):
            model.train()
            for z_temp, x_temp, delta_temp, Phi_temp, H_t_derivatives_temp, t_splines_temp, t_spline_derivatives_temp in loader:
                b, theta = model(x_temp)
                loss = loss_fn(z_temp, delta_temp, Phi_temp, H_t_derivatives_temp, t_splines_temp, t_spline_derivatives_temp, b, theta)                            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        b, theta = model(x)
        loss = loss_fn(z0, delta, Phi, H_t_derivatives, t_splines, t_spline_derivatives, b, theta)
        loss = loss.cpu().detach().numpy()
        se[i] = 1 / np.sqrt(loss * z.shape[0])
    return se