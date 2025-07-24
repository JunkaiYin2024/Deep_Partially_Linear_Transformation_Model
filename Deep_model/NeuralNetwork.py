import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, z_dim, x_dim, n_splines_H, n_hidden, n_neurons, p_dropout):
        super(DNN, self).__init__()
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