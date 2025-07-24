import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
    batch_size = 100
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