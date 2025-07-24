import numpy as np
from scipy.optimize import minimize

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

def LogLikelihood(beta, gamma_tilde, x_coefs, r, z, x, t_splines, t_spline_derivatives, delta):
    gamma = np.zeros_like(gamma_tilde)
    gamma[0] = gamma_tilde[0]
    gamma[1: ] = np.exp(gamma_tilde[1: ])
    gamma = np.cumsum(gamma)
        
    H_t = np.dot(t_splines, gamma)
    H_t_derivative = np.dot(t_spline_derivatives, gamma)
    phi_t = H_t + np.dot(z, beta) + np.dot(x, x_coefs)

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
        x_coefs = parameters[z_dim + n_splines_H: ]
        return LogLikelihood(beta, gamma_tilde, x_coefs, r, z, x, t_splines, t_spline_derivatives, delta)
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