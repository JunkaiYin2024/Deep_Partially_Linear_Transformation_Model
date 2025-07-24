import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

def B_spline(i, k, node_list, t):
    if k == 0:
        result = 1 if node_list[i] <= t < node_list[i + 1] else 0
    elif k > 0:
        coef_1 = 0 if (node_list[i + k] == node_list[i]) else (t - node_list[i]) / (node_list[i + k] - node_list[i])
        coef_2 = 0 if (node_list[i + k + 1] == node_list[i + 1]) else (node_list[i + k + 1] - t) / (node_list[i + k + 1] - node_list[i + 1])
        result = coef_1 * B_spline(i, k - 1, node_list, t) + coef_2 * B_spline(i + 1, k - 1, node_list, t)
    return result
    
def WISE_func(r, gamma, tau, g_bar):
    n_splines_H = gamma.shape[0]
    node_list_H = np.zeros(n_splines_H + 4)
    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
    time_grid = np.arange(1e-3, tau, 1e-3)
    if r == 0:
        H0 = np.log(time_grid)
    elif r == 0.5:
        H0 = np.log(2 * np.exp(0.5 * time_grid) - 2)
    elif r == 1:
        H0 = np.log(np.exp(time_grid) - 1)
    
    n_times = time_grid.shape[0]
    time_grid_splines = np.array([[B_spline(j, 3, node_list_H, time_grid[i]) for j in range(n_splines_H)] for i in range(n_times)])
    H = np.dot(time_grid_splines, gamma) + g_bar
    SE = (H - H0) ** 2
    WISE = 1e-3 * np.sum(SE[: -1] + SE[1: ]) / (2 * tau)
    return WISE

def c_index_func(risk, time, delta):
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += delta[j] * (risk[j] >= risk[k]) * (time[j] <= time[k])
            denominator += delta[j] * (time[j] <= time[k])
    c_index = numerator / denominator
    return c_index
    
def ICI_func(r, q, risk, gamma, tau, time, delta):
    quantile = np.percentile(time, q)
    n_splines_H = gamma.shape[0]
    node_list_H = np.zeros(n_splines_H + 4)
    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
    quantile_splines = np.array([B_spline(i, 3, node_list_H, quantile) for i in range(n_splines_H)])
    H_quantile = np.dot(quantile_splines, gamma)

    if r == 0:
        P_hat = 1 - np.exp(- np.exp(risk + H_quantile))
    elif r > 0:
        P_hat = 1 - np.exp(- np.log(1 + r * np.exp(risk + H_quantile)) / r)

    P_hat = np.minimum(P_hat, 1 - 1e-5)
    P_hat = np.maximum(P_hat, 1e-5)
    hazard = np.log(-np.log(1 - P_hat))

    rdelta = ro.FloatVector(delta)
    rtime = ro.FloatVector(time)
    rhazard = ro.r.matrix(hazard, nrow = hazard.shape[0], ncol = 1)
    
    pol = importr("polspline")
    calibrate = pol.hare(data = rtime, delta = rdelta, cov = rhazard)
    rP_cal = pol.phare(quantile, rhazard, calibrate)

    P_cal = np.array(rP_cal)
    ICI = np.mean(np.abs(P_hat - P_cal))

    return ICI