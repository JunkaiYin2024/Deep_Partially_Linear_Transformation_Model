import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

def B_spline(i, k, node_list, t):
    if k == 0:
        result = 1 if node_list[i] <= t <= node_list[i + 1] else 0
    elif k > 0:
        coef_1 = 0 if (node_list[i + k] == node_list[i]) else (t - node_list[i]) / (node_list[i + k] - node_list[i])
        coef_2 = 0 if (node_list[i + k + 1] == node_list[i + 1]) else (node_list[i + k + 1] - t) / (node_list[i + k + 1] - node_list[i + 1])
        result = coef_1 * B_spline(i, k - 1, node_list, t) + coef_2 * B_spline(i + 1, k - 1, node_list, t)
    return result
    
def c_index_func(risk, time, delta):
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += delta[j] * (risk[j] >= risk[k]) * (time[j] <= time[k])
            denominator += delta[j] * (time[j] <= time[k])
    c_index = numerator / denominator
    return c_index

def ICI_func(r, t, risk, gamma, tau, time, delta): 
    n_splines_H = gamma.shape[0]
    node_list_H = np.zeros(n_splines_H + 4)
    node_list_H[n_splines_H + 1: ] = np.ones(3) * tau
    node_list_H[3: n_splines_H + 1] = np.linspace(0, tau, n_splines_H - 2)
    t_splines = np.array([B_spline(i, 3, node_list_H, t) for i in range(n_splines_H)])
    H_t = np.dot(t_splines, gamma)

    if r == 0:
        P_hat = 1 - np.exp(- np.exp(risk + H_t))
    elif r > 0:
        P_hat = 1 - np.exp(- np.log(1 + r * np.exp(risk + H_t)) / r)

    P_hat = np.minimum(P_hat, 1 - 1e-5)
    P_hat = np.maximum(P_hat, 1e-5)
    hazard = np.log(-np.log(1 - P_hat))

    rdelta = ro.FloatVector(delta)
    rtime = ro.FloatVector(time)
    rhazard = ro.r.matrix(hazard, nrow = hazard.shape[0], ncol = 1)
    
    pol = importr("polspline")
    calibrate = pol.hare(data = rtime, delta = rdelta, cov = rhazard)
    rP_cal = pol.phare(t, rhazard, calibrate)

    P_cal = np.array(rP_cal)
    ICI = np.mean(np.abs(P_hat - P_cal))

    return ICI

def ICI_func_Cox(t, risk, time, delta): 
    ascending_sort_time = np.argsort(time)
    descending_sort_time = np.flip(ascending_sort_time)
    risk_desc = risk[descending_sort_time]
    hazard = 1 / (np.cumsum(np.exp(risk_desc)))
    hazard = np.flip(hazard)
    
    time_asc = time[ascending_sort_time]
    delta_asc = delta[ascending_sort_time]
    NA_est = np.cumsum(delta_asc * hazard)
    n_samples = time.shape[0]
    index = np.searchsorted(time_asc, t)
    index = np.minimum(index, n_samples - 1)
    cum_hazard = NA_est[index]

    P_hat = 1 - np.exp(- cum_hazard * np.exp(risk))
    P_hat = np.minimum(P_hat, 1 - 1e-5)
    P_hat = np.maximum(P_hat, 1e-5)
    hazard = np.log(-np.log(1 - P_hat))

    rdelta = ro.FloatVector(delta)
    rtime = ro.FloatVector(time)
    rhazard = ro.r.matrix(hazard, nrow = hazard.shape[0], ncol = 1)
    
    pol = importr("polspline")
    calibrate = pol.hare(data = rtime, delta = rdelta, cov = rhazard)
    rP_cal = pol.phare(t, rhazard, calibrate)

    P_cal = np.array(rP_cal)
    ICI = np.mean(np.abs(P_hat - P_cal))

    return ICI