import numpy as np
from scipy.stats import norm

def DataGenerator(n, r, c, sim, set, case):
    if set == 'train':
        rng = np.random.RandomState(42 * (sim + 1))
    elif set == 'test':
        rng = np.random.RandomState(3407 * (sim + 1))
    
    beta_0 = np.array([1, -1])
    z = np.zeros((n, 2))
    z[:, 0] = rng.binomial(1, 0.5, n)
    z[:, 1] = rng.normal(0.5, 0.5, n)

    corr = (np.ones((5, 5)) + np.diag(np.ones(5))) / 2
    A = np.linalg.cholesky(corr)
    x = np.matmul(rng.normal(0, 1, n * 5).reshape((n, 5)), A.T)
    x = norm.cdf(x) * 2
    if case == 'Linear':
        gx = 0.25 * (x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + 4 * x[:, 3] + 5 * x[:, 4] - 15)
    elif case == 'Additive':
        gx = 2.5 * (np.sin(2 * x[:, 0]) + np.cos(x[:, 1] / 2) / 2 + np.log(x[:, 2] ** 2 + 1) / 3 + (x[:, 3] - x[:, 3] ** 3) / 4 + (np.exp(x[:, 4]) - 1) / 5 - 1.27)
    elif case == 'Deep':
        gx = 2.45 * (np.sin(2 * x[:, 0] * x[:, 1]) + np.cos(x[:, 1] * x[:, 2]/ 2) / 2 + np.log(x[:, 2] * x[:, 3] + 1) / 3 + (x[:, 3] - x[:, 2] * x[:, 3] * x[:, 4]) / 4 + (np.exp(x[:, 4]) - 1) / 5 - 1.16)

    temp = rng.rand(n)
    if r == 0: 
        dtime = -np.log(temp) * np.exp(- np.dot(z, beta_0) - gx)
    elif r == 0.5:
        dtime = 2 * np.log(np.exp(- np.dot(z, beta_0) - gx) * (np.sqrt(1 / temp) - 1) + 1)
    elif r == 1:
        dtime = np.log((1 / temp - 1) * np.exp(- np.dot(z, beta_0) - gx) + 1)
        
    ctime = c * rng.rand(n)
    time = np.minimum(ctime, dtime)
    delta = (dtime <= ctime)

    return z, x, gx, time, delta