import matplotlib.pyplot as plt
from scipy.stats import norm
from DataPreprocessing import *
from Linear_model import *
from Additive_model import *
from Deep_model import *
from DPLCM import *
from RSF_and_SSVM import *

if __name__ == "__main__":
    r_set = [0, 0.5, 1]
    Likelihood_Deep = np.zeros(3)
    z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test = Dataset(split_ratio = 0.8, random_seed = 3407)
    z_dim = z_train.shape[1]
    beta_est = np.zeros((3, z_dim))
    beta_se = np.zeros((3, z_dim))
    c_index_set = np.zeros(3)
    n_points = 80
    ICI_set = np.zeros((3, n_points))

    for i in range(len(r_set)):
        r = r_set[i]
        beta, se, Likelihood, c_index, ICI = Deep_Estimation(r, z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test) 
        beta_est[i] = beta
        beta_se[i] = se
        Likelihood_Deep[i] = Likelihood
        c_index_set[i] = c_index
        ICI_set[i] = ICI
    
    print('r=0, Likelihood: {:.2f}\nr=0.5, Likelihood: {:.2f}\nr=1, Likelihood: {:.2f}'.format(Likelihood_Deep[0], Likelihood_Deep[1], Likelihood_Deep[2]))
    index = np.argmax(Likelihood_Deep)
    print('Best r: {}\n'.format(r_set[index]))

    z_value = beta_est / beta_se
    p_value = 2 * (1 - norm.cdf(np.abs(z_value)))
    print('Beta:', beta_est[index], '\nStandard error:', beta_se[index], '\nP_value', p_value[index], '\n')

    c_index_Deep = c_index_set[index]
    ICI_Deep = ICI_set[index]
    beta_Linear, se_Linear, c_index_Linear, ICI_Linear = Linear_Estimation(r_set[index], z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test)
    beta_Additive, se_Additive, c_index_Additive, ICI_Additive = Additive_Estimation(r_set[index], z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test)
    Likelihood_DPLCM, c_index_DPLCM, ICI_DPLCM = DPLCM_Estimation(z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test)
    c_index_RSF, ICI_RSF, c_index_SSVM = RSF_and_SSVM_Estimation(z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test)

    print('C_index:\nLinear model: {:.4f}\nAdditive model: {:.4f}\nDeep model: {:.4f}\nDPLCM: {:.4f}\nRandom Survival Forest: {:.4f}\nSurvival Support Vector Machine: {:.4f}'
            .format(c_index_Linear, c_index_Additive, c_index_Deep, c_index_DPLCM, c_index_RSF, c_index_SSVM))

    fig, ax1 = plt.subplots()
    time_points = np.arange(1, n_points + 1)
    ax1.plot(time_points, ICI_Deep, color = 'blue', label='DPLTM')
    ax1.plot(time_points, ICI_Linear, color = 'green', label='LTM')
    ax1.plot(time_points, ICI_Additive, color = 'orange', label='PLATM')
    ax1.plot(time_points, ICI_RSF, color = 'red', label='RSF')
    ax1.plot(time_points, ICI_DPLCM, color = 'purple', label='DPLCM')
    ax1.grid()
    ax1.set_xlabel('Follow-up Time (Month)')
    ax1.set_ylabel('ICI(t)')
    fig.legend(loc = 'upper right')
    plt.savefig('./ICI_comparison.pdf', dpi=300, bbox_inches='tight')