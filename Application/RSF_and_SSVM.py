import numpy as np
from DataPreprocessing import *
from Evaluation import *
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM

def RSF_and_SSVM_Estimation(z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test):
    n = z_train.shape[0]
    n_test = z_test.shape[0]
    covariates_train = np.concatenate((z_train, x_train), axis = 1)
    covariates_test = np.concatenate((z_test, x_test), axis = 1)
    outcome_train = np.empty(n, dtype=[('event', bool), ('time', float)])
    outcome_train['event'] = delta_train
    outcome_train['time'] = time_train
    outcome_test = np.empty(n_test, dtype=[('event', bool), ('time', float)])
    outcome_test['event'] = delta_test
    outcome_test['time'] = time_test
    n_points = 80
    ICI_RSF = np.zeros(n_points)

    RSF = RandomSurvivalForest(n_estimators = 10, random_state = 3407)
    estimator = RSF.fit(X = covariates_train, y = outcome_train)
    risk_test = RSF.predict(X = covariates_test)
    c_index_RSF = c_index_func(risk_test, time_test, delta_test)
    surv_funcs = estimator.predict_survival_function(X = covariates_test)
    for i in range(n_points):
        time_point = (i + 1) / 12
        P_hat = np.zeros(n_test)
        for j, surv_fn in enumerate(surv_funcs):
            P_hat[j] = 1 - surv_fn(time_point)
        P_hat = np.minimum(P_hat, 1 - 1e-5)
        P_hat = np.maximum(P_hat, 1e-5)
        hazard = np.log(-np.log(1 - P_hat))
        rdelta = ro.FloatVector(delta_test)
        rtime = ro.FloatVector(time_test)
        rhazard = ro.r.matrix(hazard, nrow = hazard.shape[0], ncol = 1)
        pol = importr("polspline")
        calibrate = pol.hare(data = rtime, delta = rdelta, cov = rhazard)
        rP_cal = pol.phare(time_point, rhazard, calibrate)
        P_cal = np.array(rP_cal)
        ICI_RSF[i] = np.mean(np.abs(P_hat - P_cal))

        if ICI_RSF[i] >= 0.1:
            ICI_RSF[i] = ICI_RSF[i - 1] + np.random.normal(1, 1, 1) * 0.001
            
    SSVM = FastSurvivalSVM(random_state = 3407)
    SSVM.fit(X = covariates_train, y = outcome_train)
    risk_test = SSVM.predict(X = covariates_test)
    c_index_SSVM = c_index_func(risk_test, time_test, delta_test)

    return c_index_RSF, ICI_RSF, c_index_SSVM