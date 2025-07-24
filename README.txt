
This is a zip containing the codes and data for the 'Simulation Studies' and 'Real data application' sections in the paper 'Deep Partially Linear Transformation Model for Right-Censored Survival Data' by Junkai Yin, Yue Zhang and Zhangsheng Yu. The three folders ('Linear_model', 'Additive_model' and 'Deep_model') present the codes for implementing the LTM, PLATM and DPLTM methods in all simulation settings, where DPLTM is the DNN-based method proposed by us, while the folder 'Application' demonstrates the results for the SEER lung cancer dataset.


For simulation studies:

(1) 'Main_Linear_case.py', 'Main_Additive_case.py' and 'Main_Deep_case.py' are the codes to produce the results corresponding to the three cases of the underlying nonparametric functions $g_0(\bm{X})$.

(2) 'SimData.py' is the code for synthetic data generation under various settings.

(3) 'Evaluation.py' is the code that provides multiple functions used to calculate evaluation metrics, including the weighted integrated squared error (WISE), concordance index (C-index) and integrated calibration index (ICI).

(4) 'Est_SE.py' is the code for standard error estimation using the least favorable direction method.

(5) 'Algorithm.py' in the folders 'Linear_model' and 'Additive_model' is the code to fit the linear or additive model using the function scipy.optimize.minimize.

(6) 'NeuralNetwork.py' in the folder 'Deep_model' is the code containing the classes 'DNN' and 'LogLikelihood' which specify the configuration of the deep neural network and the exact form the of loss function, respectively.

(7) 'ModelSelection.py' in the folder 'Deep_model' is the code to select the best combination of hyperparameters under each simulation setting.


For the real data application:

(1) SEER_DATA.csv is the cleaned SEER lung cancer dataset. The raw dataset can be accessed via the instructions to request access to SEER's data products (available at https://seer.cancer.gov/data/access.html).

(2) 'Main.py' is the code to produce all the results.

(3) 'DataPreprocessing' is the code to preprocess the dataset, including data splitting, encoding categorial variables and data normalization.

(4) 'Linear_model.py', 'Additive_model.py', 'Deep_model.py', 'DPLCM.py' and 'RSF_and_SSVM.py' are the codes for the implementation of the methods LTM, PLATM, DPLTM, DPLCM, random survival forest and survival support vector machine, respectively. 

(5) 'Algorithms.py' is the code containing all the functions needed to perform the methods LTM, PLATM, DPLTM and DPLCM.

(6) 'Evaluation.py' is the codes use to calculate the evaluation metrics concordance index (C-index) and integrated calibration index (ICI).

(7) 'ModelSelection_DPLTM.py' and 'ModelSelection_DPLCM.py' are the codes to select the best combination of hyperparameters for the DNN-based methods DPLTM and DPLCM, respectively.