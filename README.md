This is a zip containing the codes and data for the 'Simulation Studies' and 'Real data application' sections in the paper 'Deep Partially Linear Transformation Model for Right-Censored Survival Data' by Junkai Yin, Yue Zhang and Zhangsheng Yu. The three folders 'Linear\_model', 'Additive\_model' and 'Deep\_model' present the codes for implementing the LTM, PLATM and DPLTM methods in all simulation settings, where DPLTM is the DNN-based method proposed by us, while the folder 'Application' demonstrates the results for the SEER lung cancer dataset.



For simulation studies:

(1) 'Main\_Linear\_case.py', 'Main\_Additive\_case.py' and 'Main\_Deep\_case.py' are the codes to produce the results corresponding to the three cases of the underlying nonparametric functions $g\_0(\\bm{X})$.

(2) 'SimData.py' is the code for synthetic data generation under various settings.

(3) 'Evaluation.py' is the code that provides multiple functions used to calculate evaluation metrics, including the weighted integrated squared error (WISE), concordance index (C-index) and integrated calibration index (ICI).

(4) 'Est\_SE.py' is the code for standard error estimation using the least favorable direction method.

(5) 'Algorithm.py' in the folders 'Linear\_model' and 'Additive\_model' is the code to fit the linear or additive model using the function scipy.optimize.minimize.

(6) 'NeuralNetwork.py' in the folder 'Deep\_model' is the code containing the classes 'DNN' and 'LogLikelihood' which specify the configuration of the deep neural network and the exact form the of loss function, respectively.

(7) 'ModelSelection.py' in the folder 'Deep\_model' is the code to select the best combination of hyperparameters under each simulation setting.



For the real data application:

(1) SEER\_DATA.csv is the cleaned SEER lung cancer dataset. The raw dataset can be accessed via the instructions to request access to SEER's data products (available at https://seer.cancer.gov/data/access.html).

(2) 'Main.py' is the code to produce all the results.

(3) 'DataPreprocessing' is the code to preprocess the dataset, including data splitting, encoding categorial variables and data normalization.

(4) 'Linear\_model.py', 'Additive\_model.py', 'Deep\_model.py', 'DPLCM.py' and 'RSF\_and\_SSVM.py' are the codes for the implementation of the methods LTM, PLATM, DPLTM, DPLCM, random survival forest and survival support vector machine, respectively.

(5) 'Algorithms.py' is the code containing all the functions needed to perform the methods LTM, PLATM, DPLTM and DPLCM.

(6) 'Evaluation.py' is the codes use to calculate the evaluation metrics concordance index (C-index) and integrated calibration index (ICI).

(7) 'ModelSelection\_DPLTM.py' and 'ModelSelection\_DPLCM.py' are the codes to select the best combination of hyperparameters for the DNN-based methods DPLTM and DPLCM, respectively.

