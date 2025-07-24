import math
import numpy as np
import pandas as pd

def Dataset(split_ratio, random_seed):
    data = pd.read_csv('./SEER_DATA.csv', sep=",")
    data['Vital status'] = np.where(data['Vital status'] == 'Dead', 1, 0)
    data['Sex'] = np.where(data['Sex'] == 'Male', 1, 0)
    data['Marital status at diagnosis'] = np.where(data['Marital status at diagnosis'] == 'Married (including common law)', 1, 0)
    data['Primary'] = np.where(data['Primary'] == 'Yes', 1, 0)
    data['Separate Tumor Nodules Ipsilateral Lung'] = np.where(data['Separate Tumor Nodules Ipsilateral Lung'] == 'None; No intrapulmonary mets; Foci in situ/minimally invasive adenocarcinoma', 0, 1)
    data['Chemotherapy'] = np.where(data['Chemotherapy'] == 'Yes', 1, 0)
    data['Age'] = data['Age'].str.replace(r'\s*years', '', regex = True).astype(np.float64)

    data = np.array(data, dtype = np.float64)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)

    sample_size = data.shape[0]
    split_index = math.ceil(split_ratio * sample_size)

    time_train = data[: split_index, 0] / 12
    time_test = data[split_index: , 0] / 12

    delta_train = data[: split_index, 1]
    delta_test = data[split_index: , 1]

    z_train = data[: split_index, 2: 7]
    z_test = data[split_index: , 2: 7]

    x_train = data[: split_index, 7: ]
    x_test = data[split_index: , 7: ]
    x_train = (x_train - x_train.min(0)) / (x_train.max(0) - x_train.min(0))
    x_test = (x_test - x_test.min(0)) / (x_test.max(0) - x_test.min(0))

    return z_train, z_test, x_train, x_test, time_train, time_test, delta_train, delta_test