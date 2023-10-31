from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
import numpy as np
import os
import random
#We used 35813 (part of the Fibonacci Sequence) as the seed
seed = 35813
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def create_better_simulated(N_Subjects, N_ROIs):
    """
        Simulated dataset distributions are inspired from real measurements
        so this function creates better dataset for demo.
        However, number of views are hardcoded.
    """
    features = np.triu_indices(N_ROIs)[0].shape[0]
    view1 = np.random.normal(0.08, 0.067, (N_Subjects, features))
    view1 = view1.clip(min=0)
    view1 = np.array([antiVectorize(v, N_ROIs) for v in view1])

    view2 = np.random.normal(0.01, 0.006, (N_Subjects, features))
    view2 = view2.clip(min=0)
    view2 = np.array([antiVectorize(v, N_ROIs) for v in view2])

    view3 = np.random.normal(0.9, 1, (N_Subjects, features))
    view3 = view3.clip(min=0)
    view3 = np.array([antiVectorize(v, N_ROIs) for v in view3])

    view4 = np.random.normal(0.17, 0.27, (N_Subjects, features))
    view4 = view4.clip(min=0)
    view4 = np.array([antiVectorize(v, N_ROIs) for v in view4])

    view5 = np.random.normal(0.25, 0.2, (N_Subjects, features))
    view5 = view5.clip(min=0)
    view5 = np.array([antiVectorize(v, N_ROIs) for v in view5])

    view6 = np.random.normal(0.02, 0.016, (N_Subjects, features))
    view6 = view6.clip(min=0)
    view6 = np.array([antiVectorize(v, N_ROIs) for v in view6])

    return np.stack((view1, view2, view3, view4, view5, view6), axis=3)


def simulate_dataset(N_Subjects, N_ROIs, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_ROIs: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_ROIs, N_ROIs, N_views]
    """
    features = np.triu_indices(N_ROIs)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1, 2, (N_Subjects, features))

        view = np.array([antiVectorize(v, N_ROIs) for v in view])
        views.append(view)
    return np.stack(views, axis=3)


def rebuild_influential_training_dataset(fold_indices, el_taken, train_casted):
    influential_training_casted = []
    for x in fold_indices[:el_taken]:
        for y in range(len(train_casted)):
            if x == train_casted[y].ID:
                influential_training_casted.append(train_casted[y])
    return influential_training_casted


#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

#Antivectorize given vector (this gives an asymmetric adjacency matrix)
#def antiVectorize(vec, m):
#    M = np.zeros((m,m))
#     M[np.triu_indices(m)] = vec
#     M[np.tril_indices(m)] = vec
#     M[np.diag_indices(m)] = 0
#     return M

#Antivectorize given vector (this gives a symmetric adjacency matrix)
def antiVectorize(vec, m):
    M = np.zeros((m,m))
    t = 0
    for i  in range(0,m - 1):
        for j in range(i+1, m):
            M[i,j] = vec[t]
            M[j,i] = vec[t]
            t = t + 1
    return M

def Vectorize(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

#CV splits and mean-std calculation for the loss function

def preprocess_data_array(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds, random_state=seed, shuffle=True)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    train_channel_means = np.mean(X_train, axis=(0,1,2))
    train_channel_std =   np.std(X_train, axis=(0,1,2))
    return X_train, X_test, train_channel_means, train_channel_std


def preprocess_data_list(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds, random_state=seed, shuffle=True)
    split_indices = kf.split(range(len(X)))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = [X[x] for x in train_indices]
    X_test = [X[x] for x in test_indices]
    return X_train, X_test


def preprocess_casted_data(train_casted, test_casted):
    train_data = []
    for it in train_casted:
        train_data.append(it.con_mat.cpu().numpy())

    test_data = []
    for it in test_casted:
        test_data.append(it.con_mat.cpu().numpy())

    train_mean = np.mean(train_data, axis=(0, 1, 2))
    train_std = np.std(train_data, axis=(0, 1, 2))

    return train_data, test_data, train_mean, train_std


def update_score_dictionary(data_dict, data, score_excl):
    flag = 0
    for key in list(data_dict):
        if data.ID == data_dict[str(key)][0].ID:
            data_dict[str(key)][1] += score_excl
            flag = 1
    if flag != 1:
        data_dict['Data' + '_' + str(data.ID)] = [data, score_excl, data.ID]


#Create data objects for the DGN
#https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
def cast_data(array_of_tensors, subject_type = None, flat_mask = None):
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    dataset = []
    for idx, mat in enumerate(array_of_tensors):
            #Allocate numpy arrays 
            edge_index = np.zeros((2, N_ROI * N_ROI))
            edge_attr = np.zeros((N_ROI * N_ROI,CHANNELS))
            x = np.zeros((N_ROI, 1))
            y = np.zeros((1,))
            
            counter = 0
            for i in range(N_ROI):
                for j in range(N_ROI):
                    edge_index[:, counter] = [i, j]
                    edge_attr[counter, :] = mat[i, j]
                    counter += 1
    
            #Fill node feature matrix (no features every node is 1)
            for i in range(N_ROI):
                x[i,0] = 1
                
            #Get graph labels
            y[0] = None
            
            if flat_mask is not None:
                edge_index_masked = []
                edge_attr_masked = []
                for i,val in enumerate(flat_mask):
                    if val == 1:
                        edge_index_masked.append(edge_index[:,i])
                        edge_attr_masked.append(edge_attr[i,:])
                edge_index = np.array(edge_index_masked).T
                edge_attr = edge_attr_masked

            edge_index = torch.tensor(edge_index, dtype = torch.long)
            edge_attr = torch.tensor(edge_attr, dtype = torch.float)
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            con_mat = torch.tensor(mat, dtype=torch.float)
            data = Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = subject_type, ID = idx)
            dataset.append(data)
    return dataset

