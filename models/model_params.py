import numpy as np
## Configure range of estimaters

'''
    Random Forest:
        n_estimators = number of trees in the foreset
        max_features = max number of features considered for splitting a node
        max_depth = max number of levels in each decision tree
        min_samples_split = min number of data points placed in a node before the node is split
        min_samples_leaf = min number of data points allowed in a leaf node
        bootstrap = method for sampling data points (with or without replacement)
'''
n_estimators = [int(x) for x in np.linspace(start=1,stop=2000, num=10)]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
class_weight = ["balanced", "balanced_subsample", None]
bootstrap = [True, False]
rfc_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'class_weight' : class_weight,
               'bootstrap': bootstrap}

'''
    {'n_estimators': 1333, 
    'min_samples_split': 5, 
    'min_samples_leaf': 1, 
    'max_features': 'sqrt', 
    'max_depth': 50, 
    'class_weight': 
    'balanced_subsample', 
    'bootstrap': True}

'''
narrowed_rfc_grid =  {
    'n_estimators': [800, 1000, 1200, 1600, 2000], 
    'min_samples_split': [4, 5, 7, 9], 
    'min_samples_leaf': [1,2,3], 
    'max_features': ['auto', 'sqrt'], 
    'max_depth': [40, 50, 60, 70], 'class_weight': ['balanced',"balanced_subsample"], 
    'bootstrap': [True]}

'''
    Support Vector Machine:
        C : Regularization parameter. The strength of the regularization is inversely proportional to C. penalty is a squared l2 penalty
        kernel : Specifies the kernel type to be used in the algorithm. 
        gamma: {‘scale’, ‘auto’} or float, default=’scale’. Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            kernel='precomputed'can only be used when passing a 
            (n_samples, n_samples) data matrix that represents 
            pairwise similarities for the samples instead of the 
            traditional (n_samples, n_features) rectangular data matrix.
'''

svm_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001,'scale', 'auto']
}

narrowd_svm_grid = {
    'C': list(map(int, np.linspace(start=2,stop=20, num=10))),
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [*np.linspace(start=0.1,stop=10, num=10), 'scale', 'auto']
}

'''
    random state: 0, 1, 42
'''
tune_search_params = {
    'cv': 5,
    'verbose': 2,
    # 'random_state': 1,
    'n_jobs': -1,
    # 'n_iter': 30
}