import pandas as pd 
import numpy as np
import sys
# from os import listdir
# from os.path import isfile, 
from os.path import join
import os
from datetime import datetime

# for cleaning discharge summaries

# for label models
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from sklearn import metrics

# for labeling functions

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

sys.path.insert(1, '../utils')
from tools import append_csv_bydf, create_folder
from utils import *

sys.path.insert(1, '../models')
from pretrained_label_functions import get_model_lfs

sys.path.insert(1, '../dataprocess')
# from pretrained_label_functions import *
from prepare_dataset import prepare_data_for_model

from arg_parser import get_lfdict_and_parser



def get_selected_lf_list():
    index_lfname_dict, parser = get_lfdict_and_parser()
    args = parser.parse_args()
    index_list = list(map(lambda s: s.strip(),args.label_function.split(',')))
    lfname_list = [index_lfname_dict[id] for id in index_list]

    lfname_lf_dict = get_model_lfs()
    print('selected label functions:')
    for index, lfname in zip(index_list, lfname_list):
        print(f'--{index}: {lfname}')
    lf_list = [lfname_lf_dict[lfname] for lfname in lfname_list]
    return lf_list






def main():

    lf_list = get_selected_lf_list()
    quit()
    # TODO: SAVE TRAINING LOG

    # trainData, testData = prepare_data_for_model(True) # get data for training models


    '''
        LF Applier
    '''   
    applier = PandasLFApplier(lfs=lfs) 
    L_train = applier.apply(df=trainData)
    print("\n")
    print("Labeling Function Analysis on train dataset")
    print(f"{LFAnalysis(L_train, lfs).lf_summary()}")
    L_test = applier.apply(df=testData)

    num_classes = 2
    #####################
    ###  Label Models ###
    #####################
    ### Label Model ###
    print("\n")
    print("###################")
    print("### Label Model ###")
    print("###################")
    # define model
    label_model = LabelModel(cardinality=num_classes, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500,
                    lr=0.001, log_freq=100, seed=42)
    # weights of labeling functions used
    label_model_weights = np.around(label_model.get_weights(), 2)  
    # prediction
    funcs, metrics_names = get_metric_funcs_list()
    normal_labels = label_model.predict(L_test)
    # quit()
    for func, metrics_name in zip(funcs, metrics_names):
        print(metrics_name, func(testData.label, normal_labels))


    ### Majority Label Voter ###
    print("\n")
    print("############################")
    print("### Majority Label Voter ###")
    print("############################")
    # define model
    majority_model = MajorityLabelVoter(cardinality=num_classes)
    # prediction
    majority_labels = majority_model.predict(L_test)
    # print(majority_labels)
    # print(majority_labels)
    # print(testData.label)
    # quit()
    for func, metrics_name in zip(funcs, metrics_names):
        print(metrics_name, func(testData.label, majority_labels))
    

if __name__ == '__main__':
    # main(args)
    main()