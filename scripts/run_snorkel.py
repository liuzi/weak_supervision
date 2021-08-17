# import pandas as pd 

import sys
# from os import listdir
# from os.path import isfile, 
# from os.path import join
import os
import subprocess
import numpy as np
import pandas as pd
# from datetime import datetime

# for sklearn package
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# for label models
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from sklearn import metrics

# for labeling functions

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

sys.path.insert(1, '../utils')
from tools import append_tsv_bydf, create_folder
from utils import get_metric_funcs_list

sys.path.insert(1, '../models')
from obtain_label_functions import get_all_lfs
from train_classifiers_to_pickles import train_model_Kfold

sys.path.insert(1, '../dataprocess')
# from pretrained_label_functions import *
from prepare_dataset import prepare_data_for_model, get_data_for_cross_fold_test
from arg_parser import get_lfdict_and_parser

funcs, metrics_names = get_metric_funcs_list()
if(not os.path.exists("term_output")):
    create_folder("term_output")

training_log_path = os.path.join("term_output","snorkel_training_log.tsv")
def save_performance_tologfile(model_metrics, lfs_names):
    new_metrics_names=metrics_names.copy()
    new_metrics_names[-1]=metrics_names[-1][:-1]
    log_cols=["Trial_No","LF_Type", "LFs"]+new_metrics_names
    
    if(os.path.exists(training_log_path)):
        trial_no = int(
            subprocess.check_output(['tail', '-1', training_log_path]
        ).decode("utf-8").split('\t')[0])+1
    else:
        trial_no=1

    model_results=[]
    for  lf_type, metric in zip(
        ["Label Model","Majority Label Voter"], model_metrics
    ):
        confusion_flatten = f'[{", ".join(map(str, metric[-1].flatten()))}]'
        metric[-1] = confusion_flatten
        model_results.append([trial_no, lf_type,lfs_names,*metric])
    
    result_df=pd.DataFrame(model_results, columns=log_cols)
    append_tsv_bydf(result_df,training_log_path)
    print(f"Save Performance of Trial {trial_no} into {training_log_path}")


def get_selected_lf_list(rearchive):
    lfname_lf_dict = get_all_lfs(rearchive=rearchive)
    index_lfname_dict, parser = get_lfdict_and_parser()
    args = parser.parse_args()
    
    index_list = list(map(lambda s: s.strip(),args.label_function.split(',')))
    lfname_list = [index_lfname_dict[id] for id in index_list]

    print('selected label functions:')
    for index, lfname in zip(index_list, lfname_list):
        print(f'--{index}: {lfname}')
    lf_list = [lfname_lf_dict[lfname] for lfname in lfname_list]
    return lf_list

def run_snorkel_LF(lf_list, trainData, testData, logged=False):
    num_classes = 2
    ## snorkel pandas applier
    applier = PandasLFApplier(lfs=lf_list) 
    L_train = applier.apply(df=trainData)
    print("\n")
    print("Labeling Function Analysis on train dataset")
    print(f"{LFAnalysis(L_train, lf_list).lf_summary()}")
    L_test = applier.apply(df=testData)

    ### label model ###
    label_model = LabelModel(cardinality=num_classes, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500,
                    lr=0.001, log_freq=100, seed=42)
    majority_model = MajorityLabelVoter(cardinality=num_classes)
    # print(majority_model.predict(L_test))
    
    # weights of labeling functions used
    # label_model_weights = np.around(label_model.get_weights(), 2)  
    metrics_list=[]
    for model, model_name in zip(
        [label_model, majority_model],
        ["Label Model", "Majority Label Voter"]):
        normal_labels = model.predict(L_test)
        normal_labels=list(map(lambda x: 0 if x==-1 else x,normal_labels))
        # print(normal_labels)
        print("\n","#"*60,f"\n{model_name}\n", "#"*60)
        metrics = [func(testData.label, normal_labels) for func in funcs]
        for metrics_name, metric in zip(metrics_names, metrics):
            print(metrics_name, metric)
        metrics_list.append(metrics)
    if(logged):
        save_performance_tologfile(metrics_list, [lf.name for lf in lf_list])

# from sklearn.model_selection import StratifiedKFold    
# def cross_fold_test(trainData, testData, n_splits=5):
#     allData=pd.concat([trainData, testData],axis=0).set_index('patient')
#     scv=StratifiedKFold(n_splits, random_state=42, shuffle=True)

#     X=allData['summary'].copy()
#     y=allData['label'].copy()
#     data_dict={}
#     # dict_id=0
#     for (train_index, test_index), dict_id in zip(scv.split(X, y), list(range(n_splits))):
#         train_data=pd.DataFrame({'summary':X[train_index],'label':y[train_index]})
#         test_Data=pd.DataFrame({'summary':X[test_index],'label':y[test_index]})
#         data_dict[f'train_{dict_id:int}']=trainData
#         tes
#         data_dictprint(dict_id)
    # for train_index, test_index, dict_id in zip(scv.split(allData),:
    #     train_data_list.append(allData[train_index])
    #     test_data_list.append(allData[test_index])
    #     # X_train, X_test= X[train_index], X[test_index]
	#     # y_train, y_test= y[train_index], y[test_index]
    # print(train_data_list[0], train_data_list[1])
    # print(test_data_list[0],test_data_list[1])

def main():

    lf_list = get_selected_lf_list(rearchive=False)
    # TODO: SAVE TRAINING LOG
    # trainData, testData = prepare_data_for_model(True) # get data for training models
    # run_snorkel_LF(lf_list, trainData, testData,logged=True)
    # cross_fold_test(trainData,testData)
    '''
        contain num_splits of keys and values
        key: cv{id}, id from 0 to num_splits-1
        value:{'train':train_data,'test':test_data,'tf_idf':tfidf_pack}
                tfidf=[tfidf_train_data, tfidf_test_data, trainData.label, testData.label]
    '''
    # data_dict=get_data_for_cross_fold_test(5)
    #
    #  train_model_Kfold(data_dict)  ## train models for loading pickles if pickle files are not created
    cv_results=pd.read_csv('../models/term_output/cv_results.csv')
    print(cv_results.mean(axis=0))
    # print(data_dict['cv0']['train'])



    ### Majority Label Voter ###
    # print("\n","#"*60,"\nMajority Label Voter\n", "#"*60)
    # define model
    
    # majority_labels = majority_model.predict(L_test)
    # for func, metrics_name in zip(funcs, metrics_names):
    #     print(metrics_name, func(testData.label, majority_labels))
    
if __name__ == '__main__':
    # main(args)
    main()