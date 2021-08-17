import os
import subprocess
import pandas as pd
import numpy as np
from os.path import join

# for svm
from sklearn import svm

# for grid search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# for saving models
import pickle
from pathlib import Path
import sys
root_dir = os.path.dirname(os.path.abspath(Path(__file__).parent)) #project root dir
sys.path.insert(0, join(root_dir,"utils"))
sys.path.insert(0, join(root_dir,"dataprocess"))
from tools import create_folder, append_tsv_bydf, write_pickles
from prepare_dataset import prepare_data_for_model


training_log_path = join("term_output","classifier_training_log.tsv")
def save_performance_tologfile(model_name, model_metrics, params_list):
    log_cols=["Trial_No", "Classifier","Tune_Type","Parameters","Accuracy","Precision",\
        "Recall","ROC_AUC","F1_Score","Confusion_TN_FN_FP_TP"]
    
    if(os.path.exists(training_log_path)):
        trial_no = int(subprocess.check_output(['tail', '-1', training_log_path]).decode("utf-8").split('\t')[0])+1
    else:
        # append_csv_byrow(log_cols,training_log_path)
        trial_no=1

    model_results=[]
    for name, metric, params, type in zip(
        [model_name]*3, model_metrics, params_list, ["Base", "RandomizedCV", "GridCV"]
    ):
        confusion_flatten = f'[{", ".join(map(str, metric[-1].flatten()))}]'
        metric[-1] = confusion_flatten
        model_results.append([trial_no, name.upper(),type,params,*metric])
    
    result_df=pd.DataFrame(model_results, columns=log_cols)
    append_tsv_bydf(result_df,training_log_path)

def train_model_Kfold(data_dict):
    model = svm.SVC()
    metric_list=[]
    for key, value in data_dict.items():
        tfidf_train_data, tfidf_test_data, train_labels, test_labels = value['tf_idf']
        search_model=GridSearchCV(
            estimator=model, param_grid=narrowd_svm_grid, **tune_search_params
        )
        search_model.fit(tfidf_train_data, train_labels)
        y_pred = search_model.predict(tfidf_test_data)
        metrics=[func(test_labels, y_pred) for func in funcs[:-1]]
        metric_list.append(metrics)
        print(f'Best parameters found for SVM using {key} data and Grid Search\
            :\n{search_model.best_params_}')
        write_pickles(search_model, join('../models','svm_pickles',f'{key}_svm_Grid_Tuned.pkl'))

    performance_df=pd.DataFrame(metric_list,columns=[metrics_names[:-1]])
    performance_df.to_csv("../models/term_output/cv_results.csv",index=False)
        ## train
    ## TODO:

def grid_tune(grid_params, narrowed_grid_params, grid_search_params,model, model_name, logged=False):
    pickle_path=f"../models/{model_name.lower()}_pickles"
    print(f"Tuning Hyperparameters for {model_name.upper()}:\n",('-')*100)
    print(f"Initial parameters of base {model_name.upper()}:\n{model.get_params()}")
    print(f"\nRange of parameters used for {model_name.upper()} tuning:\n{grid_params}")

    searchCVs=[]
    for searchCV, param_range, search_name in zip(
        [RandomizedSearchCV, GridSearchCV],
        [grid_params, narrowed_grid_params],
        ["Randomized", "Grid"]
        ):
        searchCV_model = searchCV(
            estimator = model, param_distributions=param_range, **grid_search_params
        )
        searchCV_model.fit(tfidf_train_data, train_labels)
        print(f"Best parameters found for {model_name.upper()} using {search_name} Search\
            :\n{searchCV_model.best_params_}")
        searchCVs.append(searchCV_model)
    
    model_metrics=[]
    ## Training models on train data
    for current_model, desc in zip(
        [model, *searchCVs],
        ["Base", "Random Tuned", "Grid Tuned"]):
        current_model.fit(tfidf_train_data, train_labels)
        y_pred = current_model.predict(tfidf_test_data)
        print(f"Performance of {desc} {model_name.upper()} Model:")
        metrics=[func(test_labels, y_pred) for func in funcs]
        model_metrics.append(metrics)
        for name, metric in zip(metrics_names[:-1], metrics[:-1]):
            print(f"{name}: {metric}")
        print(metrics_names[-1], metrics[-1])
        write_pickles(current_model, join(pickle_path,f'{model_name.lower()}_{desc.replace(" ","_")}.pkl'))
    ## check model improvement after tunning
    for tuned_model_metric, desc in zip(
        model_metrics[1:],["Random Tuned", "Grid Tuned"]):  ## compare two tuned model with base model
        metrics_improvement =[(tuned-base)/base*100 for base, tuned \
            in zip(model_metrics[0][:-1],tuned_model_metric[:-1])]
        print(f"Improvement of {desc} {model_name.upper()} Model after Tuning using {desc} Search:")
        for name, improve in zip(metrics_names[:-1],metrics_improvement[:-1]):
            print(f"{name}: {improve:2.2f}%")
        print(metrics_names[-1],tuned_model_metric[-1]-model_metrics[0][-1])
    ## update training log
    if logged:
        save_performance_tologfile(model_name, model_metrics, \
            [model.get_params(),*[searchCV.best_params_ for searchCV in searchCVs]])


from model_params import rfc_grid, narrowed_rfc_grid, svm_grid,  narrowd_svm_grid,tune_search_params
from utils import get_metric_funcs_list
from sub_path import data_path
train_labels, test_labels, tfidf_train_data, tfidf_test_data = [None]*4
funcs, metrics_names = get_metric_funcs_list()

def main():

    global tfidf_train_data, tfidf_test_data, train_labels, test_labels 
    tfidf_train_data, tfidf_test_data, train_labels, test_labels = prepare_data_for_model() # get data for training models
    '''
        Tuning hyperparameters for model
    '''
    best_rfc_params = {'n_estimators': 1333, 'min_samples_split': 5, 'min_samples_leaf': 1, \
        'max_features': 'sqrt', 'max_depth': 50, 'class_weight': 'balanced_subsample', 'bootstrap': True}        
    
    model_list=[
        # RandomForestClassifier(n_estimators = 10, random_state = 42),
        RandomForestClassifier(**best_rfc_params), 
        svm.SVC()]

    for model, model_name, model_tuning_params, model_search_params in list(zip(
        model_list,
        ["RFC", "SVM"],
        [[rfc_grid, narrowed_rfc_grid],[svm_grid, narrowd_svm_grid]],
        [tune_search_params]*2
    )):
        grid_tune(*model_tuning_params,model_search_params,model,model_name,logged=True)
        print("\n"*5,"-"*50,"Following are package outputs","-"*50)

    ## (tn, fp, fn, tp)
    # for run_model_func in [run_svm_model, run_RFC_model]:
    #     run_model_func(train_labels, test_labels, tfidf_train_data, tfidf_test_data)
    # run_svm_model(trainData, testData, tfidf_train_data,tfidf_test_data)
    # run_RFC_model(trainData, testData, tfidf_train_data,tfidf_test_data)


# if __name__ == '__main__':
    # main()

