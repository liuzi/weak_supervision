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
from tools import create_folder, append_tsv_bydf, write_pickles
from prepare_dataset import prepare_data_for_model


# FIXME:
def run_RFC_model(pickle_path = "rfc_pickles"):
    create_folder(pickle_path)
    '''
        RANDOM FOREST
    '''
    print("#####################")
    print("### Random Forest ###")
    print("#####################")
    # Create a svm classifier
    clf = RandomForestClassifier(max_depth=5, \
        n_estimators=10, max_features=1, class_weight='balanced')
    # Train the model using training sets
    clf.fit(tfidf_train_data, train_labels)
    # Predict the response for test dataset
    y_pred = clf.predict(tfidf_test_data)

    metrics_names=list(map(lambda x: "Random Forest %s"%(x), metrics_names))
    for name, func in zip(metrics_names,funcs):
        print(name, func(test_labels, y_pred))
    # save model as pickle file
    filename = '%s.pkl'%"rfc"
    with open(join(pickle_path,filename), 'wb') as file :  
        pickle.dump(clf, file)
    print("Random Forest model saved")

def run_svm_model(pickle_path = "svm_pickles"):
    create_folder(pickle_path)
    kernel_list=['linear','poly','sigmoid','rbf']
    for kernel in kernel_list:
        print("###################")
        print("### SVM: %s ###"%kernel)
        print("###################")
        # Create a svm classifier
        clf = svm.SVC(kernel=kernel)
        # Train the model using training sets
        clf.fit(tfidf_train_data, train_labels)
        # Predict the response for test dataset
        y_pred = clf.predict(tfidf_test_data)

        model_metrics_names=list(map(lambda x: "%s SVM %s"%(kernel, x), metrics_names))

        for name, func in zip(model_metrics_names,funcs):
            print(name, func(test_labels, y_pred))

        # save model as pickle file
        filename = 'svm_%s.pkl'%kernel
        with open(join(pickle_path,filename), 'wb') as file :  
            pickle.dump(clf, file)
        print("%s SVM model saved"%kernel)

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


def grid_tune(grid_params, narrowed_grid_params, grid_search_params,model, model_name, logged=False):
    pickle_path=f"{model_name.lower()}_pickles"
    print(f"Tuning Hyperparameters for {model_name.upper()}:\n",('-')*100)
    print(f"Initial parameters of base {model_name.upper()}:\n{model.get_params()}")
    print(f"\nRange of parameters used for {model_name.upper()} tuning:\n{grid_params}")

    searchCVs=[]
    for searchCV, param_range, search_name in zip(
        [RandomizedSearchCV, GridSearchCV],
        [grid_params, narrowed_grid_params],
        ["Randomized", "Grid"]
        ):
        searchCV = RandomizedSearchCV(
            estimator = model, param_distributions=param_range, **grid_search_params
        )
        searchCV.fit(tfidf_train_data, train_labels)
        print(f"Best parameters found for {model_name.upper()} using {search_name} Search\
            :\n{searchCV.best_params_}")
        searchCVs.append(searchCV)
    
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
    ## Predict test data using trained models
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
data_path = "../N2C2"
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


if __name__ == '__main__':
    main()

