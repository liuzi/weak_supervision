import os
import subprocess
import pandas as pd
import numpy as np
from os.path import join


# for svm
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from utils import *

# for grid search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# for saving models
import pickle
from pathlib import Path
import sys
root_dir = os.path.dirname(os.path.abspath(Path(__file__).parent)) #project root dir
sys.path.insert(0, join(root_dir,"utils"))
from tools import create_folder, append_csv_byrow, append_csv_bydf

funcs, metrics_names = get_metric_funcs_list()

###########################
### discharge summaries ###
###########################

### path of folder with ### 
# train_txt: folder with discharge summaries for train dataset
# train_ann: folder with annotated files (derived from discharge summaries) for train dataset
# test_txt: folder with discharge summaries for test dataset
# test_ann: folder with annotated files (derived from discharge summaries) for test dataset
# MODIFIED: restrict filename to be ended with ".ann"
def count_ade_drug_fromAnn(data_name):
    labelled = []
    labelled_dict = {}
    ade_drug_counts = []

    for file in [f for f in os.listdir(join(data_path,data_name)) if ".ann" in f]:
        with open(join(data_path, data_name, file), 'rb') as document_anno_file:
            lines = document_anno_file.readlines()
            patient = file[:-4]
            boolean = False
            ade_drug_count = 0

            for line in lines :
                # MODIFIED: "ADE" -> "ADE-Drug"
                if b"ADE-Drug" in line:
                    boolean = True
                    ade_drug_count += 1
            # label which has ADE-Drug
            if boolean:
                labelled.append(1)
                labelled_dict[patient] = 1
            else :
                labelled.append(0)
                labelled_dict[patient] = 0
            ade_drug_counts.append(ade_drug_count)

    print("Number of files processed in folder %s: %d"%(
        data_name, sum(list(map(len,[labelled, labelled_dict, ade_drug_counts])))/3)
    )
    return labelled, labelled_dict, ade_drug_counts


# TODO: directly create csv file
def prepare_dataset(train_labelled_dict, test_labelled_dict, \
    data_folder_l=["train_txt", "test_txt"]):
    data_df_l=[]
    prepared_data_path=join(data_path,"dataframe")
    create_folder(prepared_data_path)

    for data_folder, labelled_dict in zip(
        data_folder_l, [train_labelled_dict, test_labelled_dict]
    ):
        curr_dir = join(data_path,data_folder)
        patient_summary_label=[]
        for file in os.listdir(curr_dir):
            label=labelled_dict[file[:-4]]
            with open(join(curr_dir, file), 'r') as document_summary_file:
                summary = " ".join(line.strip() for line in document_summary_file)
            patient_summary_label.append([file[:-4], summary, label])
        data_df = pd.DataFrame(np.array(patient_summary_label),
            columns=["patient","summary","label"])
        data_df_l.append(data_df)
        data_df.to_csv(join(prepared_data_path,"%s.csv"%data_folder), index=False)
        
    return data_df_l    

def prepare_data_for_model():
    '''
        prepare training data and test data
    '''
    ### preprocess datasets ###
    print("started preparing train and test datasets for training models")
    prepared_data_path=join(data_path,"dataframe")
    data_folder_l=["train_txt", "test_txt"]
    if(os.path.exists(prepared_data_path)):
        trainData, testData = [
            pd.read_csv(join(prepared_data_path, "%s.csv"%data_folder))
            for data_folder in data_folder_l]
    else:
        train_labelled, train_labelled_dict, \
            train_ade_drug_counts = count_ade_drug_fromAnn("train_ann") 
        test_labelled, test_labelled_dict, \
            test_ade_drug_counts = count_ade_drug_fromAnn("test_ann") 
        trainData, testData=prepare_dataset(train_labelled_dict, test_labelled_dict)
    '''
       [optional] clean training data and test data
    '''
    # FIXME: tag01 prepare data version 3
    cleaned_Data_l=[]
    for Data in [trainData, testData]:
        for func in [cleanHtml,cleanPunc,keepAlpha,removeStopWords]:
            Data["summary"]=Data["summary"].apply(func)
        cleaned_Data_l.append(Data)
    trainData, testData = cleaned_Data_l

    '''
        transform traindata and testdata for model: Feature Extraction from text
    '''
    pipeline_name = 'feature_extraction_pipeline.pkl'
    if(not os.path.exists(pipeline_name)):
        feature_extraction_pipe = make_pipeline(
            CountVectorizer(binary=True),
            TfidfTransformer(use_idf=True),
        )
        tfidf_train_data = feature_extraction_pipe.fit_transform(trainData.summary) 
        tfidf_test_data = feature_extraction_pipe.transform(testData.summary)
        # save pipeline as pickle file
        with open(pipeline_name, 'wb') as file :
            pickle.dump(feature_extraction_pipe, file)
        print("pipeline for feature extraction from text is created")
    else:
        with open(pipeline_name, 'rb') as file:
            feature_extraction_pipe = pickle.load(file)
        tfidf_train_data, tfidf_test_data = list(
            map(feature_extraction_pipe.transform, [trainData.summary, testData.summary]))

    train_labels, test_labels = trainData.label, testData.label
    return tfidf_train_data, tfidf_test_data, train_labels, test_labels

def run_RFC_model(train_labels, test_labels, tfidf_train_data,tfidf_test_data, pickle_path = "rfc_pickles"):
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


def run_svm_model(train_labels, test_labels, tfidf_train_data,tfidf_test_data,pickle_path = "svm_pickles"):
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
        confusion_flatten = ",".join(map(str, metric[-1].flatten()))
        metric[-1] = confusion_flatten
        model_results.append([trial_no, name.upper(),type,params,*metric])
    
    result_df=pd.DataFrame(model_results, columns=log_cols)
    append_csv_bydf(result_df,training_log_path)


def grid_tune(train_labels, test_labels, tfidf_train_data,\
    tfidf_test_data,grid_params, narrowed_grid_params, model, model_name="svm", logged=False):
    pickle_path=f"{model_name}_pickles"
    print(f"Tuning Hyperparameters for {model_name.upper()}:\n",('-')*100)
    print(f"Range of parameters used for {model_name.upper()} tuning:\n{grid_params}")

    searchCVs=[]
    for searchCV, param_range, search_name in zip(
        [RandomizedSearchCV, GridSearchCV],
        [grid_params, narrowed_grid_params],
        ["Randomized", "Grid"]
        ):
        searchCV = RandomizedSearchCV(
            estimator = model, param_distributions=param_range, cv=5,
            verbose=2, random_state=0, n_jobs = -1, n_iter=20
        )
        searchCV.fit(tfidf_train_data, train_labels)
        print(f"Best parameters found for {model_name.upper()} using {search_name} Search\
            :\n{searchCV.best_params_}")
        searchCVs.append(searchCV)
    
    model_metrics=[]
    for current_model, description in zip(
        [model, *searchCVs],
        ["Base", "Random Tuned", "Grid Tuned"]):
        current_model.fit(tfidf_train_data, train_labels)
        y_pred = current_model.predict(tfidf_test_data)
        print(f"Performance of {description} {model_name.upper()} Model:")
        metrics=[func(test_labels, y_pred) for func in funcs]
        model_metrics.append(metrics)
        for name, metric in zip(metrics_names[:-1], metrics[:-1]):
            print(f"{name}: {metric}")
        print(metrics_names[-1], metrics[-1])
        # model_accuracys.append(evaluate(current_model, tfidf_test_data, test_labels))

    for tuned_model_metric, desc in zip(
        model_metrics[1:],["Random Tuned", "Grid Tuned"]):  ## compare two tuned model with base model
        metrics_improvement =[(tuned-base)/base*100 for base, tuned \
            in zip(model_metrics[0][:-1],tuned_model_metric[:-1])]
        print(f"Improvement of {description} {model_name.upper()} Model after Tuning using {desc} Search:")
        for name, improve in zip(metrics_names[:-1],metrics_improvement[:-1]):
            print(f"{name}: {improve:2.2f}%")
        print(metrics_names[-1],tuned_model_metric[-1]-model_metrics[0][-1])

    if logged:
        save_performance_tologfile(model_name, model_metrics, \
            [model.get_params(),*[searchCV.best_params_ for searchCV in searchCVs]])


from model_params import rfc_grid, narrowed_rfc_grid

data_path = "../N2C2"
def main():

    tfidf_train_data, tfidf_test_data, train_labels, test_labels = prepare_data_for_model() # get data for training models
    '''
        Tuning hyperparameters for model
    '''
    # model =  RandomForestClassifier(n_estimators = 10, random_state = 42)
    best_rfc_params = {'n_estimators': 1333, 'min_samples_split': 5, 'min_samples_leaf': 1, \
        'max_features': 'sqrt', 'max_depth': 50, 'class_weight': 'balanced_subsample', 'bootstrap': True}
    model=RandomForestClassifier(**best_rfc_params)
    grid_tune(train_labels,test_labels,tfidf_train_data,tfidf_test_data,\
        rfc_grid,narrowed_rfc_grid,model,"rfc",logged=True)
    print("\n"*5,"-"*50,"Following are package outputs","-"*50)
    quit()

    ## (tn, fp, fn, tp)
    for run_model_func in [run_svm_model, run_RFC_model]:
        run_model_func(train_labels, test_labels, tfidf_train_data, tfidf_test_data)
    # run_svm_model(trainData, testData, tfidf_train_data,tfidf_test_data)
    # run_RFC_model(trainData, testData, tfidf_train_data,tfidf_test_data)


if __name__ == '__main__':
    main()

