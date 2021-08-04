################
### packages ###
################

# for files
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob
import re
import sys
import warnings
import shutil

# for discharge summary dataset preparation
import sklearn
from sklearn import datasets
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# for svm
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from utils import *

# for grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# for saving models
import pickle

sys.path.insert(1, '../utils')
from tools import copytree, create_folder_overwrite, create_folder

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


def run_RFC_model(trainData, testData, tfidf_train_data,tfidf_test_data, pickle_path = "rfc_pickles"):
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
    train_label=np.array(trainData.label)
    # Train the model using training sets
    clf.fit(tfidf_train_data, train_label)
    # Predict the response for test dataset
    y_pred = clf.predict(tfidf_test_data)
    funcs, metrics_names = get_metric_funcs_list()
    metrics_names=list(map(lambda x: "Random Forest %s"%(x), metrics_names))
    test_label=np.array(testData.label)
    for name, func in zip(metrics_names,funcs):
        print(name, func(test_label, y_pred))
    # save model as pickle file
    filename = '%s.pkl'%"rfc"
    with open(join(pickle_path,filename), 'wb') as file :  
        pickle.dump(clf, file)
    print("Random Forest model saved")



def run_svm_model(trainData, testData, tfidf_train_data,tfidf_test_data,pickle_path = "svm_pickles"):
    create_folder(pickle_path)
    kernel_list=['linear','poly','sigmoid','rbf']

    ###################
    ###  SVM Models ###
    ###################
    for kernel in kernel_list:
    
        print("###################")
        print("### SVM: %s ###"%kernel)
        print("###################")
        # Create a svm classifier
        clf = svm.SVC(kernel=kernel)
        # Train the model using training sets
        train_label=np.array(trainData.label)
        clf.fit(tfidf_train_data, train_label)
        # Predict the response for test dataset
        y_pred = clf.predict(tfidf_test_data)
        funcs, metrics_names = get_metric_funcs_list()
        model_metrics_names=list(map(lambda x: "%s SVM %s"%(kernel, x), metrics_names))
        test_label=np.array(testData.label)
        for name, func in zip(model_metrics_names,funcs):
            print(name, func(test_label, y_pred))

        # save model as pickle file
        filename = 'svm_%s.pkl'%kernel
        with open(join(pickle_path,filename), 'wb') as file :  
            pickle.dump(clf, file)
        print("%s SVM model saved"%kernel)

data_path = "../N2C2"
def main():
    
    '''
        prepare training data and test data
    '''
    ### preprocess datasets ###
    print("started preprocessing train and test datasets")
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

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
        Feature Extraction from text
    '''
    feature_extraction_pipe = make_pipeline(
        CountVectorizer(binary=True),
        TfidfTransformer(use_idf=True),
    )

    tfidf_train_data = feature_extraction_pipe.fit_transform(trainData.summary) 
    tfidf_test_data = feature_extraction_pipe.transform(testData.summary)
    # save pipeline as pickle file
    pipeline_name = 'feature_extraction_pipeline.pkl'
    with open(pipeline_name, 'wb') as file :
        pickle.dump(feature_extraction_pipe, file)
    print("pipeline for feature extraction from text is created")

    run_svm_model(trainData, testData, tfidf_train_data,tfidf_test_data)
    run_RFC_model(trainData, testData, tfidf_train_data,tfidf_test_data)


if __name__ == '__main__':
    main()


