import pandas as pd 
import numpy as np
# import os
import sys
from os import listdir
from os.path import isfile, join
import os
# from pathlib import Path

import re
import argparse
# import inspect
# import textwrap
# import pickle
import sklearn
from datetime import datetime

# for cleaning discharge summaries
import nltk
from nltk.corpus import stopwords

# for label models
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from sklearn import metrics

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from nltk.tokenize import RegexpTokenizer
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

sys.path.insert(1, '../utils')
from tools import append_csv_bydf, create_folder
'''
    import models and rules
'''


# ['linear','poly','sigmoid','rbf']

########################
### models and rules ###
########################

##############
### models ###
##############
sys.path.insert(1, '../models')
from pretrained_label_functions import *
# from pretrained_label_functions import  lf_model_svm_linear 
# from pretrained_label_functions import  lf_model_svm_poly 
# from pretrained_label_functions import  lf_model_svm_rbf 
# from pretrained_label_functions import  lf_model_svm_sigmoid 
# from pretrained_label_functions import  lf_model_rfc 

from utils import *

lf_models=[lf_model_svm_linear, lf_model_svm_poly,
    lf_model_svm_rbf, lf_model_svm_sigmoid, lf_model_rfc]
# models_dict = {
#     "1": lf_model_svm_ade_only_linear}

models_dict_desc = {
    "0": "No Models Selected",
    "1": "ADE-Only Prediction using Linear SVM",
    "2": "ADE-Only Prediction using Polynomial SVM",
    "3": "ADE-Only Prediction using Sigmoid SVM",
    "4": "ADE-Only Prediction using RBF SVM",
}

#############
### rules ###
#############
# TODO: put rules together into one file
sys.path.insert(1, '../rules')
from rule1 import lf_ade_drug_single
from rule2 import lf_ade_drug_pair
from rule3 import lf_ade_drug_pair_lem
from rule4 import lf_ade_drug_pair_lem_keyword_triggers
from rule5 import lf_sider2_triggers
from rule6 import lf_sider2_triggers_25words
from rule7 import lf_semmeddb_triggers
from rule8 import lf_keyword_triggers
from rule9 import lf_paper_triggers
from rule10 import lf_paper_triggers_200char
from rule11 import lf_paper_triggers_200char_negate
from rule12 import lf_paper_triggers_25words
rules_dict = {
    "1": lf_ade_drug_single,
    "2": lf_ade_drug_pair,
    "3": lf_ade_drug_pair_lem,
    "4": lf_ade_drug_pair_lem_keyword_triggers,
    "5": lf_sider2_triggers, 
    "6": lf_sider2_triggers_25words,
    "7": lf_semmeddb_triggers,
    "8": lf_keyword_triggers,
    "9": lf_paper_triggers,
    "10": lf_paper_triggers_200char,
    "11": lf_paper_triggers_200char_negate,
    "12": lf_paper_triggers_25words
}

rules_dict_desc = {
    "0": "No Rules Selected",
    "1": "lf_ade_drug_single - any keywords in ade_drug_single found in discharge summary",
    "2": "lf_ade_drug_pair - any pair of keywords in ade_drug_pair found in discharge summary",
    "3": "lf_ade_drug_pair_lem - any pair of lemmatised keywords in ade_drug_pair found in discharge summary",
    "4": "lf_ade_drug_pair_lem_keyword_triggers - any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary",
    "5": "lf_sider2_triggers - any pair of trigger words in sider2_triggers found in discharge summary",
    "6": "lf_sider2_triggers_25words - any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary",
    "7": "lf_semmeddb_triggers - any pair of trigger words in semmeddb_triggers found in discharge summary",
    "8": "lf_keyword_triggers - any trigger word in keyword_triggers found in discharge summary",
    "9": "lf_paper_triggers - any trigger word in paper_triggers found in discharge summary",
    "10": "lf_paper_triggers_200char - any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary",
    "11": "lf_paper_triggers_200char_negate - any trigger word in paper_triggers within 200 characters of any keyword in negate found in discharge summary",
    "12": "lf_paper_triggers_25words - any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary"
}


### execute the parse_args() method ###
# args = parser.parse_args()

################################
### create folder for output ###
################################
def create_output_folder():

    # log_df=pd.DataFrame({log_df_title:[]})

    # if not Path(join(output_folder,"log.csv")).exists():
    #     append_csv_bydf(log_df_title,join(output_folder,"log.csv",sep=","))

    date = datetime.now().strftime("%Y%m%d-%I%M%S%p")
    folder_name=f"result_{date}"
    create_folder(join(output_folder,folder_name))
    return folder_name




data_path = "../N2C2"
def main():

    # result_folder_path=join(output_folder,create_output_folder())

    ### save args configuration ###
    # print('Save settings of arguments into file %s'%join(result_folder_path, 'args_info.csv'))
    # info = pd.DataFrame({'models': [args.models], 'rules': [args.rules]})
    # info.to_csv(join(result_folder_path, 'args_info.csv'), index=False)

    lfs= []

    # FIXME: PUT CLASSIFIER AT THE END
    # get models #
    print("models selected:")
    model_list = [
        # lf_models[0],
        # lf_models[1],
        # lf_models[2],
        lf_models[4]
    ]
    for lf in model_list:
        print(f"model {lf.name} is selected")
        lfs.append(lf)
    # models_list = args.models.split(",")s

    # FIXME:

    print("\n")
    print("rules selected:")
    # get rules #
    # rules_list = args.rules.split(",")
    rules_list = [
        '3',
        '8',
        '12']
    for i in range(0, len(rules_list)) :
        print(rules_list[i], ":", rules_dict_desc[rules_list[i]])
        if rules_list[i] != "0" :
            lfs.append(rules_dict[rules_list[i]])

    ###########################
    ### discharge summaries ###
    ###########################
    # train dataset 
    '''
        get train discharge summaries labels
    '''

    prepared_data_path=join(data_path,"dataframe")
    data_folder_l=["train_txt", "test_txt"]
    if(os.path.exists(prepared_data_path)):
        trainData, testData = [
            pd.read_csv(join(prepared_data_path, "%s.csv"%data_folder))
            for data_folder in data_folder_l]
    else:
        print("please run ../models/train_classifiers_to_pickles.py to prepare data")

    '''
       [optional] clean training data and test data
    '''
    cleaned_Data_l=[]
    for Data in [trainData, testData]:
        for func in [cleanHtml,cleanPunc,keepAlpha,removeStopWords]:
            Data["summary"]=Data["summary"].apply(func)
        cleaned_Data_l.append(Data)
    trainData, testData = cleaned_Data_l
    print(trainData.shape)
    print(trainData.columns)
    print(testData.shape)
    print(testData.columns)

    ###################
    ###  LF applier ###
    ###################
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