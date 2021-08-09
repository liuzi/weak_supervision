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

sys.path.insert(1, '../models')
from pretrained_label_functions import *
from prepare_dataset import prepare_data_for_model
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


'''
    create folder for output
'''
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

    ## TODO:save args configuration 
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

    trainData, testData = prepare_data_for_model(True) # get data for training models


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