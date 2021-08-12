import sys
import os
# from utils.tools import write2file
sys.path.insert(1, '../utils')
from tools import create_folder, write2file

import pickle
import pandas as pd

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from os.path import join
# from snorkel.labeling.lf.nlp import nlp_labeling_function
# from nltk.tokenize import RegexpTokenizer
# from snorkel.labeling import PandasLFApplier
# from snorkel.labeling import LFAnalysis


feature_extraction_model_path= "../models/feature_extraction_pipeline.pkl"
'''
    model_args=[model_type, *args]
'''
def lf_model(input_data, model_type, tune_state):
    model_path = join(f'../models/{model_type}_pickles',f'{model_type}_{tune_state}.pkl')
    with open(feature_extraction_model_path, 'rb') as file:
        fe = pickle.load(file)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    X = fe.transform([input_data.summary])
    score = model.predict(X)
    return score


def make_model_kernel_lf(model_args):
    return LabelingFunction(
        name="lf_model_%s"%("_".join(list(model_args.values()))),
        f=lf_model,
        resources=dict(**model_args),
    )

from sub_path import labelfunction_dict_dir, classifier_lf_file, lf_cols
def get_model_lfs():
    model_lfs=[]
    for classifier in ['svm','rfc']:
        for tune_state in ['Base','Grid_Tuned','Random_Tuned']:
            model_lfs.append(
                make_model_kernel_lf({'model_type':classifier, 'tune_state':tune_state})
            )
    classifier_names = [model_lf.name for model_lf in model_lfs]
    if(not os.path.exists(os.path.join(labelfunction_dict_dir,classifier_lf_file))):
        classifier_df = pd.DataFrame(
            {lf_cols[0]:['c%d'% id for id in list(range(1,len(classifier_names)+1))],
            lf_cols[1]:classifier_names
            })
        create_folder(labelfunction_dict_dir)
        write2file(classifier_df, os.path.join(labelfunction_dict_dir,classifier_lf_file))   

    return dict(zip(classifier_names ,model_lfs))
# lf_model_svm_linear = make_model_kernel_lf(["svm" ,"linear"])
# lf_model_svm_poly = make_model_kernel_lf(["svm" ,"poly"])
# lf_model_svm_rbf = make_model_kernel_lf(["svm" ,"rbf"])
# lf_model_svm_sigmoid = make_model_kernel_lf(["svm" ,"sigmoid"])
# lf_model_rfc = make_model_kernel_lf(["rfc"])
# print(list(range(1,10)))


