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
from snorkel.labeling import LabelingFunction
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# for saving models
import pickle
from pathlib import Path
import sys
sys.path.insert(0, "../utils")
from data_path import n2c2_data_prefix
from tools import create_folder, append_tsv_bydf, write_pickles
# from prepare_dataset import prepare_data_for_model

# MATCHING = 1
'''
    NOT_SPAM: if text is short it is probably not spam
'''
# NOT_MATCHING = 0
# ABSTAIN = -1


n2c2_keywords = pd.read_csv("../dataprocess/keywords/n2c2_ade_drug_pair.csv",sep='\t')
print(n2c2_keywords.head(10))
for index, row in n2c2_keywords.iterrows():
    print(row)

def check_pair_lf(input_data, keyword_file):
    ade_drug_pair_df = pd.read_csv(f'../dataprocess/keywords/{keyword_file}.tsv')
    found = False
    for _, row in ade_drug_pair_df.iterrows():
        if (row['ADE'] in input_data.summary.lower()) \
            and (row['DRUG'] in input_data.summary.lower()) :
            found = True
            break
    if found:
        return 1
    else :
        return 0



def change_keyword_lf(keyword_file):
    return LabelingFunction(
        name="lf_model_%s"%("_".join(keyword_file)),
        f=check_pair_lf,
        resources=dict(keyword_file),
    )

lf_rule_n2c2_keyword_pair = change_keyword_lf('n2c2_ade_drug_pair')