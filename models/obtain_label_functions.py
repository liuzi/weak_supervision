import os
import subprocess
from typing import Pattern
import pandas as pd
import numpy as np
from os.path import join


# for svm
from sklearn import svm

# for saving models
import sys
sys.path.insert(0, "../utils")
from data_path import n2c2_data_prefix
from tools import create_folder, write2file, create_folder_overwrite

from define_label_functions import make_classifier_kernel_lf, change_keyword_lf, change_trigger_pair_withWindow_lf,top_drug_disease_same_topic_lf, textblob_subjectivity, textblob_polarity
# from prepare_dataset import prepare_data_for_model

# MATCHING = 1
'''
    NOT_SPAM: if text is short it is probably not spam
'''
# NOT_MATCHING = 0
# ABSTAIN = -1

# n2c2_keywords = pd.read_csv("../dataprocess/keywords/n2c2_ade_drug_pair.csv",sep='\t')
# print(n2c2_keywords.head(10))
# for index, row in n2c2_keywords.iterrows():
#     print(row)


from sub_path import labelfunction_dict_dir, classifier_lf_file, rule_lf_file, pattern_lf_file,lf_cols
def archive_lfs(lfs,lf_file,index_prefix=''):
    lfs_names = [lf.name for lf in lfs]
    lf_df = pd.DataFrame(
        {lf_cols[0]:[f'{index_prefix}{id}' for id in \
            list(range(1,len(lfs_names)+1))],
        lf_cols[1]:lfs_names
        })
    write2file(lf_df, os.path.join(labelfunction_dict_dir,lf_file)) 

def get_classifier_lfs():
    classifier_lfs=[]
    for classifier in ['svm','rfc']:
        for tune_state in ['Base','Grid_Tuned','Random_Tuned']:
            classifier_lfs.append(
                make_classifier_kernel_lf({'model_type':classifier, 'tune_state':tune_state})
            )
    ## specially for svm_linear
    classifier_lfs.append(
        make_classifier_kernel_lf({'model_type':'svm', 'tune_state':'linear'})
    )
    if(not os.path.exists(os.path.join(labelfunction_dict_dir,classifier_lf_file))):
        archive_lfs(classifier_lfs,classifier_lf_file,'c')
    return dict(zip(list(map(lambda x: x.name ,classifier_lfs)) ,classifier_lfs))

def get_rule_lfs():
    rule_lfs=[]
    pair_files = ['n2c2_ade_drug_pair']
    trigger_files=['n2c2_manual_trigger','paper_trigger']

    # pair_trigger_files
    for keyword in pair_files:
        rule_lfs.append(
            change_keyword_lf(keyword,pair=True)
            )
    for keyword in trigger_files:
        rule_lfs.append(
            change_keyword_lf(keyword,pair=False)
            )
    for trigger in trigger_files:
        for pair in pair_files:
            rule_lfs.append(
                change_trigger_pair_withWindow_lf({'trigger_file':trigger, 'pair_file':pair, 'window':200})
            )
    if(not os.path.exists(os.path.join(labelfunction_dict_dir,rule_lf_file))):
        archive_lfs(rule_lfs,rule_lf_file,'r')
    return dict(zip(list(map(lambda x: x.name ,rule_lfs)) ,rule_lfs))

def get_pattern_lfs():
    pattern_lfs=[top_drug_disease_same_topic_lf,textblob_subjectivity,textblob_polarity]
    if(not os.path.exists(os.path.join(labelfunction_dict_dir,pattern_lf_file))):
        archive_lfs(pattern_lfs,pattern_lf_file,'p')
    return dict(zip(list(map(lambda x: x.name ,pattern_lfs)) ,pattern_lfs))


def get_all_lfs(rearchive=False):
    if rearchive:
        create_folder_overwrite(labelfunction_dict_dir)
    c_lfs=get_classifier_lfs()
    r_lfs=get_rule_lfs()
    return {**c_lfs, **r_lfs,**(get_pattern_lfs())}

