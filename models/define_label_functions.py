import sys
import os
# from utils.tools import write2file
sys.path.insert(1, '../utils')
from tools import create_folder, write2file

import pickle
import pandas as pd
import re

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from os.path import join
# from snorkel.labeling.lf.nlp import nlp_labeling_function
# from nltk.tokenize import RegexpTokenizer
# from snorkel.labeling import PandasLFApplier
# from snorkel.labeling import LFAnalysis



'''
    Pretrained Classifier LFs
'''
def lf_model(input_data, model_type, tune_state, \
    feature_extraction_model_path= "../dataprocess/feature_extraction_pipeline.pkl"):

    model_path = join(f'../models/{model_type}_pickles',f'{model_type}_{tune_state}.pkl')
    # model_path = join(f'../models/{model_type}_pickles',f'{model_type}_{tune_state}.pkl')
    with open(feature_extraction_model_path, 'rb') as file:
        fe = pickle.load(file)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    X = fe.transform([input_data.summary])
    score = model.predict(X)
    return score

def make_classifier_kernel_lf(model_args):
    return LabelingFunction(
        name="lf_classifier_%s"%("_".join(list(model_args.values()))),
        f=lf_model,
        resources=dict(**model_args),
    )


'''
    Heuristic rules LFs
'''
def check_keyword_lf(input_data, keyword_file,pair):
    keyword_df = pd.read_csv(f'../dataprocess/keywords/{keyword_file}.tsv',sep='\t')
    found = False
    current_summary=input_data.summary.lower()
    if(pair):
        for _, row in keyword_df.iterrows():
            if (row['ADE'] in current_summary) \
                and (row['DRUG'] in current_summary): 
                found = True
                break
    else:
        for trigger in keyword_df.iloc[:,0]:
            if trigger in current_summary:
                found = True
                break

    if found:
        return 1
    else :
        return -1


def change_keyword_lf(keyword_file,pair):
    return LabelingFunction(
        name=f"lf_rule_{keyword_file}",
        f=check_keyword_lf,
        resources=dict(keyword_file=keyword_file,pair=pair)
    )


def check_trigger_pair_withWindow(input_data, trigger_file, pair_file, window):
    pair_df = pd.read_csv(f'../dataprocess/keywords/{pair_file}.tsv',sep='\t')
    trigger_list = pd.read_csv(f'../dataprocess/keywords/{trigger_file}.tsv',sep='\t').iloc[:,0]

    found = False
    current_summary=input_data.summary.lower()
    
    for _, row in pair_df.iterrows():
        # pair_sets=set([row['ADE'], row['DRUG']])
        pos = [m.start() for m in re.finditer(row['ADE'], current_summary)]
        start_pos = 0
        end_pos = 0
        # if any of the papers triggers words are found within [-200, +200] of the drug in discharge summary
        for j in range(0, len(pos)) :
            start_pos = pos[0] - window
            end_pos = pos[0] + len(row['ADE']) + window
            if start_pos < 0 :
                start_pos = 0
        if row['DRUG'] in current_summary[start_pos:end_pos]:
            if any(trigger in current_summary[start_pos:end_pos] for trigger in trigger_list):
                found = True
                break
    if found:
        return 1
    else :
        return -1

def change_trigger_pair_withWindow_lf(args):
    # trigger_file, pair_file, window
    return LabelingFunction(
        name=f'lf_rule_{("_").join(list(map(str,args.values())))}',
        f=check_trigger_pair_withWindow,
        resources=dict(**args)
    )

from snorkel.preprocess import preprocessor
from textblob import TextBlob
@preprocessor(memoize=True)
def textblob_sentiment(input):
    scores = TextBlob(input.summary)
    input.polarity = scores.sentiment.polarity
    input.subjectivity = scores.sentiment.subjectivity
    return input

@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(input):

    with open('polarity_score.tsv', 'a+') as f:
        f.write(f"{input.label}, {input.polarity}\n")
    return 0 if input.polarity < 0.2 else -1

@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(input):
    with open('subjectivity_score.tsv', 'a+') as f:
        f.write(f"{input.label}, {input.subjectivity}\n")
    return 0 if input.subjectivity < 0.33 else -1



from data_path import joint_lda_result_path, clamp_n2c2_prefix
sys.path.insert(1, '../clamp')
from import_clamp_result import get_df_list, getdrug, getdisease
@labeling_function()
def top_drug_disease_same_topic_lf(input):
    patient=input.patient
    clamp_df=get_df_list(join(clamp_n2c2_prefix,'%s.txt'%patient))
    # drug_df=getdrug(clamp_df,True).groupby()
    disease_df=getdisease(clamp_df,True)
    print(clamp_df)
    # print(drug_df)
    print(disease_df)
    quit()
