
######################################################
### RULE 4:  lf_ade_drug_pair_lem_keyword_triggers ###
######################################################

'''
ade_drug_pair_lem: contains pairs of phrases from ADE-Drug in annotated files from N2C2 where keywords are cleaned using lemmatisation // (eg) penicillins --> penicillin // note: lemmatised keywords will be added as a new keyword phrase so discharge summaries dont have to be cleaned
keyword_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from EDA 
MATCHING: any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary
'''

################
### packages ###
################
import pandas as pd
import snorkel
from snorkel.labeling import labeling_function

# LF outputs for inv-trans matching
MATCHING = 1
NOT_MATCHING = 0
ABSTAIN = -1

# get keywords
ade_drug_pair_lem_raw = pd.read_csv("../rules/keywords/ade_drug_pair_lem.csv")
ade_drug_pair_lem = list(ade_drug_pair_lem_raw['ade_drug_pair_lem'])
keyword_triggers_raw = pd.read_csv("../rules/keywords/keyword_triggers.csv")
keyword_triggers = list(keyword_triggers_raw['keyword_triggers'])

@labeling_function()
# MATCHING: any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary
def lf_ade_drug_pair_lem_keyword_triggers(x) :
    found = 0
    for j in range(0, len(ade_drug_pair_lem)) :
        arg1 = ade_drug_pair_lem[j][0]
        arg2 = ade_drug_pair_lem[j][1]
        if (arg1 in x.summary.lower()) and (arg2 in x.summary.lower()) :
            found = 1
        elif any(word in x.summary.lower() for word in keyword_triggers) :
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING
