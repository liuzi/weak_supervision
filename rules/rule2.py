
#################################
### RULE 2:  lf_ade_drug_pair ###
#################################

'''
ade_drug_pair: contains pairs of phrases from ADE-Drug in annotated files from N2C2
MATCHING: any pair of keywords in ade_drug_pair found in discharge summary
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
ade_drug_pair_raw = pd.read_csv("../rules/keywords/ade_drug_pair.csv")
ade_drug_pair = list(ade_drug_pair_raw['ade_drug_pair'])

@labeling_function()
# MATCHING: any pair of keywords in ade_drug_pair found in discharge summary
def lf_ade_drug_pair(x) :
    found = 0
    for j in range(0, len(ade_drug_pair)) :
        arg1 = ade_drug_pair[j][0]
        arg2 = ade_drug_pair[j][1]
        if (arg1 in x.summary.lower()) and (arg2 in x.summary.lower()) :
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING
