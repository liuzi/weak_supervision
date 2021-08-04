
###################################
### RULE 1:  lf_ade_drug_single ###
###################################

'''
ade_drug_single: contains individual phrases from ADE-Drug in annotated files from N2C2
MATCHING: any keywords in ade_drug_single found in discharge summary
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
ade_drug_single_raw = pd.read_csv("../rules/keywords/ade_drug_single.csv")
ade_drug_single = list(ade_drug_single_raw['ade_drug_single'])

@labeling_function()
# MATCHING: any keywords in ade_drug_single found in discharge summary
def lf_ade_drug_single(x) :
    return MATCHING if any(word in x.summary.lower() for word in ade_drug_single) else ABSTAIN
