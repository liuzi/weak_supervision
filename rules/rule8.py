

####################################
### RULE 8:  lf_keyword_triggers ###
####################################

'''
keyword_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from EDA 
MATCHING: any trigger word in keyword_triggers found in discharge summary
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
keyword_triggers_raw = pd.read_csv("../rules/keywords/keyword_triggers.csv")
keyword_triggers = list(keyword_triggers_raw['keyword_triggers'])

@labeling_function()
# MATCHING: any trigger word in keyword_triggers found in discharge summary
def lf_keyword_triggers(x) :
    if any(word in x.summary.lower() for word in keyword_triggers) :
        return MATCHING
    else :
        return ABSTAIN

