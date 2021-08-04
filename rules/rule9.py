
##################################
### RULE 9:  lf_paper_triggers ###
##################################

'''
paper_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from paper (Tang et al., 2019)
MATCHING: any trigger word in paper_triggers found in discharge summary
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
paper_triggers_raw = pd.read_csv("../rules/keywords/paper_triggers.csv")
paper_triggers = list(paper_triggers_raw['paper_triggers'])

@labeling_function()
# MATCHING: any trigger word in paper_triggers found in discharge summary
def lf_paper_triggers(x) :
    if any(word in x.summary.lower() for word in paper_triggers) :
        return MATCHING
    else :
        return ABSTAIN
