
###################################
### RULE 5:  lf_sider2_triggers ###
###################################

'''
sider2_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from SIDER2
MATCHING: any pair of trigger words in sider2_triggers found in discharge summary
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
sider2_triggers_raw = pd.read_csv("../rules/keywords/sider2_triggers.csv")
sider2_triggers = list(sider2_triggers_raw['sider2_triggers'])

@labeling_function()
# MATCHING: any pair of trigger words in sider2_triggers found in discharge summary
def lf_sider2_triggers(x) :
    found = 0
    for j in range(0, len(sider2_triggers)) :
        arg1 = sider2_triggers[j][0]
        arg2 = sider2_triggers[j][1]
        if (arg1 in x.summary.lower()) and (arg2 in x.summary.lower()) :
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING
