

#####################################
### RULE 7:  lf_semmeddb_triggers ###
#####################################

'''
semmeddb_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from SemMedDB // note: SemMedDB's CUI used to match mentions in discharge summaries
MATCHING: any pair of trigger words in semmeddb_triggers found in discharge summary
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
semmeddb_triggers_raw = pd.read_csv("../rules/keywords/semmeddb_triggers.csv")
semmeddb_triggers = list(semmeddb_triggers_raw['semmeddb_triggers'])

@labeling_function()
# MATCHING: any pair of trigger words in semmeddb_triggers found in discharge summary
def lf_semmeddb_triggers(x) :
    found = 0
    for j in range(0, len(semmeddb_triggers)) :
        arg1 = semmeddb_triggers[j][0]
        arg2 = semmeddb_triggers[j][1]
        if (arg1 in x.summary.lower()) and (arg2 in x.summary.lower()) :
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING

