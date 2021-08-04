
##########################################
### RULE 10:  lf_paper_triggers_200char ###
##########################################

'''
paper_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from paper (Tang et al., 2019)
ade_drug_single: contains individual phrases from ADE-Drug in annotated files from N2C2
MATCHING: any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary
'''

################
### packages ###
################
import pandas as pd
import re
import snorkel
from snorkel.labeling import labeling_function

# LF outputs for inv-trans matching
MATCHING = 1
NOT_MATCHING = 0
ABSTAIN = -1

# get keywords
paper_triggers_raw = pd.read_csv("../rules/keywords/paper_triggers.csv")
paper_triggers = list(paper_triggers_raw['paper_triggers'])
ade_drug_single_raw = pd.read_csv("../rules/keywords/ade_drug_single.csv")
ade_drug_single = list(ade_drug_single_raw['ade_drug_single'])

@labeling_function()
# MATCHING: any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary
def lf_paper_triggers_200char(x) :
    # set distance
    dist = 200
    found = 0
    for i in range(0, len(ade_drug_single)) :
        keyword = ade_drug_single[i][0]
        pos = [m.start() for m in re.finditer(keyword, x.summary.lower())]
        start_pos = 0
        end_pos = 0
        # if any of the papers triggers words are found within [-200, +200] of the drug in discharge summary
        for j in range(0, len(pos)) :
            start_pos = pos[0] - dist
            end_pos = pos[0] + len(keyword) + dist
            if start_pos < 0 :
                start_pos = 0
        if any(word in x.summary.lower()[start_pos:end_pos] for word in paper_triggers) :
            # print(x.summary.lower()[start_pos:end_pos])
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING
