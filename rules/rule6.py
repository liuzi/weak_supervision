
###########################################
### RULE 6:  lf_sider2_triggers_25words ###
###########################################

'''
sider2_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from SIDER2
MATCHING: any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary
'''

################
### packages ###
################
import pandas as pd
import snorkel
from snorkel.labeling import labeling_function
from nltk.tokenize import RegexpTokenizer

# LF outputs for inv-trans matching
MATCHING = 1
NOT_MATCHING = 0
ABSTAIN = -1

# get keywords
sider2_triggers_raw = pd.read_csv("../rules/keywords/sider2_triggers.csv")
sider2_triggers = list(sider2_triggers_raw['sider2_triggers'])

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

@labeling_function()
# MATCHING: any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary
def lf_sider2_triggers_25words(x) :
    # set distance (words)
    dist = 25
    found = 0
    # tokenize the discharge summary and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokenized = tokenizer.tokenize(x.summary.lower())
    for i in range(0, len(sider2_triggers)) :
        arg1 = sider2_triggers[i][0]
        arg2 = sider2_triggers[i][1]
        if (arg1 in x.summary.lower()) and (arg2 in x.summary.lower()) :
            pos = get_index_positions(tokenized, arg1)
            start_pos = 0
            end_pos = 0
            # if any of the papers triggers words are found within [-25, +25] words of the keyword in discharge summary
            for j in range(0, len(pos)) :
                start_pos = pos[j] - dist
                end_pos = pos[j] + dist
                if start_pos < 0 :
                    start_pos = 0
                if end_pos >= len(pos) :
                    end_pos = len(pos)-1
            if arg2 in tokenized[start_pos:end_pos] :
                found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING
