

###########################################
### RULE 12:  lf_paper_triggers_25words ###
###########################################

'''
paper_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from paper (Tang et al., 2019)
ade_drug_single: contains individual phrases from ADE-Drug in annotated files from N2C2
MATCHING: any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary
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
paper_triggers_raw = pd.read_csv("../rules/keywords/paper_triggers.csv")
paper_triggers = list(paper_triggers_raw['paper_triggers'])
ade_drug_single_raw = pd.read_csv("../rules/keywords/ade_drug_single.csv")
ade_drug_single = list(ade_drug_single_raw['ade_drug_single'])

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
# MATCHING: any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary
def lf_paper_triggers_25words(x) :
    # set distance (words)
    dist = 25
    found = 0
    # tokenize the discharge summary and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokenized = tokenizer.tokenize(x.summary.lower())
    for i in range(0, len(ade_drug_single)) :
        keyword = ade_drug_single[i]
        pos = get_index_positions(tokenized, keyword)
        start_pos = 0
        end_pos = 0
        # if any of the papers triggers words are found within [-25, +25] words of the keyword in discharge summary
        for j in range(0, len(pos)) :
            start_pos = pos[0] - dist
            end_pos = pos[0] + dist
            if start_pos < 0 :
                start_pos = 0
            if end_pos >= len(pos) :
                end_pos = len(pos)-1
        if any(word in tokenized[start_pos:end_pos] for word in paper_triggers) :
            # print(x.summary.lower()[start_pos:end_pos])
            # print(tokenized[start_pos:end_pos])
            found = 1
    if found == 0 :
        return ABSTAIN
    else :
        return MATCHING

