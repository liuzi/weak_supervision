
################
### packages ###
################
import os
from os import listdir
from os.path import isfile, join
import csv
import re
import pandas as pd


################
### keywords ###
################
'''
1. ade_drug_single: contains individual phrases from ADE-Drug in annotated files from N2C2
2. ade_drug_pair: contains pairs of phrases from ADE-Drug in annotated files from N2C2
3. ade_drug_pair_lem: contains pairs of phrases from ADE-Drug in annotated files from N2C2 where keywords are cleaned using lemmatisation // (eg) penicillins --> penicillin // note: lemmatised keywords will be added as a new keyword phrase so discharge summaries dont have to be cleaned
4. sider2_triggers: known pairs of drug and side effects from SIDER2
5. semmeddb_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from SemMedDB // note: SemMedDB's CUI used to match mentions in discharge summaries
6. keyword_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from EDA 
7. paper_triggers: keywords that indicate an almost confirmed presence of ADE-Drug from paper (Tang et al., 2019)
8. negate: words that indicate negation
'''

ade_drug_single = []
ade_drug_pair = []
ade_drug_pair_lem = []
sider2_triggers = []
semmeddb_triggers = []
keyword_triggers = ['drug reaction', 'allergy', 'reaction', 'rash', 'drug fever', 'allergic reaction', 'anaphylactic reaction', 'anaphylaxis', 'toxicity', 'steroid psychosis', 'hives']
paper_triggers = ['adverse to', 'after starting', 'after taking', 'after', 'allergic', 'allergies', 'allergy', 'associate', 'associated', 'attribute to', 'attributed to', 'cause', 'caused by', 'caused', 'cessation of', 'change to', 'changed to', 'controlled with', 'converted to', 'da to', 'develop from', 'developed from', 'developed', 'develops', 'discontinue', 'discontinued', 'drug allergic', 'drug allergy', 'drug induced', 'drug-induced', 'due to', 'due', 'following', 'held off in view of', 'held off', 'hypersensitivity', 'improved with', 'increasing dose', 'induced', 'interrupt', 'likely', 'not continued', 'not to start', 'post', 'reduce', 'reduced', 'related', 'sec to', 'secondary to', 'secondary', 'side effect', 'stop', 'stopped', 'stopping', 'subsequently developed', 'switch', 'switch to', 'switches to', 'switched', 'switched to', 'take off', 'taken off', 'took off', 'treated with']
negate = ['no', 'not', 'did not', 'however', 'but', 'despite', 'without']

##########################
### 1. ade_drug_single ###
##########################

print("started: ade_drug_single")

for file in os.listdir("../../N2C2/all_ann") :
    with open(join("../../N2C2/all_ann", file), 'r', encoding='utf-8', errors='ignore') as document_anno_file:
        if file != '.DS_Store' :
            lines = document_anno_file.readlines()
            entity_index = []
            entity_type = []
            entity_desc = []
            rs_index = []
            rs_type = []
            rs_arg1 = []
            rs_arg2 = []

            for line in lines :
                row = re.split('\t', line)
                if len(row) == 3 :
                    entity_index.append(re.split(' ', row[0])[0])
                    entity_type.append(re.split(' ', row[1])[0])
                    if type(row[2]) == str :
                        entity_desc.append(row[2][:-1])
                    else :
                        entity_desc.append(re.split(' ', row[2])[0][:-1])
                elif len(row) == 2 :
                    rs_index.append(row[0])
                    rs_type.append(re.split(' ', row[1])[0])
                    rs_arg1.append(re.split(' ', row[1])[1][5:])
                    rs_arg2.append(re.split(' ', row[1])[2][5:][:-1])

            # make new dataframes
            entities = pd.DataFrame({'index': entity_index, 
                                     'type': entity_type,
                                     'desc': entity_desc})
            relationships = pd.DataFrame({'index': rs_index,
                                          'type': rs_type,
                                          'arg1': rs_arg1,
                                          'arg2': rs_arg2})

            # set new index
            entities = entities.set_index('index')

            # find only ADE-Drug
            ade_drug = relationships.loc[relationships['type'] == 'ADE-Drug']
            ade_drug = ade_drug.reset_index()
            ade_drug = ade_drug.drop(columns=['level_0'])

            # map the arguments and get the ADE-Drug descriptions
            # add them into ade_drug_single 

            for i in range(0, len(ade_drug)) :
                arg1_index = ade_drug['arg1'][i]
                arg2_index = ade_drug['arg2'][i]
                arg1 = entities.loc[arg1_index]['desc']
                arg2 = entities.loc[arg2_index]['desc']
                if arg1 not in ade_drug_single :
                    # print(file, " ", arg1.lower())
                    ade_drug_single.append(arg1.lower())
                if arg2 not in ade_drug_single :
                    # print(file, " ", arg2.lower())
                    ade_drug_single.append(arg2.lower())

# make new dataframe to save as csv
ade_drug_single_df = pd.DataFrame({'ade_drug_single': ade_drug_single})
ade_drug_single_df.to_csv("ade_drug_single.csv", index=False)

print("finished: ade_drug_single")

########################
### 2. ade_drug_pair ###
########################

print("started: ade_drug_pair")

for file in os.listdir("../../N2C2/all_ann") :
    with open(join("../../N2C2/all_ann", file), 'r', encoding='utf-8', errors='ignore') as document_anno_file:
        if file != '.DS_Store' :
            lines = document_anno_file.readlines()
            entity_index = []
            entity_type = []
            entity_desc = []
            rs_index = []
            rs_type = []
            rs_arg1 = []
            rs_arg2 = []

            for line in lines :
                row = re.split('\t', line)
                if len(row) == 3 :
                    entity_index.append(re.split(' ', row[0])[0])
                    entity_type.append(re.split(' ', row[1])[0])
                    if type(row[2]) == str :
                        entity_desc.append(row[2][:-1])
                    else :
                        entity_desc.append(re.split(' ', row[2])[0][:-1])
                elif len(row) == 2 :
                    rs_index.append(row[0])
                    rs_type.append(re.split(' ', row[1])[0])
                    rs_arg1.append(re.split(' ', row[1])[1][5:])
                    rs_arg2.append(re.split(' ', row[1])[2][5:][:-1])

            # make new dataframes
            entities = pd.DataFrame({'index': entity_index, 
                                     'type': entity_type,
                                     'desc': entity_desc})
            relationships = pd.DataFrame({'index': rs_index,
                                          'type': rs_type,
                                          'arg1': rs_arg1,
                                          'arg2': rs_arg2})

            # set new index
            entities = entities.set_index('index')

            # find only ADE-Drug
            ade_drug = relationships.loc[relationships['type'] == 'ADE-Drug']
            ade_drug = ade_drug.reset_index(drop=True)

            # map the arguments and get the ADE-Drug descriptions
            # add them into ade_drug_pair 
            for i in range(0, len(ade_drug)) :
                arg1_index = ade_drug['arg1'][i]
                arg2_index = ade_drug['arg2'][i]
                arg1 = entities.loc[arg1_index]['desc']
                arg2 = entities.loc[arg2_index]['desc']
                # print(file, " ", arg1.lower(), " & ", arg2.lower())
                ade_drug_pair.append([arg1.lower(), arg2.lower()])

# make new dataframe to save as csv
ade_drug_pair_df = pd.DataFrame({'ade_drug_pair': ade_drug_pair})
ade_drug_pair_df.to_csv("ade_drug_pair.csv", index=False)

print("finished: ade_drug_pair")

############################
### 3. ade_drug_pair_lem ###
############################

print("started: ade_drug_pair_lem")

mapping = pd.read_csv("../../data/keywords_mapping.csv")
temp = mapping.set_index('original')
temp1 = temp.to_dict()
mapping_dict = temp1['replace']

for file in os.listdir("../../N2C2/all_ann") :
    with open(join("../../N2C2/all_ann", file), 'r', encoding='utf-8', errors='ignore') as document_anno_file:
        if file != '.DS_Store' :
            lines = document_anno_file.readlines()
            entity_index = []
            entity_type = []
            entity_desc = []
            rs_index = []
            rs_type = []
            rs_arg1 = []
            rs_arg2 = []

            for line in lines :
                row = re.split('\t', line)
                if len(row) == 3 :
                    entity_index.append(re.split(' ', row[0])[0])
                    entity_type.append(re.split(' ', row[1])[0])
                    if type(row[2]) == str :
                        entity_desc.append(row[2][:-1])
                    else :
                        entity_desc.append(re.split(' ', row[2])[0][:-1])
                elif len(row) == 2 :
                    rs_index.append(row[0])
                    rs_type.append(re.split(' ', row[1])[0])
                    rs_arg1.append(re.split(' ', row[1])[1][5:])
                    rs_arg2.append(re.split(' ', row[1])[2][5:][:-1])

            # make new dataframes
            entities = pd.DataFrame({'index': entity_index, 
                                     'type': entity_type,
                                     'desc': entity_desc})
            relationships = pd.DataFrame({'index': rs_index,
                                          'type': rs_type,
                                          'arg1': rs_arg1,
                                          'arg2': rs_arg2})

            # set new index
            entities = entities.set_index('index')

            # find only ADE-Drug
            ade_drug = relationships.loc[relationships['type'] == 'ADE-Drug']
            ade_drug = ade_drug.reset_index(drop=True)

            # map the arguments and get the ADE-Drug descriptions
            # add them into ade_drug_pair_lem 

            for i in range(0, len(ade_drug)) :
                arg1_index = ade_drug['arg1'][i]
                arg2_index = ade_drug['arg2'][i]
                arg1 = entities.loc[arg1_index]['desc']
                arg2 = entities.loc[arg2_index]['desc']
                ade_drug_pair_lem.append([arg1.lower(), arg2.lower()])
                # print(file, " ", arg1.lower(), " & ", arg2.lower())
                if arg1 in mapping_dict :
                    arg1 = mapping_dict[arg1]
                    ade_drug_pair_lem.append([arg1.lower(), arg2.lower()])
                    # print(file, " ", arg1.lower(), " & ", arg2.lower())
                elif arg2 in mapping_dict :
                    arg2 = mapping_dict[arg2]
                    ade_drug_pair_lem.append([arg1.lower(), arg2.lower()])
                    # print(file, " ", arg1.lower(), " & ", arg2.lower())
                elif (arg1 in mapping) and (arg2 in mapping) :
                    arg1 = mapping_dict[arg1]
                    arg2 = mapping_dict[arg2]
                    ade_drug_pair_lem.append([arg1.lower(), arg2.lower()])
                    # print(file, " ", arg1.lower(), " & ", arg2.lower())

# make new dataframe to save as csv
ade_drug_pair_lem_df = pd.DataFrame({'ade_drug_pair_lem': ade_drug_pair_lem})
ade_drug_pair_lem_df.to_csv("ade_drug_pair_lem.csv", index=False)

print("finished: ade_drug_pair_lem")

##########################
### 4. sider2_triggers ###
##########################

print("started: sider2_triggers")

SIDER2 = pd.read_csv("../../data/known_ades_SIDER2.csv")

for i in range(0, len(SIDER2)) :
    arg1 = SIDER2['drug_name'][i]
    arg2 = SIDER2['se_name'][i]
    # print(arg1.lower(), " & ", arg2.lower())
    sider2_triggers.append([arg1.lower(), arg2.lower()])

# make new dataframe to save as csv
sider2_triggers_df = pd.DataFrame({'sider2_triggers': sider2_triggers})
sider2_triggers_df.to_csv("sider2_triggers.csv", index=False)

print("ended: sider2_triggers")

############################
### 5. semmeddb_triggers ###
############################

print("started: semmeddb_triggers")

semmeddb_predications = pd.read_csv("../../data/PREDICATIONS_OCCURS.csv")
predications_cleaned = semmeddb_predications.drop_duplicates()
predications_cleaned = predications_cleaned.reset_index(drop=True)

# semantic types: https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt
# target: subject semtype == drugs
predictions_clnd = predications_cleaned.loc[predications_cleaned['SUBJECT_SEMTYPE']=='clnd']
# target: predicate == causes/complicates/affects/interacts_with 
target = predictions_clnd.loc[(predictions_clnd['PREDICATE']=='CAUSES') | (predictions_clnd['PREDICATE']=='COMPLICATES') | (predictions_clnd['PREDICATE']=='AFFECTS') | (predictions_clnd['PREDICATE']=='INTERACTS_WITH')]
target = target.reset_index(drop=True)
# target with subject semtype == 'clnd' only has predicate == causes/affects/interacts_with

for i in range(0, len(target)) :
    arg1 = target['SUBJECT_NAME'][i]
    arg2 = target['OBJECT_NAME'][i]
    # print(arg1.lower(), " & ", arg2.lower())
    semmeddb_triggers.append([arg1.lower(), arg2.lower()])

# make new dataframe to save as csv
semmeddb_triggers_df = pd.DataFrame({'semmeddb_triggers': semmeddb_triggers})
semmeddb_triggers_df.to_csv("semmeddb_triggers.csv", index=False)

print("finished: semmeddb_triggers")

###########################
### 6. keyword_triggers ###
###########################

print("started: keyword_triggers")

# make new dataframe to save as csv
keyword_triggers_df = pd.DataFrame({'keyword_triggers': keyword_triggers})
keyword_triggers_df.to_csv("keyword_triggers.csv", index=False)

print("finished: keyword_triggers")

#########################
### 7. paper_triggers ###
#########################

print("started: paper_triggers")

# make new dataframe to save as csv
paper_triggers_df = pd.DataFrame({'paper_triggers': paper_triggers})
paper_triggers_df.to_csv("paper_triggers.csv", index=False)

print("finished: paper_triggers")

#################
### 8. negate ###
#################

print("started: negate")

# make new dataframe to save as csv
negate_df = pd.DataFrame({'negate': negate})
negate_df.to_csv("negate.csv", index=False)

print("finished: negate")