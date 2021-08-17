from os.path import join
import pandas as pd

import sys
sys.path.insert(0, "../utils")
from data_path import n2c2_data_prefix
from tools import write2file

def get_n2c2_ade_drug_pair():
    data_fiiename="n2c2_ade_drug"
    n2c2_adedrug_path=join(n2c2_data_prefix,"%s.csv"%data_fiiename)
    n2c2_adedrug=pd.read_csv(n2c2_adedrug_path,sep='\t')
    n2c2_adedrug_pair = n2c2_adedrug[["ADE","DRUG"]].applymap(lambda s: s.lower()).drop_duplicates()
        # lambda s: s.lower() if type(s) == str else s)
    write2file(n2c2_adedrug_pair,f"keywords/{data_fiiename}_pair.tsv",sep="\t")
    
def get_triggrs(keyword_triggers, data_filename):
    keyword_triggers_df=pd.DataFrame({'keyword_triggers':keyword_triggers})
    write2file(keyword_triggers_df,f"keywords/{data_filename}_trigger.tsv",sep="\t")

n2c2_triggers = ['drug reaction', 'allergy', 'reaction', \
    'rash', 'drug fever', 'allergic reaction', 
    'anaphylactic reaction', 'anaphylaxis', 
    'toxicity', 'steroid psychosis', 'hives']

paper_triggers = ['adverse to', 'after starting', 'after taking', 'after', \
    'allergic', 'allergies', 'allergy', 'associate', 'associated', 'attribute to', 
    'attributed to', 'cause', 'caused by', 'caused', 'cessation of', 'change to', 
    'changed to', 'controlled with', 'converted to', 'da to', 'develop from', 
    'developed from', 'developed', 'develops', 'discontinue', 'discontinued', 
    'drug allergic', 'drug allergy', 'drug induced', 'drug-induced', 'due to', 'due', 
    'following', 'held off in view of', 'held off', 'hypersensitivity', 'improved with', 
    'increasing dose', 'induced', 'interrupt', 'likely', 'not continued', 'not to start', 'post', 
    'reduce', 'reduced', 'related', 'sec to', 'secondary to', 'secondary', 'side effect', 'stop', 
    'stopped', 'stopping', 'subsequently developed', 'switch', 'switch to', 'switches to', 'switched', 
    'switched to', 'take off', 'taken off', 'took off', 'treated with']


# get_n2c2_ade_drug_pair()
# get_triggrs(n2c2_triggers, data_fiiename='n2c2_manual')
get_triggrs(paper_triggers, data_filename='paper')


