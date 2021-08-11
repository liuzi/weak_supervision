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
    
get_n2c2_ade_drug_pair()

