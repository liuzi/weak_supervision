import argparse
import pandas as pd
import glob
import sys

sys.path.insert(1, '../utils')
from sub_path import labelfunction_dict_dir


'''
    Get dict of index and labelfunction names from folder scripts/lf_dict
'''
def get_lf_dict():
    sub_lf_df_list = [pd.read_csv(file_name) for file_name in \
        glob.glob(f'{labelfunction_dict_dir}/*lf.csv')]
    lf_df = pd.concat(sub_lf_df_list,axis=0)
    lf_dict_desc = dict(zip(lf_df.iloc[:,0], lf_df.iloc[:,1]))
    return lf_dict_desc

def get_lfdict_and_parser():
    lf_dict_desc = get_lf_dict()
    help="\n\tFORMAT: use , to separate models (eg. c1,r6)"
    help=help+('').join([f'\n\t{key}: {value}' for key, value in lf_dict_desc.items()])

    parser = argparse.ArgumentParser(description='', \
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('label_function', type=str, help=help)

    return lf_dict_desc, parser

## label_function



