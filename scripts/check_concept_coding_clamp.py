import pandas as pd
import sys

from pandas.core.algorithms import unique
sys.path.insert(1, '../utils')
from os.path import join
from tools import read_data
from data_path import concat_clamp_prefix, singledrug_featurepreprocess_prefix

for clampfile, mimicfile, newfile, itemid in zip(
    ["DISSUM_ALL_Diseases", "DISSUM_ALL_Drugs"],
    ["diag_matrix.csv", "pres_rxnorm_matrix.csv"],
    ["allepis_newCUI.csv","allepis_newRxNorm.csv"],
    ["CUI","RxNorm"]
):
    dissum_all_item_df=read_data(join(concat_clamp_prefix, clampfile),dtype=str)

    print("#"*60,"\nCLAMP_RESULT_ON_DISCHAREGE_SUMMARIES: %s\n"%clampfile,"#"*60)
    print("Number of patients: %d"%(len(dissum_all_item_df["SUBJECT_ID"].unique())))
    print("Number of episodes: %d"%(len(dissum_all_item_df["HADM_ID"].unique())))
    unique_clamp_items=dissum_all_item_df[itemid].unique()
    print("Number of %s: %d"%(itemid, len(unique_clamp_items)))
    print("Number of records: %d"%len(dissum_all_item_df))

    for matched_matrix_file in [mimicfile, newfile]:
        print(f"\n{itemid} codes from {matched_matrix_file}")
        matched_item_cols=pd.read_csv(
            join(singledrug_featurepreprocess_prefix,matched_matrix_file),
            nrows=0).columns[1:]
        print(f"Number of {itemid}: {len(matched_item_cols)}")
        intersection_items=set(unique_clamp_items).intersection(set(matched_item_cols))
        print(f"Number of Intersection of {itemid}, \
            the percentage on {matched_matrix_file}: {len(intersection_items)/len(matched_item_cols):2.2%}")

