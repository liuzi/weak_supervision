import pandas as pd
import os
# import errno
import json
import shutil
# from pathlib import Path
from csv import writer


def csv_suffix(file_path,suffix = '.csv'):
    if('.' in file_path):
        final_file_path=file_path
    else:
        final_file_path = file_path+suffix
    return final_file_path

def append_csv_bydf(df, file_name, sep="\t"):
    
    # Open file in append mode
    with open(csv_suffix(file_name), 'a') as write_obj:
        df.to_csv(write_obj, mode='a', header=write_obj.tell()==0,sep=sep,index=False)
        print("Insert %d rows into file %s"%(
            df.shape[0],file_name))

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s already exists" % path)
    else:
        print ("Successfully create the directory %s" % path)

def create_folder_overwrite(file_path):
    if os.path.exists(file_path):
        print("%s already exists. Clear and re-create this folder."%file_path)
        shutil.rmtree(file_path)
    os.makedirs(file_path)

def copytree(src, dst, symlinks=False, ignore=None):
    create_folder_overwrite(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)