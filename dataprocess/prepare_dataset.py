import os
from os.path import join
import re
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
# import nltk
## solve Resource stopwords not found. problem
# nltk.download('stopwords')
# from nltk.stem import WordNetLemmatizer  
# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import sys
sys.path.insert(1,'../utils')
from utils import *
from sub_path import data_path

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
stemmer = SnowballStemmer("english")



def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


### path of folder with ### 
# train_txt: folder with discharge summaries for train dataset
# train_ann: folder with annotated files (derived from discharge summaries) for train dataset
# test_txt: folder with discharge summaries for test dataset
# test_ann: folder with annotated files (derived from discharge summaries) for test dataset
# MODIFIED: restrict filename to be ended with ".ann"
def count_ade_drug_fromAnn(data_name):
    labelled = []
    labelled_dict = {}
    ade_drug_counts = []

    for file in [f for f in os.listdir(join(data_path,data_name)) if ".ann" in f]:
        with open(join(data_path, data_name, file), 'rb') as document_anno_file:
            lines = document_anno_file.readlines()
            patient = file[:-4]
            boolean = False
            ade_drug_count = 0

            for line in lines :
                # MODIFIED: "ADE" -> "ADE-Drug"
                if b"ADE-Drug" in line:
                    boolean = True
                    ade_drug_count += 1
            # label which has ADE-Drug
            if boolean:
                labelled.append(1)
                labelled_dict[patient] = 1
            else :
                labelled.append(0)
                labelled_dict[patient] = 0
            ade_drug_counts.append(ade_drug_count)

    print("Number of files processed in folder %s: %d"%(
        data_name, sum(list(map(len,[labelled, labelled_dict, ade_drug_counts])))/3)
    )
    return labelled, labelled_dict, ade_drug_counts


# directly create csv file
def prepare_dataset(train_labelled_dict, test_labelled_dict, \
    data_folder_l=["train_txt", "test_txt"]):
    data_df_l=[]
    prepared_data_path=join(data_path,"dataframe")
    create_folder(prepared_data_path)

    for data_folder, labelled_dict in zip(
        data_folder_l, [train_labelled_dict, test_labelled_dict]
    ):
        curr_dir = join(data_path,data_folder)
        patient_summary_label=[]
        for file in os.listdir(curr_dir):
            label=labelled_dict[file[:-4]]
            with open(join(curr_dir, file), 'r') as document_summary_file:
                summary = " ".join(line.strip() for line in document_summary_file)
            patient_summary_label.append([file[:-4], summary, label])
        data_df = pd.DataFrame(np.array(patient_summary_label),
            columns=["patient","summary","label"])
        data_df_l.append(data_df)
        data_df.to_csv(join(prepared_data_path,"%s.csv"%data_folder), index=False)
        
    return data_df_l    

def tf_idf_trasform(trainData, testData, pipeline_name = '../dataprocess/feature_extraction_pipeline.pkl'):
    '''
        transform traindata and testdata for model: Feature Extraction from text
    '''
    # pipeline_name = 'feature_extraction_pipeline.pkl'
    if(not os.path.exists(pipeline_name)):
        feature_extraction_pipe = make_pipeline(
            CountVectorizer(binary=True),
            TfidfTransformer(use_idf=True),
        )
        tfidf_train_data = feature_extraction_pipe.fit_transform(trainData.summary) 
        tfidf_test_data = feature_extraction_pipe.transform(testData.summary)
        # save pipeline as pickle file
        with open(pipeline_name, 'wb') as file :
            pickle.dump(feature_extraction_pipe, file)
        print("pipeline for feature extraction from text is created")
    else:
        with open(pipeline_name, 'rb') as file:
            feature_extraction_pipe = pickle.load(file)
        tfidf_train_data, tfidf_test_data = list(
            map(feature_extraction_pipe.transform, [trainData.summary, testData.summary]))

    # train_labels, test_labels = trainData.label, testData.label
    return tfidf_train_data, tfidf_test_data, trainData.label, testData.label

def prepare_data_for_model(get_full_data=False,  pipeline_name = '../dataprocess/feature_extraction_pipeline.pkl'):
    '''
        prepare training data and test data
    '''
    ### preprocess datasets ###
    print("started preparing train and test datasets for training models")
    prepared_data_path=join(data_path,"dataframe")
    data_folder_l=["train_txt", "test_txt"]
    if(os.path.exists(prepared_data_path)):
        trainData, testData = [
            pd.read_csv(join(prepared_data_path, "%s.csv"%data_folder))
            for data_folder in data_folder_l]
    else:
        train_labelled, train_labelled_dict, \
            train_ade_drug_counts = count_ade_drug_fromAnn("train_ann") 
        test_labelled, test_labelled_dict, \
            test_ade_drug_counts = count_ade_drug_fromAnn("test_ann") 
        trainData, testData=prepare_dataset(train_labelled_dict, test_labelled_dict)
    '''
       [optional] clean training data and test data
    '''
    # FIXME: tag01 prepare data version 3
    cleaned_Data_l=[]
    for Data in [trainData, testData]:
        for func in [cleanHtml,cleanPunc,keepAlpha,removeStopWords]:
            Data["summary"]=Data["summary"].apply(func)
        cleaned_Data_l.append(Data)
    trainData, testData = cleaned_Data_l

    ## if only need the original data then directly return trainData testData
    if get_full_data:
        return trainData, testData
    else:
        return tf_idf_trasform(trainData, testData,pipeline_name)


from sklearn.model_selection import StratifiedKFold    
def get_data_for_cross_fold_test(n_splits=5):
    pipeline_prefix='../dataprocess/cv%d_feature_extraction_pipeline.pkl'
    trainData, testData = prepare_data_for_model(True)
    allData=pd.concat([trainData, testData],axis=0)
    scv=StratifiedKFold(n_splits, random_state=42, shuffle=True)

    X=allData['summary'].copy()
    y=allData['label'].copy()
    data_dict={}
    for (train_index, test_index), dict_id in zip(scv.split(X, y), list(range(n_splits))):
        train_data=pd.DataFrame({'summary':X.iloc[train_index],'label':y.iloc[train_index]})
        test_data=pd.DataFrame({'summary':X.iloc[test_index],'label':y.iloc[test_index]})

        pipeline_name=pipeline_prefix%dict_id
        tfidf_pack=tf_idf_trasform(train_data, test_data,pipeline_name)
        
        current_dataset={'train':train_data,'test':test_data,'tf_idf':tfidf_pack}
        data_dict['cv%d'%dict_id]=current_dataset
        # data_dictprint(dict_id)
    
    return data_dict
    # for train_index, test_index, dict_id in zip(scv.split(allData),:
    #     train_data_list.append(allData[train_index])
    #     test_data_list.append(allData[test_index])
    #     # X_train, X_test= X[train_index], X[test_index]
	#     # y_train, y_test= y[train_index], y[test_index]
    # print(train_data_list[0], train_data_list[1])
    # print(test_data_list[0],test_data_list[1])





