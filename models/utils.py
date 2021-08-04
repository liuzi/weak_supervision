# for discharge summary dataset preparation
import sklearn
from sklearn import datasets
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn import metrics


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


def get_metric_funcs_list():

    funcs=[metrics.accuracy_score, \
        metrics.precision_score, 
        metrics.recall_score,
        metrics.roc_auc_score,
        metrics.f1_score,
        metrics.confusion_matrix]
    
    metrics_names=[
        "accuracy", "precision", "recall", 
        "ROC AUC", "f1 score", "confusion matrix\n"
    ]
    return funcs, metrics_names

