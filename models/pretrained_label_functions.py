################
### packages ###
################

# for reading files
import pickle

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from os.path import join
# from snorkel.labeling.lf.nlp import nlp_labeling_function
# from nltk.tokenize import RegexpTokenizer
# from snorkel.labeling import PandasLFApplier
# from snorkel.labeling import LFAnalysis

feature_extraction_model_path= "../models/feature_extraction_pipeline.pkl"
'''
    model_args=[model_type, *args]
'''
def lf_model(x, model_args):
    model_path = join("../models/%s_pickles"%model_args[0],"%s.pkl"%("_".join(model_args)))
    with open(feature_extraction_model_path, 'rb') as file:
        fe = pickle.load(file)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    X = fe.transform([x.summary])
    score = model.predict(X)
    return score


def make_model_kernel_lf(model_args):
    return LabelingFunction(
        name="lf_model_%s"%("_".join(model_args)),
        f=lf_model,
        resources=dict(model_args=model_args),
    )

lf_model_svm_linear = make_model_kernel_lf(["svm" ,"linear"])
lf_model_svm_poly = make_model_kernel_lf(["svm" ,"poly"])
lf_model_svm_rbf = make_model_kernel_lf(["svm" ,"rbf"])
lf_model_svm_sigmoid = make_model_kernel_lf(["svm" ,"sigmoid"])
lf_model_rfc = make_model_kernel_lf(["rfc"])

