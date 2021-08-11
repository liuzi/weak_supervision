# for discharge summary dataset preparation
from sklearn import metrics


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

