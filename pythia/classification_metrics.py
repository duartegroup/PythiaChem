#!/usr/bin/ent python

# parts of this code are inspired by Chemical Space Analysis and Property Prediction for Carbon Capture Amine Molecules.
#https://doi.org/10.1039/D3DD00073G
#https://zenodo.org/records/10213104
#https://github.com/flaviucipcigan/ccus_amine_prediction_workflow

# Standard libraries
import logging
import matplotlib.pyplot as plt
import pandas as pd

# NumPy
import numpy as np

# Joblib
import joblib

# Scikit-learn
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance

# Imbalanced-learn
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

# Deprecated package (handle with care)
try:
    import scikitplot as skplt
except ModuleNotFoundError:
    print("Scikit plot not available")

# Custom module
from . import plots as pltsk



def get_multi_label_confusion_matrix(df, predicted_column_name="prediction", known_column_name="known", labels=(0,1), return_dict=False):
    """
    Function to produce get the confusion matrix from a classification result
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param labels: tuple - the labels used for the classes
    :param return_dict: bool - return the confusion matrix as a dictionary with keys tp, tn, fp, fn rather than an array
    """

    log = logging.getLogger(__name__)

    multi_cm = sklearn.metrics.multilabel_confusion_matrix(df[known_column_name].values, df[predicted_column_name].values, labels=labels)
    log.debug(multi_cm)

    if return_dict is True:
        c_dict = {}
        for i in range(len(multi_cm)):
            c_dict[i] = confusion_matrix_to_dict(multi_cm[i])
        return c_dict
    else:
        return multi_cm

def get_confusion_matrix(df, predicted_column_name="prediction", known_column_name="known", labels=(0,1), return_dict=False):
    """
    Function to produce get the confusion matrix from a classification result
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param labels: tuple - the labels used for the classes
    :param return_dict: bool - return the confusion matrix as a dictionary with keys tp, tn, fp, fn rather than an array
    """

    log = logging.getLogger(__name__)

    c_matrix = confusion_matrix(df[known_column_name].values, df[predicted_column_name].values, labels=labels)
    
    log.debug(c_matrix)

    if return_dict is True:
        c_matrix = confusion_matrix_to_dict(c_matrix)

    return c_matrix


def confusion_matrix_to_dict(cm):
    """
    Convert confusion matrix to dict
    :param cm: confusion matrix from sklearn
    """
    log = logging.getLogger(__name__)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # 
    # TN | FP
    #----|----
    # FN | TP
    #

    cm_d = {"tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1]
           }

    return cm_d


def accuracy_percentage(df, prediction_column="prediction", known_column="known"):
    """
    Function to calculate the accuracy of a classification
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param prediction_column: str - the column name containing the predicted classes
    :param known_column: str - the column name containing the ground truth known classes
    """
    log = logging.getLogger(__name__)

    log.debug(df)

    df_tmp = df[df[prediction_column] == df[known_column]]
    acc = (len(df_tmp) / len(df.index)) * 100.0
    return acc


def g_mean(cm_d):
    """
    Function to calculate geometric mean
    :param cm_d: confusion matrix dictionary from get_confusion_matrix
    """

    sensitivity = tpr(cm_d)
    specificity = tnr(cm_d)

    g = np.sqrt(sensitivity * specificity)

    return g


def accuracy(cm):
    """
    Function to calculate the accuracy 
    :param cm: confusion matrix dataframe from sklearn or dictionary from get_confusion_matrix(return_dict=True)
    """
    if isinstance(cm, dict):
        ac = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
    else:
        cm_acc = cm.diagonal() / matrix.sum(axis=1)
        ac = cm_acc.diagonal()

    return ac


def tpr(cm_d):
    """
    Function to calculate the true postive rate 
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fn"])

def tnr(cm_d):
    """
    Function to calculate the true negative rate
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tn"] / (cm_d["tn"] + cm_d["fp"])

def fpr(cm_d):
    """
    Function to calculate the false postive rate
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["fp"] / (cm_d["fp"] + cm_d["tn"])

def fnr(cm_d):
    """
    Function to calculate the false negative rate also known as sensitivity
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["fn"] / (cm_d["fn"] + cm_d["tp"])

def precision(cm_d):
    """
    Function to calculate the precision
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fp"])

def recall(cm_d):
    """
    Function to calculate the recall
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fn"])

def generalized_f(cm_d, beta=1.0):
    """
    Function to calculate generalized f score
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    :param beta: float - coefficent to weight recall and precision trade off
    """
    beta2 = beta * beta

    return ((1 + beta2) * precision(cm_d) * recall(cm_d)) / ((precision(cm_d) * beta2) + recall(cm_d))

def matthews_correlation_coefficient(cm_d, ytest=None, ypredicted=None):
    """
    Function to calculate Matthew's correlation coefficient
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    :param ytest: iterable - to test the local function against sklean, this is the known ground truth values
    :param ypredicted: iterable - to test the local function against sklean, this is the predicted values
    """
    log = logging.getLogger(__name__)

    verbose = False

    if ytest is not None and ypredicted is not None:
        from  sklearn.metrics  import matthews_corrcoef
        mcc_sklearn = matthews_corrcoef(ytest, ypredicted)
        verbose = True
        log.info("Matthew's correlation coefficent from sklearn = {}".format(mcc_sklearn))

    mcc_function = (cm_d["tp"] * cm_d["tn"] - cm_d["fp"] * cm_d["fn"]) / np.sqrt((cm_d["tp"] + cm_d["fp"]) * (cm_d["tp"] + cm_d["fn"]) * (cm_d["tn"] + cm_d["fp"]) * (cm_d["tn"] + cm_d["fn"]))

    if verbose is True:
        log.info("Matthew's correlation coefficent from function = {}".format(mcc_function))

    return mcc_function


def auc(metric1, metric2):
    """
    Function to calculate the area under the curve
    :param metric1: iterable - x axis metric values as an iterble
    :param metric2: iterable - y axis metric values as an iterble
    """

    return sklearn.metrics.auc(metric1, metric2)

def calculate_multi_label_confusion_based_metrics(cmtx=None, df=None, predicted_column_name="prediction", known_column_name="known", 
        probabilities_column_name=None, probabilities=None, labels=(0,1), positive_label=1, imbalanced=False, verbose=False, plt_filename=None):
    """
    :param cmtx: np array or dict - confusion matrix from sklearn or confusion_matrix_to_dict()
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param probabilities_column_name: str - the column name containing a classes predicted probabilities (sklearn predict_proba() output)
    :param probabilities: np array - class probability prediction from sklearn predict_proba() output
    :param labels: tuple - the labels used for the classes
    :param positive_label: str/int/float - label applied for the positive class
    :param imbalanced: bool True/False - use the imbalanced learn scorers where possible rather than normal sklearn
    :param verbose: bool True/False - more log output than normal
    """

    log = logging.getLogger(__name__)

    if cmtx is not None:
        if isinstance(cmtx, dict):
            log.info("Using provided confusion matrix: {}".format(cmtx))
        else:
            log.info("Looks like provided confusion matrix is not a dictionary converting")

            cmtx = {}
            for i in range(len(multi_cm)):
                cmtx[i] = confusion_matrix_to_dict(multi_cm[i])
            log.info("converted: {}".format(cmtx))
    elif df is not None:
        if isinstance(df, pd.DataFrame):
            cmtx = get_multi_label_confusion_matrix(df, predicted_column_name=predicted_column_name, known_column_name=known_column_name, labels=labels, return_dict=True)
            pltsk.plot_metrics(df, predicted_column_name=predicted_column_name, known_column_name=known_column_name, probabilities=probabilities, 
                 positive_label=1, labels=labels, name=plt_filename)
        else:
            log.info("Looks like provided data is not a pandas dataframe - ERROR")
            return ()
    else:
        log.info("Neither confusion matrix or data given - ERROR")
        return ()
  
    cm_metrics = {}
    for k, v in cmtx.items():
        log.info("\n\n----- Calculating metrics for class {} -----\n".format(k))
        log.info("Confusion matrix: {}".format(v))
        cm_metrics[k] = calculate_confusion_based_metrics(cmtx=v, 
                                                          df=df, 
                                                          predicted_column_name=predicted_column_name, 
                                                          known_column_name=known_column_name, 
                                                          probabilities_column_name=probabilities_column_name, 
                                                          probabilities=probabilities, 
                                                          labels=labels, 
                                                          positive_label=positive_label, 
                                                          imbalanced=imbalanced,
                                                          verbose=verbose,
                                                          key=k)
    return cm_metrics



def calculate_confusion_based_metrics(cmtx=None, df=None, predicted_column_name="prediction", known_column_name="known", probabilities_column_name=None, 
        probabilities=None, labels=(0,1), positive_label=1, imbalanced=False, verbose=False, key=1, plt_filename=None, all_classes=True, get_roc_curve=True,
                                    get_pr_curve=True, vmin=None, vmax=None, col_map="Blues", annotate=True,title=None):
    """
    :param cmtx: np array or dict - confusion matrix from sklearn or confusion_matrix_to_dict()
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param probabilities_column_name: str - the column name containing a classes predicted probabilities (sklearn predict_proba() output)
    :param probabilities: np array - class probability prediction from sklearn predict_proba() output
    :param labels: tuple - the labels used for the classes
    :param positive_label: str/int/float - label applied for the positive class
    :param imbalanced: bool True/False - use the imbalanced learn scorers where possible rather than normal sklearn
    :param verbose: bool True/False - more log output than normal
    :param key: int - class key to use to get the support from sklearn classification report
    """

    log = logging.getLogger(__name__)

    log.info("Attempting to calculate confusion based metrics")

    if cmtx is not None:
        if isinstance(cmtx, dict):
            log.info("Using provided confusion matrix: {}".format(cmtx))
        else:
            log.info("Looks like provided confusion matrix is not a dictionary converting")
            cmtx = confusion_matrix_to_dict(cmtx)
            log.info("converted: {}".format(cmtx))
    elif df is not None:
        if isinstance(df, pd.DataFrame):
            cmtx = get_confusion_matrix(df, predicted_column_name=predicted_column_name, known_column_name=known_column_name, labels=labels, return_dict=True)
            log.info(cmtx)
            if plt_filename is not None:
                pltsk.plot_metrics(df,predicted_column_name = predicted_column_name, known_column_name =known_column_name, probabilities=probabilities,
                 positive_label=positive_label, labels=labels, name=plt_filename, all_classes=all_classes, roc_curve=get_roc_curve,pr_curve=get_pr_curve,
                                   col_map=col_map, annotate=annotate,vmin=vmin,vmax=vmax,title=title)
        else:
            log.info("Looks like provided data is not a pandas dataframe - ERROR")
            return ()
    else:
        log.info("Neither confusion matrix or data given - ERROR")
        return ()

    output_metrics = {}
    ac = accuracy(cmtx)
    output_metrics["accuracy"] = ac
    true_pos_rate = tpr(cmtx)
    output_metrics["tpr"] = true_pos_rate
    false_pos_rate = fpr(cmtx)
    output_metrics["fpr"] = false_pos_rate
    true_neg_rate = tnr(cmtx)
    output_metrics["tnr"] = true_neg_rate
    false_neg_rate = fnr(cmtx)
    output_metrics["fnr"] = false_neg_rate
    gmean = g_mean(cmtx)
    output_metrics['g-mean'] = gmean
    f_half = generalized_f(cmtx, beta=0.5)
    output_metrics["f_half"] = f_half
    f_one =  generalized_f(cmtx, beta=1.0)
    output_metrics["f1"] = f_one
    f_two =  generalized_f(cmtx, beta=2.0)
    output_metrics["f2"] = f_two
    mcc = matthews_correlation_coefficient(cmtx)
    output_metrics["matthews_correlation_coefficient"] = mcc
    precis = precision(cmtx)
    output_metrics["precision"] = precis
    rec = recall(cmtx)
    output_metrics["recall"] = rec

    if df is not None:
        if imbalanced is False:
            rep = sklearn.metrics.classification_report(df[known_column_name], df[predicted_column_name], output_dict=True, labels=labels)
            try:
                output_metrics["support"] = rep[str(float(key))]["support"]
            except KeyError:
                log.warning("Support cannot be gotten from classification report")
        else:
            rep = classification_report_imbalanced(df[known_column_name], df[predicted_column_name], labels=labels)
            output_metrics["imbalenced-str"] = rep

        
        if verbose is True:
            tmp_rep = sklearn.metrics.classification_report(df[known_column_name], df[predicted_column_name])
            log.info(tmp_rep)

    if probabilities_column_name is not None and df is not None:
        fposr, tposr, roc_thresholds = roc_curve(df[known_column_name].values,
                                         df[probabilities_column_name].values,
                                         pos_label=positive_label)
        if verbose is True:
            log.info("ROC curve fpr and tpr data:\n{}\n{}".format(fposr, tposr))
        output_metrics["tpr-roc"] = tposr
        output_metrics["fpr-roc"] = fposr
        output_metrics["roc_thresholds"] = roc_thresholds

        p, r, pr_thresholds = precision_recall_curve(df[known_column_name].values,
                                                         df[probabilities_column_name].values,
                                                         pos_label=positive_label)
        if verbose is True:
            log.info("Precision recall curve precision and recall data:\n{}\n{}".format(p, r))
        output_metrics["precision-pr"] = p
        output_metrics["recall-pr"] = r
        output_metrics["pr_thresholds"] = pr_thresholds

        #roc_auc = sklearn.metrics.roc_auc_score(df[known_column_name].values, df[probabilities_column_name].values)
        # problematic but general
        roc_auc = auc(fposr, tposr)
        output_metrics["roc_auc"] = roc_auc
        pr_auc = auc(r, p)
        output_metrics["pr_auc"] = pr_auc

    elif probabilities is not None:
        fposr, tposr, roc_thresholds = roc_curve(df[known_column_name].values,
                                         probabilities[:,1],
                                         pos_label=positive_label)
        if verbose is True:
            log.info("ROC curve fpr and tpr data:\n{}\n{}".format(fposr, tposr))
        output_metrics["tpr-roc"] = tposr
        output_metrics["fpr-roc"] = fposr
        output_metrics["roc_thresholds"] = roc_thresholds

        p, r, pr_thresholds = precision_recall_curve(df[known_column_name].values,
                                                         probabilities[:,1],
                                                         pos_label=positive_label)
        
        if verbose is True:
            log.info("Precision recall curve precision and recall data:\n{}\n{}".format(p, r))
        output_metrics["precision-pr"] = p
        output_metrics["recall-pr"] = r
        output_metrics["pr_thresholds"] = pr_thresholds

        #roc_auc = sklearn.metrics.roc_auc_score(df[known_column_name].values, df[probabilities_column_name].values)
        
        roc_auc = auc(fposr, tposr)
        output_metrics["roc_auc"] = roc_auc
        pr_auc = auc(r, p)
        output_metrics["pr_auc"] = pr_auc
    

    if verbose is True:
        log.info("{}".format("\n".join(["{}:\n{}".format(k, v) for k, v in output_metrics.items()])))

    return output_metrics




def calculate_permutation_importance(model_file, x, y, n_repeats=5, n_toplot =7, save=True, random_seed=None, filename = "test", figsize=(9,9), fontsize=30, color='lightgrey', error_color='darkgrey'):
    """
    Calculate the permutation importance of the features of a model.
    Parameters:
    model_file: the name of the model .sav file
    x: the features
    y: the target
    n_repeats: the number of times to repeat the permutation. Default is 5.
    n_toplot: the number of features to plot. Default is 7.
    save: whether to save the plot. Default is True.
    random_seed: the random seed. Default is None.

    Returns:
    df: the dataframe of feature importance   
    
    """
    from sklearn.inspection import permutation_importance
    
    model_name = model_file.split(".")[0].replace("model_", "").replace("_", " ")
    print(model_name)
    clf = joblib.load(model_file)
    importance = permutation_importance(clf, x, y, n_repeats=n_repeats, random_state=random_seed)
    

    features_names = x.columns
    features = []
    for a in features_names:
        if str(a).startswith("time_between_transporter_added_and_lysis_(s)"):
            a= str(a).replace("time_between_transporter_added_and_lysis_(s)","$\Delta t$")
        if str(a).startswith("cl_conc_(mm)"):
            a= str(a).replace("cl_conc_(mm)", "[Cl$^{-}]$")
        features.append(str(a))

    # make a dataframe with two columns from two lists features and feature_importance
    df = pd.DataFrame({'feature': features, 
                       'importance': importance.importances_mean, 
                       'importance_std': importance.importances_std})
    # add a column with absolute values of feature importance
    df['abs_importance'] = df['importance'].abs()
    # sort the dataframe by absolute value of feature importance in descending order
    df = df.sort_values('abs_importance', ascending = False).reset_index(drop=True)
    
    # set type of the feature column as string
    df['feature'] = df['feature'].astype(str)

    if save:
        df.to_csv("feature_importance_permutation_{}_{}.csv".format(filename, model_name))

    # plot the N most important features
    N = n_toplot
    fig, axes = plt.subplots(figsize=figsize, dpi=300)
    axes.barh(df['feature'][:N], df['importance'][:N], align='center', color = color)
    axes.errorbar(df['importance'][:N], df['feature'][:N], xerr=df['importance_std'][:N], 
                  fmt='o', 
                  capsize=10, 
                  color = error_color)
    axes.set_title("{}, {}".format(model_name, filename), fontsize=fontsize+2)
    axes.set_xlabel('Permutation importance', fontsize=fontsize+1, labelpad=10)
    axes.set_ylabel('Feature', fontsize=fontsize+1, labelpad=10)
    axes.tick_params(labelsize=fontsize)
    axes.grid(False)
    
    # invert the y axis
    axes.invert_yaxis()

    # add background color to the figure
    axes.patch.set_facecolor('white')

    # add outline to the figure
    axes.patch.set_edgecolor('black')
    axes.patch.set_linewidth(5)

    # set tickmarks to face inwards
    axes.tick_params(direction='in')

    # set tickmark padding
    axes.tick_params(pad=10)
    
    
    #remove the frame
    for _, spine in axes.spines.items():
        spine.set_visible(False)


    if save:
        fig.savefig("fig_feature_importance_permutation_{}_{}.png".format(filename, model_name))

    #plt.show() 

    return df





