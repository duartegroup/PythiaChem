#!/usr/bin/ent python

# numerics and data packages
import pandas as pd
import numpy as np

# sklearn
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

# imbalanced
from imblearn.metrics import classification_report_imbalanced 
from imblearn.metrics import sensitivity_specificity_support

# matplotlib
import matplotlib.pyplot as plt

# seaborn
import seaborn as sns

# local modules
from . import classification_metrics as cmet

# logging
import logging 


def roc_curve_data(ytest, probs, pos_label=1):
    """
    """
    
    log = logging.getLogger(__name__)
    fp, tp, thresholds = roc_curve(ytest, probs, pos_label=pos_label)
    roc_auc = cmet.auc(fp, tp)
    log.debug("ROC analysis class: {}\n\tTrue positives:\n\t{}\n\tFalse positives:\n\t{}\n\tAUC: {}".format(pos_label, tp, fp, roc_auc))
    
    return fp, tp, thresholds, roc_auc
        
def plot_roc_curve(probas, y_test, title="ROC Curve", x_lab="False Positive Rate (FPR)", 
                   y_lab="True Positive Rate (TPR)", axes=None, figsize=None, plot_class="all",
                  col_map="viridis", savefigure=False, filename=None, size=(10,10), fontsize=15,
                  return_raw_data=False):
    """
    """
    
    log = logging.getLogger(__name__)
                   
    if not isinstance(probas, np.ndarray):
        probas = np.array(probas)
    
    if not isinstance(y_test, np.ndarray):
        y_test = np.ndarray(y_test)
        
    if isinstance(plot_class, str) or  isinstance(plot_class, None):
        plot_class = list(set(y_test))
        data = {int(k):{} for k in plot_class}
    else:
        data = {int(k):{} for k in plot_class}
        
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=size)

    for cls in plot_class:
        cls = int(cls)
        #log.info("probas {} {}".format(probas, probas.shape))
        data[cls]["fpr"], data[cls]["tpr"], data[cls]["thresholds"], data[cls]["auc"] = roc_curve_data(y_test, probas[:, cls], 
                                                                                                       pos_label=cls)
        col = plt.cm.get_cmap(col_map)(float(cls)/len(plot_class))
        axes.plot(data[cls]["fpr"],  data[cls]["tpr"], color=col, label="Class {} (auc={:.2f})".format(cls, data[cls]["auc"]))
        axes.plot([0, 1], [0, 1], "k:")
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])
        axes.set_title("{}".format(title), fontsize=fontsize+2)
        axes.set_xlabel("{}".format(x_lab), fontsize=fontsize+1)
        axes.set_ylabel("{}".format(y_lab), fontsize=fontsize+1)
        axes.legend(loc="lower right", fontsize=fontsize)
        axes.tick_params(labelsize=fontsize)
        axes.grid(True)
        #plt.show()
        
        if savefigure is True and ax is None:
            if filename is not None:
                plt.savefig("{}".format(filename))
            else:
                filename = "roc.png"
                plt.savefig("roc.png")
    
    if not return_raw_data:
        return axes
    else:
        return axes, data

def precision_recall_data(y_test, probs, pos_label=1):
    """
    """
    log = logging.getLogger(__name__)
    prec, rec, thresholds = precision_recall_curve(y_test, probs, pos_label=pos_label)
    average_precision = average_precision_score(y_test, probs)
    log.debug("precision recall analysis class: {}\n\tPrecision:\n\t{}\n\tRecall:\n\t{}\n\tAverage precision: {}".format(pos_label, prec, rec, average_precision))
    
    return prec, rec, thresholds, average_precision
    


def plot_pr_curve(probas, y_test, title="Precision Recall Curve", x_lab="Recall", 
                   y_lab="Precision", axes=None, figsize=None, plot_class="all",
                  col_map="viridis", savefigure=False, filename=None, size=(10,10), fontsize=15,
                  return_raw_data=False):
    """
    """
    
    log = logging.getLogger(__name__)
                   
    if not isinstance(probas, np.ndarray):
        probas = np.array(probas)
    
    if not isinstance(y_test, np.ndarray):
        y_test = np.ndarray(y_test)
        
    if isinstance(plot_class, str) or  isinstance(plot_class, None):
        plot_class = list(set(y_test))
        data = {k:{} for k in plot_class}
    else:
        data = {k:{} for k in plot_class}
        
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=size)

    for cls in plot_class:
        cls = int(cls)
        data[cls]["precision"], data[cls]["recall"], data[cls]["thresholds"], data[cls]["ap"] = precision_recall_data(y_test, probas[:, cls], 
                                                                                                       pos_label=cls)
        col = plt.cm.get_cmap(col_map)(float(cls)/len(plot_class))
        axes.plot(data[cls]["recall"],  data[cls]["precision"], color=col, label="Class {} (Average Precision={:.2f})".format(cls, data[cls]["ap"]))
        axes.plot([0, 1], [0, 1], "k:")
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])
        axes.set_title("{}".format(title), fontsize=fontsize+2)
        axes.set_xlabel("{}".format(x_lab), fontsize=fontsize+1)
        axes.set_ylabel("{}".format(y_lab), fontsize=fontsize+1)
        axes.legend(loc="lower right", fontsize=fontsize)
        axes.tick_params(labelsize=fontsize)
        axes.grid(True)
        #plt.show()
        
        if savefigure is True and ax is None:
            if filename is not None:
                plt.savefig("{}".format(filename))
            else:
                filename = "pr.png"
                plt.savefig("pr.png")
    
    if not return_raw_data:
        return axes
    else:
        return axes, data
            
def plot_confusion_matrix(cmx, axes=None, col_map="RdYlBu", labels=(0, 1), title="Confusion Matrix", x_label="Predicted Class", y_label="Known Class", fontsize=20,
                          annotate = False, vmin=None, vmax=None):
    """
    """
    log = logging.getLogger(__name__)
    
    log.info("{}\n{}".format(cmx, labels))
    
    cmx_df = pd.DataFrame(data=cmx, columns=labels, index=labels)
    sns.set(font_scale=1.4)
    axes = sns.heatmap(cmx_df, annot=True, ax=axes, fmt="d")
    axes.set_title(title, fontsize=fontsize+2)
    axes.set_xlabel(x_label, fontsize=fontsize+1)
    axes.set_ylabel(y_label, fontsize=fontsize+1)
    
    display(axes)
    return axes


def plot_metrics(df, predicted_column_name="prediction", known_column_name="known", probabilities=None,
                 positive_label=1, labels=(0, 1), name="metric_plots.png", figsize=None, fontsize=20, col_map="viridis",
                 all_classes=True, roc_curve=True, pr_curve=True, annotate=True, vmin=None, vmax=None, title=None):
    """
    Function to plot confusion matrix, roc curves and precision reacall curve
    """

    log = logging.getLogger(__name__)

    if all_classes is True:

        classes = set(df[predicted_column_name].values)
        lclasses = list(classes)
        if roc_curve is True and pr_curve is True:
            if figsize is None:
                figsize = ((len(classes) + 2) * 10, 10)
            fig, axs = plt.subplots(1, len(classes) + 2, figsize=figsize)

        elif roc_curve is True or pr_curve is True:
            if figsize is None:
                figsize = ((len(classes) + 1) * 10, 10)
            fig, axs = plt.subplots(1, len(classes) + 1, figsize=figsize)
        else:
            if figsize is None:
                figsize = (len(classes) * 10, 10)
            fig, axs = plt.subplots(1, len(classes), figsize=figsize)

        if title is not None:
            fig.suptitle(title, fontsize=fontsize + 4)

        cmx = cmet.get_multi_label_confusion_matrix(df, predicted_column_name=predicted_column_name,
                                                    known_column_name=known_column_name, labels=labels,
                                                    return_dict=False)
        for ith in range(len(classes)):
            axs[ith] = plot_confusion_matrix(cmx[ith], axes=axs[ith], col_map=col_map,
                                             title="Confusion Matrix Class {}".format(int(lclasses[ith])),
                                             annotate=annotate, fontsize=fontsize, vmin=vmin, vmax=vmax)

        if roc_curve is True:
            log.info(len(classes))
            axs[len(classes)] = plot_roc_curve(probabilities, df[known_column_name].values, axes=axs[len(classes)],
                                               plot_class="all", col_map=col_map, fontsize=fontsize)

        if pr_curve is True:
            axs[-1] = plot_pr_curve(probabilities, df[known_column_name].values, axes=axs[-1], plot_class="all",
                                    col_map=col_map, fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(name)

    else:
        classes = set(df[predicted_column_name].values)
        lclasses = list(classes)
        if roc_curve is True and pr_curve is True:
            if figsize is None:
                figsize = (30, 10)
            fig, axs = plt.subplots(1, 3, figsize=figsize)

        elif roc_curve is True or pr_curve is True:
            if figsize is None:
                figsize = (20, 10)
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        else:
            if figsize is None:
                figsize = (10, 10)
            fig, axs = plt.subplots(1, 1, figsize=figsize)

        if title is not None:
            fig.suptitle(title, fontsize=fontsize + 4)

        cmx = cmet.get_confusion_matrix(df, predicted_column_name=predicted_column_name,
                                        known_column_name=known_column_name, labels=labels, return_dict=False)

        axs[0] = plot_confusion_matrix(cmx, axes=axs[0], col_map=col_map, fontsize=fontsize, annotate=annotate,
                                       vmin=vmin, vmax=vmax)

        if roc_curve is True:
            log.info(len(classes))
            axs[1] = plot_roc_curve(probabilities, df[known_column_name].values, axes=axs[1], plot_class="all",
                                    col_map=col_map, fontsize=fontsize)

        if pr_curve is True:
            axs[-1] = plot_pr_curve(probabilities, df[known_column_name].values, axes=axs[-1], plot_class="all",
                                    col_map=col_map, fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(name)