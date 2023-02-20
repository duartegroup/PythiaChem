# Functions to merge into pythia

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from imblearn.metrics import (
    classification_report_imbalanced,
    sensitivity_specificity_support,
)


def grid_search_classifier_parameters(
    clf,
    Xtrain,
    ytrain,
    clf_options,
    clf_names,
    iteration,
    no_train_output=False,
    cv=5,
    name=None,
    prefix=None,
    scoring=("roc_auc", "precision", "recall"),
):
    """Does GridSearchCV (hyperparameter tuning) and finds the best report metrics if requested

    Args:
        clf (estimator): Estimator object for GridSearchCV
        Xtrain (array-like): X_train values
        ytrain (array-like): y_train values
        clf_options (dict): Dict of classifier parameters being looped over.
        clf_names (list): List of classifiers being looped over.
        iteration (int): Which iteration the loop over the classifiers is at.
        no_train_output (boolean, optional): Whether to suppress output. Defaults to False.
        cv (int, optional): Cv to use in GridSearchCV. Defaults to 5.
        name (str, optional): Classifier name. Defaults to None.
        prefix (str, optional): Subdirectory for output data (when testing multiple
            descriptors)
        scoring (tuple, optional): Scoring to use. Defaults to ("roc_auc", "precision", "recall").

    Returns:
        optparam_search (estimator): GridSearchCV object
        opt_parameters (dict): Best parameters

    """
    log = logging.getLogger(__name__)
    if prefix is None:
        outdir = name
    else:
        outdir = prefix

    # Grid search model optimizer
    parameters = clf_options[clf_names[iteration]]
    log.debug("\tname: {} parameters: {}".format(name, parameters))

    optparam_search = GridSearchCV(
        clf,
        parameters,
        cv=cv,
        error_score=np.nan,
        scoring=scoring,
        refit=scoring[0],
        return_train_score=True,
    )
    log.debug("\tCV xtrain: {}".format(Xtrain))

    optparam_search.fit(Xtrain.values, ytrain.values.ravel())
    opt_parameters = optparam_search.best_params_

    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv(f"{outdir}/{name}_grid_search_metrics.csv")
        log.info("\tBest parameters; {}".format(opt_parameters))
        for mean, std, params in zip(
            optparam_search.cv_results_["mean_test_{}".format(scoring[0])],
            optparam_search.cv_results_["std_test_{}".format(scoring[0])],
            optparam_search.cv_results_["params"],
        ):
            log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass

    return optparam_search, opt_parameters


def test_imbalanced_classifiers_single_fold(
    df,
    test_df,
    classes,
    testclasses,
    classifiers,
    clf_options,
    cv=5,
    clf_names=None,
    class_labels=(0, 1),
    no_train_output=False,
    random_seed=107901,
    prefix=None,
    overwrite=False,
    plot_roc=False,
):
    """
    function to run classification test over classifiers using imbalanced resampling, with hyperparameter tuning
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    :param df: dataframe - training dataframe of features and identifiers (smiles and/or names)
    :param test_df: dataframe - test dataframe of features and identifiers
    :param classes: iterable - list of training labels
    :param testclasses: iterable - list of test labels
    :param classifiers: list - list of classifier methods
    :param clf_options: dict - dict of classifier parameters
    :param cv: int or CV splitter - folds for GridSearchCV
    :param clf_names: list – list of classifier method names
    :param class_labels: tuple - tuple of class labels (e.g. (0,1) for binary classification)
    :param no_train_output: bool - whether to suppress GridSearchCV training output, defaults to False
    :param random_seed: int - random seed to use
    :param prefix: str - subdirectory for output data (when testing multiple
        descriptors)
    :param overwrite: bool - whether to overwrite existing data, defaults to False
    :param plot_roc: bool - whether to plot roc curves, defaults to False

    :return clfs: list - list of estimator objects
    :return opt_params: list - list of optimal parameters
    """

    log = logging.getLogger(__name__)

    log.info("Features: {}".format(df.columns))

    log_df = pd.DataFrame()

    clfs, opt_params = [], []
    reports, roc_aucs, scores, c_matrices = [], [], [], []

    iteration = 0
    data = df.copy()
    data.reset_index(inplace=True)

    # Training data
    Xtrain = df
    log.debug("Train X\n{}".format(Xtrain))
    #df.to_csv("Xtrain.csv")
    ytrain = classes
    log.debug("Train Y\n{}".format(ytrain))

    # Testing data
    Xtest = test_df
    log.debug("Test X\n{}".format(Xtest))
    #test_df.to_csv("Xtest.csv")
    ytest = testclasses
    log.debug("Test Y\n{}".format(ytest))

    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]

    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))

        name = "{}".format("_".join(name.split()))

        if prefix is None:
            outdir = name
        else:
            outdir = f"{prefix}/{name}"

        # Make directory for each classifier
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        elif overwrite is False and os.path.isdir(outdir) is True:
            log.warning(
                "Directory already exists and overwrite is False will stop before overwriting."
            )
            return None
        else:
            log.info(f"Directory {outdir} already exists will be overwritten")

        # Grid search model optimizer, which will refit at the end to get the final model with optimised parameters
        clf, opt_param = grid_search_classifier_parameters(
            classf,
            Xtrain,
            ytrain,
            clf_options,
            clf_names,
            iteration,
            no_train_output,
            cv=cv,
            name=name,
            prefix=prefix,
        )
        clfs.append(clf)
        opt_params.append(opt_param)

        # Evaluate the model
        ## evaluate the model on multiple metric score as list for averaging
        predicted_clf = clf.predict(Xtest)
        sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
        sc_df = pd.DataFrame(
            data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"]
        )
        sc_df.to_csv(os.path.join(outdir, "PRFS_score.csv"))

        ## evaluate the principal score metric only (incase different to those above although this is unlikely)
        clf_score = clf.score(Xtest, ytest)

        ## Get the confusion matrices
        c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)

        ## Calculate the roc area under the curve (requires clf to have predict_proba available)
        probs = clf.predict_proba(Xtest)
        fpr, tpr, thresholds = roc_curve(ytest, probs[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        roc_aucs.append(roc_auc)
        log.info(f"\tROC analysis area under the curve: {roc_auc}")

        # output metrics for consideration
        log.info(f"\tConfusion matrix ({name}):\n{c_matrix}\n")

        c_matrices.append(c_matrix)
        log.info("\n\tscore ({}): {}".format(name, clf_score))

        scores.append(clf_score)

        log.info("\tImbalance reports:")
        log.info(
            "\tImbalance classification report:\n{}".format(
                classification_report_imbalanced(ytest, predicted_clf)
            )
        )
        output_dict = classification_report_imbalanced(
            ytest, predicted_clf, output_dict=True
        )

        # Plot the roc curves (now made optional)
        if plot_roc:
            figure, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                fpr, tpr, color="red", lw=1.5, label=f"ROC curve (auc = {roc_auc:.2f})"
            )

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")

            precision_label = f"Precision: class 0 = {output_dict[0]['pre']:.2f}, class 1 = {output_dict[1]['pre']:.2f}"
            recall_label = f"Recall: class 0 = {output_dict[0]['rec']:.2f}, class 1 = {output_dict[1]['rec']:.2f}"
            f1_label = f"F1 score: class 0 = {output_dict[0]['f1']:.2f}, class 1 = {output_dict[1]['f1']:.2f}"
            annot_str = f"{precision_label}\n{recall_label}\n{f1_label}"

            ax.annotate(
                annot_str,
                xy=(0.5, 0),
                xycoords="figure fraction",
                size=9,
                ha="center",
                va="bottom",
            )
            figure.tight_layout()
            plt.savefig(f"{outdir}/{name}_roc_curve.png")
            plt.show()

        reports.append(classification_report_imbalanced(ytest, predicted_clf))

        sensitvity, specificity, support = sensitivity_specificity_support(
            ytest, predicted_clf
        )
        log.debug("\t{} {} {}".format(sensitvity, specificity, support))
        log.info("\t Index | Predicted | Label\n\t----------------------")

        log.info(
            "\t{}\n-----\n".format(
                "\n\t".join(
                    [
                        "  {}   |   {}   |   {}".format(i, p, k)
                        for i, (p, k) in enumerate(zip(predicted_clf, ytest))
                    ]
                )
            )
        )

        pred = [
            list(range(len(ytest))),
            list(ytest),
            list(predicted_clf),
            list(probs[:, 0]),
            list(probs[:, 1]),
        ]
        pred_cols = [
            "Index",
            "True value",
            "Predicted value",
            "P(class 0)",
            "P(class 1)",
        ]

        pred = pd.DataFrame(pred).T
        pred.columns = pred_cols
        pred.to_csv(f"{outdir}/{name}_predictions.csv")
        iteration += 1

    log_df["roc_auc"] = pd.Series(roc_aucs)

    log_df["report"] = pd.Series(reports)
    log_df["score"] = pd.Series(scores)

    log_df["c_matrix"] = pd.Series(c_matrices)

    if prefix is None:
        log_df.to_csv("logs2.csv")
    else:
        log_df.to_csv(f"{prefix}/logs2.csv")

    return clfs, opt_params


def test_classifiers_nested(
    df,
    classes,
    classifiers,
    clf_options,
    inner_cv=5,
    outer_cv=5,
    scoring="roc_auc",
    score_name="roc_auc",
    clf_names=None,
    prefix=None,
    random_seed=107901,
):
    """
    Nested cross validation for model selection (looking at scores only)
    Only accepts one type of scoring each time
    Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

    :param df: dataframe - training dataframe of features and identifiers (smiles and/or names)
    :param classes: iterable - list of training labels
    :param classifiers: list - list of classifier methods
    :param clf_options: dict - dict of classifier parameters
    :param inner_cv: int or CV splitter - folds for the inner CV (GridSearchCV)
    :param outer_cv: int or CV splitter - folds for the outer CV (cross_val_score)
    :param scoring: str or callable - a single string or a callable that returns a
        single value for scoring
    :param score_name: str - name for scoring method
    :param clf_names: list – list of classifier method names
    :param prefix: str - subdirectory for nested CV scores (when testing multiple
        descriptors)
    :param random_seed: int - random seed to use

    :return scores_df: DataFrame - scores for each outer CV and mean score
    """

    log = logging.getLogger(__name__)

    scores = {}

    data = df.copy()
    data.reset_index(inplace=True)

    # Training data
    Xtrain = df
    log.debug("Train X\n{}".format(Xtrain))
    df.to_csv("Xtrain.csv")
    ytrain = classes
    log.debug("Train Y\n{}".format(ytrain))

    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]

    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))

        parameters = clf_options[name]
        print(parameters)
        # Just want the nested CV scores here
        clf = GridSearchCV(
            classf,
            parameters,
            cv=inner_cv,
            scoring=scoring,
            refit=True,
        )
        nested_score = cross_val_score(
            clf, X=Xtrain.values, y=ytrain.values.ravel(), cv=outer_cv, scoring=scoring
        )
        scores[name] = nested_score

    score_label = f"Mean_{score_name}"
    scores_df = pd.DataFrame.from_dict(scores, orient="index")
    scores_df[score_label] = scores_df.mean(axis=1)

    if prefix is None:
        if not os.path.isdir("nested_cv_results"):
            os.makedirs("nested_cv_results")
        scores_df.to_csv(f"nested_cv_results/nested_cv_{score_name}.csv")
    else:
        if not os.path.isdir(f"nested_cv_results/{prefix}"):
            os.makedirs(f"nested_cv_results/{prefix}")
        scores_df.to_csv(f"nested_cv_results/{prefix}/nested_cv_{score_name}.csv")

    return scores_df
