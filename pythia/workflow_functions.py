#!/usr/bin/env python

# parts of this code are inspired by Chemical Space Analysis and Property Prediction for Carbon Capture Amine Molecules.
#https://doi.org/10.1039/D3DD00073G
#https://zenodo.org/records/10213104
#https://github.com/flaviucipcigan/ccus_amine_prediction_workflow


# Python packages and utilities
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support
from IPython.display import Image, display
import logging
import math
import matplotlib.pyplot as plt
# Update the matplotlib font configuration
plt.rcParams['font.family'] = 'Arial'  # or any other font available on your system
from mlxtend.evaluate import permutation_test as pt
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, explained_variance_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_curve, auc)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import scipy
import scipy.stats as stats
# Own modules
from pythia import classification_metrics as cmetrics
from pythia import scaling


def df_corr(x, y, corr_method):
    """
    Calculate the correlation between two arrays using the specified method.

    Parameters:
    x (array-like): The first array of data.
    y (array-like): The second array of data.
    corr_method (str): The method to use for correlation calculation.
                       Possible methods are 'pearson', 'kendall', 'spearman'.

    Returns:
    pandas.DataFrame: A DataFrame containing the correlation matrix.
    """
    df = pd.DataFrame(data=np.array([x, y]).T)
    corr = df.corr(method=corr_method)
    
    return corr


def find_correlating_features(features, targets, thresh=0.4, plot=True, process_non_numeric=True,
                              corr_method="spearman", sig_metric=None, significance=False,
                              random_seed=105791, n_sample=10000, sig_level=0.05):
    """
    Pulls out features which correlate with a property bay at least the threshold as measured by the specified correlation method.

    :param features: pandas dataframe - pandas data frame of features to correlate against a property
    :param targets: pandas series - pandas series of property target values
    :param thresh: float - Pearson correlation coefficent threshold to be equal to or more than
    :param plot: true/false - plot the features with a suitably high correlation
    :param process_non_numeric: true/false - Proces or not features which mordred could not calcualte for all molecules
    :param corr_method: str - any correlation method that can be given pandas .corr(method=arg)
    :param significance: true/false - one tail significance of the corr_method: how many times out of the n_sample do I obtain a correlation coefficient that is greater than the observed value
    :param random_seed: int - random seed for sampling in the significance test
    :param n_sample: int - number of resamples/shuffle of the data significance test
    :param sig_level: float - level to consider the results significant 0.05 is 95% significance level here
    """

    log = logging.getLogger(__name__)

    log.info(targets)
    usable_features = []
    significant_features = []

    if significance is True:
        log.info("Significance will be calculated for the correlations")
        if sig_metric == "pearson":
            fcorr = lambda x, y: abs(scipy.stats.pearsonr(x, y)[0])
        elif sig_metric == "spearman":
            fcorr = lambda x, y: abs(scipy.stats.spearmanr(x, y)[0])
        elif sig_metric == "kendalltau":
            fcorr = lambda x, y: abs(scipy.stats.kendalltau(x, y)[0])
        elif sig_metric == "onetail_gt_pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(x, y)[0]
        elif sig_metric == "onetail_gt_spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(x, y)[0]
        elif sig_metric == "onetail_gt_kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(x, y)[0]
        elif sig_metric == "onetail_lt_pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(y, x)[0]
        elif sig_metric == "onetail_lt_spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(y, x)[0]
        elif sig_metric == "onetail_lt_kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(y, x)[0]
        elif sig_metric == "notequal":
            fcorr = "x_mean != y_mean"
        elif sig_metric == "greaterthan":
            fcorr = "x_mean > y_mean"
        elif sig_metric == "lessthan":
            fcorr = "x_mean < y_mean"
        elif sig_metric is None:
            fcorr = "x_mean != y_mean"
        else:
            log.error("ERROR - Correlation method unrecognised")
            return

    for ent in features.columns:

        tf = True if ent in features.columns else False
        log.debug("{} {}".format(ent, tf))
        series = features[ent].copy()

        if series.isnull().sum().sum() > 0:
            # log.info("NaN identifed")
            if series.isnull().sum() < 0.7 * len(features[ent].values):
                log.info("filling NaNs")
                series.fillna(features[ent].mean())
            else:
                log.warning("{} - Too many NaN for filling".format(ent))
                continue
        #else:
            #pass

        # Drop any rows with NA
        feat = series.values
        tmp_df = pd.DataFrame(data=np.array([feat, targets]).T, columns=[ent, "property"])
        tmp_df.dropna(inplace=True)

        try:
            correlations = tmp_df.corr(method=corr_method)

            if abs(correlations.loc[ent, "property"]) > thresh:
                # print("mia xara")
                usable_features.append(ent)
                if significance is True:
                    p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
                    significant = True if p_value < sig_level else False
                    log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}".format(ent, correlations.loc[
                        ent, "property"], p_value, sig_level, significant))
                    if significant is True:
                        significant_features.append(ent)
                else:
                    log.info("{}: {:.4f}".format(ent, correlations.loc[ent, "property"]))

                if plot is True:
                    tmp_df.plot.scatter(ent, "property", grid=True)
            else:
                log.debug("Feature {} does not correlate to the threshold {}".format(ent, thresh))

            del tmp_df

        except KeyError:
            if process_non_numeric is True:
                # print("oxi")
                # If non-numeric error messages occur from mordred the correlation matrix is empy and has a key error
                log.debug(
                    "WARNING - some molecules do not have this feature ({}) thus the correlation is for a sub-set.".format(
                        ent))

                tmp_df = tmp_df[pd.to_numeric(tmp_df[ent], errors='coerce').notnull()]
                tmp_df[ent] = pd.to_numeric(tmp_df[ent])
                tmp_df["property"] = pd.to_numeric(tmp_df["property"])

                del tmp_df
            else:
                del tmp_df

    if significance is False:
        return usable_features
    else:
        return usable_features, significant_features


def permutation_test(features, targets, n_sample=10000, corr_method="pearson", random_seed=105791,  sig_level=0.05, significance = True):
    """
    :param features: pandas dataframe - pandas data frame of features to correlate against a property
    :param targets: pandas series - pandas series of property target values
    :param process_non_numeric: true/false - Process or not features which mordred could not calcualte for all molecules
    :param corr_method: str - any correlation method that can be given pandas .corr(method=arg)
    :param significance: true/false - one tail significance of the corr_method: how many times out of the n_sample do I obtain a correlation coefficient that is greater than the observed value
    :param random_seed: int - random seed for sampling in the significance test
    :param n_sample: int - number of resamples/shuffle of the data significance test
    :param sig_level: float - level to consider the results significant 0.05 is 95% significance level here
    """
    log = logging.getLogger(__name__)
    significant_features = []
    
    if significance is True:
        log.info("Significance will be calculated for the correlations")
        if corr_method == "pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(x, y)[0]
        elif corr_method == "spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(x, y)[0]
        elif corr_method == "kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(x, y)[0]
        else:
            log.error("ERROR - Correlation method unrecognised")
            return

    for ent in features.columns:
        feat = features[ent].values
        tmp_df = pd.DataFrame(data=np.array([feat, targets]).T, columns=[ent, "property"])
        tmp_df.dropna(inplace=True)
    
    try:
            correlations = tmp_df.corr(method=corr_method)
            p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
            significant = True if p_value < sig_level else False
            log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))

            if significant:
                significant_features.append(ent)
    except KeyError:
            
            # If non-numeric error messages occur from mordred the correlation matrix is empy and has a key error
            log.warning("WARNING - some molecules do not have this feature thus the correlation is for a sub-set.")
            tmp_df = tmp_df[pd.to_numeric(tmp_df[ent], errors='coerce').notnull()]
            tmp_df[ent] = pd.to_numeric(tmp_df[ent])
            tmp_df["property"] = pd.to_numeric(tmp_df["property"])

            correlations = tmp_df.corr(method=corr_method) 
            p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
            significant = True if p_value < sig_level else False
            log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
            if significant:
                significant_features.append(ent)

    return correlations, p_value,  significant_features



def general_two_class(ent, bound=4, **kwarg):
    """
    function to perform a basic classification
    :param ent: float - value to perform classification against
    """
    
    if ent < bound:
        c = 0
    elif ent >= bound:
        c = 1
    
    return c
        
def constant_value_columns(df):
    """
    Function to find constant value columns
    :param df: Pandas dataframe - dataframe to find non unique columns
    """
    
    log = logging.getLogger(__name__)
    
    cols = [name for name in df.columns if df[name].nunique() == 1]
    
    return cols


def grid_search_classifier_parameters(clf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=5, name=None, scoring=("roc_auc", "precision", "recall")):
    """
    Perform a grid search to optimize classifier hyperparameters and report metrics if requested.

    Parameters:
    clf (estimator object): The classifier to be optimized.
    Xtrain (array-like or pandas DataFrame): Training feature set.
    ytrain (array-like or pandas DataFrame): Training target labels.
    clf_options (dict): A dictionary where keys are classifier names and values are dictionaries of hyperparameters to be optimized.
    clf_names (list): A list of classifier names corresponding to the keys in clf_options.
    iteration (int): The current iteration index for selecting the classifier from clf_names.
    no_train_output (bool): If False, outputs the grid search metrics to a CSV file.
    cv (int, default=5): Number of cross-validation folds.
    name (str, optional): The name to be used for output files.
    scoring (tuple, default=("roc_auc", "precision", "recall")): A tuple of scoring metrics to evaluate during grid search.

    Returns:
    dict: The best hyperparameters found during the grid search.

    """
    log = logging.getLogger(__name__)
    
    # Grid search model optimizer
    parameters = clf_options[clf_names[iteration]]
    # log.debug("\tname: {} parameters: {}".format(name, parameters))
    
    optparam_search = GridSearchCV(clf, parameters, cv=cv, error_score=np.nan, scoring=scoring, refit=scoring[0], return_train_score=True)
    log.debug("\tCV xtrain: {}".format(Xtrain))
    
    optparam_search.fit(Xtrain, ytrain.values.ravel())
    opt_parameters = optparam_search.best_params_
    
    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv("{}/{}_grid_search_metrics.csv".format(name,name))
        #log.info("\tBest parameters; {}".format(opt_parameters))
        #for mean, std, params in zip(optparam_search.cv_results_["mean_test_{}".format(scoring[0])],
        #                            optparam_search.cv_results_["std_test_{}".format(scoring[0])],
        #                           optparam_search.cv_results_['params']):
        # log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass
    
    return opt_parameters


def grid_search_regressor_parameters(rgs, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=5,
                                     name=None, scoring=("r2", "neg_root_mean_squared_error")):
    """
    Perform a grid search to optimize regressor hyperparameters and report metrics if requested.

    Parameters:
    rgs (estimator object): The regressor to be optimized.
    Xtrain (array-like or pandas DataFrame): Training feature set.
    ytrain (array-like or pandas DataFrame): Training target values.
    rgs_options (dict): A dictionary where keys are regressor names and values are dictionaries of hyperparameters to be optimized.
    rgs_names (list): A list of regressor names corresponding to the keys in rgs_options.
    iteration (int): The current iteration index for selecting the regressor from rgs_names.
    no_train_output (bool): If False, outputs the grid search metrics to a CSV file.
    cv (int, default=5): Number of cross-validation folds.
    name (str, optional): The name to be used for output files.
    scoring (tuple, default=("r2", "neg_root_mean_squared_error")): A tuple of scoring metrics to evaluate during grid search.

    Returns:
    dict: The best hyperparameters found during the grid search.

    """
    log = logging.getLogger(__name__)

    # Grid search model optimizer
    parameters = rgs_options
    #parameters = rgs_options[rgs_names[iteration]]
    log.info("\tname: {} parameters: {}".format(name, parameters))

    optparam_search = GridSearchCV(estimator=rgs, param_grid=parameters, cv=cv, error_score=np.nan, scoring=scoring, refit=scoring[0],
                                   return_train_score=True)
    log.debug("\tCV xtrain: {}".format(Xtrain))

    optparam_search.fit(Xtrain, ytrain)

    #optparam_search.fit(Xtrain, ytrain.values.ravel())
    opt_parameters = optparam_search.best_params_

    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv("{}/{}_grid_search_metrics.csv".format(name, name))
        #log.info("\tBest parameters; {}".format(opt_parameters))

        #for mean, std, params in zip(optparam_search.cv_results_["mean_test_{}".format(scoring[0])],
        #                        optparam_search.cv_results_["std_test_{}".format(scoring[0])],
        #                       optparam_search.cv_results_['params']):
            #log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass

    return opt_parameters


def kfold_test_regressor_with_optimization(df, targets, regressors, rgs_options, scale=True, cv=5, n_repeats=20,
                                           rgs_names=None,
                                           no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                           random_seed=107901, overwrite=True):
    """
    function to run regression model
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param targets: target values
    :param regressor: list - list of regression methods
    """

    log = logging.getLogger(__name__)
    log.info("Features: {}".format(df.columns))

    list_opt_param, predictions = [], []

    iteration = 0
    pd.set_option('display.max_columns', 20)

    if rgs_names is None:
        rgs_names = [i for i in range(0, len(regressors))]

    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of
    # independent cross validators.

    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting regression")
    for name, regs in zip(rgs_names, regressors):
        log.info("\n-----\nBegin {}\n-----\n".format(name))

        kf_iteration = 0
        scores = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))

        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok=True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))

        # Loop over  Kfold here
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))

            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())

            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))

            # Train
            Xtrain = df.iloc[train_indx]
            ytrain = targets.iloc[train_indx]

            # Test
            Xtest = df.iloc[test_indx]
            ytest = targets.iloc[test_indx]

            # way to calculate the test indexes
            # test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_regressor_parameters(regs, Xtrain, ytrain, rgs_options[name], rgs_names, iteration,
                                                             no_train_output, cv=cv, name=name)

            list_opt_param.append(opt_param)

            # Fit final model using optimized parameters
            reg = regs
            reg.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.info("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            reg.fit(Xtrain, ytrain)
            #reg.fit(Xtrain, ytrain.values.ravel())

            # Evaluate the model
            # evaluate the model on multiple metric score as list for averaging
            predicted_reg = reg.predict(Xtest)
            sc = mean_absolute_error(ytest, predicted_reg)
            score_list.append(sc)

            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            reg_score = reg.score(Xtest, ytest)
            scores.append(reg_score)
            log.info("\n\tscore ({}): {}".format(name, reg_score))

            pred = [list(test_indx), list(ytest), list(predicted_reg)]

            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1

        del predictions[:]
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        log.info("The average scores are: {}, and the std scores are: {}".format(average_scores, std_scores))
        iteration = iteration + 1


def split_test_regressors_with_optimization(df, targets, test_df, test_targets, regressors, rgs_options, scale=True,
                                            cv=5,
                                            n_repeats=20, rgs_names=None,
                                            no_train_output=False, random_seed=107901, overwrite=True):
    """
    Perform regression using multiple regressors with optional hyperparameter optimization and evaluation.

    Args:
    df (pd.DataFrame): Training data features.
    targets (pd.Series or np.array): Training data targets.
    test_df (pd.DataFrame): Test data features.
    test_targets (pd.Series or np.array): Test data targets.
    regressors (list of sklearn regressor objects): List of regressor objects to be trained and evaluated.
    rgs_options (dict): Dictionary where keys are regressor names and values are dictionaries of hyperparameters to optimize.
    scale (bool, optional): Whether to scale the data (default is True).
    cv (int, optional): Number of cross-validation folds (default is 5).
    n_repeats (int, optional): Number of times to repeat the experiment (default is 20).
    rgs_names (list, optional): Names of the regressors (default is None, uses index-based names).
    no_train_output (bool, optional): If True, do not output training metrics to CSV files (default is False).
    random_seed (int, optional): Random seed for reproducibility (default is 107901).
    overwrite (bool, optional): Whether to overwrite existing directories (default is True).

    Returns: None
    """

    log = logging.getLogger(__name__)
    all_test_metrics, all_train_metrics = [], []
    iteration = 0
    p=[]
    if rgs_names is None:
        rgs_names = [i for i in range(0, len(regressors))]

    log.info("Starting regression")
    for name, reg in zip(rgs_names, regressors):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        name = "{}".format("_".join(name.split()))

        # Make directory for each regressor
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok=True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))

        #         Train set
        Xtrain = df
        ytrain = targets

        #         Find optimal parameters
        opt_param = grid_search_regressor_parameters(reg, Xtrain, ytrain, rgs_options[name], rgs_names, iteration,
                                                     no_train_output, cv=cv, name=name)

        # Fit model using optimized parameters
        reg.set_params(**opt_param)
        reg.fit(Xtrain, ytrain)
        #         Train set
        train_predictions = reg.predict(Xtrain)


        #         Save Model
        targ = 'ddg'
        # dump(reg, "{}/model_{}_{}_fixed.joblib".format(name, name, targ))

        #         Test set
        Xtest = test_df
        ytest = test_targets

        #         Evaluate the model based on the validation set
        test_predictions = reg.predict(Xtest)
        log.info("\n\t The predictions are: {}".format(test_predictions))

        #         Calculate metrics for regression
        train_metrics = {}
        train_metrics['name'] = name
        train_metrics["variance"] = round(explained_variance_score(targets, train_predictions), 2)
        train_metrics["MAE"] = round(mean_absolute_error(targets, train_predictions), 2)
        train_metrics["MSE"] = round(mean_squared_error(targets, train_predictions), 2)
        train_metrics["RMSE"] = math.sqrt(round(mean_squared_error(targets, train_predictions), 2))
        train_metrics["R2"] = round(r2_score(targets, train_predictions), 2)

        log.info(train_metrics)
        all_train_metrics.append(train_metrics)
        
        test_metrics = {}
        test_metrics['name'] = name
        test_metrics["variance"] = round(explained_variance_score(test_targets, test_predictions), 2)
        test_metrics["MAE"] = round(mean_absolute_error(test_targets, test_predictions), 2)
        test_metrics["MSE"] = round(mean_squared_error(test_targets, test_predictions), 2)
        test_metrics["RMSE"] = math.sqrt(round(mean_squared_error(test_targets, test_predictions), 2))
        test_metrics["R2"] = round(r2_score(test_targets, test_predictions), 2)

        log.info(test_metrics)
        all_test_metrics.append(test_metrics)
        
        # save the model to disk
        model_savefile = 'model_{}.sav'.format(name)
        pickle.dump(reg, open(model_savefile, 'wb'))

        
        plt.rcParams.update()
        #ref: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
        # Set the style globally
        # Alternatives include bmh, fivethirtyeight, ggplot,
        # dark_background, seaborn-deep, etc
        plt.style.use('default')

        plt.rcParams['font.family'] = 'Arial'
        #plt.rcParams['font.serif'] = 'Ubuntu'
        #plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['figure.titlesize'] = 24

        plt.figure(figsize=(9, 9), dpi=300)
        plt.scatter(targets, train_predictions, s=60, alpha=0.6,edgecolors="k", color='silver', label="Train")
        plt.scatter(test_targets, test_predictions, s=60,alpha=0.6, edgecolors="k", color='blue', label="Test")
        plt.ylabel('Predicted Δ∆$G^{‡}$ (kJ/mol)')
        plt.xlabel('Experimental Δ∆$G^{‡}$ (kJ/mol)')
        plt.legend(loc='lower right')
        plt.title(name.replace("_", " "))
        plt.plot([-0.05, 11], [-0.05, 11], "k:")
        plt.savefig('{}/{}.png'.format(name, name), dpi=300)
        plt.show()

        for i, j in zip(ytest, test_predictions):
            p.append([i, j])
        pred = pd.DataFrame(p, columns=['actual', 'predicted'])
        pred.to_csv("{}/predictions.csv".format(name), index=False)
        del p[:]

    all_train_metrics = pd.DataFrame(all_train_metrics)
    all_train_metrics.to_csv('train_metrics.csv')
    
    all_test_metrics = pd.DataFrame(all_test_metrics)
    all_test_metrics.to_csv('test_metrics.csv')

def kfold_test_classifiers_with_optimization(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None,
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False):
    """
    function to run classification test over classifiers using imbalanced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix=[],[],[],[],[]
    list_average_roc_auc,list_average_scores=[],[]
    disp,predictions,actual,index,same = [], [], [],[],[]
    probablity=[]
    
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    log.info("Features: {}".format(data))
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = scaling.minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
            
            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalance reports:")
            log.info("\tImbalance classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            #ax = plt.subplot(2, plt_rows, kf_iteration+1)
            #ax.plot(fpr, tpr, color="red",
            #         lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))

            #ax.plot([0, 1], [0, 1], "k:")
            #ax.set_xlim(xmin=0.0, xmax=1.01)
            #ax.set_ylim(ymin=0.0, ymax=1.01)
            #ax.set_xlabel('False Positive Rate')
            #ax.legend(loc="lower right")
            #ax.set_ylabel('True Positive Rate')
            
                   
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n{}".format(score_str1, score_str2, score_str3)
        #plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        #figure.tight_layout()
        #plt.savefig("{0}/{0}_roc_curves.png".format(name))
        #plt.show()

        iteration = iteration + 1
    
    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)



def kfold_test_classifiers_with_optimization_weights(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None,class_weight=None,
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False):
    """
    function to run classification test over classifiers using imbalanced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix=[],[],[],[],[]
    list_average_roc_auc,list_average_scores=[],[]
    disp,predictions,actual,index,same = [], [], [],[],[]
    probablity=[]
    
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    log.info("Features: {}".format(data))
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = scaling.minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = int(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            clf.set_params(class_weight=class_weight)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
            
            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalance reports:")
            log.info("\tImbalance classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            #ax = plt.subplot(2, plt_rows, kf_iteration+1)
            #ax.plot(fpr, tpr, color="red",
            #         lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            #
            #    # ugliest legend i ve made in my life - maybe one under the other?
            #
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
            #ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))
#
            #ax.plot([0, 1], [0, 1], "k:")
            #ax.set_xlim(xmin=0.0, xmax=1.01)
            #ax.set_ylim(ymin=0.0, ymax=1.01)
            #ax.set_xlabel('False Positive Rate')
            #ax.set_ylabel('True Positive Rate')
            #ax.legend(loc="lower right")
            
                   
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n{}".format(score_str1, score_str2, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1
    
    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)



def test_classifiers_with_optimization(df, test_df, classes, testclasses, classifiers, clf_options, overwrite=False, scale=False, cv=5,
                                       n_repeats=20,random_seed=107901, clf_names=None,no_train_output=False,
                                       class_labels=(0, 1)):
    """
    function to run classification test over classifiers using imbalanced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """

    log = logging.getLogger(__name__)

    log.info("Features: {}".format(df.columns))

    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()

    predicted, list_opt_param = [], []
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [], [], [], [], []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)

    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]

    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
        log.info("data:\n{}".format(data))

    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))

        kf_iteration = 0
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))

        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok=True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))

            # Train
        Xtrain = df
        log.debug("Train X\n{}".format(Xtrain))
        df.to_csv('Xtrain.csv')
        ytrain = classes
        log.debug("Train Y\n{}".format(ytrain))

        # Test
        Xtest = test_df
        log.debug("Test X\n{}".format(Xtest))
        test_df.to_csv('Xtest.csv')
        ytest = testclasses
        log.debug("Test Y\n{}".format(ytest))

        # Grid search model optimizer
        opt_param = grid_search_classifier_parameters(classf, Xtrain, ytrain, clf_options, clf_names, iteration,
                                                          no_train_output, cv=cv, name=name)

        list_opt_param.append(opt_param)

        # Fit final model using optimized parameters
        clf = classf
        clf.set_params(**opt_param)
        log.info("\n\t----- Predicting using: {} -----".format(name))
        log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
        clf.fit(Xtrain, ytrain)
        
        # save the model to disk
        model_savefile = 'model_{}.sav'.format(name)
        pickle.dump(clf, open(model_savefile, 'wb'))

        # Evaluate the model
        ## evaluate the model on multiple metric score as list for averaging
        predicted_clf = clf.predict(Xtest)
        sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
        sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
        sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
        score_list.append(sc)

        ## evaluate the principle score metric only (incase different to those above although this is unlikely)
        clf_score = clf.score(Xtest, ytest)
        scores.append(clf_score)

        ## Get the confusion matrices
        c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
        confusion_matrices.append(c_matrix)

        ## Calculate the roc area under the curve
        probs = clf.predict_proba(Xtest)
        fpr, tpr, thresholds = roc_curve(ytest, probs[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        list_roc_auc.append(roc_auc)

        roc_aucs.append(roc_auc)
        log.info("\tROC analysis area under the curve: {}".format(roc_auc))

        # output metrics for consideration
        log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))

        list_c_matrix.append(c_matrix)
        log.info("\n\tscore ({}): {}".format(name, clf_score))

        list_score.append(clf_score)

        log.info("\tImbalance reports:")
        log.info(
            "\tImbalance classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
        output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)

        ## Plot the roc curves
        #ax = plt.subplot(2, plt_rows, kf_iteration + 1)
        #ax.plot(fpr, tpr, color="red",
        #        lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))

        #ax.plot(fpr, tpr, alpha=0.0, color="white", lw=1.5,
        #        label="pre_class0 = {:.2f}\n".format(output_dict[0]['pre']) + "pre_class1 = {:.2f}".format(
        #            output_dict[1]['pre']))
        #ax.plot(fpr, tpr, alpha=0.0, color="white", lw=1.5,
        #        label="f1_class0 = {:.2f}\n".format(output_dict[0]['f1']) + "f1_class1 = {:.2f}".format(
        #            output_dict[1]['f1']))
        #ax.plot(fpr, tpr, alpha=0.0, color="white", lw=1.5,
        #        label="rec_class0 = {:.2f}\n".format(output_dict[0]['rec']) + "rec_class1 = {:.2f}".format(
        #            output_dict[1]['rec']))

        #ax.plot([0, 1], [0, 1], "k:")
        #ax.set_xlim(xmin=0.0, xmax=1.01)
        #ax.set_ylim(ymin=0.0, ymax=1.01)
        #ax.set_xlabel('False Positive Rate')
        #ax.set_ylabel('True Positive Rate')
        #ax.legend(loc="lower right")

        list_report.append(classification_report_imbalanced(ytest, predicted_clf))

        sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
        log.debug("\t{} {} {}".format(sensitvity, specificity, support))
        log.info("\t -----Index | Predicted | Label\n\t------------------")

        log.info("\t{}\n-----\n".format("\n\t".join(
            ["  {}   |   {}   |   {}".format(i, p, k) for i, (p, k) in enumerate(zip(predicted_clf, ytest))])))

        pred = [list(range(len(testclasses))), list(ytest), list(predicted_clf), list(probs[:, 0]), list(probs[:, 1])]

        pred = pd.DataFrame(pred)
        pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
        kf_iteration = kf_iteration + 1

        # Plot and assess classifier over all folds

        # NOTE - rows are scores columns are classes
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)

        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(
            average_scores[0][0],
            std_scores[0][0],
            average_scores[1][0],
            std_scores[1][0],
            average_scores[2][0],
            std_scores[2][0],
            average_scores[3][0],
            std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(
            average_scores[0][1],
            std_scores[0][1],
            average_scores[1][1],
            std_scores[1][1],
            average_scores[2][1],
            std_scores[2][1],
            average_scores[3][1],
            std_scores[3][1])
        score_str3 = "Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n{}".format(score_str1, score_str2, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points',
                     size=12, ha='center', va='bottom')
        #figure.tight_layout()
        #plt.savefig("{0}/{0}_roc_curves.png".format(name))
        #plt.show()

        iteration = iteration + 1
    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)

def directory_names(classifier_names):
    """
    Function to get the names of the directories from the classifier scripts
    :param classifier_names: list - list of classifier names
    """
    
    names = ["{}".format("_".join(name.split())) for name in classifier_names]
    
    return names


def build_data_from_directory(data_directory, max_folds=10, save=False):
    """
    Function to build a set of data from csv files named K.csv where K is the fold number and the csv
    is the predictions for the test data from that fold
    :param data_directory: str - name of the directory to read the csv files from
    :param max_folds: int - the number of folds run in the Kfold cv
    :param save: bool - save the data to a csv file
    """

    log = logging.getLogger(__name__)

    for i in range(max_folds):
        log.info(f"Reading {i}.csv")
        data_fold = pd.read_csv(os.path.join(data_directory, f"{i}.csv"), header=0)
        if i == 0:
            data = data_fold.copy()
        else:
            data = pd.concat([data, data_fold], ignore_index=True)

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.columns = ["m_index", "known", "prediction", "prob0", "prob1"]

    # Remove duplicates by keeping the first occurrence
    data.drop_duplicates(subset="m_index", keep='first', inplace=True)

    data["m_index"] = data["m_index"].astype(int)
    data.set_index("m_index", inplace=True, drop=True, verify_integrity=True)
    data.sort_index(inplace=True)

    if save:
        data.to_csv(os.path.join(data_directory, "all_predictions.csv"))

    return data


def build_data_from_directory_regr(data_directory, max_folds=10):
    """
    Function to build a set of data from csv files names K.csv where K is the fold number and the csv
    is the predictions for the test data from that fold
    :param directory: str - name of the directory to read the csv files from
    :param max_fold: int - the number of folds run in the Kfold cv
    """

    log = logging.getLogger(__name__)

    for i in range(0, max_folds):
        log.info("Reading {}.csv".format(i))
        data_fold = pd.read_csv(os.path.join(data_directory, "{}.csv".format(i)), header=0)
        # log.info(data_fold)
        if i == 0:
            data = data_fold.copy()
        else:
            data = pd.concat([data, data_fold])

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.columns = ["index", "known", "prediction"]
    data["index"] = [int(ent) for ent in data["index"].values]
    data.set_index("index", inplace=True, drop=True, verify_integrity=True)
    data.sort_index(inplace=True)

    return data

def metrics_for_regression(directories=('LassoCV', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR', 'BayesianRegr'),
    max_folds=10,names=None, smiles=None):
    log = logging.getLogger(__name__)
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.set_size_inches(15.0, 7.0)
    for directory in directories:
        log.info("\n-----\nAnalyzing predictions for model {}\n-----".format(directory))
        data = build_data_from_directory_regr(directory, max_folds=max_folds)

        if names is not None:
            data["names"] = names
        if smiles is not None:
            data["smiles"] = smiles

        variance = explained_variance_score(data['known'], data['prediction'])
        MAE = mean_absolute_error(data['known'], data['prediction'])
        MSE = mean_squared_error(data['known'], data['prediction'])
        RMSE = math.sqrt(MSE)
        R2 = r2_score(data['known'], data['prediction'])
        log.info("\n-----\n Scores for Regressor: Explained Variance: {}, MAE: {}, MSE: {},R2: {}\n-----".format(variance,MAE,MSE,R2))
        f = open("{}/metrics.txt".format(directory), "w")
        f.write(str(variance))
        f.write("\n")
        f.write(str(MAE))
        f.write("\n")
        f.write(str(MSE))
        f.write("\n")
        f.write(str(RMSE))
        f.write("\n")
        f.write(str(R2))
        f.write("\n")
        f.close()


        plt.rcParams.update()
        #ref: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
        # Set the style globally
        # Alternatives include bmh, fivethirtyeight, ggplot,
        # dark_background, seaborn-deep, etc
        plt.style.use('default')

        plt.rcParams['font.family'] = 'Arial'
        #plt.rcParams['font.serif'] = 'Ubuntu'
        #plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['figure.titlesize'] = 24

        if directory.endswith("Regr"):
            fig_title = directory.replace("Regr", "Regressor")
        else:
            fig_title = directory
        titlename = fig_title.replace("_", "")

        fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
        ax.scatter(data['known'], data['prediction'], s=60, alpha=0.6, edgecolors="k", color='silver')
        ax.set_ylabel('Predicted Δ∆$G^{‡}$ (kJ/mol)')
        ax.set_xlabel('Experimental Δ∆$G^{‡}$ (kJ/mol)')
        plt.plot([-0.05, 11], [-0.05, 11], "k:")
        ax.set_title(titlename)

        # add metrics to plot
        ax.text(0.05, 0.95, "R$^2$ = {:.2f}".format(R2), transform=ax.transAxes, fontsize=14, verticalalignment='top')
        ax.text(0.05, 0.90, "Variance = {:.2f}".format(variance), transform=ax.transAxes, fontsize=14, verticalalignment='top')
        ax.text(0.05, 0.85, "MSE = {:.2f}".format(MSE), transform=ax.transAxes, fontsize=14, verticalalignment='top')
        ax.text(0.05, 0.80, "MAE = {:.2f}".format(MAE), transform=ax.transAxes, fontsize=14, verticalalignment='top')


        fig.savefig("{}/regression.png".format(directory))
        fig.show()

def metrics_for_all_classes(directories=("AdaBoost", "DecisionTree", "ExtraTreesClassifier", "GaussianProcess", "LogisticRegression", "NearestNeighbors"), max_folds=10,
                            names=None, smiles=None):
    """
    Function to run over all directories and build the outputs over the folds in folds ML training scheme.
    :param directories: iterable - all of the directories to loop over
    :param max_folds: int - The number of folds which the data was passed over and predictions saved to $(fold_number).csv
    """
    
    log = logging.getLogger(__name__)

    for directory in directories:
          
        log.info("\n-----\nAnalyzing predictions for model {}\n-----".format(directory))
        data = build_data_from_directory(directory, max_folds=max_folds)
        if names is not None:
            data["names"] = names
        if smiles is not None:
            data["smiles"] = smiles
            
        if "names" or "smiles" in data.keys():
            misclassed = which_are_misclassified(data)
        elif "names" and "smiles" in data.keys():
            misclassed = which_are_misclassified(data)
        else:
            log.info("Ideally give at least one of names or smiles to find out which molecules were misclassified")
            misclassed = which_are_misclassified(data, return_indx=True)
        
        with open(os.path.join(directory, "misclassed_molecules.csv"), "w") as fout:
            misclassed.to_csv(fout)
        
        probs = data[["prob0", "prob1"]].to_numpy()
        
        #cmetrics.plot_metrics(data, probabilities=probs, name="{}/metrics_plot.png".format(directory))
        c = cmetrics.get_multi_label_confusion_matrix(df=data, return_dict=True)
        multi_metrics = cmetrics.calculate_multi_label_confusion_based_metrics(df=data, probabilities=probs, positive_label=1, imbalanced=True,
                                                                               plt_filename=os.path.join(directory, "metrics.png"), verbose=False)
        f = open("{}/multi_metrics.txt".format(directory), "w")
        f.write( str(multi_metrics) )
        f.close()

        a_file = open("{}/multi_metrics.pkl".format(directory), "wb")

        pickle.dump(multi_metrics, a_file)

        a_file.close()

        metrics0=[]
        metrics = pd.DataFrame(columns=['tpr','fpr','tnr','fnr','f_half','f1','f2','MCC','precision','recall','roc_auc','pr_auc'], index = [0,1])
        metrics0.append((round(multi_metrics[0]['tpr'], 2)))
        metrics0.append((round(multi_metrics[0]['fpr'], 2)))
        metrics0.append((round(multi_metrics[0]['tnr'], 2)))
        metrics0.append((round(multi_metrics[0]['fnr'], 2)))
        metrics0.append((round(multi_metrics[0]['f_half'], 2)))
        metrics0.append((round(multi_metrics[0]['f1'], 2)))
        metrics0.append((round(multi_metrics[0]['f2'], 2)))
        metrics0.append((round(multi_metrics[0]['matthews_correlation_coefficient'], 2)))
        metrics0.append((round(multi_metrics[0]['precision'], 2)))
        metrics0.append((round(multi_metrics[0]['recall'], 2)))
        metrics0.append((round(multi_metrics[0]['roc_auc'], 2)))
        metrics0.append((round(multi_metrics[0]['pr_auc'], 2)))

        metrics.loc[0] = metrics0

        metrics1=[]
        metrics1.append((round(multi_metrics[1]['tpr'], 2)))
        metrics1.append((round(multi_metrics[1]['fpr'], 2)))
        metrics1.append((round(multi_metrics[1]['tnr'], 2)))
        metrics1.append((round(multi_metrics[1]['fnr'], 2)))
        metrics1.append((round(multi_metrics[1]['f_half'], 2)))
        metrics1.append((round(multi_metrics[1]['f1'], 2)))
        metrics1.append((round(multi_metrics[1]['f2'], 2)))
        metrics1.append((round(multi_metrics[1]['matthews_correlation_coefficient'], 2)))
        metrics1.append((round(multi_metrics[1]['precision'], 2)))
        metrics1.append((round(multi_metrics[1]['recall'], 2)))
        metrics1.append((round(multi_metrics[1]['roc_auc'], 2)))
        metrics1.append((round(multi_metrics[1]['pr_auc'], 2)))

        metrics.loc[1] = metrics1

        metrics.to_csv("{}/metrics.csv".format(directory))
        metrics.to_latex("{}/metric.tex".format(directory), index=True)
        log.info("Over all data points including smote points")
        display(Image(os.path.join(directory, "metrics.png")))
        
        log.info("Over all data points including smote points:\n{}".format(metrics))


        probs = data[["prob0", "prob1"]].to_numpy()

        metrics = cmetrics.calculate_confusion_based_metrics(df=data, probabilities=probs, positive_label=1)

        #cmetrics.plot_metrics_skplt(data, probabilities=probs, name="{}/_original_data_metrics_plot.png".format(directory))
        c = cmetrics.get_multi_label_confusion_matrix(df=data, return_dict=True)

        multi_metrics = cmetrics.calculate_multi_label_confusion_based_metrics(df=data, probabilities=probs, positive_label=1, imbalanced=True, 
                                                                               plt_filename=os.path.join(directory, "metrics_real_only.png"), verbose=False)

        f = open("{}/multi_metrics_original.txt".format(directory),"w")
        f.write( str(multi_metrics) )
        f.close()

        a_file = open("{}/multi_metrics_original.pkl".format(directory), "wb")

        pickle.dump(multi_metrics, a_file)

        a_file.close()

        metrics0=[]
        metrics = pd.DataFrame(columns=['tpr','fpr','tnr','fnr','f_half','f1','f2','MCC','precision','recall','roc_auc','pr_auc'], index = [0,1])
        metrics0.append((round(multi_metrics[0]['tpr'], 2)))
        metrics0.append((round(multi_metrics[0]['fpr'], 2)))
        metrics0.append((round(multi_metrics[0]['tnr'], 2)))
        metrics0.append((round(multi_metrics[0]['fnr'], 2)))
        metrics0.append((round(multi_metrics[0]['f_half'], 2)))
        metrics0.append((round(multi_metrics[0]['f1'], 2)))
        metrics0.append((round(multi_metrics[0]['f2'], 2)))
        metrics0.append((round(multi_metrics[0]['matthews_correlation_coefficient'], 2)))
        metrics0.append((round(multi_metrics[0]['precision'], 2)))
        metrics0.append((round(multi_metrics[0]['recall'], 2)))
        metrics0.append((round(multi_metrics[0]['roc_auc'], 2)))
        metrics0.append((round(multi_metrics[0]['pr_auc'], 2)))

        metrics.loc[0] = metrics0


        metrics1=[]
        metrics1.append((round(multi_metrics[1]['tpr'], 2)))
        metrics1.append((round(multi_metrics[1]['fpr'], 2)))
        metrics1.append((round(multi_metrics[1]['tnr'], 2)))
        metrics1.append((round(multi_metrics[1]['fnr'], 2)))
        metrics1.append((round(multi_metrics[1]['f_half'], 2)))
        metrics1.append((round(multi_metrics[1]['f1'], 2)))
        metrics1.append((round(multi_metrics[1]['f2'], 2)))
        metrics1.append((round(multi_metrics[1]['matthews_correlation_coefficient'], 2)))
        metrics1.append((round(multi_metrics[1]['precision'], 2)))
        metrics1.append((round(multi_metrics[1]['recall'], 2)))
        metrics1.append((round(multi_metrics[1]['roc_auc'], 2)))
        metrics1.append((round(multi_metrics[1]['pr_auc'], 2)))

        metrics.loc[1] = metrics1

        metrics.to_csv("{}/metrics_original.csv".format(directory))
        log.info("Over all REAL data points NOT including smote points")
        display(Image(os.path.join(directory, "metrics_real_only.png")))
        
        log.info("Over all REAL data points NOT including smote points:\n{}".format(metrics))
    
def which_are_misclassified(data, known_column_label="known", predicted_column_label="prediction", return_indx=False):
    """
    Function to get the molecules which are misclassified
    :param data: pandas dataframe - columns of at least known_column_label and predicted_column_label good to have smiles and name
    """
    log = logging.getLogger(__name__)
    
    log.info(data)
    
    if return_indx is False:
        df = data[data[known_column_label] != data[predicted_column_label]]
        if "names" in df.keys() and "smiles" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{} {}".format(name, smile) for name, smile in zip(df["names"].values,df["smiles"].values)])))
        elif "names" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(name) for name in df["names"].values])))
        elif "smiles" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(smile) for smile in df["smiles"].values])))
    
    else:
        df = data[data[known_column_label] != data[predicted_column_label]]
        log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(ith) for ith in df.index])))
    
    return df


def get_feature_names_from_column_transformers(ctransformer):
    """
    Function to get feature names from sklearn column transofrmers see
    https://towardsdatascience.com/extracting-plotting-feature-names-importance-from-scikit-learn-pipelines-eb5bfa6a31f4
    :param ctransformer: columnTransformer - sklearn transformer
    """

    log = logging.getLogger(__name__)
    log.info("\n-----\nGetting feature names from column transformer\n-----\n")

    new_feature_names = []

    if not hasattr(ctransformer, 'transformers_'):
        raise ValueError("The ColumnTransformer must be fitted before extracting feature names")

    for ith, (trans_name, trans, original_feature_names) in enumerate(ctransformer.transformers_):
        log.info(f"Transformer index: {ith}\nTransformer name: {trans_name}\nTransformer: {trans}\nOriginal feature names: {original_feature_names}\n")

        if hasattr(trans, "get_feature_names_out"):
            if isinstance(trans, OneHotEncoder):
                # Use get_feature_names_out with the original feature names
                names = list(trans.get_feature_names_out(original_feature_names))
            else:
                names = list(trans.get_feature_names_out())
        elif hasattr(trans, "features_"):
            if hasattr(trans, "indicator_"):
                missing_indicator_indices = trans.indicator_.features_
                missing_features = [f"{original_feature_names[ith]}_missing_flag" for ith in missing_indicator_indices]
                names = original_feature_names + missing_features
            else:
                names = original_feature_names
        else:
            names = original_feature_names

        # Add feature names only if they are strings or can be converted to strings
        new_feature_names.extend([str(name) for name in names if isinstance(name, str)])

    # Remove duplicates and sort to match the expected order
    seen = set()
    new_feature_names = [x for x in new_feature_names if not (x in seen or seen.add(x))]

    log.info(f"Extracted feature names: {new_feature_names}")

    return new_feature_names

def feature_categorization(features_df, feature_types="some_categorical", categorical_indxs=None):
    '''

    function to determine the type of scaling for the features

    :param features_df: dataframe of original features
    :param feature_types: are they all categorical, some categorical or non categorical features
    :param categorical_indxs: indices of the categorial features. We sugguest the user provide the indices for the categorial features when feature_types="some_categorial". If not provided, the function will try to guess the categorial features automatically.
    :return: new dataframe with scaled and one-hotencoded features
    '''

    log = logging.getLogger(__name__)

    # Convert boolean columns to integers
    for column in features_df.columns:
        if features_df[column].dtype == bool:
            features_df[column] = features_df[column].astype(int)

    feature_columns = features_df.columns

    # Automatic selection of categorical features
    if categorical_indxs is None:
        log.info("-----")
        log.info("Categorical features not provided. Trying to guess the categorical features automatically.")
        log.info("-----")

        categorical_indxs = []
        for i, column in enumerate(features_df.columns):
            if features_df[column].dtype == 'object' or features_df[column].dtype == 'bool':
                categorical_indxs.append(i)

        categorical_features = [feature_columns[i] for i in categorical_indxs]
        log.info("Automatically assigned categorical indices:\n{} {}".format(categorical_indxs, len(categorical_indxs)))
        log.info("Automatically assigned categorical features:\n{} {}".format(categorical_features,
                                                                              len(categorical_features)))
        log.info(
            "**Please check if the automatic assignment is correct. If not, please provide the indices of the categorical features.**")
        log.info("-----")

    # No categorical features, only scale the data as numbers
    if feature_types == "no_categorical":
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_transformer = MinMaxScaler()
        features_df[numeric_features] = numerical_transformer.fit_transform(features_df[numeric_features])
        log.info(pd.DataFrame(features_df, columns=feature_columns))
        features_df = pd.DataFrame(features_df, columns=feature_columns)
        categorical_indxs = None  # Ensure categorical_indxs is None when there are no categorical features

    elif feature_types == "some_categorical":
        numeric_features = [feature_columns[i] for i in range(len(feature_columns)) if i not in categorical_indxs]
        categorical_features = [feature_columns[i] for i in range(len(feature_columns)) if i in categorical_indxs]

        numerical_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Apply transformations
        if numeric_features:
            features_df[numeric_features] = numerical_transformer.fit_transform(features_df[numeric_features])

        if categorical_features:
            transformed_categorical_features = categorical_transformer.fit_transform(features_df[categorical_features])
            categorical_feature_names = categorical_transformer.get_feature_names_out(categorical_features)

            if len(categorical_feature_names) != transformed_categorical_features.shape[1]:
                raise ValueError(
                    f"Feature names length ({len(categorical_feature_names)}) does not match encoded data shape ({transformed_categorical_features.shape[1]})")

            # Ensure that transformed_categorical_features shape matches the length of categorical_feature_names
            transformed_categorical_df = pd.DataFrame(transformed_categorical_features,
                                                      columns=categorical_feature_names, index=features_df.index)

            features_df = pd.concat([features_df.drop(columns=categorical_features), transformed_categorical_df],
                                    axis=1)

        log.info(features_df.columns)

    elif feature_types == "categorical":
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformed_features = categorical_transformer.fit_transform(features_df)
        feature_names = categorical_transformer.get_feature_names_out(features_df.columns)

        if len(feature_names) != transformed_features.shape[1]:
            raise ValueError(
                f"Feature names length ({len(feature_names)}) does not match encoded data shape ({transformed_features.shape[1]})")

        features_df = pd.DataFrame(transformed_features, columns=feature_names, index=features_df.index)
        log.info(features_df)

    elif feature_types == "none":
        log.info("No scaling or encoding applied")
        return features_df, None

    return features_df, categorical_indxs, numerical_transformer, categorical_transformer

def scale_test_set(features_df, categorical_indxs, numerical_transformer, categorical_transformer):
    """
        Scale and transform the test set features using the fitted transformers from the training set.

        This function applies the same numerical scaling and categorical encoding transformations
        that were applied to the training set on the test set. It ensures that the test set features
        are processed consistently with the training set.

        :param features_df: pandas DataFrame
            The dataframe containing the test set features to be scaled and transformed.

        :param categorical_indxs: list of int
            The list of column indices for categorical features in the test set. If the test set has
            no categorical features, this should be an empty list.

        :param numerical_transformer: sklearn.preprocessing.Transformer
            The fitted numerical transformer (e.g., MinMaxScaler) from the training set.

        :param categorical_transformer: sklearn.preprocessing.OneHotEncoder
            The fitted categorical transformer (OneHotEncoder) from the training set. If the test set
            has no categorical features, this should be None.

        :return: pandas DataFrame
            The dataframe containing the scaled and transformed features of the test set.
        """

    numeric_features = [features_df.columns[i] for i in range(len(features_df.columns)) if i not in categorical_indxs]

    # Transform the numerical features using the numerical_transformer from the training set.
    numerical_features_transformed = numerical_transformer.transform(features_df[numeric_features])

    # Transform the categorical features using the categorical_transformer from the training set.
    if categorical_indxs:
        categorical_features_transformed = categorical_transformer.transform(features_df.iloc[:, categorical_indxs])
        categorical_feature_names = categorical_transformer.get_feature_names_out(
            features_df.columns[categorical_indxs])

        # Convert transformed features to DataFrames
        categorical_df_transformed = pd.DataFrame(categorical_features_transformed, columns=categorical_feature_names,
                                                  index=features_df.index)
    else:
        categorical_df_transformed = pd.DataFrame(index=features_df.index)

    # Extract feature names after transformation
    numerical_feature_names = numeric_features

    # Convert transformed features to DataFrames
    numerical_df_transformed = pd.DataFrame(numerical_features_transformed, columns=numerical_feature_names,
                                            index=features_df.index)

    # Concat the tranformed features
    scaled_features_df = pd.concat([numerical_df_transformed, categorical_df_transformed], axis=1)

    return scaled_features_df

def ensemble(csv_files):
    """
        Compute ensemble predictions from multiple CSV files containing predicted and actual values,
        calculate regression metrics, and plot the results.

        Args:
        csv_files (list of str): List of CSV file paths. Each CSV file should contain 'actual' and 'predicted' columns.

        Returns:
        dict: Dictionary containing the calculated regression metrics:
              - 'mae': Mean Absolute Error
              - 'mse': Mean Squared Error
              - 'rmse': Root Mean Squared Error
              - 'r2': R-squared score

        Notes:
        - Reads predicted values from each CSV file.
        - Calculates the mean of predicted values.
        - Uses actual values from the first CSV file.
        - Computes regression metrics (MAE, MSE, RMSE, R2) based on the ensemble predictions.
        - Plots the scatter plot of actual vs. predicted values.
    """

    log = logging.getLogger(__name__)
    predicted_values = []

    ensemble_metrics = {}
    # Read predicted values from each CSV file
    for file in csv_files:
        log.info("\n\t Reading:{} ".format(file))
        data = pd.read_csv(file)
        predicted_values.append(data['predicted'])

    log.info("\n\t Gathered all predictions ")
    # Calculate mean value of predicted values
    mean_predicted = np.mean(predicted_values, axis=0)

    # Get actual values from the first CSV file
    actual = pd.read_csv(csv_files[0])['actual']
    log.info("\n\tCalculating metrics")
    # Calculate regression metrics
    mae = mean_absolute_error(actual, mean_predicted)
    mse = mean_squared_error(actual, mean_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, mean_predicted)

    ensemble_metrics['mae'] = mae
    ensemble_metrics['mse'] = mse
    ensemble_metrics['rmse'] = rmse
    ensemble_metrics['r2'] = r2

    plt.figure(figsize=(6, 4))
    plt.scatter(actual, mean_predicted, color='blue', marker='x')
    plt.ylabel('Predicted Δ∆$G^{‡}$ (kJ/mol)', fontsize=13)
    plt.xlabel('Experimental Δ∆$G^{‡}$ (kJ/mol)', fontsize=13)
    plt.plot([-0.05, 11], [-0.05, 11], "k:")

    plt.xticks(np.arange(0, 12, step=2))
    plt.yticks(np.arange(0, 12, step=2))


    return ensemble_metrics