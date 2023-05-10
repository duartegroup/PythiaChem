#!/usr/bin/env python

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
import pickle
import csv

#RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors

# scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, RFE
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # This is sklearns auto-scaling function
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
# scikit-imbalenced learn
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

import logging 
logging.basicConfig(format='%(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

import math

import py3Dmol

from joblib import dump, load



# Own modules
from . import classification_metrics as cmetrics

# stats and plotting
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend
from mlxtend.evaluate import permutation_test as pt
from IPython.display import Image, display

def autoscale(df):
    """
    scale a pandas dataframe using autoscaling/z-scaling
    :param df: pandas dataframe - data frame to be returned scaled
    """
    df_tmp = df.copy()
    normalized_df = (df_tmp-df_tmp.mean())/df_tmp.std()
    return normalized_df

def minmaxscale(df):
    """
    scale a pandas dataframe using min max scaling
    :param df: pandas dataframe - data frame to be returned scaled
    """
    
    df_tmp = df.copy()
    normalized_df = (df_tmp-df_tmp.min())/(df_tmp.max()-df_tmp.min())
    return normalized_df

def get_mol_from_smiles(smiles, canonicalize=True):
    """ 
    Function to make a mol object based on smiles
    :param smiles: str - SMILES string
    :param canonicalize: True/False - use RDKit canonicalized smile or the input resprectively
    """

    log = logging.getLogger(__name__)

    if canonicalize is True:
        s = Chem.CanonSmiles(smiles, useChiral=1)
    else:
        s = smiles
    mol = Chem.MolFromSmiles(s)
    log.debug("Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
              "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(smiles, s))
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    return mol

def smiles_to_molcule(s, addH=True, canonicalize=True, threed=True, add_stereo=False, remove_stereo=False, random_seed=10459, verbose=False):
    """ 
    :param s: str - smiles string
    :param addH: bool - Add Hydrogens or not
    """

    log = logging.getLogger(__name__)

    mol = get_mol_from_smiles(s, canonicalize=canonicalize)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)
    
    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)
    
        if verbose is True:
            for atom in mol.GetAtoms():
                log.info("Atom {} {} in molecule from smiles {} tag will be cleared. "
                        "Set properties {}.".format(atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)


    if threed:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    return mol 


def draw_3D_mol(m,p=None,confId=-1):
        mb = Chem.MolToMolBlock(m,confId=confId)
        if p is None:
            p = py3Dmol.view(width=400,height=400)
        p.removeAllModels()
        p.addModel(mb,'sdf')
        p.setStyle({'stick':{}})
        p.setBackgroundColor('0xeeeeee')
        p.zoomTo()
        return p.show()
    



def RDkit_descriptors(mols):
    """
    Generate both 2D and 3D RDkit Descriptors. There are 208 in total.

    :param mols: a list of RDkit mol objects with EmBedding
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # Calculate all descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
        
    df = pd.DataFrame(Mol_descriptors,columns=desc_names)
    df = df.fillna(0)
        
    return df




def feature_categorization(features_df, feature_types = "some_catagorical"):

    """
    In the case there are both categorial and non-categorial features, this function split the categorial features.
    """
    #Automatic selection of categorical feaures
    catagorical_indxs = []
    for column in features_df:
        arr = features_df[column]
        #print(arr.name)
        if arr.dtype.name == 'int64':
            index_no = features_df.columns.get_loc(arr.name)
           #print(index_no)
            catagorical_indxs.append(index_no)



    feature_columns = features_df.columns

    # Backup
    backup_feats_df = features_df.copy()

    # None catagorical only scale the data as numbers
    if feature_types == "no_catagorical":
        mm_scaler = MinMaxScaler()
        features_df = mm_scaler.fit_transform(features_df)
        log.info(pd.DataFrame(features_df, columns=feature_columns))
        features_df = pd.DataFrame(features_df, columns=feature_columns)

    # Some catagorical - Need to provide the indexes
    elif feature_types == "some_catagorical":
        numeric_features = [feature_columns[i] for i in range(len(feature_columns)) if i not in catagorical_indxs]
        numerical_transformer = MinMaxScaler()
        categorical_features = [feature_columns[i] for i in range(len(feature_columns)) if i in catagorical_indxs]
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        if any(ent in categorical_features for ent in numeric_features):
            log.warning("WARNING - numeric and catagorical feature specififed overlap")
            log.info(numeric_features)
            log.info(categorical_features)
        else:
            log.info("Numerical features:\n{} {}".format(numeric_features, len(numeric_features)))
            log.info("Catagorical features:\n{} {}".format(categorical_features, len(catagorical_indxs)))

        preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numeric_features),
            ('catagorical', categorical_transformer, categorical_features)])

        features_df = preprocessor.fit_transform(features_df)
        #display(features_df)
        feature_names = get_feature_names_from_column_transformers(preprocessor)
        catagorical_indxs = [i for i in range(len(numeric_features), len(feature_names))]
        log.info(feature_names)

        log.info(pd.DataFrame(features_df, columns=feature_names))
        features_df = pd.DataFrame(features_df, columns=feature_names)
        log.info("catagorical indexes {}".format(catagorical_indxs))
        log.info("Catagorical features start on column name {} and end on {}".format(features_df.columns[catagorical_indxs[0]], features_df.columns[catagorical_indxs[-1]]))

    # All catagorical
    elif feature_types == "catagorical":
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        features_df = categorical_transformer.fit_transform(features_df).toarray()
        feature_names = [categorical_transformer.get_feature_names(feature_columns)]
        features_df = pd.DataFrame(features_df, columns=feature_names)
        log.info(features_df)

    # No scaling or other encoding
    else:
        log.info("No scaling")

    return features_df



def clean_correlated_features(features, thresh = 0.8, clean01 = True):
    """
    This function calculates the pearson correlation coefficients for all pairwise combination of all the features.
    It then remove highly-correlated features and output the cleaned features.

    :param: thresh: pearson correlation coefficient threashold above which features are classified as highly-correlated.
    """
    #calculate correlation coefficients for all pairwise combinations
    df = features.corr()
    
    to_remove = []
    for col in df.columns:
        corr = df.index[df[col].between(thresh, 1.0, inclusive = 'both')].tolist()
        if len(corr) > 1:
            #print(col, corr)
            to_remove.append(corr)
    
    to_remove2 = sum(to_remove,[])
    #print(to_remove2)
    
    to_keep = set(df.columns) - set(to_remove2)    
    features_cleaned = features[list(to_keep)]

    if clean01 == True:
        feats1 = features_cleaned.loc[:, (features_cleaned != 0).any(axis=0)]
        feats2 = feats1.loc[:, (feats1 != 1).any(axis=0)]
        log.info(feats2)
        return feats2
    
    if clean01 == False:
        log.info(features_cleaned)
        return features_cleaned
    

def correlation_heatmap(features, annot=True):
    """
    This function plots the feature correlation matrix as a heatmap.
    """

    correlations = features.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmin=-1.0, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, 
                annot=annot, 
                cbar_kws={"shrink": .70},
                cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.show()











def df_corr(x, y, corr_method):
    """
    """
    df = pd.DataFrame(data=np.array([x, y]).T)
    corr = df.corr(method=corr_method)
    
    return corr




def find_correlating_features(features, targets, thresh=0.4, plot=True, process_non_numeric=True,
                              corr_method="spearman", sig_metric=None, significance=False,
                              random_seed=105791, n_sample=10000, sig_level=0.05):
    """
    Pulls out features which correlate with a property bay at least the threshold as measure by the Pearson correlation
    coefficent (R) [-1, 1].
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
        else:
            pass

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

                # correlations = tmp_df.corr(method=corr_method)
                # if abs(correlations.loc[ent, "property"]) > thresh:
                #     print("gamiesai")
                #     usable_features.append(ent)
                #     if significance is True:
                #         # print(feat)
                #         p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
                #         significant = True if p_value < sig_level else False
                #         log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
                #     else:
                #         log.info("{}: {:.4f}" .format(ent, correlations.loc[ent, "property"]))
                del tmp_df
            else:
                del tmp_df

    if significance is False:
        return usable_features
    else:
        return usable_features, significant_features


def permutation_test(features, targets, n_sample=10000, corr_method="pearson", random_seed=105791,  sig_level=0.05):
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
    
    feat = series.values
    tmp_df = pd.DataFrame(data=np.array([feat, targets]).T, columns=[ent, "property"])
    tmp_df.dropna(inplace=True)
    
    try:
            correlations = tmp_df.corr(method=corr_method)
            p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
            significant = True if p_value < sig_level else False
            log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
            
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
    
    return correlations, p_value, significant


        
def constant_value_columns(df):
    """
    Function to find constant value columns
    :param df: Pandas dataframe - dataframe to find non unique columns
    """
    
    log = logging.getLogger(__name__)
    
    cols = [name for name in df.columns if df[name].nunique() == 1]
    
    return cols



def grid_search_reg_parameters(rgs, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=5, name=None,scoring=("r2","neg_root_mean_squared_error")):
    """
    Grid search regressor hyperparams and find the best report metrics if requested
    """
    
    # Grid search model optimizer

    parameters = rgs_options[rgs_names[iteration]]
    log.info("\tname: {} parameters: {}".format(name, parameters))
    
    optparam_search = GridSearchCV(rgs, parameters, cv=cv, error_score=np.nan, scoring=scoring, refit=scoring[0], return_train_score=True)
#     print("\tCV xtrain: {}".format(Xtrain))
    
    optparam_search.fit(Xtrain, ytrain)
    opt_parameters = optparam_search.best_params_
    
    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv("{}/{}_grid_search_metrics.csv".format(name,name))
        log.info("\tBest parameters; {}".format(opt_parameters))

        for mean, std, params in zip(optparam_search.cv_results_["mean_test_{}".format(scoring[0])], optparam_search.cv_results_["std_test_{}".format(scoring[0])], optparam_search.cv_results_['params']):
            log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass
    
    return opt_parameters


def kfold_test_imbalenced_regressors_with_optimization(df, targets, regressors, rgs_options, scale=True, cv=5, n_repeats=20, rgs_names=None, no_train_output=False, test_set_size=0.2, smiles=None, names=None, random_seed=107901, overwrite=True):
    """
    function to run classification test over classifiers using imbalenced resampling
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
        
    if rgs_names is None:
        rgs_names = [i for i in range(0, len(regressors))]
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.

    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting regression")
    for name, classf in zip(rgs_names, regressors):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        scores = []
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
        
        # Loop over  Kfold here a
        
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = targets.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = targets.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_reg_parameters(classf, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
#             sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            
            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = mean_absolute_error(ytest, predicted_clf)
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   
            
            pred = [list(test_indx),list(ytest),list(predicted_clf)]
            
            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
        
        del predictions[:]
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        log.info("The average scores are: {}, and the std scores are: {}".format(average_scores, std_scores))
        iteration = iteration + 1

#             list_score.append(clf_score)
            

    
def directory_names_from_regressor_names(regressor_names):
    """
    Function to get the names of the directories from the classifier scripts
    :param regressor_names: list - list of classifier names
    """
    
    names = ["{}".format("_".join(name.split())) for name in regressor_names]
    
    return names
    
def build_data_from_directory(data_directory, max_folds=10):
    """
    Fucntion to build a set of data from csv files names K.csv where K is the fold number and the csv
    is the predictions for the test data from that fold
    :param directory: str - name of the directory to read the csv files from
    :param max_fold: int - the number of folds run in the Kfold cv
    """
    
    log = logging.getLogger(__name__)
    
    for i in range(0, max_folds):
        log.info("Reading {}.csv".format(i))
        data_fold = pd.read_csv(os.path.join(data_directory, "{}.csv".format(i)), header=0)
        #log.info(data_fold)
        if i == 0:
            data = data_fold.copy()
        else:
            data = pd.concat([data, data_fold])

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.columns = ["m_index", "known", "prediction", "prob0", "prob1"]
    data["m_index"] = [int(ent) for ent in data["m_index"].values]
    data.set_index("m_index", inplace=True, drop=True, verify_integrity=True)
    data.sort_index(inplace=True)
    
    return data


def build_data_from_directory_regr(data_directory, max_folds=10):
    """
    Fucntion to build a set of data from csv files names K.csv where K is the fold number and the csv
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

def metrics_for_regression(directories=('LassoCV'),
 # 'KNeighborsRegressor',
 # 'Decision_Tree_Regressor',
 # 'SVR',
 # 'Bayesian_Regr',
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

        plt.scatter(data['known'], data['prediction'], color='grey',marker='o')
        plt.ylabel('Predicted logEC50', fontsize=13)
        plt.xlabel('Experimental logEC50', fontsize=13)
        plt.title(directory)
        plt_text = "Scores for Regressor: \nExplained Variance: {:1f} \nMAE: {:1f} \nMSE: {:1f} \nR2: {:1f}".format(variance,MAE,MSE,R2)
        plt.text(-4.8,0.2, plt_text)
        plt.plot([data['known'].min(), data['known'].max()], [data['known'].min(), data['known'].max()], "k:")

        plt.xticks(np.arange(-5, 4, step=1))
        plt.yticks(np.arange(-3, 3, step=1))
        plt.savefig("{}/regression.png".format(directory))
        plt.show()

def metrics_for_all_classes(directories=("AdaBoost", "Decision_Tree", "ExtraTreesClassifier", "Gaussian_Process", "Logistic_Regression", "Nearest_Neighbors"), max_folds=10, 
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

        ####
        """
        IMPORTANT BE CAREFUL HERE! This index needs to change depending on how many SMOTE  are created!!!
        89-110 are for OPERA & mordred with conditions classification.
        
        When only Mordred are used we have 101 original data!
        """
        # data.drop(data.index[101:134], axis=0, inplace=True)
        data.drop(data.index[119:168], axis=0, inplace=True)
        # the above is with added data
        # data.drop(data.index[89:110], axis=0,inplace=True)
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


def get_feature_names_from_column_transformers(ctransformer):
    """
    Function to get feature names from sklearn column transofrmers see
    https://towardsdatascience.com/extracting-plotting-feature-names-importance-from-scikit-learn-pipelines-eb5bfa6a31f4
    :param ctransformer: columnTransformer - sklearn transformer
    """

    log = logging.getLogger(__name__)

    log.info("\n-----\nGetting feature names from column transformer\n-----\n")

    new_feature_names = []
    trans_list = []

    for ith, ent in enumerate(ctransformer.transformers_):
        trans_name, trans, original_feature_name = ent
        log.info("Transformer index: {}\nTranformer name: {}\nTransformer: {}\nOriginal feature names: {}\n".format(ith,
                                                                                                                    trans_name,
                                                                                                                    trans,
                                                                                                                    original_feature_name))
        if hasattr(trans, "get_feature_names"):
            if isinstance(trans, OneHotEncoder):
                names = list(trans.get_feature_names(original_feature_name))
            else:
                names = list(trans.get_feature_names())

        elif hasattr(trans, "features_"):
            missing_indicator_indices = trans.indicator_.features_
            missing_features = [original_feature_name[ith] + "_missing_flag" for ith in missing_indicator_indices]
            names = original_feature_name + missing_features

        else:
            names = original_feature_name

        new_feature_names = new_feature_names + names
        trans_list = trans_list + [trans_name] * len(names)

    return new_feature_names



def grid_search_reg_parameters_ML(rgs, Xtrain, ytrain, rgs_options, rgs_names, iteration, no_train_output, cv=5, name=None,scoring=("r2","neg_root_mean_squared_error")):
    """
    Grid search regressor hyperparams and find the best report metrics if requested
    """
    
    # Grid search model optimizer

    parameters = rgs_options
    log.info("\tname: {} parameters: {}".format(name, parameters))
    
    optparam_search = GridSearchCV(rgs, parameters, cv=cv, error_score=np.nan, scoring=scoring, refit=scoring[0], return_train_score=True)
#     print("\tCV xtrain: {}".format(Xtrain))
    
    optparam_search.fit(Xtrain, ytrain)
    opt_parameters = optparam_search.best_params_
    
    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv("{}/{}_grid_search_metrics.csv".format(name,name))
        log.info("\tBest parameters; {}".format(opt_parameters))

        for mean, std, params in zip(optparam_search.cv_results_["mean_test_{}".format(scoring[0])], optparam_search.cv_results_["std_test_{}".format(scoring[0])], optparam_search.cv_results_['params']):
            log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass
    
    return opt_parameters

def test_regressors_with_optimization(df, targets, test_df, test_targets, regressors, rgs_options, scale=True, cv=5, 
                                      n_repeats=20, rgs_names=None, 
                                    no_train_output=False, random_seed=107901, overwrite=True):
    """
    function to run regression 
    """
    all_test_metrics = []
    iteration = 0
        
    if rgs_names is None:
        rgs_names = [i for i in range(0, len(regressors))]

    log.info("Starting regression")
    for name, reg in zip(rgs_names, regressors):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each regressor
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
            
#         Train set
        Xtrain = df
#         log.info("Train X\n{}".format(Xtrain))
        ytrain = targets
#         log.info("Train Y\n{}".format(ytrain))
        
#         Find optimal parameters
        opt_param = grid_search_reg_parameters_ML(reg, Xtrain, ytrain, rgs_options[name], rgs_names, iteration, no_train_output, cv=cv, name=name)
            
        # Fit model using optimized parameters 
        reg.set_params(**opt_param)
        reg.fit(Xtrain, ytrain)
        
#         Save Model        
        targ = 'ddg'
        dump(reg, "{}/model_{}_{}_fixed.joblib".format(name,name, targ))
        
#         Test set    
        Xtest = test_df
#         log.info("Test X\n{}".format(Xtest))
        ytest = test_targets
#         log.info("Test Y\n{}".format(ytest))
        
#         Evaluate the model based on the validation set
        test_predictions = reg.predict(Xtest)
        train_predictions = reg.predict(Xtrain)
        log.info("\n\t The predictions are: {}".format(test_predictions))


        df_output_train = pd.DataFrame({'train_target': ytrain, 'train_prediction': train_predictions})
        df_output_test = pd.DataFrame({'test_target': ytest, 'test_prediction': test_predictions})
        df_output_train.to_csv('output_{}_train.csv'.format(name))
        df_output_test.to_csv('output_{}_test.csv'.format(name))
        
#         Calculate metrics for regression
        test_metrics = {}
        test_metrics['name'] = name
        test_metrics["variance"] = round(explained_variance_score(test_targets, test_predictions),2)
        test_metrics["MAE"] = round(mean_absolute_error(test_targets, test_predictions),2)
        test_metrics["MSE"] = round(mean_squared_error(test_targets, test_predictions),2)
        test_metrics["RMSE"] = math.sqrt(round(mean_squared_error(test_targets, test_predictions),2))
        test_metrics["R2"] = round(r2_score(test_targets, test_predictions),2)
    
        log.info(test_metrics)
        all_test_metrics.append(test_metrics)
    
    
        fig, ax = plt.subplots(1,2, figsize=(10, 4))
        ax[0].scatter(targets, train_predictions,color='b')
        ax[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], color='gray', linestyle='dashed', linewidth=2)
        ax[0].set_xlabel('y train')
        ax[0].set_ylabel('y train pred')
        ax[0].set_xticks(np.arange(-5, 4, step=1))
        ax[0].set_yticks(np.arange(-3, 3, step=1))

        ax[1].scatter(test_targets,test_predictions,color='r')
        ax[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], color='gray', linestyle='dashed', linewidth=2)
        ax[1].set_xlabel('y test')
        ax[1].set_ylabel('y test pred')
        ax[1].set_xticks(np.arange(-5, 4, step=1))
        ax[1].set_yticks(np.arange(-3, 3, step=1))


        fig.suptitle(name)
        fig_name = name + '_fitting.png'
        plt.savefig('{}/{}.png'.format(name,name))
    
    all_test_metrics = pd.DataFrame(all_test_metrics)
    all_test_metrics.to_csv('test_metrics.csv')
