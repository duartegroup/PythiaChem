#!/usr/bin/env python
import logging
import math
import numpy as np
import os
import pandas as pd
import sys

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

def logarithm2(df):
    """
    scale a pandas dataframe using logarithmic scaling, chose between 2 or 10
    :param df: pandas dataframe - data frame to be returned scaled
    """

    df_tmp = df.copy()
    log_df = np.log2(df_tmp)
    return log_df

def logarithm10(df):
    """
    scale a pandas dataframe using logarithmic scaling, chose between 2 or 10
    :param df: pandas dataframe - data frame to be returned scaled
    """

    df_tmp = df.copy()
    log_df = np.log10(df_tmp)
    return log_df