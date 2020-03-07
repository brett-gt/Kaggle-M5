""" Preprocess contains functions used to preprocess dataset including feature selection and
    normalization.  Specific details of operations to be performed are defined in Settings.py.
"""

import Globals as globals

import numpy as np
import pandas as pd
import pickle
import numpy as np
import datetime as datetime

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

import pywt
from statsmodels.robust import mad


import os

#--------------------------------------------------------------------------------
path = "Data/"


#--------------------------------------------------------------------------------
def load_data():
    calendar = pd.read_csv(path + "calendar.csv")
    calendar = calendar.apply(lambda x: x.fillna(0),axis=0)

    prices = pd.read_csv(path + "sell_prices.csv")
    sample_sub = pd.read_csv(path + "sample_submission.csv")
    train_val_set = pd.read_csv(path + "sales_train_validation.csv")

    return calendar, prices, sample_sub, train_val_set;

#--------------------------------------------------------------------------------
def get_d_from_date(calendar, date):
    """ The common date identified is d_XXXX where XXXX is a numbered list from
        1 to 1969 (currently).  Use this function to get the d_XXXX that corresponds
        to a specific date.  Uses calendar dataframe (loaded from calendar.csv) for
        lookup.

        Arguments:
            calendar - calendar dataframe (pre-loaded from calendar.csv)
            date - (datetime) format year-month-day with 0 padding

        Returns:
            string - d_XXXX or null if not found
    """  
    print("get_d_from_date: Looking for " + str(date))
    result = calendar.loc[calendar['date']==date].d

    if(result.empty):
        return None
    else:
        return result.values[0]

#--------------------------------------------------------------------------------
def get_date_from_d(calendar, d):
    """ The common date identified is d_XXXX where XXXX is a numbered list from
        1 to 1969 (currently).  Use this function to get the date associated with a
        d_XXXX  Uses calendar dataframe (loaded from calendar.csv) for
        lookup.

        Arguments:
            calendar - calendar dataframe (pre-loaded from calendar.csv)
            d - d_XXXX 

        Returns:
            datetime - format year-month-day with 0 padding
    """  
    print("get_date_from_d: Looking for " + str(d))
    result = calendar.loc[calendar['d']==d].date

    if(result.empty):
        return None
    else:
        return datetime.datetime.strptime(result.values[0],globals.DATE_FORMAT)

#--------------------------------------------------------------------------------
#https://www.kdnuggets.com/2019/06/select-rows-columns-pandas.html
def get_d_range(data, d_start, d_end, columns=True):
    """ Get a range of rows from a dataset using the d_XXXX value.

        Arguments:
            data- dataset that contains a "d" column (d_XXXX)
            d_start - first d_
            d_end   - end _d
            columns - the target is a dataframe that has columns named d_

        Returns:
            data frame
    """  
    if(columns):
        cols = data.columns

        if "d_1" not in cols:
            return None
        if d_start not in cols:
            return None
        if d_end not in cols:
            return None

        first_d_col = data.columns.get_loc("d_1")
        d_first_col = data.columns.get_loc(d_start)
        d_end_col = data.columns.get_loc(d_end) + 1

        result = data.iloc[:, np.r_[0:first_d_col, d_first_col:d_end_col]]
        return result

    else:
        start = data.loc[data['d']==d_start].index
        end   = data.loc[data['d']==d_end].index
        print(data[start:end].head())
        return data[start:end]




#--------------------------------------------------------------------------------
def combine_data(calendar, prices, sample_sub, train_val_set):
    print("Combining data...")

#--------------------------------------------------------------------------------
#TODO: THIS ISNT GOOD
def split_data(data, pcnt_training):
    """ This function is responsible for split a dataset into training and 
        validation subsets.
    """
    cutoff = -1*pcnt_training;

    d_cols = [c for c in data.columns if 'd_' in c]
    
    train_data = data[d_cols[-100:cutoff]]
    val_data = data[d_cols[cutoff:]]

    return train_data, val_data


