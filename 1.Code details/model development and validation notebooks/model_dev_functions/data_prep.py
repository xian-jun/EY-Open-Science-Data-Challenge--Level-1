# Supress Warnings
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import glob
from scipy import stats
from tqdm import tqdm
from itertools import cycle
import requests
import planetary_computer as pc
import re
import pandas as pd
import numpy as np


def read_multiple_pickles(folder_path, cols_to_drop, filter_condition=None):
    if filter_condition is None:
        pkl_files = glob.glob(folder_path + "/*.pkl")
    else:
        pkl_files = [file for file in glob.glob(
            folder_path + "/*.pkl") if filter_condition not in file]
    df_list = [pd.read_pickle(file).drop(cols_to_drop, axis=1)
               for file in pkl_files]
    return pkl_files, df_list


def get_all_one_point_bands(full_df, label_col,
                            label_encode={'Rice': 1, 'Non Rice': 0}):
    '''
    all bands without windows retrieved by extracting the pixel value directly
    '''
    X = full_df[['B02', 'B03', 'B04', 'B08', 'B05', 'B06',
                 'B07', 'B11', 'B12', 'B8A', 'SCL', 'B01', 'B09']]
    y = full_df[label_col].map(label_encode)

    return X, y


def get_aggregation_from_window(full_df, window_suffix, label_col, suffix_date, agg_method,
                                label_encode={'Rice': 1, 'Non Rice': 0}):
    '''
    aggregate the w5 features to get a single value and return only these features

    parameters:
    full_df = the pickle file read to be processed
    window_suffix = the window name suffix at the columns with windows, e.g. '_w5', '_w10'
    label_col = 'Class of Land'
    suffix_date = date added to the processed columns as suffix to prevent column name clash after concatenating multiple dates
    agg_method = aggregation method, e.g. lambda x: x.mean(), mode(), median(), etc.

    return:
    X, y 
    '''

    windowed_columns = [
        band for band in full_df.columns if window_suffix in band]

    X = full_df[windowed_columns].applymap(agg_method)
    X = X.add_suffix(suffix_date)

    if label_col:
        y = full_df[label_col].map(label_encode)
    else:
        y = None

    return X, y


def batch_aggregate_pickle(df_list, file_paths, window_suffix,
                           label_colname='Class of Land', agg_method=lambda x: x.mean()):
    '''
    batch load multiple pkl files and aggregate the windows we target

    parameters:
    df_list = a list of dfs opened from pkl file paths
    file_paths = the pkl file paths, used to extract date suffix
    window_suffix = the window name suffix at the columns with windows, e.g. '_w5', '_w10'
    label_column = the column name of target feature. if there's no target feature, put `None`

    return:
    dfs_combned = w5 features from the dataframes aggregated and data from multiple dates combined
    list_of_dfs = a list of extracted dfs (useful for various feature engineering)
    '''
    dfs_combined = pd.DataFrame()
    list_of_dfs = []

    for i in range(len(df_list)):

        # get suffix to add to each column to mark the date
        pattern = r'-\d{2}-\d{2}'
        date_string = re.search(pattern, file_paths[i]).group()
        suffix = f'_{date_string[1:3]}{date_string[4:]}'

        # aggregate the windows and prepare dataframe
        X_agg, y = get_aggregation_from_window(
            df_list[i], window_suffix, label_colname, suffix, agg_method)
        dfs_combined = pd.concat([dfs_combined, X_agg], axis=1)
        list_of_dfs.append(X_agg)

    # add label data back to the dfs_combined
    if y is not None:
        dfs_combined = pd.concat([dfs_combined, y], axis=1)

    return dfs_combined, list_of_dfs


def prediction_to_submission_df(path, pred_arr):
    '''
    Change the prediction array into submission df format

    params:
    path = the file path of submission template
    pred_arr = prediction array
    '''

    df = pd.read_csv(path)
    df.rename(columns={'Latitude and Longitude': 'id'}, inplace=True)
    df['target'] = pred_arr

    int_to_label = {1: 'Rice', 0: 'Non Rice'}
    df['target'] = df['target'].map(int_to_label)

    return df
