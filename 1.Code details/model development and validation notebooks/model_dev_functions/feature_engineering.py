import re
import glob
from scipy import stats
from tqdm import tqdm
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def add_vhvv(s1_dfs_list_to_add_vhvv):
    '''
    add vh/vv to ratio to a list of sentinel-1 dfs

    parameters:
    s1_dfs_list_to_add_vhvv = list of df from S1 before adding vh/vv

    return:
    X_comb_vhvv = feature with vh/vv for each date 
    '''
    # prevent change to original df
    list_of_dfs_to_add_vhvv_copy = [i.copy() for i in s1_dfs_list_to_add_vhvv]

    for i, data in enumerate(list_of_dfs_to_add_vhvv_copy):
        # index to extract the value from index string
        vh = data.filter(like='vh').columns[0]
        vv = data.filter(like='vv').columns[0]
        data['vh/vv'] = data[vh] / data[vv]
        data.rename(columns={'vh/vv': f'vh/vv_{i}'}, inplace=True)

    X_comb_vhvv = pd.concat(list_of_dfs_to_add_vhvv_copy, axis=1)

    return X_comb_vhvv


def add_NDVI(list_of_dfs_to_add_NDVI):
    '''
    add NDVI to each date in a list of Sentinel-2 dfs
    '''

    # prevent change to original df
    list_of_dfs_to_add_NDVI_copy = [i.copy() for i in list_of_dfs_to_add_NDVI]

    for i, data in enumerate(list_of_dfs_to_add_NDVI_copy):
        # print(data)
        # index to extract the value from index string
        RED = data.filter(like='B04').columns[0]
        NIR = data.filter(like='B08').columns[0]
        data['NDVI'] = (data[RED] - data[NIR]) / (data[RED] + data[NIR])
        data.rename(columns={'NDVI': f'NDVI_{i}'}, inplace=True)

    X_comb_NDVI = pd.concat(list_of_dfs_to_add_NDVI_copy, axis=1)

    return X_comb_NDVI


def get_high_corr_cols(df, threshold):
    '''
    Get highly correlated columns for feature selection

    parameters:
    df = dataframe to feature select from
    threshold = 0-1, the threshold of correlation of which features with correlation higher than that will be dropped
    '''

    corr_df = df.corr()
    high_corr_cols = set()

    for i in range(len(corr_df.columns)):
        for j in range(i):
            if abs(corr_df.iloc[i, j]) > threshold:
                colname = corr_df.columns[i]
                high_corr_cols.add(colname)

    print(f'number of high_corr_cols: {len(high_corr_cols)}')

    return high_corr_cols


def extract_RGB_NIR(s2_df_list):
    '''
    extract RGB and NIR from s2 list of dfs
    '''
    # prevent change to original df
    s2_df_list_copy = [i.copy() for i in s2_df_list]

    RGB_NIR_cols = []

    for i, data in enumerate(s2_df_list_copy):  # extract the column names
        # print(data)
        # index to extract the value from index string
        RED = data.filter(like='B04').columns[0]
        NIR = data.filter(like='B08').columns[0]
        GREEN = data.filter(like='B03').columns[0]
        BLUE = data.filter(like='B02').columns[0]

        extracted_cols = data.loc[:, [RED, GREEN, BLUE, NIR]]
        RGB_NIR_cols.append(extracted_cols)

    X_comb = pd.concat(RGB_NIR_cols, axis=1)

    return X_comb
