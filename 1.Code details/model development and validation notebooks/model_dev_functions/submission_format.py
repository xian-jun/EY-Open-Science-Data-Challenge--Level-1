import pandas as pd
import numpy as np


def prediction_to_submission_df(path, pred_arr):

    df = pd.read_csv(path)
    df.rename(columns={'Latitude and Longitude': 'id'}, inplace=True)
    df['target'] = pred_arr

    int_to_label = {1: 'Rice', 0: 'Non Rice'}
    df['target'] = df['target'].map(int_to_label)

    return df
