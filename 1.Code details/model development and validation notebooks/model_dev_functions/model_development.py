# Feature Engineering
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import pandas as pd


def base_model_pred(X_train, y_train, models):
    # Generate predictions from base models
    X = []  # This will store the prediction outputs of each model
    y = []  # This will store the true labels
    for model in models:
        y_pred = cross_val_predict(model, X_train, y_train, cv=5)
        X.append(y_pred)

    # Convert to a 2D array, where each row represents a sample and each column represents a model's prediction
    X = np.array(X).T
    y = y_train  # Flatten the true labels into a 1D array

    return X, y


def base_model_pred_for_submission(X_train, y_train, X_pred, models):
    # Generate predictions from base models
    X = []  # This will store the prediction outputs of each model
    y = []  # This will store the true labels
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        X.append(y_pred)

    # Convert to a 2D array, where each row represents a sample and each column represents a model's prediction
    X = np.array(X).T
    # y = y_train  # Flatten the true labels into a 1D array

    return X  # , y


def model_scores(X, y, scoring):

    # Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_model, X, y, cv=5)

    # Logistic Regression classifier
    lr_model = LogisticRegression(random_state=42, n_jobs=-1)
    lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring=scoring)

    # SVM classifier
    svm_model = SVC(kernel='linear', random_state=42)
    svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring=scoring)

    # XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring=scoring)

    # LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    lgb_scores = cross_val_score(lgb_model, X, y, cv=5, scoring=scoring)

    return rf_scores, lr_scores, svm_scores, xgb_scores, lgb_scores


def kf_split(df, y):
    # creaete kfold column and initialize with -1
    df['kfold'] = -1

    # initiate kfold class from model selection module
    kf = StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold

    return df


def run_cv(X, y, fold, clf):

    df = pd.concat([X, y], axis=1).reset_index(drop=True)

    df_train = df[df.kfold != fold]
    df_valid = df[df.kfold == fold]

    X_train = df_train.drop(y.name, axis=1).values
    X_valid = df_valid.drop(y.name, axis=1).values

    y_train = df_train[y.name].values
    y_valid = df_valid[y.name].values

    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)

    accuracy = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    report = classification_report(y_valid, preds)
    conf_matrix = confusion_matrix(y_valid, preds)

    print(f'accuracy= {accuracy}, f1-score= {f1}')

    return preds, report, conf_matrix


def df_to_arr(X, y):

    X = X.values
    y = y.values

    return X, y
