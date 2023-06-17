# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from functools import partial

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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

# initialize domain space for range of values, e.g.

# space = {'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
#          'gamma': hp.uniform('gamma', 1, 9),
#          'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
#          'reg_lambda': hp.quniform('reg_lambda', 0.01, 0.1, 1),
#          'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#          'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
#          'n_estimators': 180,
#          'seed': 0
#          }


def kf_split(df, y):
    # creaete kfold column and initialize with -1
    df['kfold'] = -1

    # initiate kfold class from model selection module
    kf = StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold

    return df


class hyperopt:

    def __init__(self, X, y, clf, **param_space):
        self.param_space = param_space
        self.clf = clf
        self.X = X
        self.y = y

    def optimize(self):

        self.clf.set_params(self.param_space)
        accuracies = []

        for fold in range(5):
            df = pd.concat([self.X, self.y], axis=1).reset_index(drop=True)

            df_train = df[df.kfold != fold]
            df_valid = df[df.kfold == fold]

            X_train = df_train.drop(self.y.name, axis=1).values
            X_valid = df_valid.drop(self.y.name, axis=1).values

            y_train = df_train[self.y.name].values
            y_valid = df_valid[self.y.name].values

            self.clf.fit(X_train, y_train)

            preds = self.clf.predict(X_valid)

            fold_accuracy = accuracy_score(y_valid, preds)
            accuracies.append(fold_accuracy)

        return -1 * np.mean(accuracies)

    def get_best_params(self, max_eval):
        optimization_function = partial(self.optimize, X=self.X, y=self.y)

        trials = Trials()
        best_hyperparams = fmin(fn=optimization_function,
                                space=self.param_space,
                                algo=tpe.suggest,
                                max_evals=max_eval,
                                trials=trials)

        return best_hyperparams


#optimization_function = partial(optimize, X=X_mean_fad_df_split, y=y_mean_0209)
#
#trials = Trials()
#
# best_hyperparams = fmin(fn=optimization_function,
#                        space=space,
#                        algo=tpe.suggest,
#                        max_evals=100,
#                        trials=trials)
#
# print(best_hyperparams)
#
#best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
# best_hyperparams['gamma'] = 6.5   # max gamma
