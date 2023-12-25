# Python
# ---- coding: UTF-8 ----
# Author: Zhao Chang
# Created: 2023-10-02
# Description: This script performs hyperparameter optimization for various machine learning classifiers about SCHZ project.


import numpy as np
import pandas as pd
import scipy.io as scio
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.text import MIMEText

BASE_PATH = "D:/taskswitch100/SCHZ" ###  you should change the path to your own path
DATA_PATH = os.path.join(BASE_PATH, 'SCH_HF.csv')
MODE_PATH = os.path.join(BASE_PATH, 'deg.mat')
SAVE_PATH = os.path.join(BASE_PATH, 'DEGnoRT_results')
Clus_PATH = os.path.join(BASE_PATH, 'schaefer_100')
mode = scio.loadmat(MODE_PATH)['deg']

def read_data(conditions):
    df = pd.read_csv(DATA_PATH)
    group = df['group'].map({condition: i for i, condition in enumerate(conditions)}).reset_index(drop=True)
    mode = scio.loadmat(MODE_PATH)['deg']
    mode = np.square(mode)
    subjects = df['subjects']
    return group, mode,subjects,df

# def correct_mode(mode,subjects):
#     corrected_modes = np.zeros_like(mode)
#     for idx, m in enumerate(mode[0]):
#         corrected_m = np.zeros_like(m)
#         num_subjects = m.shape[0]
#         for i in range(num_subjects):
#             subj = subjects[i]
#             Clus_data = scio.loadmat(os.path.join(Clus_PATH, f'{subj}_cluster_size.mat'))
#             Clus_size = Clus_data['Clus_size']
#             Clus_num = Clus_data['Clus_num']
#             N = np.size(Clus_num)
#             p = np.zeros(N)
#             indices = len(np.where(Clus_num < 1)[0])
#
#             for j in range(indices):
#                 p[j] = np.sum(np.abs(Clus_size[j][0] - (1 / Clus_num[0][j]))) / N
#
#             corrected_m[i, :] = m[i, :] * Clus_num * (1 - p)
#         corrected_modes[0, idx] = corrected_m
#         # scio.savemat(os.path.join(SAVE_PATH, 'corrected_modes.mat'), {'corrected_modes': np.array(corrected_modes)}
#     return corrected_modes

# def calculate_U(a, mode,X_age_gender):
#     U =  np.sum(mode[0] * np.array(a))
#     linreg = LinearRegression().fit(X_age_gender, U)
#     U_pred = linreg.predict(X_age_gender)
#     U_residual = U - U_pred
#     return U_residual

def calculate_U(mode,X_age_gender):
    linreg = LinearRegression().fit(X_age_gender, mode)
    U_pred = linreg.predict(X_age_gender)
    U_residual = mode - U_pred
    return U_residual

def classifier_evaluate(model, **params):
    model.class_weight = 'balanced'
    # a_keys = [f'a{i + 1}' for i in range(7)]
    # a = [params[key] for key in a_keys]
    # U = calculate_U(a, mode, X_age_gender)
    U = calculate_U(mode,X_age_gender)
    for param, value in params.items():
        if param in ['n_estimators', 'max_depth', 'n_neighbors']:
            value = int(value)
        setattr(model, param, value)
    cv = KFold(n_splits=10, shuffle=True, random_state=2023)
    aucs = []
    for train, test in cv.split(U, group):
        auc_score = auc_from_fold(train, test, model, U, group)
        aucs.append(auc_score)
    return np.mean(aucs)

def auc_from_fold(train, test, model, U, group):
    probas_ = model.fit(U[train], group[train]).predict_proba(U[test])
    fpr, tpr, _ = roc_curve(group[test], probas_[:, 1])
    return auc(fpr, tpr)

def optimize_params(model, param_bounds, a_bounds, i):
    param_bounds.update({'a' + str(i + 1): a_bounds for i in range(7)})
    opt = BayesianOptimization(lambda **params: classifier_evaluate(model, **params), param_bounds, random_state=i,allow_duplicate_points=True)
    opt.maximize(init_points=5, n_iter=200)
    return opt.max['params'], opt.max['target']

def parallel_optimization(i, model, param_bounds):
    params, score = optimize_params(model, param_bounds, a_bounds, i)
    return params, score

def save_best_params_to_csv(best_params_list, best_scores_list, model_name, folder_path):
    df = pd.DataFrame(best_params_list)
    df['score'] = best_scores_list
    csv_file = f"{model_name}_best_params.csv"
    csv_path = os.path.join(folder_path, csv_file)
    df.to_csv(csv_path, index=False)


def set_save_folder(conditions):
    folder_name = f"{conditions[1]}_parameters"
    save_folder_path = os.path.join(SAVE_PATH, folder_name)
    os.makedirs(save_folder_path, exist_ok=True)
    return save_folder_path

if __name__ == '__main__':

    conditions_list = [['CONTROL', 'SCHZ']]
    for conditions in conditions_list:
        group, mode_raw,subjects,df = read_data(conditions)
        save_folder_path = set_save_folder(conditions)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        df['sex'] = df['sex'].map({'M': 0, 'F': 1})
        X_age_gender = df[['age', 'sex']]
        # mode = correct_mode(mode_raw,subjects)

        models = [
            (RandomForestClassifier(random_state=2023), {'n_estimators': (10, 200), 'max_depth': (1, 20)}),
            (SVC(random_state=2023,probability=True), {'C': (0.1, 10), 'gamma': (0.01, 1)}),
            (KNeighborsClassifier(), {'n_neighbors': (1, 20)}),
            (DecisionTreeClassifier(random_state=2023), {'max_depth': (1, 20)}),
            (LogisticRegression(random_state=2023, max_iter=5000), {'C': (0.1, 10)}),
            (GradientBoostingClassifier(random_state=2023), {'n_estimators': (10, 200), 'learning_rate': (0.01, 1)}),
            (AdaBoostClassifier(random_state=2023), {'n_estimators': (10, 200), 'learning_rate': (0.01, 1)}),
            (MLPClassifier(random_state=2023, max_iter=5000), {'alpha': (0.0001, 0.1), 'learning_rate_init': (0.001, 0.1)})
        ]

        a_bounds = (0, 1)
        num_cores = multiprocessing.cpu_count()
        all_accuracies = []
        for model, param_bounds in models:
            model_name = model.__class__.__name__
            print(f'Optimizing {model_name}...')

            results = Parallel(n_jobs=num_cores)(
                delayed(parallel_optimization)(i, model, param_bounds) for i in range(100)
            )

            best_params_list, best_scores_list = zip(*results)
            save_best_params_to_csv(best_params_list, best_scores_list, model_name, folder_path=save_folder_path)

            best_idx = np.argmax(best_scores_list)
            best_params = best_params_list[best_idx]
            best_score = best_scores_list[best_idx]

            print(f'Best parameters for {model_name}: {best_params}')
            print(f'Best score for {model_name}: {best_score}')

