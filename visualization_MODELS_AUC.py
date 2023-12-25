# Python
# -- coding: utf-8 --
# Author: Zhao Chang
# Created: 2023-10-3

import numpy as np
import pandas as pd
import scipy.io as scio
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


BASE_PATH =  "D:\\taskswitch100\\SCHZ" ###  you should change the path to your own path
results_PATH = os.path.join(BASE_PATH, 'single_RT_results','SCHZ_parameters')
DATA_PATH = os.path.join(BASE_PATH, 'SCH_HF.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'RT')

def data_preproce():

    files = os.listdir(results_PATH)
    best_scores = {}
    best_params = {}
    for file in files:
        model_name = file.split("_")[0]
        df = pd.read_csv(os.path.join(results_PATH, file))
        max_score = df['score'].max()
        best_params[model_name] = df[df['score'] == max_score].iloc[0, :-1].to_dict()
        best_scores[model_name] = max_score
    # single
    df = pd.read_csv(DATA_PATH)
    group = df['group'].map({'CONTROL':0,'SCHZ':1}).reset_index(drop=True)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    X_age_gender = df[['age', 'sex','reaction_time']]
    mode = scio.loadmat(os.path.join(BASE_PATH, 'single_mode.mat'))['HB']
    # Multi
    # df = pd.read_csv(DATA_PATH)
    # group = df['group'].map({'CONTROL': 0, 'SCHZ': 1}).reset_index(drop=True)
    # df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    # X_age_gender = df[['age', 'sex', 'reaction_time']]
    # mode =scio.loadmat(os.path.join(BASE_PATH, 'corrected_modes.mat'))['corrected_modes']
    # return best_params,best_scores,mode,X_age_gender,group
# single
def calculate_U(mode,X_age_gender):
    linreg = LinearRegression().fit(X_age_gender, mode)
    U_pred = linreg.predict(X_age_gender)
    U_residual = mode - U_pred
    return U_residual
# Multi
# def calculate_U(a, mode,X_age_gender):
#     U =  np.sum(mode[0] * np.array(a))
#     linreg = LinearRegression().fit(X_age_gender, U)
#     U_pred = linreg.predict(X_age_gender)
#     U_residual = U - U_pred
#     return U_residual

if __name__ == '__main__':

    best_params,best_scores,mode,X_age_gender,group= data_preproce()

    models = [
        (RandomForestClassifier(random_state=2023), {'n_estimators': best_params['RandomForestClassifier']['n_estimators'],
                                                      'max_depth': best_params['RandomForestClassifier']['max_depth']}),
        (SVC(random_state=2023, probability=True), {'C': best_params['SVC']['C'], 'gamma': best_params['SVC']['gamma']}),
        (KNeighborsClassifier(), {'n_neighbors': best_params['KNeighborsClassifier']['n_neighbors']}),
        (DecisionTreeClassifier(random_state=2023), {'max_depth': best_params['DecisionTreeClassifier']['max_depth']}),
        (LogisticRegression(random_state=2023, max_iter=1000), {'C': best_params['LogisticRegression']['C']}),
        (GradientBoostingClassifier(random_state=2023), {'n_estimators': best_params['GradientBoostingClassifier']['n_estimators'], 
                                                         'learning_rate': best_params['GradientBoostingClassifier']['learning_rate']}),
        (AdaBoostClassifier(random_state=2023), {'n_estimators':best_params['AdaBoostClassifier']['n_estimators'], 
                                                 'learning_rate': best_params['AdaBoostClassifier']['learning_rate']}),
        (MLPClassifier(random_state=2023, max_iter=1000), {'alpha': best_params['MLPClassifier']['alpha'], 
                                                           'learning_rate_init': best_params['MLPClassifier']['learning_rate_init']})
    ]

    model_names = []
    model_accuracies = []
    model_auc_scores = []


    plt.figure(figsize=(10, 6))
    
    for model, param_bounds in models:
        model_name = model.__class__.__name__
        model_names.append(model_name)

        params = best_params[model_name]
        model.class_weight = 'balanced'
        # a_keys = [f'a{i + 1}' for i in range(7)]
        # a = [params[key] for key in a_keys]
        # U = calculate_U(a,mode, X_age_gender)
        U = calculate_U(mode, X_age_gender)
        for param, value in params.items():
            if param in ['n_estimators', 'max_depth', 'n_neighbors']:
                value = int(value)
            setattr(model, param, value)
        cv = KFold(n_splits=10, shuffle=True, random_state=2023)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        accs=[]


        def auc_from_fold(train, test, model, U, group):
            probas_ = model.fit(U[train], group[train]).predict_proba(U[test])
            fpr, tpr, _ = roc_curve(group[test], probas_[:, 1])
            return auc(fpr, tpr)

        aucs=[]
        for train, test in cv.split(U, group):
            model.fit(U[train], group[train])
            y_prob = model.predict_proba(U[test])[:, 1]
            y_pred = model.predict(U[test])
            acc = accuracy_score(group[test],y_pred)
            accs.append(acc)
            fpr, tpr, _ = roc_curve(group[test], y_prob)

            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

        mean_tpr /= cv.get_n_splits()
        mean_tpr[-1] = 1.0
        roc_auc = auc(mean_fpr, mean_tpr)

        model_auc_scores.append(best_scores[model_name])
        model_accuracies.append(np.mean(accs))

        plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Models')
    plt.legend(loc='lower right')
    plt.figure(figsize=(5, 6))
    # plt.savefig(os.path.join(SAVE_PATH, 'compare_plot.png'))
    plt.show()


    data = {
    'Model': model_names,
    'Accuracy': model_accuracies,
    'AUC': model_auc_scores
}

    results_df = pd.DataFrame(data)
    results_csv_path = os.path.join(SAVE_PATH,'single_RT_performance.csv')
    results_df.to_csv(results_csv_path, index=False)

    print(f'Results saved to {results_csv_path}')