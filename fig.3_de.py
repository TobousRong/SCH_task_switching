import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

BASE_PATH = "D:\\taskswitch100\\SCHZ"  ###  you should change the path to your own path
results_PATH = os.path.join(BASE_PATH, 'RT_results', 'SCHZ_parameters')
DATA_PATH = os.path.join(BASE_PATH, 'SCH_HF.csv')
SAVE_PATH = os.path.join(BASE_PATH, 'RT')
Clus_PATH ="D:\\taskswitch100\\SCHZ\\corrected_modes.mat"
# network = pd.read_excel('D:\\taskswitch100\\Thomas_roi100.xlsx')
region_num = []
for j in range(100):
    list = str(f'region{j + 1}')
    region_num.append(list)
# network['region_num'] = region_num

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

    df = pd.read_csv(DATA_PATH)
    group = df['group'].map({'CONTROL': 0, 'SCHZ': 1}).reset_index(drop=True)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    X_age_gender = df[['age', 'sex','reaction_time']]
    mode =scio.loadmat(os.path.join(BASE_PATH, 'corrected_modes.mat'))['corrected_modes']
    return best_params,best_scores,mode,X_age_gender,group

def calculate_U(a, mode,X_age_gender):
    U =  np.sum(mode[0] * np.array(a))
    scaler = MinMaxScaler()
    U = scaler.fit_transform(U)
    linreg = LinearRegression().fit(X_age_gender, U)
    U_pred = linreg.predict(X_age_gender)
    U_residual = U - U_pred
    return U_residual

if __name__ == '__main__':

    best_params, best_scores, mode, X_age_gender, group = data_preproce()

    models = [
         (RandomForestClassifier(random_state=2023),{'n_estimators': best_params['RandomForestClassifier']['n_estimators'],
                                                   'max_depth': best_params['RandomForestClassifier']['max_depth']}),
        # (AdaBoostClassifier(random_state=2023), {'n_estimators':best_params['AdaBoostClassifier']['n_estimators'],
        #                                          'learning_rate': best_params['AdaBoostClassifier']['learning_rate']}),
    ]

    model_names = []
    model_accuracies = []
    model_auc_scores = []



    for model, param_bounds in models:
        model_name = model.__class__.__name__
        model_names.append(model_name)
        params = best_params[model_name]
        model.class_weight = 'balanced'
        a_keys = [f'a{i + 1}' for i in range(7)]
        a = [params[key] for key in a_keys]
        U = calculate_U(a, mode,X_age_gender)
        for param, value in params.items():
            if param in ['n_estimators', 'max_depth', 'n_neighbors']:
                value = int(value)
            setattr(model, param, value)
        cv = KFold(n_splits=10, shuffle=True, random_state=2023)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        accs = []
        Ten_importance = np.zeros([1,100])
        for train, test in cv.split(U, group):
            model.fit(U[train], group[train])
            importance = model.feature_importances_
            Ten_importance += importance
        Ten_importance=Ten_importance/10
        a = pd.DataFrame(Ten_importance).T

        # for train, test in cv.split(U, group):
        #     model.fit(U[train], group[train])
        # importance = model.feature_importances_
        # a = pd.DataFrame(importance)
        a.columns = ["importance"]
        a['region'] = region_num
        a = a.sort_values(by='importance', ascending=False).reset_index()
        a['sort'] = a.index + 1  # 索引变列
        region = a['region'][:10]
        impor = np.array(a['importance'][:10])
        plt.figure(figsize=(12, 8), dpi=300)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.bar(region, impor,0.7,color='LightskyBlue')
        # plt.title('ABC特征重要性排序', {'size': 20})
        plt.ylabel('Importance', {'size': 20})
        plt.xlabel('Regions', {'size': 20})
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        # plt.savefig(os.path.join(SAVE_PATH, 'RTFeature Importance_RF.png'))
        plt.show()
        a.to_csv(os.path.join(SAVE_PATH, 'MRTfeature_importance.csv'), index=False)
        # else:
        #         importance = model.feature_importances_
        #         a = pd.DataFrame(importance)
        #         a.columns = ["importance"]
        #         a['region'] = network['abb Name']
        #         a = a.sort_values(by='importance', ascending=False).reset_index()
        #         a['sort'] = a.index + 1  # 索引变列
        #         region = a['region'][:10]
        #         impor = np.array(a['importance'][:10])
        #         plt.figure(figsize=(12, 8), dpi=300)
        #         plt.rcParams['font.family'] = ['sans-serif']
        #         plt.rcParams['font.sans-serif'] = ['SimHei']
        #         plt.bar(region, impor,0.7,color="#C9AE74")
        #         plt.title('ABC特征重要性排序', {'size': 20})
        #         plt.ylabel('重要性', {'size': 18})
        #         plt.yticks(fontsize=14)
        #         plt.xticks(fontsize=14)
        #         plt.savefig(os.path.join(SAVE_PATH, 'Feature Importance_ABC.png'))
        #         plt.show()






