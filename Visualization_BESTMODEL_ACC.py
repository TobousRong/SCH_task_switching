import os
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


BASE_PATH = "D:\\taskswitch100\\SCHZ\\"
RESULTS_PATH = os.path.join(BASE_PATH, 'PCRT_results', 'SCHZ_parameters')
DATA_PATH = os.path.join(BASE_PATH, 'SCH_HF.csv')

def load_data():
    files = os.listdir(RESULTS_PATH)
    best_scores, best_params = {}, {}
    for file in files:
        model_name = file.split("_")[0]
        df = pd.read_csv(os.path.join(RESULTS_PATH, file))
        max_score = df['score'].max()
        best_params[model_name] = df[df['score'] == max_score].iloc[0, :-1].to_dict()
        best_scores[model_name] = max_score
    
    df = pd.read_csv(DATA_PATH)
    df['group'] = df['group'].map({'CONTROL': 0, 'SCHZ': 1})
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    X_age_gender = df[['age', 'sex','reaction_time']]
    group = df['group']
    # mode = scio.loadmat(os.path.join(BASE_PATH, 'corrected_modes.mat'))['corrected_modes']
    mode = scio.loadmat(os.path.join(BASE_PATH, 'pc.mat'))['pc']
    return best_params, best_scores, mode, X_age_gender, group
# Multi
# def calculate_U(a,mode, X_age_gender):
#     U = np.sum(mode[0] * np.array(a))
#     linreg = LinearRegression().fit(X_age_gender, U)
#     U_pred = linreg.predict(X_age_gender)
#     U_residual= U - U_pred
#     return  U_residual
# Single
def calculate_U(mode,X_age_gender):
    linreg = LinearRegression().fit(X_age_gender, mode)
    U_pred = linreg.predict(X_age_gender)
    U_residual = mode - U_pred
    return U_residual
def calculate_metrics(true_labels, predicted_labels):
    TP, FN, FP, TN = confusion_matrix(true_labels, predicted_labels).ravel()
    return TP / (TP + FN), TN / (TN + FP), (TP + TN) / (TP + TN + FP + FN)


def plot_auc_curve(mean_fpr, mean_tpr, roc_auc, save_path):
    
    plt.plot(mean_fpr, mean_tpr, color="black", label=f'Mean AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--',alpha=0.7, color='#6b7c87')
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.legend(loc='lower right',fontsize = 7)
    # plt.savefig(os.path.join(save_path, 'RTAUC_plot.png'))
    plt.show()

def plot_metrics(sens_list,spec_list,acc_list, save_path):
    labels = ['Sensitivity', 'Specificity', 'Accuracy']
    # labels = ['Precision','Recall','Accuracy']
    # means = [np.mean(prc_list), np.mean(recall_list), np.mean(acc_list)]
    means = [np.mean(sens_list), np.mean(spec_list), np.mean(acc_list)]
    plt.figure(figsize=(6, 6), dpi=300)
    plt.bar(labels, means, color=['SkyBlue', 'LightskyBlue', 'SteelBlue'])
    for index, value in enumerate(means):
        plt.text(index, value, f'{value:.2f}')
    
    plt.ylabel('Metric Value')
    # plt.savefig(os.path.join(save_path, 'RTMetrics_plot.png'))
    plt.show()

def main():
    best_params, _, mode, X_age_gender, group = load_data()
    # a_keys = [f'a{i+1}' for i in range(7)]
    # a = [best_params['RandomForestClassifier'][key] for key in a_keys]
    # U = calculate_U(a,mode, X_age_gender)
    U = calculate_U(mode, X_age_gender)
    # model = GradientBoostingClassifier(random_state=2023,
    #                                    n_estimators=int(best_params['GradientBoostingClassifier']['n_estimators']),
    #                                    learning_rate=best_params['GradientBoostingClassifier']['learning_rate'])
    model = RandomForestClassifier(random_state=2023, n_estimators=int(best_params['RandomForestClassifier']['n_estimators']),
                                                      max_depth= int(best_params['RandomForestClassifier']['max_depth']))
    # model =AdaBoostClassifier(random_state=2023, n_estimators=int(best_params['AdaBoostClassifier']['n_estimators']),
    #                                              learning_rate=best_params['AdaBoostClassifier']['learning_rate'])
    # model = MLPClassifier(random_state=2023,max_iter=1000,
    #                                    alpha=int(best_params['MLPClassifier']['alpha']),
    #                                    learning_rate_init=best_params['MLPClassifier']['learning_rate_init'])
    cv = KFold(n_splits=10, shuffle=True, random_state=2023)


    acc_list ,prc_list,recall_list,sens_list,spec_list= [], [], [],[],[]
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(6, 6), dpi=300)
    

    for i, (train, test) in enumerate(cv.split(U, group)):
        model.fit(U[train], group[train])
        y_prob = model.predict_proba(U[test])[:, 1]
        fpr, tpr, _ = roc_curve(group[test], y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linestyle='--', alpha=0.7, label=f'Fold{i+1} (AUC={roc_auc:.3f})')
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

        y_pred = model.predict(U[test])
        TN, FP, FN, TP = confusion_matrix(group[test], y_pred).ravel()
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        acc = (TP + TN) / (TP + TN + FP + FN)
        # prc=(TP)/(TP+FP)
        # recall=(TP)/(TP+FN)
        sens_list.append(sens)
        spec_list.append(spec)
        # prc_list.append(prc)
        # recall_list.append(recall)
        acc_list.append(acc)


    mean_tpr /= cv.get_n_splits()
    mean_tpr[-1] = 1.0
    mean_tpr = np.concatenate([[0], mean_tpr])
    mean_fpr = np.concatenate([[0], mean_fpr])
    roc_auc = auc(mean_fpr, mean_tpr)

    save_path = "D:\\taskswitch100\\SCHZ\\DEGRT_resultsRT"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    plot_auc_curve(mean_fpr, mean_tpr, roc_auc, save_path)
    # plot_metrics(prc_list, recall_list, acc_list, save_path)
    plot_metrics(sens_list, spec_list, acc_list, save_path)


if __name__ == "__main__":
    main()
    
