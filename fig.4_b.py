import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_PATH = 'D:\\taskswitch100\\SCHZ\\Predication\\'
MODE_PATH = 'D:\\taskswitch100\\SCHZ\\'
DATA_PATH = os.path.join(BASE_PATH, 'sana_saps_score.xlsx')
MAT_PATH = os.path.join(MODE_PATH, 'corrected_modes.mat')
# MAT_PATH = os.path.join(MODE_PATH, 'pc.mat')
SAVE_PATH = 'D:\\taskswitch100\\SCHZ\\NomlizPred\\'

lr = LinearRegression()
# Load data
df = pd.read_excel(DATA_PATH, sheet_name='total')
sans = df['sans']
target = sans.to_numpy()

feature = scio.loadmat(MAT_PATH)['corrected_modes']
matrices = [feature[0, i] for i in range(7)]
feature = np.hstack(matrices)[111:161, :]
sum_matrix = np.sum(matrices, axis=0)
average_matrix = sum_matrix / len(matrices)
feature =average_matrix[111:161, :]
scaler = MinMaxScaler()
feature= scaler.fit_transform(feature)

# feature = scio.loadmat(MAT_PATH)['pc']
# feature = feature[111:161, :]
# scaler = MinMaxScaler()
# feature= scaler.fit_transform(feature)
print(feature.shape)

df = pd.read_csv('D:\\taskswitch100\\SCHZ\\SCH_HF.csv')
df = df[df['group'].isin(['SCHZ'])]
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
X_age_gender = df[['age','sex','reaction_time']]#'reaction_time'
# X_age_gender['age_sex_interaction'] = X_age_gender['age'] * X_age_gender['sex']

combined_features = np.hstack((feature, X_age_gender.values))


model = LinearRegression()
rfe = RFE(model, n_features_to_select=39)
fit = rfe.fit(combined_features, target)
selected_features = np.where(fit.support_)[0]
combined_selected_features = combined_features[:, selected_features]

pred_scores = np.zeros(np.size(target))
pred_coeff = []
for i in range(len(target)):
    lasso = LassoCV(cv=10, max_iter=20000, tol=10e-4, n_jobs=-1).fit(np.delete(combined_selected_features, [i], 0), np.delete(target, [i], 0))
    predicted = lasso.predict(np.array(combined_selected_features[i, :]).reshape(1, -1))[0]
    pred_scores[i] = predicted
    coef = lasso.coef_
    pred_coeff.append(coef)


pred_coeff = np.mean(np.array(pred_coeff), axis=0)
scaler = StandardScaler()
pred_coeff_standardized = scaler.fit_transform(pred_coeff.reshape(-1, 1))# Save normalized_pred_coeff as a CSV file
np.savetxt(os.path.join(SAVE_PATH, 'MRTsans_pred_coeff.csv'),pred_coeff_standardized, delimiter=',')
np.savetxt(os.path.join(SAVE_PATH, 'MRTsans_selected_features.csv'), selected_features, delimiter=',')
# # Evaluation
corr, p = pearsonr(pred_scores, target)
print(f'Correlation coefficient: {corr}, P-value: {p}')
fig, ax = plt.subplots()
ax.scatter(pred_scores, target, edgecolors=(0, 0, 0))
ax.set_xlabel('predicted')
ax.set_ylabel('score')
plt.show()

results_df = pd.DataFrame({'Pred_scores': pred_scores, 'Target':target})
output_csv_file = 'D:\\taskswitch100\\SCHZ\\NomlizPred\\MRTsans_results.csv'
results_df.to_csv(output_csv_file, index=False)
print(f'Results saved to {output_csv_file}')