import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import scipy.io as scio
import os
import matplotlib.pyplot as plt
BASE_PATH = 'D:\\taskswitch100\\SCHZ\\Predication\\'
MODE_PATH = 'D:\\taskswitch100\\SCHZ\\'
DATA_PATH = os.path.join(BASE_PATH, 'sana_saps_score.xlsx')
MAT_PATH = os.path.join(MODE_PATH, 'corrected_modes.mat')
SAVE_PATH = 'D:\\taskswitch100\\SCHZ\\NomlizPred\\'
# MAT_PATH = os.path.join(MODE_PATH, 'IN.mat')

# Load data
df = pd.read_excel(DATA_PATH, sheet_name='total')
sans = df['sans']
target = sans.to_numpy()

feature = scio.loadmat(MAT_PATH)['corrected_modes']
matrices = [feature[0, i] for i in range(7)]
sum_matrix = np.sum(matrices, axis=0)
average_matrix = sum_matrix / len(matrices)
feature = average_matrix[111:161, :]
scaler = MinMaxScaler()
feature= scaler.fit_transform(feature)

# feature = scio.loadmat(MAT_PATH)['IN']
# feature = feature[111:161,:]
# scaler = MinMaxScaler()
# feature= scaler.fit_transform(feature)

df = pd.read_csv('D:\\taskswitch100\\SCHZ\\SCH_HF.csv')
df = df[df['group'].isin(['SCHZ'])]
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
X_age_gender = df[['age', 'sex','reaction_time']]#'reaction_time'
combined_features = np.hstack((feature, X_age_gender.values))

# Set the range of n_features_to_select values
n_features_range = range(1, 103)
cv_scores = []

for n_features in n_features_range:
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features)
    combined_selected_features = rfe.fit_transform(combined_features, target)

    # Perform cross-validation with mean squared error scoring
    scores = cross_val_score(model, combined_selected_features, target, cv=10, scoring='neg_mean_squared_error')
    cv_scores.append(np.mean(scores))

# Find the optimal number of features
optimal_n_features = n_features_range[np.argmin(np.abs(cv_scores))]

# Plotting the cross-validated mean squared error against the number of selected features
plt.figure(figsize=(10, 6))
# Mark the minimum point as the optimal number of features
plt.plot(n_features_range, np.abs(cv_scores), marker='o', markersize=8, linestyle='-')
# plt.scatter(optimal_n_features, min(np.abs(cv_scores)), color='red', s=80, label='Optimal Number of Features')
# Adding labels and title
plt.xlabel('Number of Selected Features ')
# plt.ylabel('Cross-Validated Mean Squared Error')
plt.ylabel('Cross-Validation Score (Absolute Value)')
plt.title('Optimal Number of SANS  Features for Linear Regression')
# plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(SAVE_PATH, 'Feature SANS.png'))
print("Optimal number of features:", optimal_n_features)